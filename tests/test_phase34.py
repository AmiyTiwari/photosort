"""Phase 3 (people, events, questions, custom rules) and phase 4 (provider-neutral agent)."""
import hashlib
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_pipeline import save, scene

from photosort import questions as qs
from photosort.agent import Agent
from photosort.api import create_app
from photosort.library import Library
from photosort.llm import AnthropicMessages, OpenAICompatible, Reply

RNG = np.random.default_rng(7)
IDENTITY = {name: RNG.normal(size=128).astype(np.float32) for name in ("alice", "bob", "stranger")}


def face(name):
    e = IDENTITY[name] + RNG.normal(scale=0.25, size=128).astype(np.float32)
    return e / np.linalg.norm(e)


class FakeFaces:
    """Stands in for the ONNX models: who is in each file is decided by the test."""

    def __init__(self):
        self.by_sha: dict[str, list[str]] = {}

    def available(self):
        return True

    def register(self, path, people):
        self.by_sha[hashlib.sha256(path.read_bytes()).hexdigest()] = people

    def detect(self, path):
        names = self.by_sha.get(path.stem, [])
        return [{"x": 0.1 * i, "y": 0.1, "w": 0.2, "h": 0.2, "score": 0.95, "embedding": face(n), "crop": b"jpg"}
                for i, n in enumerate(names)]


def shoot(engine, folder, name, seed, when, people):
    p = folder / name
    save(scene(seed), p, when)
    engine.register(p, people)
    return p


@pytest.fixture
def setup(tmp_path):
    engine = FakeFaces()
    lib = Library(tmp_path / "library", workers=4, face_engine=engine)
    trip = tmp_path / "phone"
    # 9 photos on one day (an event), alice in all, bob in some, one stranger.
    for i in range(9):
        who = ["alice"] + (["bob"] if i % 2 == 0 else [])
        shoot(engine, trip, f"IMG_{i}.jpg", 100 + i, f"2025:12:20 1{i}:00:00", who)
    shoot(engine, trip, "IMG_s.jpg", 200, "2025:12:20 19:30:00", ["stranger"])
    # A second, separate day: 5 photos (small event).
    for i in range(5):
        shoot(engine, trip, f"JAN_{i}.jpg", 300 + i, f"2026:01:10 1{i}:00:00", [])
    return lib, engine, tmp_path


def people(lib):
    c = lib.catalog.connect()
    return {r["label"]: (r["n"], r["named"]) for r in c.execute(
        "SELECT v.label, p.named, COUNT(f.id) n FROM people p JOIN factor_values v ON v.id=p.value_id "
        "LEFT JOIN faces f ON f.value_id=p.value_id GROUP BY p.value_id")}


def test_people_events_and_questions(setup):
    lib, engine, tmp = setup
    b = lib.import_sync(str(tmp / "phone"))
    assert b["status"] == "done" and b["faces_found"] == 15
    counts = sorted(n for n, _ in people(lib).values())
    assert counts == [1, 5, 9]  # stranger, bob, alice

    c = lib.catalog.connect()
    events = c.execute("SELECT n FROM events ORDER BY start_at").fetchall()
    assert [e["n"] for e in events] == [10, 5]

    open_q = qs.list_open(c)
    kinds = [q["kind"] for q in open_q]
    assert kinds.count("name_person") == 2 and "name_event" in kinds
    assert open_q[0]["kind"] == "name_person" and open_q[0]["payload"]["photos"] == 9  # biggest first


def test_names_anchor_future_imports(setup):
    lib, engine, tmp = setup
    lib.import_sync(str(tmp / "phone"))
    c = lib.catalog.connect()
    q = next(q for q in qs.list_open(c) if q["kind"] == "name_person" and q["payload"]["photos"] == 9)
    qs.answer(lib, c, q["id"], {"name": "Alice"})
    ev_q = next(q for q in qs.list_open(c) if q["kind"] == "name_event")
    qs.answer(lib, c, ev_q["id"], {"name": "Goa trip"})
    alice = c.execute("SELECT id FROM factor_values WHERE label='Alice'").fetchone()[0]

    rule_value = lib.create_custom_value("Family", "Alice in winter", {"person": "Alice", "months": [12, 1]})
    assert rule_value["photos"] == 9

    # Later import from another device: alice again, inside the Goa trip window, plus a new day.
    later = tmp / "friend"
    shoot(engine, later, "F1.jpg", 400, "2025:12:20 18:30:00", ["alice"])
    shoot(engine, later, "F2.jpg", 401, "2026:01:10 16:00:00", ["alice", "bob"])
    lib.import_sync(str(later))

    assert people(lib)["Alice"] == (11, 1)  # joined the named cluster, no new person
    goa = c.execute("SELECT v.label, e.n FROM events e JOIN factor_values v ON v.id=e.value_id "
                    "WHERE e.named=1").fetchone()
    assert tuple(goa) == ("Goa trip", 11)  # same event kept its name and grew
    n = c.execute("SELECT COUNT(*) FROM photo_factor WHERE value_id=?", (rule_value["value_id"],)).fetchone()[0]
    assert n == 11  # custom rule re-applied to the new import
    assert c.execute("SELECT COUNT(*) FROM photo_factor WHERE value_id=?", (alice,)).fetchone()[0] == 11

    # Renaming a person does not break the saved rule (it stores the id).
    lib.name_person(alice, "Alice M")
    assert lib.search({"value_ids": [rule_value["value_id"]]})["total"] == 11


def test_reject_and_merge(setup):
    lib, engine, tmp = setup
    lib.import_sync(str(tmp / "phone"))
    c = lib.catalog.connect()
    bob = c.execute("SELECT value_id FROM faces GROUP BY value_id HAVING COUNT(*)=5").fetchone()[0]
    face_id = c.execute("SELECT id FROM faces WHERE value_id=? LIMIT 1", (bob,)).fetchone()[0]
    lib.name_person(bob, "Bob")
    lib.reject_face(face_id)
    assert c.execute("SELECT value_id FROM faces WHERE id=?", (face_id,)).fetchone()[0] != bob
    assert c.execute("SELECT COUNT(*) FROM face_negatives").fetchone()[0] == 1

    # Naming another cluster "bob" merges it into Bob.
    other = c.execute("SELECT value_id FROM faces WHERE id=?", (face_id,)).fetchone()[0]
    assert lib.name_person(other, "bob") == bob


class ScriptedLLM:
    """A fake model: returns canned replies and records what it was sent."""

    name = "scripted"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, system, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


def test_agent_tool_loop(setup):
    lib, engine, tmp = setup
    lib.import_sync(str(tmp / "phone"))
    c = lib.catalog.connect()
    big = c.execute("SELECT value_id FROM faces GROUP BY value_id ORDER BY COUNT(*) DESC LIMIT 1").fetchone()[0]

    llm = ScriptedLLM([
        Reply(tool_calls=[{"id": "1", "name": "list_people", "arguments": {}}]),
        Reply(tool_calls=[{"id": "2", "name": "name_person", "arguments": {"person_id": big, "name": "Alice"}},
                          {"id": "3", "name": "show", "arguments": {"person_ids": [big]}},
                          {"id": "4", "name": "offer_choices", "arguments": {"options": ["Yes", "No"]}}]),
        Reply(content="Named Alice. Want a Family factor?"),
    ])
    out = Agent(lib, llm).run("the person in most photos is Alice")
    assert out["reply"].startswith("Named Alice") and out["choices"] == ["Yes", "No"] and out["changed"]
    assert out["faces"][0]["name"] == "Alice"
    assert people(lib)["Alice"] == (9, 1)
    tool_msgs = [m for m in llm.seen[-1] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["people"][0]["photos"] == 9

    # Bad tool input is reported back to the model, not raised.
    llm2 = ScriptedLLM([
        Reply(tool_calls=[{"id": "5", "name": "find_photos", "arguments": {"rule": {"nope": 1}}}]),
        Reply(content="Sorry."),
    ])
    Agent(lib, llm2).run("find")
    err = json.loads([m for m in llm2.seen[-1] if m["role"] == "tool"][-1]["content"])
    assert "unknown rule keys" in err["error"]


def test_agent_without_model(setup):
    lib, _, _ = setup
    assert Agent(lib, None).run("hi")["error"] == "not_configured"


HISTORY = [
    {"role": "user", "content": "who is this?", "images": [{"media_type": "image/jpeg", "data": "QUJD"}]},
    {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "c1", "name": "list_people", "arguments": {}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "list_people", "content": "{\"people\": []}",
     "images": [{"media_type": "image/jpeg", "data": "REVG"}]},
]
TOOLS = [{"name": "list_people", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def test_openai_adapter():
    sent = {}

    def handler(req: httpx.Request):
        sent["url"], sent["auth"], sent["body"] = str(req.url), req.headers.get("authorization"), json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "list_people", "arguments": "{\"limit\": 3}"}}]}}]})

    llm = OpenAICompatible("http://local/v1/", "some-model", "k", transport=httpx.MockTransport(handler))
    r = llm.chat("sys", HISTORY, TOOLS)
    assert sent["url"] == "http://local/v1/chat/completions" and sent["auth"] == "Bearer k"
    roles = [m["role"] for m in sent["body"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]  # tool images follow the tool results
    assert sent["body"]["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert sent["body"]["tools"][0]["function"]["name"] == "list_people"
    assert r.tool_calls == [{"id": "c2", "name": "list_people", "arguments": {"limit": 3}}]


def test_anthropic_adapter():
    sent = {}

    def handler(req: httpx.Request):
        sent["headers"], sent["body"] = req.headers, json.loads(req.content)
        return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "t9", "name": "list_people", "input": {"limit": 2}}]})

    llm = AnthropicMessages("", "some-model", "k", transport=httpx.MockTransport(handler))
    r = llm.chat("sys", HISTORY, TOOLS)
    body = sent["body"]
    assert sent["headers"]["x-api-key"] == "k" and body["system"] == "sys" and body["max_tokens"] > 0
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert body["messages"][2]["content"][0]["content"][1]["type"] == "image"
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert r.content == "Let me look." and r.tool_calls[0]["arguments"] == {"limit": 2}
    # Raw content is replayed unchanged on the next turn to the same provider.
    replay = AnthropicMessages.to_wire([{"role": "user", "content": "x"}, r.as_message()])
    assert replay[1]["content"] == r.raw


def test_api_phase34(setup):
    lib, engine, tmp = setup

    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = TestClient(create_app(lib, llm_transport=httpx.MockTransport(handler)))
    lib.import_sync(str(tmp / "phone"))

    ppl = client.get("/api/people").json()
    assert [p["photos"] for p in ppl] == [9, 5]
    assert client.post(f"/api/people/{ppl[0]['id']}/name", json={"name": "Alice"}).status_code == 200
    assert client.get("/api/people").json()[0]["name"] == "Alice"
    qlist = client.get("/api/questions").json()
    ev = next(q for q in qlist if q["kind"] == "name_event")
    assert client.post(f"/api/questions/{ev['id']}/answer", json={"answer": {"name": "Goa"}}).status_code == 200
    factors = {f["key"]: f for f in client.get("/api/factors").json()}
    assert [v["label"] for v in factors["people"]["values"]][0] == "Alice"
    assert "Goa" in [v["label"] for v in factors["events"]["values"]]
    assert client.post("/api/custom", json={"factor": "Family", "value": "A", "rule": {"bad": 1}}).status_code == 400
    assert client.post("/api/custom", json={"factor": "Family", "value": "A", "rule": {"person": "Alice"}}).json()["photos"] == 9

    assert client.post("/api/chat", json={"message": "hi"}).json()["error"] == "not_configured"
    r = client.post("/api/llm", json={"preset": "ollama", "model": "local-model"}).json()
    assert r["config"]["base_url"] == "http://localhost:11434/v1" and not r["key_set"]
    assert client.post("/api/llm", json={"preset": "openai", "model": "m", "api_key": "secret"}).json()["key_set"]
    assert "secret" not in json.dumps(client.get("/api/llm").json())
    assert oct((lib.root / "secrets.json").stat().st_mode & 0o777) == "0o600"
    assert client.post("/api/llm/test").json()["reply"] == "ok"
    assert client.post("/api/chat", json={"message": "hi"}).json()["reply"] == "ok"
    assert [m["role"] for m in client.get("/api/chat").json()] == ["user", "assistant"]
