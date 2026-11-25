"""Phase 5: content groups, learning from answers, suggestions, text search, CLIP near-dups, place names."""
import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_pipeline import save, scene

from photosort import questions as qs
from photosort.agent import Agent
from photosort.api import create_app
from photosort.learning import NEGATIVE_PROMPTS
from photosort.library import Library
from photosort.llm import Reply

RNG = np.random.default_rng(3)
DIRS = {k: RNG.normal(size=512) for k in ("beach", "food", "street", "person", "generic")}
DIRS = {k: v / np.linalg.norm(v) for k, v in DIRS.items()}
# Like real CLIP, every photo partly matches the generic prompt "a photo".
TEXT = {"a photo of a beach": "beach", "food": "food", "a plate of food": "food",
        "a photo of a person": "person", "a photo of a street": "street", "a photo": "generic"}


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeClip:
    def __init__(self):
        self.by_sha = {}

    def available(self):
        return True

    def register(self, path, vec):
        self.by_sha[hashlib.sha256(path.read_bytes()).hexdigest()] = unit(unit(vec) + 0.6 * DIRS["generic"])

    def embed_images(self, paths):
        return np.stack([self.by_sha.get(p.stem, unit(RNG.normal(size=512))) for p in paths])

    def embed_texts(self, texts):
        out = []
        for t in texts:
            if t in TEXT:
                out.append(DIRS[TEXT[t]])
            else:  # other negatives: unrelated random directions
                out.append(np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(size=512))
        return np.stack([unit(v) for v in out])


class FakePlaces:
    def available(self):
        return True

    def lookup(self, lat, lon):
        if 15 < lat < 16:
            return {"country": ("IN", "India"), "region": ("IN.33", "Goa"), "city": ("IN.33.1", "Calangute")}
        return {"country": ("IN", "India"), "region": ("IN.16", "Maharashtra"), "city": None}


class NoFaces:
    def available(self):
        return False


def noise(scale=0.15, _rng=[None]):
    if _rng[0] is None or scale is None:
        _rng[0] = np.random.default_rng(11)
    return _rng[0].normal(size=512) * (scale or 0) / np.sqrt(512)  # vector of norm ~scale


@pytest.fixture(autouse=True)
def fresh_noise():
    noise(None)  # each test gets the same noise sequence regardless of test order


GOA = {1: "N", 2: (15.0, 33.0, 0.0), 3: "E", 4: (73.0, 45.0, 0.0)}


@pytest.fixture
def setup(tmp_path):
    clip = FakeClip()
    lib = Library(tmp_path / "library", workers=4, face_engine=NoFaces(), clip_engine=clip, place_index=FakePlaces())
    src = tmp_path / "phone"

    def shoot(name, seed, when, vec, gps=None):
        p = src / name
        save(scene(seed), p, when, gps)
        clip.register(p, vec)
        return p

    for i in range(6):  # beach day in Goa
        shoot(f"beach{i}.jpg", 10 + i, f"2025:12:20 1{i}:00:00", DIRS["beach"] + noise(), GOA)
    for i in range(6):
        shoot(f"food{i}.jpg", 30 + i, f"2025:11:0{i + 1} 20:00:00", DIRS["food"] + noise())
    for i in range(6):
        shoot(f"street{i}.jpg", 50 + i, f"2025:10:0{i + 1} 09:00:00", DIRS["street"] + noise())
    for i in range(4):  # ambiguous: a person on a beach
        shoot(f"amb{i}.jpg", 70 + i, f"2025:09:0{i + 1} 09:00:00",
              unit(DIRS["beach"]) + unit(DIRS["person"]) + noise(0.05))
    lib.import_sync(str(src))
    return lib, clip, tmp_path


def members(lib, vid):
    c = lib.catalog.connect()
    return {r[0] for r in c.execute(
        "SELECT s.original_path FROM photo_factor pf JOIN sources s ON s.sha256=pf.sha256 WHERE pf.value_id=?",
        (vid,))}


def names(paths):
    return sorted(p.rsplit("/", 1)[1].split(".")[0] for p in paths)


def test_default_groups_and_places(setup):
    lib, _, _ = setup
    c = lib.catalog.connect()
    assert c.execute("SELECT COUNT(*) FROM learned_groups WHERE prompts IS NOT NULL").fetchone()[0] == 6
    labels = {r[0] for r in c.execute(
        "SELECT v.label FROM factor_values v JOIN factors f ON f.id=v.factor_id WHERE f.key='place'")}
    assert {"India", "Goa", "Calangute", "No location"} <= labels
    ev = c.execute("SELECT v.label FROM events e JOIN factor_values v ON v.id=e.value_id "
                   "WHERE v.label LIKE 'Calangute%'").fetchone()
    assert ev and ev[0].startswith("Calangute · ")
    assert lib.search({"place": "Goa"})["total"] == 6


def test_content_group_learns_from_answers(setup):
    lib, _, _ = setup
    g = lib.create_content_group("Beach", ["a photo of a beach"])
    vid = g["value_id"]
    assert {n for n in names(members(lib, vid)) if n.startswith("beach")} == {f"beach{i}" for i in range(6)}
    assert not {n for n in names(members(lib, vid)) if n.startswith(("food", "street"))}

    c = lib.catalog.connect()
    q = next(q for q in qs.list_open(c) if q["kind"] == "confirm_group" and q["payload"]["value_id"] == vid)
    asked = q["payload"]["shas"]
    path = {r[0]: r[1].rsplit("/", 1)[1] for r in c.execute("SELECT sha256, original_path FROM sources")}
    assert {path[s][:3] for s in asked} == {"amb"}  # it asks about the uncertain ones

    assert len(asked) == 4
    lib.answer_question(q["id"], {"accept": asked[:2], "reject": asked[2:]})
    m = members(lib, vid)
    assert {path[s] for s in asked[:2]} <= {p.rsplit("/", 1)[1] for p in m}
    assert not ({path[s] for s in asked[2:]} & {p.rsplit("/", 1)[1] for p in m})
    assert c.execute("SELECT round FROM learned_groups WHERE value_id=?", (vid,)).fetchone()[0] == 1
    assert c.execute("SELECT status FROM questions WHERE id=?", (q["id"],)).fetchone()[0] == "answered"
    assert lib.search({"content": "Beach"})["total"] == 8  # 6 beach photos + the 2 accepted


def test_manual_collection_suggests_more(setup):
    lib, _, _ = setup
    c = lib.catalog.connect()
    path = {r[1].rsplit("/", 1)[1]: r[0] for r in c.execute("SELECT sha256, original_path FROM sources")}
    v = lib.create_custom_value("Albums", "Dinners", shas=[path[f"food{i}.jpg"] for i in range(3)])
    q = next(q for q in qs.list_open(c) if q["kind"] == "confirm_group" and q["payload"]["value_id"] == v["value_id"])
    assert q["payload"]["suggest"] is True
    suggested = {s for s in q["payload"]["shas"]}
    assert suggested == {path[f"food{i}.jpg"] for i in range(3, 6)}
    lib.answer_question(q["id"], {"accept": list(suggested)})
    assert c.execute("SELECT COUNT(*) FROM photo_factor WHERE value_id=? AND source='user'",
                     (v["value_id"],)).fetchone()[0] == 6


def test_text_search_and_rules(setup):
    lib, _, _ = setup
    c = lib.catalog.connect()
    path = {r[0]: r[1].rsplit("/", 1)[1] for r in c.execute("SELECT sha256, original_path FROM sources")}
    hits = [path[s] for s in lib.text_search("a plate of food")]
    assert hits and all(h.startswith("food") for h in hits) and len(hits) == 6
    assert lib.text_search("a photo of a beach", rule={"place": "Goa"})  # narrowed to Goa
    assert all(path[s].startswith("beach") for s in lib.text_search("a photo of a beach", rule={"place": "Goa"}))


def test_clip_near_duplicates(tmp_path):
    clip = FakeClip()
    lib = Library(tmp_path / "lib", workers=2, face_engine=NoFaces(), clip_engine=clip, place_index=FakePlaces())
    v = DIRS["street"]
    for i, seed in enumerate((901, 902)):  # different pixels, same moment, same content
        p = tmp_path / "src" / f"shot{i}.jpg"
        save(scene(seed), p, f"2025:05:05 10:00:0{i * 5}")
        clip.register(p, v + noise(0.01))
    p = tmp_path / "src" / "later.jpg"  # same content, but an hour later: not a burst
    save(scene(903), p, "2025:05:05 11:00:00")
    clip.register(p, v)
    lib.import_sync(str(tmp_path / "src"))
    c = lib.catalog.connect()
    groups = c.execute("SELECT g.kind, COUNT(*) FROM similar_groups g JOIN photo_similar ps ON ps.group_id=g.id "
                       "GROUP BY g.id").fetchall()
    assert [tuple(g) for g in groups] == [("burst", 2)]


def test_agent_and_api(setup):
    lib, _, _ = setup

    class Scripted:
        def __init__(self, replies):
            self.replies = replies

        def chat(self, system, messages, tools):
            assert any(t["name"] == "create_content_group" for t in tools)
            return self.replies.pop(0)

    sha = lib.catalog.connect().execute("SELECT sha256 FROM photos LIMIT 1").fetchone()[0]
    assert Agent(lib, None)._expand([sha[:12], sha, "nomatch"]) == [sha, sha, "nomatch"]
    out = Agent(lib, Scripted([
        Reply(tool_calls=[{"id": "1", "name": "create_content_group",
                           "arguments": {"name": "Meals", "descriptions": ["a plate of food"]}},
                          {"id": "2", "name": "find_photos", "arguments": {"text": "food", "limit": 3}}]),
        Reply(content="Made a Meals group."),
    ])).run("group my food photos")
    assert out["changed"]
    meals = lib.catalog.connect().execute("SELECT id FROM factor_values WHERE label='Meals'").fetchone()[0]
    assert len(members(lib, meals)) == 6, names(members(lib, meals))

    client = TestClient(create_app(lib))
    s = client.get("/api/stats").json()
    assert s["content_analysis"] and s["place_names"]
    r = client.get("/api/search", params={"q": "food"}).json()
    assert r["total"] == 6 and len(r["items"]) == 6
    assert any(g["name"] == "Meals" for g in client.get("/api/content").json())
    assert client.post("/api/content", json={"name": "X", "prompts": []}).status_code == 400
    assert client.post(f"/api/groups/{meals}/teach", json={"reject": [r["items"][0]["sha256"]]}).json()["photos"] == 5
    assert client.delete(f"/api/custom/{meals}").status_code == 200
