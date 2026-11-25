"""The Claude Code adapter: MCP server protocol, the harness subprocess contract, agent/API wiring."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_phase34 import FakeFaces, setup  # noqa: F401  (library with people, events, questions)

from photosort.agent import Agent
from photosort.api import create_app
from photosort.claude_code import ClaudeCodeHarness
from photosort.llm import LLMError, from_config

FAKE = Path(__file__).parent / "fake_claude.py"
ROOT = Path(__file__).resolve().parent.parent


def _mcp(lib, env_extra=None):
    env = os.environ | {"PYTHONPATH": str(ROOT)} | (env_extra or {})
    return subprocess.Popen([sys.executable, "-m", "photosort.mcp_server", "--library", str(lib.root)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def test_mcp_server_protocol(setup, tmp_path):
    lib, _, _ = setup
    lib.import_sync(str(tmp_path / "phone"))
    ui_file = tmp_path / "ui.json"
    proc = _mcp(lib, {"PHOTOSORT_UI_FILE": str(ui_file), "PHOTOSORT_VISION": "0"})
    try:
        def rpc(method, params=None, mid=1):
            msg = {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            return json.loads(proc.stdout.readline())

        init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
        assert init["result"]["protocolVersion"] == "2025-06-18" and "tools" in init["result"]["capabilities"]
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()
        tools = rpc("tools/list", mid=2)["result"]["tools"]
        names = {t["name"] for t in tools}
        assert {"library_overview", "name_person", "find_photos", "show"} <= names and "look_at_photos" not in names
        assert all(t["inputSchema"]["type"] == "object" for t in tools)

        r = rpc("tools/call", {"name": "library_overview", "arguments": {}}, mid=3)["result"]
        assert r["isError"] is False and json.loads(r["content"][0]["text"])["photos"] == 15
        assert json.loads(ui_file.read_text()) == {"ui": {"choices": [], "photos": [], "faces": []}, "changed": False}

        big = json.loads(rpc("tools/call", {"name": "list_people", "arguments": {}}, mid=4)["result"]["content"][0]["text"])["people"][0]["id"]
        r = rpc("tools/call", {"name": "name_person", "arguments": {"person_id": big, "name": "Alice"}}, mid=5)["result"]
        assert json.loads(r["content"][0]["text"])["name"] == "Alice"
        rpc("tools/call", {"name": "show", "arguments": {"person_ids": [big]}}, mid=6)
        rpc("tools/call", {"name": "offer_choices", "arguments": {"options": ["Yes", "No"]}}, mid=7)
        side = json.loads(ui_file.read_text())
        assert side["changed"] is True and side["ui"]["choices"] == ["Yes", "No"] and side["ui"]["faces"][0]["name"] == "Alice"

        r = rpc("tools/call", {"name": "find_photos", "arguments": {"rule": {"nope": 1}}}, mid=8)["result"]
        assert r["isError"] is True and "unknown rule keys" in r["content"][0]["text"]
        err = rpc("nope/method", mid=9)
        assert err["error"]["code"] == -32601
        assert rpc("ping", mid=10)["result"] == {}
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    assert lib.catalog.connect().execute("SELECT label FROM factor_values WHERE id=?", (big,)).fetchone()[0] == "Alice"


def test_harness_turns_and_sessions(setup, tmp_path, monkeypatch):
    lib, _, _ = setup
    lib.import_sync(str(tmp_path / "phone"))
    log = tmp_path / "argv.log"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    FAKE.chmod(0o755)
    harness = ClaudeCodeHarness(lib.root, binary=str(FAKE), vision=True, effort="medium")
    monkeypatch.setenv("CLAUDE_EFFORT", "max")
    agent = Agent(lib, harness, vision=True)

    big = lib.catalog.connect().execute("SELECT value_id FROM faces GROUP BY value_id ORDER BY COUNT(*) DESC LIMIT 1").fetchone()[0]
    out = agent.run(f"name person {big} Alice")
    assert out["reply"].startswith("Let me check.\n\nfake reply after 3 tool calls") and out["changed"] and out["choices"] == ["Yes", "No"]
    assert lib.catalog.connect().execute("SELECT label FROM factor_values WHERE id=?", (big,)).fetchone()[0] == "Alice"
    sid = lib.catalog.setting("claude_code_session")
    assert sid

    sha = lib.catalog.connect().execute("SELECT sha256 FROM photos LIMIT 1").fetchone()[0]
    out = agent.run(f"look {sha}")
    assert "look_at_photos:img" in out["reply"] and not out["changed"]
    argvs = [json.loads(l) for l in log.read_text().splitlines()]
    assert "--session-id" in argvs[0] and argvs[0][argvs[0].index("--session-id") + 1] == sid
    assert "--resume" in argvs[1] and argvs[1][argvs[1].index("--resume") + 1] == sid
    assert argvs[1][argvs[1].index("--tools") + 1] == "" and "--strict-mcp-config" in argvs[1]
    assert argvs[1][argvs[1].index("--output-format") + 1] == "stream-json" and "--verbose" in argvs[1]
    assert argvs[1][argvs[1].index("--allowedTools") + 1] == "mcp__photosort"
    assert "mcp__photosort__" in argvs[1][argvs[1].index("--system-prompt") + 1]
    assert argvs[1][argvs[1].index("--effort") + 1] == "medium"

    hist = [m["role"] for m in agent.history()]
    assert hist == ["user", "assistant", "user", "assistant"]
    agent.reset()
    assert lib.catalog.setting("claude_code_session") == ""
    assert harness.ping() == "ok"


def test_harness_errors(setup, tmp_path, monkeypatch):
    lib, _, _ = setup
    monkeypatch.setenv("PHOTOSORT_CLAUDE_BIN", "")
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(LLMError):
        ClaudeCodeHarness(lib.root)
    broken = tmp_path / "broken.sh"
    broken.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    broken.chmod(0o755)
    out = Agent(lib, ClaudeCodeHarness(lib.root, binary=str(broken))).run("hi")
    assert out["error"] == "llm" and "exited with 3" in out["reply"]


def test_api_claude_code_preset(setup, tmp_path, monkeypatch):
    lib, _, _ = setup
    FAKE.chmod(0o755)
    monkeypatch.setenv("PHOTOSORT_CLAUDE_BIN", str(FAKE))
    client = TestClient(create_app(lib))
    r = client.post("/api/llm", json={"preset": "claude_code"}).json()
    assert r["config"]["adapter"] == "claude_code" and not r["config"]["model"]
    assert isinstance(from_config(r["config"], None, library_root=lib.root), ClaudeCodeHarness)
    assert client.post("/api/llm/test").json()["reply"] == "ok"
    out = client.post("/api/chat", json={"message": "hello"}).json()
    assert "fake reply" in out["reply"] and out["choices"] == ["Yes", "No"]
    assert [m["role"] for m in client.get("/api/chat").json()] == ["user", "assistant"]
    assert client.delete("/api/chat").json()["ok"]
