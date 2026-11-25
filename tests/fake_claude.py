#!/usr/bin/env python3
"""A stand-in for the `claude` CLI used by the tests.

It behaves like `claude -p --output-format json` with an MCP server attached: it starts the server
from the --mcp-config file, speaks the MCP protocol to it, calls a few tools, and prints the JSON
result object. Which tools it calls is steered by the prompt text. It logs its argv to the file
named by FAKE_CLAUDE_LOG so tests can check how it was invoked.
"""
import json
import os
import subprocess
import sys


def main():
    argv = sys.argv[1:]
    if os.environ.get("FAKE_CLAUDE_LOG"):
        with open(os.environ["FAKE_CLAUDE_LOG"], "a") as f:
            f.write(json.dumps(argv) + "\n")
    opts, prompt = {}, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-p", "--strict-mcp-config", "--no-session-persistence", "--verbose"):
            i += 1
        elif a.startswith("--"):
            opts[a] = argv[i + 1] if i + 1 < len(argv) else ""
            i += 2
        else:
            prompt = a
            i += 1
    sid = opts.get("--session-id") or opts.get("--resume") or "fake-session"
    assert "CLAUDE_EFFORT" not in os.environ, "parent session effort must not leak into the child"
    out = {"type": "result", "is_error": False, "session_id": sid, "total_cost_usd": 0.01, "num_turns": 1,
           "terminal_reason": "completed", "modelUsage": {"fake-model": {}}, "permission_denials": []}

    def say(text):  # an assistant event, as --output-format stream-json emits them
        print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}))

    if "--mcp-config" not in opts:
        out["result"] = "ok"
        say("ok")
        print(json.dumps(out))
        return
    say("Let me check.")

    cfg = json.load(open(opts["--mcp-config"]))["mcpServers"]["photosort"]
    env = os.environ | cfg.get("env", {})
    proc = subprocess.Popen([cfg["command"], *cfg["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, env=env)
    n = [0]

    def rpc(method, params=None, notify=False):
        n[0] += 1
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            msg["id"] = n[0]
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        if notify:
            return None
        return json.loads(proc.stdout.readline())

    init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "fake", "version": "0"}})
    assert init["result"]["serverInfo"]["name"] == "photosort", init
    rpc("notifications/initialized", notify=True)
    tools = {t["name"] for t in rpc("tools/list")["result"]["tools"]}
    calls = [("library_overview", {})]
    if "name person" in prompt:  # "name person <id> <name>"
        _, _, pid, name = prompt.split(" ", 3)
        calls.append(("name_person", {"person_id": int(pid), "name": name}))
    if "look" in prompt:
        calls.append(("look_at_photos", {"photo_ids": prompt.split("look ", 1)[1].split()}))
    calls.append(("offer_choices", {"options": ["Yes", "No"]}))
    results = []
    for name, args in calls:
        assert name in tools, (name, tools)
        r = rpc("tools/call", {"name": name, "arguments": args})["result"]
        results.append((name, r))
    proc.stdin.close()
    proc.wait(timeout=30)
    summary = "; ".join(f"{name}:{'img' if any(c['type'] == 'image' for c in r['content']) else r['content'][0]['text'][:60]}"
                        for name, r in results)
    out["result"] = f"fake reply after {len(calls)} tool calls [{summary}]"
    out["num_turns"] = len(calls) + 1
    say(out["result"])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
