"""Run the assistant through Claude Code's headless mode instead of a model API.

`claude -p` supplies the model (whatever the user's Claude Code is set to), the agent loop and the
login; PhotoSort supplies the system prompt and its tools via the MCP server in mcp_server.py.
No API key is needed. Only works where the `claude` CLI is installed and logged in.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from .llm import LLMError

MCP_SERVER_NAME = "photosort"
TURN_TIMEOUT_S = 900
HARNESS_NOTE = (
    "\n\nYour tools come from the MCP server 'photosort' (tool names start with mcp__photosort__); "
    "you have no other tools, no file access and no shell. Call them as described above and never "
    "mention tool names, MCP or Claude Code to the user."
)


class ClaudeCodeHarness:
    name = "claude_code"

    def __init__(self, library_root: Path, model: str | None = None, vision: bool = False,
                 binary: str | None = None, max_turns: int = 12, python: str | None = None,
                 effort: str | None = None):
        self.binary = binary or os.environ.get("PHOTOSORT_CLAUDE_BIN") or shutil.which("claude")
        if not self.binary:
            raise LLMError("Claude Code CLI ('claude') was not found on this machine")
        self.library_root = Path(library_root)
        self.model, self.vision, self.max_turns = model or None, vision, max_turns
        self.effort = effort or None  # None: Claude Code's own default
        self.python = python or sys.executable

    # ------------------------------------------------------------------ plumbing

    def _mcp_config(self, ui_file: Path) -> dict:
        env = {"PHOTOSORT_UI_FILE": str(ui_file), "PHOTOSORT_VISION": "1" if self.vision else "0"}
        for k in ("PATH", "HOME"):
            if os.environ.get(k):
                env[k] = os.environ[k]
        return {"mcpServers": {MCP_SERVER_NAME: {
            "command": self.python,
            "args": ["-m", "photosort.mcp_server", "--library", str(self.library_root)],
            "env": env,
        }}}

    def _run(self, args: list[str], timeout: int = TURN_TIMEOUT_S) -> dict:
        env = os.environ | {"MCP_TOOL_TIMEOUT": "600000", "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
        env.pop("CLAUDE_EFFORT", None)  # never inherit a parent Claude Code session's effort setting
        try:
            proc = subprocess.run([self.binary, "-p", *args], capture_output=True, text=True, timeout=timeout,
                                  cwd=self.library_root, env=env)
        except subprocess.TimeoutExpired:
            raise LLMError(f"Claude Code did not answer within {timeout} s")
        except OSError as e:
            raise LLMError(f"could not run Claude Code: {e}")
        # stream-json: one event per line; keep every assistant text block (what the model said
        # between tool calls as well as its final message) and the closing "result" event.
        texts, data = [], {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "assistant":
                for b in (e.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "text" and (b.get("text") or "").strip():
                        texts.append(b["text"].strip())
            elif e.get("type") == "result":
                data = e
        if proc.returncode != 0 and not data:
            raise LLMError(f"Claude Code exited with {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}")
        data["_texts"] = texts
        return data

    def _common_args(self, system: str) -> list[str]:
        args = ["--output-format", "stream-json", "--verbose", "--system-prompt", system, "--tools", "",
                "--max-turns", str(self.max_turns)]
        if self.model:
            args += ["--model", self.model]
        if self.effort:
            args += ["--effort", self.effort]
        return args

    # ------------------------------------------------------------------ API

    def ping(self) -> str:
        data = self._run(["Reply with the single word: ok", *self._common_args("You are a connectivity test."),
                          "--no-session-persistence", "--strict-mcp-config"], timeout=180)
        if data.get("is_error"):
            raise LLMError(str(data.get("result") or data)[:300])
        return str(data.get("result", "")).strip()

    def run_turn(self, system: str, user_text: str, session_id: str | None) -> dict:
        """One user turn. Claude Code runs the whole tool loop; returns the final text plus UI effects."""
        with tempfile.TemporaryDirectory(prefix="photosort-turn-") as tmp:
            ui_file, cfg_file = Path(tmp) / "ui.json", Path(tmp) / "mcp.json"
            cfg_file.write_text(json.dumps(self._mcp_config(ui_file)))
            base = [user_text, *self._common_args(system + HARNESS_NOTE),
                    "--mcp-config", str(cfg_file), "--strict-mcp-config", "--allowedTools", f"mcp__{MCP_SERVER_NAME}"]
            new_session = session_id is None
            sid = session_id or str(uuid.uuid4())
            data = self._run(base + (["--session-id", sid] if new_session else ["--resume", sid]))
            if data.get("is_error") and not new_session and "session" in str(data.get("result", "")).lower():
                sid = str(uuid.uuid4())  # the saved session is gone (e.g. cleaned up): start a fresh one
                data = self._run(base + ["--session-id", sid])
            try:
                side = json.loads(ui_file.read_text()) if ui_file.exists() else {}
            except json.JSONDecodeError:
                side = {}
        text = "\n\n".join(data.get("_texts") or []) or str(data.get("result") or "").strip()
        denied = data.get("permission_denials") or []
        if data.get("is_error") and not text:
            raise LLMError(f"Claude Code reported an error: {json.dumps(data)[:400]}")
        return {
            "text": text or ("(no reply)" if not denied else "I wasn't allowed to use my tools for that."),
            "session_id": data.get("session_id") or sid,
            "ui": side.get("ui") or {"choices": [], "photos": [], "faces": []},
            "changed": bool(side.get("changed")),
            "cost_usd": data.get("total_cost_usd"),
            "num_turns": data.get("num_turns"),
            "usage": {k: v for k, v in (data.get("usage") or {}).items()
                      if k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
            "model": next(iter(data.get("modelUsage") or {}), None),
            "denied": [d.get("tool_name") for d in denied if isinstance(d, dict)],
            "cut_off": data.get("terminal_reason") not in (None, "completed"),
        }
