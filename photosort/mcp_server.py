"""PhotoSort's assistant tools as an MCP server over stdio.

Lets an external agent harness (Claude Code's `claude -p`, or any MCP client) drive the library
with the same tools the built-in agent loop uses. Protocol: JSON-RPC 2.0, one JSON object per line.

    python -m photosort.mcp_server --library <dir>

Environment:
    PHOTOSORT_UI_FILE   where to write UI side effects (choices / photos / faces shown) and
                        whether anything changed, as JSON, after every tool call
    PHOTOSORT_VISION=1  expose look_at_photos (the client must accept image content)
"""
import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path

from .agent import LOOK_TOOL, READ_ONLY, TOOLS, Agent
from .library import Library

log = logging.getLogger("photosort.mcp")
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


class Server:
    def __init__(self, library: Library, vision: bool, ui_file: Path | None):
        self.agent = Agent(library, None, vision=vision)
        self.vision = vision
        self.ui_file = ui_file
        self.ui = {"choices": [], "photos": [], "faces": []}
        self.changed = False
        self.tools = TOOLS + ([LOOK_TOOL] if vision else [])

    # ------------------------------------------------------------------ protocol

    def handle(self, msg: dict) -> dict | None:
        method, params, mid = msg.get("method"), msg.get("params") or {}, msg.get("id")
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return self._ok(mid, {"protocolVersion": version, "capabilities": {"tools": {}},
                                  "serverInfo": {"name": "photosort", "version": "0.5"}})
        if method == "notifications/initialized" or method.startswith("notifications/"):
            return None
        if method == "ping":
            return self._ok(mid, {})
        if method == "tools/list":
            return self._ok(mid, {"tools": [
                {"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]}
                for t in self.tools]})
        if method == "tools/call":
            return self._ok(mid, self.call(params.get("name", ""), params.get("arguments") or {}))
        if mid is None:
            return None
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}

    @staticmethod
    def _ok(mid, result) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def call(self, name: str, args: dict) -> dict:
        if name not in {t["name"] for t in self.tools}:
            return {"content": [{"type": "text", "text": json.dumps({"error": f"unknown tool {name}"})}], "isError": True}
        result, images = self.agent._execute({"id": name, "name": name, "arguments": args}, self.ui)
        if name not in READ_ONLY and "error" not in result:
            self.changed = True
        self._flush_ui()
        content = [{"type": "text", "text": json.dumps(result, default=str)}]
        for img in images or []:
            content.append({"type": "image", "data": img["data"], "mimeType": img["media_type"]})
        return {"content": content, "isError": "error" in result}

    def _flush_ui(self) -> None:
        if not self.ui_file:
            return
        tmp = self.ui_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ui": self.ui, "changed": self.changed}))
        os.replace(tmp, self.ui_file)

    # ------------------------------------------------------------------ stdio loop

    def serve(self, stdin=None, stdout=None) -> None:
        stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("mcp: bad json line")
                continue
            try:
                resp = self.handle(msg)
            except Exception as e:  # never let one call kill the server
                log.exception("mcp: %s failed", msg.get("method"))
                resp = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32603, "message": str(e)}}
            if resp is not None:
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="photosort.mcp_server")
    ap.add_argument("--library", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="mcp %(levelname)s %(message)s")
    lib = Library(Path(args.library))
    if lib.clip_engine.available():  # text search needs the model; load it off the critical path
        threading.Thread(target=lib.clip_engine.warm, daemon=True).start()
    ui_file = os.environ.get("PHOTOSORT_UI_FILE")
    Server(lib, os.environ.get("PHOTOSORT_VISION") == "1", Path(ui_file) if ui_file else None).serve()


if __name__ == "__main__":
    main()
