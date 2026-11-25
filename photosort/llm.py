"""Provider-neutral chat + tool-calling interface.

The agent only ever sees `LLM.chat(system, messages, tools) -> Reply`. Messages and tools use a
neutral shape; each adapter converts to and from its wire format:

  user       {"role": "user", "content": str, "images": [{"media_type", "data"(base64)}]?}
  assistant  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}],
              "provider": str, "raw": provider-native content (replayed unchanged to the same provider)}
  tool       {"role": "tool", "tool_call_id": str, "name": str, "content": str, "images": [...]?}
  tool spec  {"name", "description", "parameters": JSON Schema}

Adapters:
  openai     any OpenAI-compatible /chat/completions endpoint: OpenAI, Azure-compatible gateways,
             Google Gemini (OpenAI endpoint), Ollama, LM Studio, vLLM, llama.cpp, OpenRouter, Groq, ...
  anthropic  Anthropic Messages API
  none       no model configured; the app works without the assistant
"""
import json
from dataclasses import dataclass, field

import httpx

TIMEOUT = httpx.Timeout(600.0, connect=10.0)  # reasoning models can take a while per step

# Starting points for the settings screen. The model id is always the user's choice.
PRESETS = {
    "openai": {"adapter": "openai", "base_url": "https://api.openai.com/v1", "needs_key": True},
    "gemini": {"adapter": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "needs_key": True},
    "anthropic": {"adapter": "anthropic", "base_url": "https://api.anthropic.com", "needs_key": True},
    "openrouter": {"adapter": "openai", "base_url": "https://openrouter.ai/api/v1", "needs_key": True},
    "ollama": {"adapter": "openai", "base_url": "http://localhost:11434/v1", "needs_key": False},
    "lmstudio": {"adapter": "openai", "base_url": "http://localhost:1234/v1", "needs_key": False},
    "custom": {"adapter": "openai", "base_url": "", "needs_key": False},
    # Uses the local Claude Code CLI and its login instead of an API; see claude_code.py.
    "claude_code": {"adapter": "claude_code", "base_url": "", "needs_key": False},
}


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw: object = None
    provider: str = ""
    truncated: bool = False          # hit max_tokens
    meta: dict = field(default_factory=dict)   # provider diagnostics, e.g. input_transformations

    def as_message(self) -> dict:
        m = {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls, "provider": self.provider}
        if self.raw is not None:
            m["raw"] = self.raw
        return m


def strip_raw(messages: list[dict]) -> list[dict]:
    """Drop provider-native replay content (thinking blocks etc.) so turns are rebuilt from text + tool calls.

    Used when the history sent is not byte-identical to the one that produced those turns (it was
    trimmed, or the system prompt / tool set changed): models that bind reasoning blocks to the
    conversation prefix reject them after such an edit, and rebuilt turns are always accepted."""
    return [{k: v for k, v in m.items() if k != "raw"} if m.get("role") == "assistant" else m for m in messages]


class LLM:
    name = "base"

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        raise NotImplementedError


def _post(url: str, headers: dict, body: dict, transport=None) -> dict:
    try:
        with httpx.Client(timeout=TIMEOUT, transport=transport) as c:
            r = c.post(url, headers=headers, json=body)
    except httpx.HTTPError as e:
        raise LLMError(f"could not reach {url}: {e}") from e
    if r.status_code >= 400:
        try:
            err = r.json()
            msg = err.get("error", {}).get("message") if isinstance(err.get("error"), dict) else err.get("error")
        except ValueError:
            msg = None
        raise LLMError(f"{r.status_code} from model API: {msg or r.text[:300]}")
    return r.json()


def _parse_args(s) -> dict:
    if isinstance(s, dict):
        return s
    try:
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {"_invalid_arguments": s}
    except json.JSONDecodeError:
        return {"_invalid_arguments": s}


# ---------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatible(LLM):
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, transport=None):
        if not base_url or not model:
            raise LLMError("base URL and model are required")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.transport = model, api_key, transport

    @staticmethod
    def to_wire(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        pending_images: list[dict] = []

        def flush():
            if pending_images:
                out.append({"role": "user", "content": [{"type": "text", "text": "Images returned by the tools above:"}]
                            + [_oai_image(i) for i in pending_images]})
                pending_images.clear()

        for m in messages:
            if m["role"] != "tool":
                flush()
            if m["role"] == "user":
                if m.get("images"):
                    out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]
                                + [_oai_image(i) for i in m["images"]]})
                else:
                    out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                        for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
                pending_images.extend(m.get("images") or [])
        flush()
        return out

    def chat(self, system, messages, tools) -> Reply:
        body = {"model": self.model, "messages": self.to_wire(system, messages)}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post(self.url, headers, body, self.transport)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response: {str(data)[:300]}")
        calls = [{"id": c.get("id") or f"call_{i}", "name": c["function"]["name"],
                  "arguments": _parse_args(c["function"].get("arguments"))}
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        return Reply(content=msg.get("content") or "", tool_calls=calls, provider=self.name)


def _oai_image(img: dict) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{img['media_type']};base64,{img['data']}"}}


# ---------------------------------------------------------------------- Anthropic Messages API


class AnthropicMessages(LLM):
    name = "anthropic"

    def __init__(self, base_url: str, model: str, api_key: str | None, transport=None, max_tokens: int = 32000,
                 effort: str | None = None, binding_check: bool = False):
        if not model or not api_key:
            raise LLMError("model and API key are required")
        self.url = (base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        self.model, self.api_key, self.transport, self.max_tokens = model, api_key, transport, max_tokens
        self.effort = effort
        # Opt into the API's conversation-binding check and report dropped reasoning blocks: a
        # diagnostic for verifying this adapter replays history exactly (Claude 5 family only).
        self.binding_check = binding_check

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        out: list[dict] = []

        def push(role, blocks):
            if out and out[-1]["role"] == role:  # API requires alternating roles
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for m in messages:
            if m["role"] == "user":
                push("user", [{"type": "text", "text": m["content"]}] + [_ant_image(i) for i in m.get("images") or []])
            elif m["role"] == "assistant":
                if m.get("provider") == "anthropic" and m.get("raw"):
                    push("assistant", m["raw"])  # replay unchanged (keeps thinking blocks valid)
                else:
                    blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                    blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                               for c in m.get("tool_calls") or []]
                    push("assistant", blocks or [{"type": "text", "text": "(no content)"}])
            elif m["role"] == "tool":
                content = [{"type": "text", "text": m["content"]}] + [_ant_image(i) for i in m.get("images") or []]
                push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": content}])
        return out

    def chat(self, system, messages, tools) -> Reply:
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": self.to_wire(messages)}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                             for t in tools]
        if self.effort:
            body["output_config"] = {"effort": self.effort}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        if self.binding_check:
            headers["anthropic-beta"] = "thinking-binding-controls-2026-08-01"
            body["thinking"] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
        data = _post(self.url, headers, body, self.transport)
        meta = {}
        if "input_transformations" in data:
            meta["input_transformations"] = data["input_transformations"]
        if data.get("stop_reason") == "refusal":
            why = (data.get("stop_details") or {}).get("category")
            return Reply(content="The model declined this request" + (f" ({why})." if why else "."),
                         provider=self.name, meta=meta)
        blocks = data.get("content") or []
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
        calls = [{"id": b["id"], "name": b["name"], "arguments": _parse_args(b.get("input"))}
                 for b in blocks if b.get("type") == "tool_use"]
        return Reply(content=text, tool_calls=calls, raw=blocks, provider=self.name,
                     truncated=data.get("stop_reason") == "max_tokens", meta=meta)


def _ant_image(img: dict) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]}}


# ---------------------------------------------------------------------- factory


def from_config(cfg: dict, api_key: str | None, transport=None, library_root=None):
    """cfg: {"preset", "adapter", "base_url", "model", "vision"}; returns None when unconfigured.

    Returns an LLM (chat-completions style, driven by agent.Agent's own loop) or, for the
    "claude_code" adapter, a harness object with `run_turn()` that runs the loop itself."""
    adapter = cfg.get("adapter") or PRESETS.get(cfg.get("preset", ""), {}).get("adapter")
    if not adapter or adapter == "none":
        return None
    if adapter == "claude_code":
        from .claude_code import ClaudeCodeHarness

        if library_root is None:
            raise LLMError("the Claude Code adapter needs the library path")
        return ClaudeCodeHarness(library_root, model=cfg.get("model") or None, vision=bool(cfg.get("vision")),
                                 effort=cfg.get("effort") or None)
    if not cfg.get("model"):
        return None
    if adapter == "openai":
        return OpenAICompatible(cfg.get("base_url", ""), cfg["model"], api_key, transport)
    if adapter == "anthropic":
        return AnthropicMessages(cfg.get("base_url", ""), cfg["model"], api_key, transport,
                                 effort=cfg.get("effort") or None, binding_check=bool(cfg.get("binding_check")))
    raise LLMError(f"unknown adapter {adapter}")
