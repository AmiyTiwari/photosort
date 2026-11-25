"""The PhotoSort assistant: a tool-calling loop over the library, independent of the model provider."""
import base64
import hashlib
import json
import logging
from datetime import datetime

from . import questions as qs
from .llm import LLM, LLMError, strip_raw
from .rules import RuleError

log = logging.getLogger("photosort.agent")

MAX_STEPS = 10
HISTORY = 300  # messages sent back to the model each turn; older ones are trimmed (see _window)

SYSTEM = """You are the PhotoSort assistant. You help one person organise their photo library.

How the library works:
- Every unique photo is stored once. "Factors" are ways of sorting (Date, People, Events, Places,
  Camera, Type, plus custom ones). A photo can sit under many factor values without being copied.
- People are face clusters found automatically ("Person 12") until the user names them.
- Events are runs of photos close in time; their names default to a date range.
- Custom factors are created by you from what the user wants, e.g. factor "Family" with value
  "Grandparents", backed by a rule so future imports are sorted the same way automatically.
- Content groups (factor "Content") are defined by short visual descriptions, e.g. "Beach & sea" =
  ["a photo of a beach", "a photo of the sea"], and assigned automatically from what is in the photo.
  They improve as the user answers "does this belong?" questions (or via teach_group).
- Manual collections (custom values without a rule) learn from their photos and suggest similar ones.
- Places are real names (country, region, city) when the photo has GPS.
- The review queue holds the questions that sort the most photos per answer.

Rules (used by find_photos and create_custom_value) are JSON objects:
  {"all": [...]}, {"any": [...]}, {"not": {...}}, and conditions:
  date_from/date_to "YYYY-MM-DD", months [12,1], person (name or id, or a list meaning all of them),
  any_person [names], min_people / max_people (face count), event (name or id),
  near {"lat","lon","km"}, camera "iPhone", path "WhatsApp" (original file path contains),
  screenshot true/false, value_ids [factor value ids], place "Goa" (country/region/city name),
  content "Food" (content group name).
- find_photos also takes `text`, a plain-language description searched by what is in the photo
  ("sunset over water", "birthday cake"); combine it with a rule to narrow by person, date or place.
- Use a rule for facts (who, when, where); use a content group for what the photo shows.

How to work:
- On a first conversation, look at library_overview, then interview the user briefly: which ways of
  sorting matter to them, who the main people are, and any custom groupings (e.g. "Receipts",
  "Kids' school", "Work"). Ask one or two things at a time and use offer_choices for quick replies.
- Prefer the highest-impact questions (largest unnamed people and events first). Use `show` so the
  user can see the faces or photos you are asking about.
- Act through tools; never claim you changed something unless a tool confirmed it.
- Before a broad change (deleting a custom value, merging people), confirm with the user.
- Review-queue questions are for the user to answer: show or describe them, but only answer one
  yourself when the user asks you to decide. Looking at photos costs money; look only when the
  user's request needs it, and at a few photos rather than many.
- Photo ids are 12-character prefixes of the photo hash; pass them back exactly as given.
- Keep replies short and concrete. Refer to people and events by name, not by id."""

SHORT_ID = 12


def _tool(name, description, properties=None, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}


RULE_SCHEMA = {"type": "object", "description": "Rule object as described in the system prompt."}

TOOLS = [
    _tool("library_overview", "Counts of photos, people, events, open questions, factors and current settings."),
    _tool("list_people", "People clusters with photo counts, largest first.",
          {"include_unnamed": {"type": "boolean"}, "limit": {"type": "integer"}}),
    _tool("list_events", "Events (time-based groups), newest first.", {"limit": {"type": "integer"}}),
    _tool("list_factors", "All factors and their values with counts; custom values include their rule.",
          {"values_per_factor": {"type": "integer"}}),
    _tool("find_photos", "Count and sample photos matching a rule and/or a plain-language `text` description.",
          {"rule": RULE_SCHEMA, "text": {"type": "string"}, "limit": {"type": "integer"}}),
    _tool("create_content_group",
          "Create or update a content group defined by what is in the photo. `descriptions` are short "
          "visual phrases like 'a photo of a birthday cake'. Higher threshold = stricter (default 0.5).",
          {"name": {"type": "string"}, "descriptions": {"type": "array", "items": {"type": "string"}},
           "threshold": {"type": "number"}}, ["name", "descriptions"]),
    _tool("teach_group", "Tell a content group or manual collection which photos belong and which do not.",
          {"value_id": {"type": "integer"}, "belong": {"type": "array", "items": {"type": "string"}},
           "do_not_belong": {"type": "array", "items": {"type": "string"}}}, ["value_id"]),
    _tool("name_person", "Name a person cluster. If the name already exists the clusters are merged.",
          {"person_id": {"type": "integer"}, "name": {"type": "string"}}, ["person_id", "name"]),
    _tool("merge_people", "Merge person `merge_id` into `keep_id` (same person).",
          {"keep_id": {"type": "integer"}, "merge_id": {"type": "integer"}}, ["keep_id", "merge_id"]),
    _tool("rename_event", "Give an event a name, e.g. 'Goa trip'.",
          {"event_id": {"type": "integer"}, "name": {"type": "string"}}, ["event_id", "name"]),
    _tool("create_custom_value",
          "Create or update a custom factor value. With `rule` it is re-applied to every future import; "
          "with `photo_ids` (sha256) those photos are added manually.",
          {"factor": {"type": "string"}, "value": {"type": "string"}, "rule": RULE_SCHEMA,
           "photo_ids": {"type": "array", "items": {"type": "string"}}}, ["factor", "value"]),
    _tool("delete_custom_value", "Delete a custom factor value (photos themselves are untouched).",
          {"value_id": {"type": "integer"}}, ["value_id"]),
    _tool("update_settings", "Change sorting settings. Events are rebuilt when event settings change.",
          {"event_gap_hours": {"type": "number", "description": "gap that starts a new event"},
           "event_min_photos": {"type": "integer"},
           "dup_policy": {"type": "string", "enum": ["keep_all", "hide_non_best"]}}),
    _tool("list_questions", "Open review-queue questions, highest impact first.", {"limit": {"type": "integer"}}),
    _tool("answer_question",
          "Answer a review-queue question. name_person/name_event: {\"name\"}; merge_people: {\"same\": bool}; "
          "confirm_face: {\"correct\": bool}; confirm_group: {\"accept\": [photo ids], \"reject\": [photo ids]}; "
          "any: {\"skip\": true}.",
          {"question_id": {"type": "integer"}, "answer": {"type": "object"}}, ["question_id", "answer"]),
    _tool("show", "Display photos and/or faces of people to the user under your reply.",
          {"photo_ids": {"type": "array", "items": {"type": "string"}},
           "person_ids": {"type": "array", "items": {"type": "integer"}}}),
    _tool("offer_choices", "Show quick-reply buttons under your reply.",
          {"options": {"type": "array", "items": {"type": "string"}}}, ["options"]),
]

LOOK_TOOL = _tool("look_at_photos", "See up to 6 photos (thumbnails) to describe or classify them.",
                  {"photo_ids": {"type": "array", "items": {"type": "string"}}}, ["photo_ids"])


class Agent:
    def __init__(self, lib, llm: LLM | None, vision: bool = False):
        self.lib, self.llm, self.vision = lib, llm, vision

    # ------------------------------------------------------------------ history

    def _conn(self):
        return self.lib.catalog.connect()

    def history(self) -> list[dict]:
        return [json.loads(r[0]) for r in self._conn().execute("SELECT message FROM chat_messages ORDER BY id")]

    def _save(self, msg: dict) -> None:
        stored = {k: v for k, v in msg.items() if k != "images"}
        if msg.get("images"):
            stored["content"] = (msg.get("content") or "") + f" [{len(msg['images'])} image(s) were shown]"
        self._conn().execute("INSERT INTO chat_messages(message, created_at) VALUES (?, ?)",
                             (json.dumps(stored), datetime.now().isoformat(timespec="seconds")))
        self._conn().commit()

    def reset(self) -> None:
        self._conn().execute("DELETE FROM chat_messages")
        self._conn().commit()
        self.lib.catalog.set_setting("claude_code_session", "")

    @staticmethod
    def _window(msgs: list[dict], ctx: str) -> list[dict]:
        """Messages to send this turn.

        The full history is replayed unchanged whenever possible. If it had to be trimmed, or an
        earlier turn was produced under a different system prompt / tool set, provider-native
        reasoning blocks are stripped from the replayed turns (see llm.strip_raw)."""
        w = msgs[-HISTORY:]
        while w and not (w[0]["role"] == "user"):
            w = w[1:]
        intact = len(w) == len(msgs) and all(
            m.get("ctx") == ctx for m in w if m["role"] == "assistant" and m.get("raw") is not None)
        return w if intact else strip_raw(w)

    # ------------------------------------------------------------------ loop

    def run(self, user_text: str) -> dict:
        if self.llm is None:
            return {"reply": "No AI model is configured yet. Open Settings → Assistant to connect one "
                             "(OpenAI, Gemini, Anthropic, a local Ollama model, or any OpenAI-compatible server). "
                             "Everything else, including the review queue, works without it.",
                    "choices": [], "photos": [], "faces": [], "error": "not_configured"}
        if hasattr(self.llm, "run_turn"):  # a harness that runs the tool loop itself (claude_code.py)
            return self._run_harness(user_text)
        user_msg = {"role": "user", "content": user_text}
        self._save(user_msg)
        tools = TOOLS + ([LOOK_TOOL] if self.vision else [])
        ctx = hashlib.sha1((SYSTEM + json.dumps(tools, sort_keys=True)).encode()).hexdigest()[:12]
        msgs = self._window(self.history(), ctx)
        ui = {"choices": [], "photos": [], "faces": []}
        changed = False

        for _ in range(MAX_STEPS):
            try:
                reply = self.llm.chat(SYSTEM, msgs, tools)
            except LLMError as e:
                return {"reply": f"Model error: {e}", **ui, "error": "llm", "changed": changed}
            msg = reply.as_message() | {"ctx": ctx}
            if reply.meta:
                msg["meta"] = reply.meta
                if reply.meta.get("input_transformations"):
                    log.warning("model dropped/flagged reasoning blocks: %s", reply.meta["input_transformations"])
            self._save(msg)
            msgs.append(msg)
            if reply.truncated and not reply.tool_calls:
                return {"reply": reply.content + "\n\n(The reply was cut off; ask me to continue.)", **ui, "changed": changed}
            if not reply.tool_calls:
                return {"reply": reply.content, **ui, "changed": changed}
            for call in reply.tool_calls:
                result, images = self._execute(call, ui)
                changed |= call["name"] not in READ_ONLY
                tmsg = {"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                        "content": json.dumps(result, default=str)}
                if images:
                    tmsg["images"] = images
                self._save(tmsg)
                msgs.append(tmsg)
        return {"reply": "(Stopped after too many steps. Ask me to continue.)", **ui, "changed": changed}

    def _run_harness(self, user_text: str) -> dict:
        self._save({"role": "user", "content": user_text})
        sid = self.lib.catalog.setting("claude_code_session") or None
        ui = {"choices": [], "photos": [], "faces": []}
        try:
            r = self.llm.run_turn(SYSTEM, user_text, sid)
        except LLMError as e:
            return {"reply": f"Model error: {e}", **ui, "error": "llm", "changed": False}
        if r.get("session_id") and r["session_id"] != sid:
            self.lib.catalog.set_setting("claude_code_session", r["session_id"])
        text = r["text"]
        if r.get("denied"):
            text += "\n\n(Some tool calls were not permitted: " + ", ".join(map(str, r["denied"])) + ")"
        if r.get("cut_off"):
            text += "\n\n(Stopped before finishing; ask me to continue.)"
        self._save({"role": "assistant", "content": text, "provider": self.llm.name,
                    "meta": {k: r.get(k) for k in ("cost_usd", "num_turns", "model", "session_id", "usage")}})
        return {"reply": text, **(r.get("ui") or ui), "changed": bool(r.get("changed"))}

    def _execute(self, call: dict, ui: dict) -> tuple[dict, list | None]:
        args = call.get("arguments") or {}
        if "_invalid_arguments" in args:
            return {"error": "arguments were not valid JSON"}, None
        fn = getattr(self, f"t_{call['name']}", None)
        if fn is None or (call["name"] == "look_at_photos" and not self.vision):
            return {"error": f"unknown tool {call['name']}"}, None
        try:
            out = fn(ui=ui, **args)
        except (ValueError, RuleError, KeyError, TypeError) as e:
            return {"error": str(e)}, None
        except Exception as e:  # keep the conversation alive on unexpected failures
            log.exception("tool %s failed", call["name"])
            return {"error": f"internal error: {e}"}, None
        if isinstance(out, tuple):
            return out
        return out, None

    # ------------------------------------------------------------------ tools

    def _expand(self, ids) -> list[str]:
        """Photo id prefixes (as the tools report them) -> full sha256 values."""
        out = []
        for i in ids or []:
            i = str(i)
            if len(i) < 64:
                rows = self._conn().execute("SELECT sha256 FROM photos WHERE sha256 LIKE ? LIMIT 2", (i + "%",)).fetchall()
                if len(rows) == 1:
                    i = rows[0][0]
            out.append(i)
        return out

    def _label(self, vid):
        r = self._conn().execute("SELECT label FROM factor_values WHERE id=?", (vid,)).fetchone()
        return r[0] if r else None

    def t_library_overview(self, ui):
        c = self._conn()
        one = lambda q, *a: c.execute(q, a).fetchone()[0]  # noqa: E731
        return {
            "photos": one("SELECT COUNT(*) FROM photos"),
            "original_files": one("SELECT COUNT(*) FROM sources"),
            "date_range": list(c.execute("SELECT MIN(taken_at), MAX(taken_at) FROM photos").fetchone()),
            "with_location": one("SELECT COUNT(*) FROM photos WHERE lat IS NOT NULL"),
            "screenshots": one("SELECT COUNT(*) FROM photos WHERE is_screenshot=1"),
            "faces": one("SELECT COUNT(*) FROM faces"),
            "people_named": one("SELECT COUNT(*) FROM people WHERE named=1"),
            "people_unnamed": one("SELECT COUNT(*) FROM people WHERE named=0"),
            "events": one("SELECT COUNT(*) FROM events WHERE n>0"),
            "events_named": one("SELECT COUNT(*) FROM events WHERE named=1 AND n>0"),
            "near_duplicate_groups": one("SELECT COUNT(*) FROM similar_groups"),
            "open_questions": one("SELECT COUNT(*) FROM questions WHERE status='open'"),
            "custom_factors": [r[0] for r in c.execute("SELECT name FROM factors WHERE kind='custom'")],
            "face_detection": self.lib.face_engine.available(),
            "content_analysis": self.lib.clip_engine.available(),
            "place_names": self.lib.place_index.available(),
            "settings": {k: self.lib.catalog.setting(k) for k in ("event_gap_hours", "event_min_photos", "dup_policy")},
        }

    def t_list_people(self, ui, include_unnamed=True, limit=30):
        rows = self._conn().execute(
            "SELECT p.value_id id, v.label name, p.named, COUNT(f.id) faces, COUNT(DISTINCT f.sha256) photos "
            "FROM people p JOIN factor_values v ON v.id=p.value_id LEFT JOIN faces f ON f.value_id=p.value_id "
            f"{'' if include_unnamed else 'WHERE p.named=1'} GROUP BY p.value_id ORDER BY photos DESC LIMIT ?",
            (min(int(limit), 200),)).fetchall()
        return {"people": [dict(r) | {"named": bool(r["named"])} for r in rows]}

    def t_list_events(self, ui, limit=30):
        rows = self._conn().execute(
            "SELECT e.value_id id, v.label name, e.named, e.start_at, e.end_at, e.n photos FROM events e "
            "JOIN factor_values v ON v.id=e.value_id WHERE e.n>0 ORDER BY e.start_at DESC LIMIT ?",
            (min(int(limit), 200),)).fetchall()
        return {"events": [dict(r) | {"named": bool(r["named"])} for r in rows]}

    def t_list_factors(self, ui, values_per_factor=15):
        c = self._conn()
        out = []
        for f in c.execute("SELECT * FROM factors ORDER BY id").fetchall():
            vals = c.execute(
                "SELECT v.id, v.label, COUNT(pf.sha256) n, r.rule, lg.prompts FROM factor_values v "
                "LEFT JOIN photo_factor pf ON pf.value_id=v.id LEFT JOIN custom_rules r ON r.value_id=v.id "
                "LEFT JOIN learned_groups lg ON lg.value_id=v.id "
                "WHERE v.factor_id=? GROUP BY v.id ORDER BY n DESC LIMIT ?",
                (f["id"], int(values_per_factor))).fetchall()
            out.append({"factor": f["name"], "kind": f["kind"], "values": [
                {"id": v["id"], "label": v["label"], "photos": v["n"]}
                | ({"rule": json.loads(v["rule"])} if v["rule"] else {})
                | ({"descriptions": json.loads(v["prompts"])} if v["prompts"] else {})
                for v in vals]})
        return {"factors": out}

    def _describe_photos(self, shas):
        c = self._conn()
        out = []
        for s in shas:
            p = c.execute("SELECT taken_at, lat, lon FROM photos WHERE sha256=?", (s,)).fetchone()
            if not p:
                continue
            tags = [r[0] for r in c.execute(
                "SELECT v.label FROM photo_factor pf JOIN factor_values v ON v.id=pf.value_id "
                "JOIN factors f ON f.id=v.factor_id WHERE pf.sha256=? AND f.key IN ('people','events') ", (s,))]
            out.append({"id": s[:SHORT_ID], "taken_at": p["taken_at"], "people_and_events": tags,
                        "has_location": p["lat"] is not None})
        return out

    def t_find_photos(self, ui, rule=None, text=None, limit=12):
        limit = min(int(limit), 24)
        if text:
            shas = self.lib.text_search(text, rule)
            return {"total": len(shas), "sample": self._describe_photos(shas[:limit]),
                    "note": "ranked by visual match; the total is approximate"}
        if not rule:
            raise ValueError("give a rule or text")
        r = self.lib.search(rule, limit)
        return {"total": r["total"], "sample": self._describe_photos(r["shas"])}

    def t_create_content_group(self, ui, name, descriptions, threshold=0.5):
        return self.lib.create_content_group(name, descriptions, threshold)

    def t_teach_group(self, ui, value_id, belong=None, do_not_belong=None):
        return self.lib.teach_group(int(value_id), self._expand(belong), self._expand(do_not_belong))

    def t_name_person(self, ui, person_id, name):
        vid = self.lib.name_person(int(person_id), name)
        return {"ok": True, "person_id": vid, "name": self._label(vid),
                "merged_into_existing": vid != int(person_id)}

    def t_merge_people(self, ui, keep_id, merge_id):
        self.lib.merge_people(int(keep_id), int(merge_id))
        return {"ok": True, "person_id": int(keep_id), "name": self._label(int(keep_id))}

    def t_rename_event(self, ui, event_id, name):
        self.lib.rename_event(int(event_id), name)
        return {"ok": True}

    def t_create_custom_value(self, ui, factor, value, rule=None, photo_ids=None):
        if rule is None and not photo_ids:
            raise ValueError("give a rule or photo_ids")
        return self.lib.create_custom_value(factor, value, rule, self._expand(photo_ids))

    def t_delete_custom_value(self, ui, value_id):
        self.lib.delete_custom_value(int(value_id))
        return {"ok": True}

    def t_update_settings(self, ui, event_gap_hours=None, event_min_photos=None, dup_policy=None):
        out = {}
        if dup_policy:
            self.lib.set_dup_policy(dup_policy)
            out["dup_policy"] = dup_policy
        if event_gap_hours is not None or event_min_photos is not None:
            out["events"] = self.lib.set_event_settings(event_gap_hours, event_min_photos)
        return out or {"error": "nothing to change"}

    def t_list_questions(self, ui, limit=10):
        qlist = qs.list_open(self._conn(), min(int(limit), 50))
        return {"questions": [{"id": q["id"], "kind": q["kind"], "question": q["text"],
                               "photos_affected": round(q["priority"])} | _brief_payload(q) for q in qlist]}

    def t_answer_question(self, ui, question_id, answer):
        answer = dict(answer or {})
        for key in ("accept", "reject"):
            if key in answer:
                answer[key] = self._expand(answer[key])
        return {"ok": True, **self.lib.answer_question(int(question_id), answer)}

    def t_show(self, ui, photo_ids=None, person_ids=None):
        c = self._conn()
        for s in self._expand(photo_ids):
            if c.execute("SELECT 1 FROM photos WHERE sha256=?", (s,)).fetchone() and s not in ui["photos"]:
                ui["photos"].append(s)
        for pid in person_ids or []:
            faces = [r[0] for r in c.execute(
                "SELECT id FROM faces WHERE value_id=? ORDER BY confidence DESC LIMIT 6", (int(pid),))]
            ui["faces"].append({"person_id": int(pid), "name": self._label(int(pid)), "faces": faces})
        return {"ok": True, "shown_photos": len(ui["photos"]), "shown_people": len(ui["faces"])}

    def t_offer_choices(self, ui, options):
        ui["choices"] = [str(o)[:80] for o in options][:6]
        return {"ok": True}

    def t_look_at_photos(self, ui, photo_ids):
        images, seen = [], []
        for s in self._expand(photo_ids[:6]):
            path = self.lib.thumbs_dir / f"{s}.jpg"
            if path.exists():
                images.append({"media_type": "image/jpeg", "data": base64.b64encode(path.read_bytes()).decode()})
                seen.append(s[:SHORT_ID])
        return {"photos": seen, "note": "images attached in this order"}, images


READ_ONLY = {"library_overview", "list_people", "list_events", "list_factors", "find_photos", "list_questions",
             "show", "offer_choices", "look_at_photos"}


def _brief_payload(q) -> dict:
    p = q["payload"]
    keys = {"name_person": ("value_id",), "merge_people": ("a", "b"), "confirm_face": ("value_id", "sha256"),
            "name_event": ("value_id", "start", "end"), "confirm_group": ("value_id", "shas")}.get(q["kind"], ())
    out = {k: p[k] for k in keys if k in p}
    if "sha256" in out:
        out["photo_id"] = out.pop("sha256")[:SHORT_ID]
    if "shas" in out:
        out["photo_ids"] = [s[:SHORT_ID] for s in out.pop("shas")]
    return out
