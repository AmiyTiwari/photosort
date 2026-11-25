"""HTTP API + static UI."""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm as llm_mod
from . import questions as qs
from .agent import Agent
from .library import FACTOR_ORDER, ImportBusy, Library
from .rules import RuleError

STATIC = Path(__file__).parent / "static"


class ImportReq(BaseModel):
    path: str
    mode: str = "copy"


class ShaReq(BaseModel):
    sha256: str


class SettingsReq(BaseModel):
    dup_policy: str | None = None
    event_gap_hours: float | None = None
    event_min_photos: int | None = None


class NameReq(BaseModel):
    name: str


class MergeReq(BaseModel):
    keep: int
    drop: int


class AnswerReq(BaseModel):
    answer: dict


class CustomReq(BaseModel):
    factor: str
    value: str
    rule: dict | None = None
    shas: list[str] | None = None


class LLMReq(BaseModel):
    preset: str = "none"
    base_url: str = ""
    model: str = ""
    vision: bool = False
    api_key: str | None = None      # None = keep the stored key, "" = clear it
    effort: str | None = None       # anthropic adapter: low | medium | high | xhigh | max
    binding_check: bool = False     # anthropic adapter diagnostic, see llm.AnthropicMessages


class ChatReq(BaseModel):
    message: str


class ContentReq(BaseModel):
    name: str
    prompts: list[str]
    threshold: float = 0.5


class TeachReq(BaseModel):
    accept: list[str] = []
    reject: list[str] = []


class ExportReq(BaseModel):
    factor: str
    dest: str
    include_hidden: bool = False


def create_app(library: Library, llm_transport=None) -> FastAPI:
    app = FastAPI(title="PhotoSort")
    lib = library
    secrets_path = lib.root / "secrets.json"

    def db():
        return lib.catalog.connect()

    # ---------------------------------------------------------------- AI model config (any provider)

    def llm_config() -> dict:
        return json.loads(lib.catalog.setting("llm") or "{}")

    def stored_key() -> str | None:
        try:
            key = json.loads(secrets_path.read_text()).get("llm_api_key")
        except (OSError, ValueError):
            key = None
        return key or os.environ.get("PHOTOSORT_LLM_API_KEY") or None

    def make_agent() -> Agent:
        cfg = llm_config()
        try:
            model = llm_mod.from_config(cfg, stored_key(), llm_transport, library_root=lib.root)
        except llm_mod.LLMError as e:
            raise HTTPException(400, str(e))
        return Agent(lib, model, vision=bool(cfg.get("vision")))

    @app.get("/api/llm")
    def get_llm():
        return {"config": llm_config(), "key_set": bool(stored_key()), "presets": llm_mod.PRESETS}

    @app.post("/api/llm")
    def set_llm(req: LLMReq):
        if req.preset != "none" and req.preset not in llm_mod.PRESETS:
            raise HTTPException(400, "unknown preset")
        preset = llm_mod.PRESETS.get(req.preset, {})
        if req.effort and req.effort not in ("low", "medium", "high", "xhigh", "max"):
            raise HTTPException(400, "effort must be low, medium, high, xhigh or max")
        cfg = {"preset": req.preset, "adapter": preset.get("adapter", "none"),
               "base_url": req.base_url.strip() or preset.get("base_url", ""),
               "model": req.model.strip(), "vision": req.vision, "effort": req.effort or None,
               "binding_check": req.binding_check}
        lib.catalog.set_setting("llm", json.dumps(cfg))
        if req.api_key is not None:  # stored only on this machine, readable only by this user
            fd = os.open(secrets_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"llm_api_key": req.api_key.strip()}, f)
        return get_llm()

    @app.post("/api/llm/test")
    def test_llm():
        agent = make_agent()
        if agent.llm is None:
            raise HTTPException(400, "no model configured")
        try:
            if hasattr(agent.llm, "ping"):
                text = agent.llm.ping()
            else:
                text = agent.llm.chat("Reply with the single word: ok", [{"role": "user", "content": "ping"}], []).content
        except llm_mod.LLMError as e:
            raise HTTPException(502, str(e))
        return {"ok": True, "reply": text[:200]}

    # ---------------------------------------------------------------- assistant chat

    @app.get("/api/chat")
    def chat_history():
        out = []
        for m in Agent(lib, None).history():
            if m["role"] == "user" or (m["role"] == "assistant" and m.get("content")):
                out.append({"role": m["role"], "content": m["content"]})
        return out

    @app.post("/api/chat")
    def chat(req: ChatReq):
        text = req.message.strip()
        if not text:
            raise HTTPException(400, "empty message")
        return make_agent().run(text[:4000])

    @app.delete("/api/chat")
    def chat_reset():
        Agent(lib, None).reset()
        return {"ok": True}

    # ---------------------------------------------------------------- overview

    @app.get("/api/stats")
    def stats():
        c = db()
        one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
        return {
            "library": str(lib.root),
            "photos": one("SELECT COUNT(*) FROM photos"),
            "sources": one("SELECT COUNT(*) FROM sources"),
            "stored_bytes": one("SELECT COALESCE(SUM(bytes), 0) FROM photos"),
            "source_bytes": one("SELECT COALESCE(SUM(size), 0) FROM sources"),
            "similar_groups": one("SELECT COUNT(*) FROM similar_groups"),
            "hidden": one("SELECT COUNT(*) FROM photos WHERE hidden=1"),
            "dup_policy": lib.catalog.setting("dup_policy"),
            "event_gap_hours": float(lib.catalog.setting("event_gap_hours")),
            "event_min_photos": int(lib.catalog.setting("event_min_photos")),
            "people": one("SELECT COUNT(*) FROM people"),
            "events": one("SELECT COUNT(*) FROM events WHERE n>0"),
            "open_questions": one("SELECT COUNT(*) FROM questions WHERE status='open'"),
            "face_detection": lib.face_engine.available(),
            "content_analysis": lib.clip_engine.available(),
            "place_names": lib.place_index.available(),
            "busy": one("SELECT COUNT(*) FROM import_batches WHERE status='running'") > 0,
        }

    # ---------------------------------------------------------------- import

    @app.post("/api/imports")
    def start_import(req: ImportReq):
        try:
            return {"id": lib.start_import(req.path, req.mode)}
        except ImportBusy as e:
            raise HTTPException(409, str(e))
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/imports")
    def list_imports():
        return [dict(r) for r in db().execute("SELECT * FROM import_batches ORDER BY id DESC LIMIT 50")]

    @app.get("/api/fs")
    def browse_fs(path: str = "~"):
        """List subdirectories so the UI can offer a folder picker."""
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise HTTPException(400, "not a directory")
        try:
            dirs = sorted(
                (e.name for e in os.scandir(p) if e.is_dir() and not e.name.startswith(".")), key=str.lower
            )
        except PermissionError:
            dirs = []
        return {"path": str(p), "parent": str(p.parent), "dirs": dirs}

    # ---------------------------------------------------------------- browse

    @app.get("/api/factors")
    def factors(show_hidden: bool = False):
        c = db()
        hid = "" if show_hidden else " AND p.hidden=0"
        counts = {
            r["value_id"]: r["n"]
            for r in c.execute(
                "SELECT pf.value_id, COUNT(*) n FROM photo_factor pf JOIN photos p ON p.sha256=pf.sha256 "
                f"WHERE 1=1{hid} GROUP BY pf.value_id"
            )
        }
        named = {r[0] for r in c.execute("SELECT value_id FROM people WHERE named=1")}
        out = []
        factors = c.execute("SELECT * FROM factors").fetchall()
        rank = {k: i for i, k in enumerate(FACTOR_ORDER)}
        for f in sorted(factors, key=lambda f: (rank.get(f["key"], len(rank)), f["id"])):
            vals = [dict(v) | {"count": counts.get(v["id"], 0)} for v in
                    c.execute("SELECT id, key, label, parent_id FROM factor_values WHERE factor_id=?", (f["id"],))]
            vals = [v for v in vals if v["count"]]
            if f["kind"] == "people":  # hide one-off strangers; they stay reachable from the People screen
                vals = [v | {"named": v["id"] in named} for v in vals if v["id"] in named or v["count"] >= 2]
                vals.sort(key=lambda v: (not v["named"], -v["count"], v["label"]))
            elif f["kind"] in ("date", "event"):
                if f["kind"] == "event":
                    starts = dict(c.execute("SELECT value_id, start_at FROM events").fetchall())
                    for v in vals:
                        v["key"] = starts.get(v["id"], "")
                vals.sort(key=lambda v: v["key"], reverse=True)
            else:
                vals.sort(key=lambda v: (-v["count"], v["label"]))
            out.append({"key": f["key"], "name": f["name"], "kind": f["kind"], "values": vals})
        return out

    @app.get("/api/photos")
    def photos(value_id: int | None = None, offset: int = 0, limit: int = 200, show_hidden: bool = False):
        limit = max(1, min(limit, 1000))
        where, args = ["1=1"], []
        if value_id is not None:
            where.append("p.sha256 IN (SELECT sha256 FROM photo_factor WHERE value_id=?)")
            args.append(value_id)
        if not show_hidden:
            where.append("p.hidden=0")
        w = " AND ".join(where)
        c = db()
        total = c.execute(f"SELECT COUNT(*) FROM photos p WHERE {w}", args).fetchone()[0]
        rows = c.execute(
            "SELECT p.sha256, p.taken_at, p.width, p.height, p.hidden, ps.group_id, "
            "(SELECT COUNT(*) FROM photo_similar s WHERE s.group_id = ps.group_id) AS group_size, "
            "(SELECT COUNT(*) FROM sources s WHERE s.sha256 = p.sha256) AS n_sources "
            f"FROM photos p LEFT JOIN photo_similar ps ON ps.sha256 = p.sha256 WHERE {w} "
            "ORDER BY p.taken_at DESC LIMIT ? OFFSET ?",
            [*args, limit, offset],
        )
        return {"total": total, "items": [dict(r) for r in rows]}

    @app.get("/api/search")
    def search(q: str, offset: int = 0, limit: int = 200):
        q = q.strip()
        if not q:
            raise HTTPException(400, "empty query")
        shas = bad_request(lambda: lib.text_search(q[:300]))
        page = shas[offset:offset + max(1, min(limit, 1000))]
        c = db()
        rows = {r["sha256"]: dict(r) for r in c.execute(
            "SELECT p.sha256, p.taken_at, p.width, p.height, p.hidden, ps.group_id, "
            "(SELECT COUNT(*) FROM photo_similar s WHERE s.group_id = ps.group_id) AS group_size, "
            "(SELECT COUNT(*) FROM sources s WHERE s.sha256 = p.sha256) AS n_sources "
            f"FROM photos p LEFT JOIN photo_similar ps ON ps.sha256 = p.sha256 "
            f"WHERE p.sha256 IN ({','.join('?' * len(page))})", page)} if page else {}
        return {"total": len(shas), "items": [rows[s] for s in page if s in rows]}

    @app.get("/api/photos/{sha}")
    def photo(sha: str):
        c = db()
        p = c.execute("SELECT * FROM photos WHERE sha256=?", (sha,)).fetchone()
        if not p:
            raise HTTPException(404)
        sources = [dict(r) for r in c.execute(
            "SELECT original_path, size, batch_id FROM sources WHERE sha256=? ORDER BY id", (sha,))]
        factor_vals = [dict(r) for r in c.execute(
            "SELECT f.name factor, v.id value_id, v.label, pf.source FROM photo_factor pf "
            "JOIN factor_values v ON v.id=pf.value_id JOIN factors f ON f.id=v.factor_id "
            "WHERE pf.sha256=? ORDER BY f.id, v.parent_id IS NOT NULL", (sha,))]
        g = c.execute(
            "SELECT g.* FROM photo_similar ps JOIN similar_groups g ON g.id=ps.group_id WHERE ps.sha256=?", (sha,)
        ).fetchone()
        faces = [dict(r) for r in c.execute(
            "SELECT f.id, f.x, f.y, f.w, f.h, f.value_id, f.confidence, f.source, v.label FROM faces f "
            "LEFT JOIN factor_values v ON v.id=f.value_id WHERE f.sha256=? ORDER BY f.x", (sha,))]
        return {"photo": dict(p), "sources": sources, "factors": factor_vals, "group": dict(g) if g else None,
                "faces": faces}

    @app.get("/api/photos/{sha}/file")
    def photo_file(sha: str):
        p = db().execute("SELECT store_path FROM photos WHERE sha256=?", (sha,)).fetchone()
        if not p:
            raise HTTPException(404)
        return FileResponse(lib.root / p["store_path"])

    @app.get("/thumbs/{sha}.jpg")
    def thumb(sha: str):
        if not all(ch in "0123456789abcdef" for ch in sha) or len(sha) != 64:
            raise HTTPException(400)
        path = lib.thumbs_dir / f"{sha}.jpg"
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    # ---------------------------------------------------------------- near-duplicates

    def group_with_members(c, g) -> dict:
        members = [dict(r) for r in c.execute(
            "SELECT p.sha256, p.width, p.height, p.bytes, p.quality, p.taken_at, p.taken_src, p.hidden "
            "FROM photo_similar ps JOIN photos p ON p.sha256=ps.sha256 WHERE ps.group_id=? "
            "ORDER BY p.taken_at", (g["id"],))]
        return dict(g) | {"members": members}

    @app.get("/api/similar")
    def similar(offset: int = 0, limit: int = 50):
        c = db()
        groups = c.execute(
            "SELECT g.*, COUNT(ps.sha256) n FROM similar_groups g JOIN photo_similar ps ON ps.group_id=g.id "
            "GROUP BY g.id ORDER BY n DESC, g.id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = c.execute("SELECT COUNT(*) FROM similar_groups").fetchone()[0]
        return {"total": total, "groups": [group_with_members(c, g) for g in groups]}

    @app.get("/api/similar/{gid}")
    def similar_group(gid: int):
        c = db()
        g = c.execute("SELECT * FROM similar_groups WHERE id=?", (gid,)).fetchone()
        if not g:
            raise HTTPException(404)
        return group_with_members(c, g)

    @app.post("/api/similar/{gid}/best")
    def set_best(gid: int, req: ShaReq):
        try:
            lib.set_group_best(gid, req.sha256)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/similar/{gid}/remove")
    def remove_member(gid: int, req: ShaReq):
        lib.remove_from_group(gid, req.sha256)
        return {"ok": True}

    @app.post("/api/settings")
    def settings(req: SettingsReq):
        try:
            if req.dup_policy:
                lib.set_dup_policy(req.dup_policy)
            if req.event_gap_hours is not None or req.event_min_photos is not None:
                lib.set_event_settings(req.event_gap_hours, req.event_min_photos)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    # ---------------------------------------------------------------- people, events, questions, custom

    def bad_request(fn):
        try:
            return fn()
        except (ValueError, RuleError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/analyze")
    def analyze():
        try:
            return {"id": lib.start_analysis()}
        except ImportBusy as e:
            raise HTTPException(409, str(e))

    @app.get("/api/people")
    def people(include_singletons: bool = False):
        rows = db().execute(
            "SELECT p.value_id id, v.label name, p.named, p.cover_face, COUNT(f.id) faces, "
            "COUNT(DISTINCT f.sha256) photos FROM people p JOIN factor_values v ON v.id=p.value_id "
            "LEFT JOIN faces f ON f.value_id=p.value_id GROUP BY p.value_id "
            f"{'' if include_singletons else 'HAVING p.named=1 OR faces >= 2'} "
            "ORDER BY p.named DESC, photos DESC").fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/people/{vid}/faces")
    def person_faces(vid: int, limit: int = 60):
        return [dict(r) for r in db().execute(
            "SELECT id, sha256, confidence, source FROM faces WHERE value_id=? ORDER BY confidence DESC LIMIT ?",
            (vid, min(limit, 500)))]

    @app.post("/api/people/{vid}/name")
    def name_person(vid: int, req: NameReq):
        return {"id": bad_request(lambda: lib.name_person(vid, req.name))}

    @app.post("/api/people/merge")
    def merge_people(req: MergeReq):
        bad_request(lambda: lib.merge_people(req.keep, req.drop))
        return {"ok": True}

    @app.post("/api/faces/{fid}/reject")
    def reject_face(fid: int):
        bad_request(lambda: lib.reject_face(fid))
        return {"ok": True}

    @app.get("/faces/{fid}.jpg")
    def face_crop(fid: int):
        path = lib.faces_dir / f"{int(fid)}.jpg"
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @app.post("/api/events/{vid}/name")
    def name_event(vid: int, req: NameReq):
        bad_request(lambda: lib.rename_event(vid, req.name))
        return {"ok": True}

    @app.get("/api/questions")
    def questions(limit: int = 30):
        return qs.list_open(db(), min(limit, 200))

    @app.post("/api/questions/{qid}/answer")
    def answer_question(qid: int, req: AnswerReq):
        return bad_request(lambda: lib.answer_question(qid, req.answer))

    @app.post("/api/custom")
    def create_custom(req: CustomReq):
        if req.rule is None and not req.shas:
            raise HTTPException(400, "give a rule or photos")
        return bad_request(lambda: lib.create_custom_value(req.factor, req.value, req.rule, req.shas))

    @app.get("/api/content")
    def content_groups():
        return [dict(r) | {"prompts": json.loads(r["prompts"]) if r["prompts"] else None} for r in db().execute(
            "SELECT lg.value_id id, v.label name, f.name factor, lg.prompts, lg.threshold, lg.auto, "
            "(SELECT COUNT(*) FROM photo_factor pf WHERE pf.value_id=lg.value_id) photos, "
            "(SELECT COUNT(*) FROM group_examples e WHERE e.value_id=lg.value_id) examples "
            "FROM learned_groups lg JOIN factor_values v ON v.id=lg.value_id JOIN factors f ON f.id=v.factor_id "
            "ORDER BY lg.prompts IS NULL, v.label")]

    @app.post("/api/content")
    def create_content(req: ContentReq):
        return bad_request(lambda: lib.create_content_group(req.name, req.prompts, req.threshold))

    @app.post("/api/groups/{vid}/teach")
    def teach(vid: int, req: TeachReq):
        return bad_request(lambda: lib.teach_group(vid, req.accept, req.reject))

    @app.get("/api/custom")
    def list_custom():
        return [dict(r) | {"rule": json.loads(r["rule"]) if r["rule"] else None} for r in db().execute(
            "SELECT v.id, f.name factor, v.label value, r.rule, "
            "(SELECT COUNT(*) FROM photo_factor pf WHERE pf.value_id=v.id) photos FROM factor_values v "
            "JOIN factors f ON f.id=v.factor_id LEFT JOIN custom_rules r ON r.value_id=v.id "
            "WHERE f.kind='custom' ORDER BY f.name, v.label")]

    @app.delete("/api/custom/{vid}")
    def delete_custom(vid: int):
        bad_request(lambda: lib.delete_custom_value(vid))
        return {"ok": True}

    # ---------------------------------------------------------------- export

    @app.post("/api/export")
    def export(req: ExportReq):
        try:
            return lib.export_view(req.factor, req.dest, req.include_hidden)
        except ValueError as e:
            raise HTTPException(400, str(e))

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
    return app
