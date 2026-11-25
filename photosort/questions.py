"""Review queue: the few questions whose answers sort the most photos.

Questions have a stable `key`; once answered or dismissed they are not asked again.
Priority ~ number of photos an answer affects.
"""
import json
from datetime import datetime

from . import events as ev
from . import people as pp

NAME_PERSON_MIN_FACES = 3
NAME_EVENT_MIN_PHOTOS = 8


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _upsert(conn, key, kind, priority, payload) -> None:
    row = conn.execute("SELECT status FROM questions WHERE key=?", (key,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO questions(key, kind, priority, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (key, kind, priority, json.dumps(payload), _now()))
    elif row["status"] in ("open", "stale"):
        conn.execute("UPDATE questions SET status='open', priority=?, payload=? WHERE key=?",
                     (priority, json.dumps(payload), key))


def refresh_questions(lib, conn) -> int:
    live: set[str] = set()

    for r in conn.execute(
        "SELECT p.value_id, COUNT(f.id) n, COUNT(DISTINCT f.sha256) photos FROM people p "
        "JOIN faces f ON f.value_id=p.value_id WHERE p.named=0 GROUP BY p.value_id HAVING n >= ?",
        (NAME_PERSON_MIN_FACES,)).fetchall():
        faces = [x[0] for x in conn.execute(
            "SELECT id FROM faces WHERE value_id=? ORDER BY confidence DESC, score DESC LIMIT 8", (r["value_id"],))]
        key = f"name_person:{r['value_id']}"
        _upsert(conn, key, "name_person", r["photos"], {"value_id": r["value_id"], "faces": faces, "photos": r["photos"]})
        live.add(key)

    for a, b, sim in pp.merge_suggestions(conn):
        na, nb = (conn.execute("SELECT COUNT(*) FROM faces WHERE value_id=?", (v,)).fetchone()[0] for v in (a, b))
        if min(na, nb) < 2:
            continue
        a, b = sorted((a, b))
        key = f"merge_people:{a}:{b}"
        fa, fb = (conn.execute("SELECT cover_face FROM people WHERE value_id=?", (v,)).fetchone()[0] for v in (a, b))
        _upsert(conn, key, "merge_people", (na + nb) * sim,
                {"a": a, "b": b, "face_a": fa, "face_b": fb, "similarity": round(sim, 3)})
        live.add(key)

    for r in conn.execute(
        "SELECT f.id, f.value_id, f.sha256, f.confidence FROM faces f JOIN people p ON p.value_id=f.value_id "
        "WHERE p.named=1 AND f.source='model' AND f.confidence < ? ORDER BY f.confidence LIMIT 200",
        (pp.AUTO_ASSIGN,)).fetchall():
        key = f"confirm_face:{r['id']}:{r['value_id']}"
        _upsert(conn, key, "confirm_face", 1.0,
                {"face_id": r["id"], "value_id": r["value_id"], "sha256": r["sha256"], "confidence": r["confidence"]})
        live.add(key)

    for r in conn.execute(
        "SELECT value_id, n, start_at, end_at FROM events WHERE named=0 AND n >= ?", (NAME_EVENT_MIN_PHOTOS,)
    ).fetchall():
        sample = [x[0] for x in conn.execute(
            "SELECT p.sha256 FROM photo_factor pf JOIN photos p ON p.sha256=pf.sha256 WHERE pf.value_id=? "
            "ORDER BY p.quality DESC LIMIT 6", (r["value_id"],))]
        key = f"name_event:{r['value_id']}"
        _upsert(conn, key, "name_event", r["n"] / 2,
                {"value_id": r["value_id"], "photos": r["n"], "start": r["start_at"], "end": r["end_at"], "sample": sample})
        live.add(key)

    if lib.clip_engine.available():
        for g in conn.execute(
            "SELECT lg.value_id, lg.round, lg.auto, v.label FROM learned_groups lg "
            "JOIN factor_values v ON v.id=lg.value_id").fetchall():
            picks, in_doubt = lib.learner.to_ask(conn, g["value_id"])
            if not picks:
                continue
            key = f"confirm_group:{g['value_id']}:{g['round']}"
            _upsert(conn, key, "confirm_group", min(in_doubt, 200) * (1.0 if g["auto"] else 0.5),
                    {"value_id": g["value_id"], "shas": picks, "in_doubt": in_doubt, "suggest": not g["auto"]})
            live.add(key)

    # Subjects that disappeared (merged person, re-segmented event) go stale.
    for r in conn.execute("SELECT key FROM questions WHERE status='open'").fetchall():
        if r["key"] not in live:
            conn.execute("UPDATE questions SET status='stale' WHERE key=?", (r["key"],))
    conn.commit()
    return len(live)


def list_open(conn, limit=50) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM questions WHERE status='open' ORDER BY priority DESC, id LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        q = dict(r)
        q["payload"] = json.loads(q["payload"])
        q["text"] = describe(conn, q)
        out.append(q)
    return out


def describe(conn, q) -> str:
    p = q["payload"]
    label = lambda vid: (conn.execute("SELECT label FROM factor_values WHERE id=?", (vid,)).fetchone() or ["?"])[0]  # noqa: E731
    if q["kind"] == "name_person":
        return f"Who is this? ({label(p['value_id'])}, in {p['photos']} photos)"
    if q["kind"] == "merge_people":
        return f"Are {label(p['a'])} and {label(p['b'])} the same person?"
    if q["kind"] == "confirm_face":
        return f"Is this {label(p['value_id'])}?"
    if q["kind"] == "name_event":
        return f"What was this event? ({label(p['value_id'])}, {p['photos']} photos)"
    if q["kind"] == "confirm_group":
        if p.get("suggest"):
            return f"Add these to {label(p['value_id'])}?"
        return f"Which of these belong in {label(p['value_id'])}?"
    return q["kind"]


def answer(lib, conn, qid: int, ans: dict) -> dict:
    """Apply an answer. ans: {"skip": true} or kind-specific fields."""
    q = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
    if not q:
        raise ValueError("unknown question")
    if q["status"] != "open":
        raise ValueError(f"question is {q['status']}")
    p = json.loads(q["payload"])
    result: dict = {}
    if ans.get("skip"):
        status = "dismissed"
    elif q["kind"] == "name_person":
        result["value_id"] = pp.name_person(lib, conn, p["value_id"], ans["name"])
        status = "answered"
    elif q["kind"] == "merge_people":
        if ans.get("same"):
            named = {r[0] for r in conn.execute("SELECT value_id FROM people WHERE named=1 AND value_id IN (?, ?)",
                                                (p["a"], p["b"]))}
            keep, drop = (p["b"], p["a"]) if p["b"] in named and p["a"] not in named else (p["a"], p["b"])
            pp.merge_people(conn, keep, drop)
            pp.sync_people_mapping(lib, conn)
            result["value_id"] = keep
        status = "answered"  # "not the same" is remembered by the key
    elif q["kind"] == "confirm_face":
        if ans.get("correct"):
            pp.confirm_face(conn, p["face_id"])
        else:
            pp.reject_face(lib, conn, p["face_id"])
        status = "answered"
    elif q["kind"] == "name_event":
        ev.rename_event(conn, p["value_id"], ans["name"])
        status = "answered"
    elif q["kind"] == "confirm_group":
        asked = set(p["shas"])
        accept = [s for s in ans.get("accept", []) if s in asked]
        reject = [s for s in ans.get("reject", []) if s in asked]
        conn.execute("UPDATE questions SET status='answered', answer=?, answered_at=? WHERE id=?",
                     (json.dumps(ans), _now(), qid))  # before teach(), which asks the next batch
        lib.learner.teach(conn, p["value_id"], accept, reject)
        refresh_questions(lib, conn)
        return {"accepted": len(accept), "rejected": len(reject)}
    else:
        raise ValueError(f"cannot answer {q['kind']}")
    conn.execute("UPDATE questions SET status=?, answer=?, answered_at=? WHERE id=?",
                 (status, json.dumps(ans), _now(), qid))
    conn.commit()
    refresh_questions(lib, conn)
    return result
