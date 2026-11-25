"""Event segmentation with stable identity.

Photos with a trustworthy capture time are split into events wherever the time gap exceeds
`event_gap_hours` (or the location jumps far within a short time). On every run the new segments
are matched to existing events by member overlap, so an event keeps its id and user-given name
when late photos (e.g. from a friend's phone) are added to it.
"""
import math
import uuid
from collections import Counter
from datetime import datetime

LOCATION_JUMP_KM = 100


def _km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def segment(rows: list[dict], gap_hours: float) -> list[list[dict]]:
    segs, cur, last_geo = [], [], None
    prev_t = None
    for r in rows:
        t = datetime.fromisoformat(r["taken_at"])
        geo = (r["lat"], r["lon"]) if r["lat"] is not None else None
        split = prev_t is not None and (t - prev_t).total_seconds() > gap_hours * 3600
        if not split and geo and last_geo and _km(geo, last_geo) > LOCATION_JUMP_KM \
                and (t - prev_t).total_seconds() > 3600:
            split = True
        if split and cur:
            segs.append(cur)
            cur = []
        cur.append(r)
        prev_t = t
        last_geo = geo if split else (geo or last_geo)
    if cur:
        segs.append(cur)
    return segs


def auto_label(start: str, end: str, place: str | None = None) -> str:
    s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if s.date() == e.date():
        when = s.strftime("%a %-d %b %Y")
    elif (s.year, s.month) == (e.year, e.month):
        when = f"{s.day}–{e.strftime('%-d %b %Y')}"
    elif s.year == e.year:
        when = f"{s.strftime('%-d %b')} – {e.strftime('%-d %b %Y')}"
    else:
        when = f"{s.strftime('%-d %b %Y')} – {e.strftime('%-d %b %Y')}"
    return f"{place} · {when}" if place else when


def _photo_places(lib, conn) -> dict[str, str]:
    """sha -> most specific place name (city, else region) for photos that have one."""
    fid = lib._factor_id(conn, "place")
    best: dict[str, tuple[int, str]] = {}
    for r in conn.execute(
        "SELECT pf.sha256, v.key, v.label FROM photo_factor pf JOIN factor_values v ON v.id=pf.value_id "
        "WHERE v.factor_id=? AND v.key LIKE '%.%'", (fid,)
    ):
        depth = r["key"].count(".")
        if depth > best.get(r["sha256"], (0, ""))[0]:
            best[r["sha256"]] = (depth, r["label"])
    return {s: label for s, (_, label) in best.items()}


def _segment_place(seg: list[dict], places: dict[str, str]) -> str | None:
    names = Counter(places[r["sha256"]] for r in seg if r["sha256"] in places)
    return names.most_common(1)[0][0] if names else None


def rebuild_events(lib, conn) -> dict:
    gap = float(lib.catalog.setting("event_gap_hours"))
    min_n = int(lib.catalog.setting("event_min_photos"))
    fid = lib._factor_id(conn, "events")

    rows = [dict(r) for r in conn.execute(
        "SELECT sha256, taken_at, lat, lon FROM photos "
        "WHERE taken_src IN ('exif', 'filename') AND is_screenshot=0 ORDER BY taken_at")]
    segs = [s for s in segment(rows, gap) if len(s) >= min_n]

    existing: dict[int, set[str]] = {}
    for r in conn.execute(
        "SELECT pf.value_id, pf.sha256 FROM photo_factor pf JOIN events e ON e.value_id=pf.value_id"
    ):
        existing.setdefault(r["value_id"], set()).add(r["sha256"])
    named = {r[0] for r in conn.execute("SELECT value_id FROM events WHERE named=1")}

    # Greedy best-overlap matching, largest overlaps first.
    seg_sets = [{r["sha256"] for r in s} for s in segs]
    pairs = sorted(
        ((len(ss & members), si, vid) for si, ss in enumerate(seg_sets) for vid, members in existing.items()
         if ss & members),
        reverse=True,
    )
    seg_to_vid, used = {}, set()
    for _, si, vid in pairs:
        if si not in seg_to_vid and vid not in used:
            seg_to_vid[si] = vid
            used.add(vid)

    conn.execute("DELETE FROM photo_factor WHERE value_id IN (SELECT value_id FROM events)")
    places = _photo_places(lib, conn)
    created = 0
    for si, seg in enumerate(segs):
        start, end = seg[0]["taken_at"], seg[-1]["taken_at"]
        label = auto_label(start, end, _segment_place(seg, places))
        vid = seg_to_vid.get(si)
        if vid is None:
            vid = lib._value_id(conn, "events", f"e-{uuid.uuid4().hex[:12]}", label)
            conn.execute("INSERT INTO events(value_id, start_at, end_at, n) VALUES (?, ?, ?, ?)",
                         (vid, start, end, len(seg)))
            created += 1
        else:
            conn.execute("UPDATE events SET start_at=?, end_at=?, n=? WHERE value_id=?", (start, end, len(seg), vid))
            if vid not in named:
                conn.execute("UPDATE factor_values SET label=? WHERE id=?", (label, vid))
        conn.executemany("INSERT OR IGNORE INTO photo_factor(sha256, value_id, source) VALUES (?, ?, 'rule')",
                         [(r["sha256"], vid) for r in seg])

    # Events that no longer match anything: drop unnamed ones; keep named ones (empty) so the name survives.
    gone = set(existing) - used
    for vid in gone - named:
        conn.execute("DELETE FROM events WHERE value_id=?", (vid,))
        conn.execute("DELETE FROM factor_values WHERE id=? AND factor_id=?", (vid, fid))
    for vid in gone & named:
        conn.execute("UPDATE events SET n=0 WHERE value_id=?", (vid,))
    conn.commit()
    return {"events": len(segs), "new_events": created}


def rename_event(conn, value_id: int, name: str) -> None:
    name = name.strip()
    if not name:
        raise ValueError("name is empty")
    if not conn.execute("SELECT 1 FROM events WHERE value_id=?", (value_id,)).fetchone():
        raise ValueError("unknown event")
    conn.execute("UPDATE factor_values SET label=? WHERE id=?", (name, value_id))
    conn.execute("UPDATE events SET named=1 WHERE value_id=?", (value_id,))
    conn.commit()
