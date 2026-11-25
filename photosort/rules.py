"""A small JSON rule language for custom factors and photo search, compiled to SQL.

A rule is a dict. Combinators: {"all": [...]}, {"any": [...]}, {"not": rule}.
Conditions (all optional, several keys in one dict are ANDed):
  date_from / date_to   "YYYY-MM-DD" (inclusive)
  months                [12, 1, 2]           month numbers, any year
  person                "Mom" or a people value id; list = photo has all of them
  any_person            ["Mom", "Dad"]       photo has at least one of them
  min_people / max_people  number of faces in the photo
  event                 event value id or event name
  near                  {"lat": .., "lon": .., "km": ..}
  camera                substring of make/model
  path                  substring of any original file path (e.g. "WhatsApp")
  screenshot            true / false
  value_ids             [ids]  photo is under any of these factor values
  place                 "Goa" / "India" / "Mumbai": country, region or city name
  content               "Food": a content group name
"""
import json
import math

KEYS = {
    "all", "any", "not", "date_from", "date_to", "months", "person", "any_person", "min_people",
    "max_people", "event", "near", "camera", "path", "screenshot", "value_ids", "place", "content",
}


class RuleError(ValueError):
    pass


def _people_id(conn, ref) -> int:
    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        row = conn.execute("SELECT value_id FROM people WHERE value_id=?", (int(ref),)).fetchone()
    else:
        row = conn.execute(
            "SELECT v.id value_id FROM factor_values v JOIN people p ON p.value_id=v.id WHERE lower(v.label)=lower(?)",
            (str(ref).strip(),)).fetchone()
    if not row:
        raise RuleError(f"unknown person: {ref}")
    return row["value_id"]


def _event_id(conn, ref) -> int:
    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        row = conn.execute("SELECT value_id FROM events WHERE value_id=?", (int(ref),)).fetchone()
    else:
        row = conn.execute(
            "SELECT v.id value_id FROM factor_values v JOIN events e ON e.value_id=v.id WHERE lower(v.label)=lower(?)",
            (str(ref).strip(),)).fetchone()
    if not row:
        raise RuleError(f"unknown event: {ref}")
    return row["value_id"]


def compile_rule(conn, rule) -> tuple[str, list]:
    """Return (SQL boolean expression over alias p, params)."""
    if isinstance(rule, str):
        try:
            rule = json.loads(rule)
        except json.JSONDecodeError as e:
            raise RuleError(f"rule is not valid JSON: {e}")
    if not isinstance(rule, dict) or not rule:
        raise RuleError("rule must be a non-empty object")
    unknown = set(rule) - KEYS
    if unknown:
        raise RuleError(f"unknown rule keys: {sorted(unknown)}; allowed: {sorted(KEYS)}")

    parts, params = [], []

    def add(sql, *args):
        parts.append(sql)
        params.extend(args)

    in_value = "p.sha256 IN (SELECT sha256 FROM photo_factor WHERE value_id=?)"
    for key, v in rule.items():
        if key in ("all", "any"):
            if not isinstance(v, list) or not v:
                raise RuleError(f"'{key}' needs a non-empty list")
            subs = [compile_rule(conn, r) for r in v]
            joiner = " AND " if key == "all" else " OR "
            add("(" + joiner.join(s for s, _ in subs) + ")", *[p for _, ps in subs for p in ps])
        elif key == "not":
            s, ps = compile_rule(conn, v)
            add(f"NOT ({s})", *ps)
        elif key == "date_from":
            add("p.taken_at >= ?", str(v)[:10])
        elif key == "date_to":
            add("p.taken_at < date(?, '+1 day')", str(v)[:10])
        elif key == "months":
            months = [int(m) for m in (v if isinstance(v, list) else [v])]
            add(f"CAST(strftime('%m', p.taken_at) AS INTEGER) IN ({','.join('?' * len(months))})", *months)
        elif key == "person":
            for ref in v if isinstance(v, list) else [v]:
                add(in_value, _people_id(conn, ref))
        elif key == "any_person":
            ids = [_people_id(conn, r) for r in v]
            add(f"p.sha256 IN (SELECT sha256 FROM photo_factor WHERE value_id IN ({','.join('?' * len(ids))}))", *ids)
        elif key == "min_people":
            add("(SELECT COUNT(*) FROM faces f WHERE f.sha256=p.sha256) >= ?", int(v))
        elif key == "max_people":
            add("(SELECT COUNT(*) FROM faces f WHERE f.sha256=p.sha256) <= ?", int(v))
        elif key == "event":
            add(in_value, _event_id(conn, v))
        elif key == "near":
            try:
                lat, lon, km = float(v["lat"]), float(v["lon"]), float(v.get("km", 10))
            except (KeyError, TypeError, ValueError):
                raise RuleError("near needs {lat, lon, km}")
            dlat = km / 111.0
            dlon = km / (111.0 * max(0.01, math.cos(math.radians(lat))))
            add("p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ?", lat - dlat, lat + dlat, lon - dlon, lon + dlon)
        elif key == "camera":
            add("(lower(coalesce(p.camera_make,'') || ' ' || coalesce(p.camera_model,'')) LIKE ?)", f"%{str(v).lower()}%")
        elif key == "path":
            add("p.sha256 IN (SELECT sha256 FROM sources WHERE lower(original_path) LIKE ?)", f"%{str(v).lower()}%")
        elif key == "screenshot":
            add("p.is_screenshot = ?", int(bool(v)))
        elif key in ("place", "content"):
            ids = [r[0] for r in conn.execute(
                "SELECT v.id FROM factor_values v JOIN factors f ON f.id=v.factor_id "
                "WHERE f.key=? AND lower(v.label)=lower(?)", (key, str(v).strip()))]
            if not ids:
                raise RuleError(f"unknown {key}: {v}")
            add(f"p.sha256 IN (SELECT sha256 FROM photo_factor WHERE value_id IN ({','.join('?' * len(ids))}))", *ids)
        elif key == "value_ids":
            ids = [int(i) for i in v]
            add(f"p.sha256 IN (SELECT sha256 FROM photo_factor WHERE value_id IN ({','.join('?' * len(ids))}))", *ids)
    return "(" + " AND ".join(parts) + ")", params


def apply_custom_rules(conn) -> dict:
    """Re-evaluate every rule-based custom value (so new imports are sorted the same way)."""
    counts = {}
    for r in conn.execute("SELECT value_id, rule FROM custom_rules").fetchall():
        try:
            sql, params = compile_rule(conn, r["rule"])
        except RuleError:
            continue  # e.g. a person referenced by the rule was renamed; leave existing mapping
        conn.execute("DELETE FROM photo_factor WHERE value_id=? AND source='rule'", (r["value_id"],))
        cur = conn.execute(
            f"INSERT OR IGNORE INTO photo_factor(sha256, value_id, source) SELECT p.sha256, ?, 'rule' "
            f"FROM photos p WHERE {sql}", (r["value_id"], *params))
        counts[r["value_id"]] = cur.rowcount
    conn.commit()
    return counts


def normalize_rule(conn, rule):
    """Resolve person/event names to ids so a saved rule survives renames."""
    if isinstance(rule, str):
        try:
            rule = json.loads(rule)
        except json.JSONDecodeError as e:
            raise RuleError(f"rule is not valid JSON: {e}")
    if not isinstance(rule, dict):
        raise RuleError("rule must be an object")
    out = {}
    for k, v in rule.items():
        if k in ("all", "any"):
            out[k] = [normalize_rule(conn, r) for r in v] if isinstance(v, list) else v
        elif k == "not":
            out[k] = normalize_rule(conn, v)
        elif k == "person":
            out[k] = [_people_id(conn, r) for r in v] if isinstance(v, list) else _people_id(conn, v)
        elif k == "any_person":
            out[k] = [_people_id(conn, r) for r in v]
        elif k == "event":
            out[k] = _event_id(conn, v)
        else:
            out[k] = v
    return out
