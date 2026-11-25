"""Incremental face clustering anchored on the existing library.

Each person (named or not) is a value of the 'people' factor with a centroid = mean of its face
embeddings. A new face joins the closest existing person if similar enough, otherwise starts a new
cluster. Named people and the user's corrections therefore steer every later import.
"""
import numpy as np

# Cosine similarity on SFace embeddings. OpenCV's suggested same-person threshold is 0.363.
AUTO_ASSIGN = 0.45       # confident: assign silently
MAYBE_ASSIGN = 0.363     # likely: assign, but ask the user to confirm if the person is named
AUTO_MERGE = 0.55        # two unnamed clusters this close are merged without asking
SUGGEST_MERGE = 0.40     # ask "same person?"


def emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Centroids:
    """In-memory running means of every person cluster."""

    def __init__(self, conn):
        self.sums: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}
        for r in conn.execute("SELECT value_id, embedding FROM faces WHERE value_id IS NOT NULL"):
            self.add(r["value_id"], emb(r["embedding"]))
        self.negatives: dict[int, set[int]] = {}
        for r in conn.execute("SELECT face_id, value_id FROM face_negatives"):
            self.negatives.setdefault(r["face_id"], set()).add(r["value_id"])

    def add(self, vid: int, e: np.ndarray) -> None:
        self.sums[vid] = self.sums.get(vid, 0) + e
        self.counts[vid] = self.counts.get(vid, 0) + 1

    def matrix(self):
        ids = list(self.sums)
        if not ids:
            return ids, np.zeros((0, 128), np.float32)
        m = np.stack([self.sums[i] for i in ids])
        return ids, m / np.linalg.norm(m, axis=1, keepdims=True)

    def best(self, e: np.ndarray, exclude: set[int] = frozenset()) -> tuple[int | None, float]:
        ids, m = self.matrix()
        if not ids:
            return None, -1.0
        sims = m @ e
        for i, vid in enumerate(ids):
            if vid in exclude:
                sims[i] = -1
        k = int(np.argmax(sims))
        return ids[k], float(sims[k])


def new_person(lib, conn) -> int:
    n = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] + 1
    while True:
        key = f"p{n}"
        if not conn.execute(
            "SELECT 1 FROM factor_values v JOIN factors f ON f.id=v.factor_id WHERE f.key='people' AND v.key=?", (key,)
        ).fetchone():
            break
        n += 1
    vid = lib._value_id(conn, "people", key, f"Person {n}")
    conn.execute("INSERT OR IGNORE INTO people(value_id) VALUES (?)", (vid,))
    return vid


def cluster_new_faces(lib, conn) -> int:
    """Assign every face without a person. Returns the number assigned."""
    cents = Centroids(conn)
    rows = conn.execute("SELECT id, embedding FROM faces WHERE value_id IS NULL ORDER BY id").fetchall()
    for r in rows:
        e = emb(r["embedding"])
        vid, sim = cents.best(e, cents.negatives.get(r["id"], set()))
        if vid is None or sim < MAYBE_ASSIGN:
            vid, sim = new_person(lib, conn), 1.0
        conn.execute("UPDATE faces SET value_id=?, confidence=? WHERE id=?", (vid, round(sim, 4), r["id"]))
        cents.add(vid, e)
    auto_merge_unnamed(conn, cents)
    conn.commit()
    return len(rows)


def auto_merge_unnamed(conn, cents: Centroids) -> None:
    """Greedy clustering can split one person into two clusters; fold near-identical unnamed ones together."""
    named = {r[0] for r in conn.execute("SELECT value_id FROM people WHERE named=1")}
    while True:
        ids, m = cents.matrix()
        if len(ids) < 2:
            return
        sims = m @ m.T
        np.fill_diagonal(sims, -1)
        is_named = np.array([i in named for i in ids])
        sims[np.outer(is_named, is_named)] = -1  # never merge two named people silently
        pos = {v: k for k, v in enumerate(ids)}
        for r in conn.execute(  # a face the user rejected from B must not come back to B via a merge
            "SELECT DISTINCT f.value_id a, n.value_id b FROM face_negatives n JOIN faces f ON f.id=n.face_id"
        ):
            if r["a"] in pos and r["b"] in pos:
                sims[pos[r["a"]], pos[r["b"]]] = sims[pos[r["b"]], pos[r["a"]]] = -1
        i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
        if sims[i, j] < AUTO_MERGE:
            return
        a, b = ids[i], ids[j]
        if a in named:
            keep, drop = a, b
        elif b in named:
            keep, drop = b, a
        else:
            keep, drop = (a, b) if cents.counts[a] >= cents.counts[b] else (b, a)
        merge_people(conn, keep, drop)
        cents.sums[keep] = cents.sums[keep] + cents.sums.pop(drop)
        cents.counts[keep] += cents.counts.pop(drop)


def merge_people(conn, keep: int, drop: int) -> None:
    conn.execute("UPDATE faces SET value_id=? WHERE value_id=?", (keep, drop))
    conn.execute("UPDATE OR IGNORE face_negatives SET value_id=? WHERE value_id=?", (keep, drop))
    conn.execute("DELETE FROM face_negatives WHERE value_id=?", (drop,))
    conn.execute("DELETE FROM people WHERE value_id=?", (drop,))
    conn.execute("DELETE FROM photo_factor WHERE value_id=?", (drop,))
    conn.execute("DELETE FROM factor_values WHERE id=?", (drop,))


def merge_suggestions(conn) -> list[tuple[int, int, float]]:
    cents = Centroids(conn)
    ids, m = cents.matrix()
    if len(ids) < 2:
        return []
    sims = m @ m.T
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if SUGGEST_MERGE <= sims[i, j] < 1:
                out.append((ids[i], ids[j], float(sims[i, j])))
    return out


def sync_people_mapping(lib, conn) -> None:
    """Rebuild photo -> person rows from faces (photo is in a person's view if any face is theirs)."""
    fid = lib._factor_id(conn, "people")
    conn.execute(
        "DELETE FROM photo_factor WHERE value_id IN (SELECT id FROM factor_values WHERE factor_id=?)", (fid,)
    )
    conn.execute(
        "INSERT OR IGNORE INTO photo_factor(sha256, value_id, confidence, source) "
        "SELECT sha256, value_id, MAX(confidence), CASE WHEN MAX(source='user') THEN 'user' ELSE 'model' END "
        "FROM faces WHERE value_id IS NOT NULL GROUP BY sha256, value_id"
    )
    # Drop empty clusters, refresh cover faces (highest-confidence face).
    conn.execute(
        "DELETE FROM people WHERE named=0 AND value_id NOT IN (SELECT DISTINCT value_id FROM faces WHERE value_id IS NOT NULL)"
    )
    conn.execute(
        "DELETE FROM factor_values WHERE factor_id=? AND id NOT IN (SELECT value_id FROM people)", (fid,)
    )
    conn.execute(
        "UPDATE people SET cover_face = (SELECT id FROM faces f WHERE f.value_id = people.value_id "
        "ORDER BY f.source='user' DESC, f.confidence DESC, f.score DESC LIMIT 1)"
    )
    conn.commit()


def name_person(lib, conn, value_id: int, name: str) -> int:
    """Name a cluster. If someone already has that name, merge into them. Returns the surviving id."""
    name = name.strip()
    if not name:
        raise ValueError("name is empty")
    fid = lib._factor_id(conn, "people")
    existing = conn.execute(
        "SELECT v.id FROM factor_values v JOIN people p ON p.value_id=v.id "
        "WHERE v.factor_id=? AND lower(v.label)=lower(?) AND v.id != ?", (fid, name, value_id)
    ).fetchone()
    if existing:
        merge_people(conn, existing["id"], value_id)
        value_id = existing["id"]
    conn.execute("UPDATE factor_values SET label=? WHERE id=?", (name, value_id))
    conn.execute("UPDATE people SET named=1 WHERE value_id=?", (value_id,))
    conn.execute("UPDATE faces SET source='user', confidence=1.0 WHERE value_id=? AND confidence >= ?",
                 (value_id, AUTO_ASSIGN))
    sync_people_mapping(lib, conn)
    return value_id


def reject_face(lib, conn, face_id: int) -> None:
    """'This is not them': remember it, then re-cluster this face elsewhere."""
    row = conn.execute("SELECT value_id FROM faces WHERE id=?", (face_id,)).fetchone()
    if not row:
        raise ValueError("unknown face")
    if row["value_id"] is not None:
        conn.execute("INSERT OR IGNORE INTO face_negatives(face_id, value_id) VALUES (?, ?)", (face_id, row["value_id"]))
    conn.execute("UPDATE faces SET value_id=NULL, source='model' WHERE id=?", (face_id,))
    cluster_new_faces(lib, conn)
    sync_people_mapping(lib, conn)


def confirm_face(conn, face_id: int) -> None:
    conn.execute("UPDATE faces SET source='user', confidence=1.0 WHERE id=?", (face_id,))
    conn.commit()
