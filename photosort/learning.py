"""Learned groups on CLIP embeddings.

Two kinds of group share one mechanism:
- content groups (factor "Content"): defined by text prompts ("a photo of a beach"), assigned
  automatically, and refined by the user's yes/no answers;
- manual collections (custom factor values without a rule): learn from the photos the user put in
  them and *suggest* more, never auto-adding.

Scoring: zero-shot CLIP probability against generic negative prompts, blended with a small
logistic-regression classifier trained on the user's examples once there are enough of them.
The uncertain photos are sent to the review queue in batches; every answer adds training examples.
"""
import json

import numpy as np

NEGATIVE_PROMPTS = [
    "a photo", "a photo of a person", "a group photo of people", "a selfie", "a photo of a room indoors",
    "a photo of a street", "a photo of an object", "a blurry photo", "a photo of a building", "a screenshot",
]
# Phrases were checked against a mixed test set (see tests/ and README): single fruits, sweets and a
# newspaper page were missed by the shorter versions.
DEFAULT_GROUPS = {
    "Documents & receipts": ["a photo of a document", "a photo of a paper receipt", "a scanned page of text",
                             "a photo of a newspaper or printed page", "a photo of handwritten notes or a whiteboard",
                             "a photo of a form, bill or ticket"],
    "Food": ["a photo of food on a plate", "a photo of a meal", "a photo of fruit", "a photo of fruits or vegetables",
             "a photo of sweets or candy", "a photo of a drink"],
    "Pets": ["a photo of a dog", "a photo of a cat"],
    "Beach & sea": ["a photo of a beach", "a photo of the sea"],
    "Nature & mountains": ["a photo of mountains", "a landscape photo of nature", "a photo of a forest",
                           "a photo of flowers or plants"],
    "Buildings & city": ["a photo of a building", "a photo of a house", "a photo of city architecture",
                         "a photo of a street"],
}
ASK_BATCH = 8
MIN_EXAMPLES_TO_TRAIN = 3
ASK_BAND = 0.35           # "in doubt" = score within this distance of the group's threshold
SEARCH_MIN_SCORE = 0.22   # ViT-B/32: real matches score >= ~0.24, unrelated text <= ~0.20
SEARCH_WINDOW = 0.06      # keep results close to the best match


class EmbeddingCache:
    """All photo embeddings as one matrix, rebuilt lazily when the version changes."""

    def __init__(self):
        self.version = -1
        self.shas: list[str] = []
        self.index: dict[str, int] = {}
        self.E = np.zeros((0, 512), np.float32)

    def get(self, conn, version: int):
        if version != self.version:
            rows = conn.execute("SELECT sha256, vec FROM photo_embeddings ORDER BY sha256").fetchall()
            self.shas = [r[0] for r in rows]
            self.index = {s: i for i, s in enumerate(self.shas)}
            self.E = (np.stack([np.frombuffer(r[1], np.float16) for r in rows]).astype(np.float32)
                      if rows else np.zeros((0, 512), np.float32))
            self.version = version
        return self.shas, self.index, self.E


def _softmax_first(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 0] / e.sum(axis=1)


def _logreg(X: np.ndarray, y: np.ndarray, w8: np.ndarray, l2: float = 1e-2, steps: int = 300) -> tuple:
    """Weighted L2 logistic regression by gradient descent; X is small (examples only)."""
    w, b = np.zeros(X.shape[1]), 0.0
    lr = 1.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * w8
        w -= lr * (X.T @ g / w8.sum() + l2 * w)
        b -= lr * g.sum() / w8.sum()
    return w, b


class Learner:
    def __init__(self, lib):
        self.lib = lib
        self.cache = EmbeddingCache()
        self._text_cache: dict[str, np.ndarray] = {}

    @property
    def clip(self):
        return self.lib.clip_engine

    def _texts(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            for t, v in zip(missing, self.clip.embed_texts(missing)):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts])

    def matrix(self, conn):
        return self.cache.get(conn, self.lib.embedding_version)

    # ------------------------------------------------------------------ scoring

    def examples(self, conn, vid: int, manual: bool) -> tuple[set, set]:
        pos, neg = set(), set()
        for r in conn.execute("SELECT sha256, label FROM group_examples WHERE value_id=?", (vid,)):
            (pos if r["label"] > 0 else neg).add(r["sha256"])
        if manual:  # photos the user put in the collection are positives
            pos |= {r[0] for r in conn.execute(
                "SELECT sha256 FROM photo_factor WHERE value_id=? AND source='user'", (vid,))}
        return pos - neg, neg

    def score(self, conn, vid: int) -> np.ndarray | None:
        """Probability per photo (aligned with the cache) that it belongs to group `vid`."""
        g = conn.execute("SELECT * FROM learned_groups WHERE value_id=?", (vid,)).fetchone()
        shas, index, E = self.matrix(conn)
        if g is None or not len(shas):
            return None
        prompts = json.loads(g["prompts"]) if g["prompts"] else []
        pos, neg = self.examples(conn, vid, manual=not prompts)
        pi = [index[s] for s in pos if s in index]
        ni = [index[s] for s in neg if s in index]

        with np.errstate(all="ignore"):
            prior = None
            if prompts:
                own = (E @ self._texts(prompts).T).max(axis=1, keepdims=True)
                negs = E @ self._texts(NEGATIVE_PROMPTS).T
                prior = _softmax_first(100 * np.hstack([own, negs]))
            elif pi:
                c = E[pi].mean(axis=0)
                prior = 1 / (1 + np.exp(-(E @ (c / np.linalg.norm(c)) - 0.80) * 25))
            if prior is None:
                return None
            if len(pi) < MIN_EXAMPLES_TO_TRAIN or (prompts and len(ni) < MIN_EXAMPLES_TO_TRAIN):
                return prior

            # Pseudo-negatives: photos the prior is confident are *not* in the group.
            rng = np.random.default_rng(vid)
            pool = np.setdiff1d(np.nonzero(prior < 0.15)[0], pi)
            pseudo = rng.choice(pool, size=min(len(pool), max(30, 3 * len(pi))), replace=False) if len(pool) else []
            X = np.vstack([E[pi], E[ni]] + ([E[pseudo]] if len(pseudo) else []))
            y = np.concatenate([np.ones(len(pi)), np.zeros(len(ni) + len(pseudo))])
            w8 = np.concatenate([np.ones(len(pi) + len(ni)), np.full(len(pseudo), 0.3)])
            w, b = _logreg(X, y, w8)
            learned = 1 / (1 + np.exp(-(E @ w + b)))
            k = len(pi) + len(ni)
            return (learned * k + prior * 6) / (k + 6)  # trust the user's examples more as they grow

    # ------------------------------------------------------------------ apply / ask

    def apply(self, conn, vid: int) -> int:
        g = conn.execute("SELECT * FROM learned_groups WHERE value_id=?", (vid,)).fetchone()
        if g is None:
            return 0
        p = self.score(conn, vid)
        shas, _, _ = self.matrix(conn)
        pos, neg = self.examples(conn, vid, manual=not g["prompts"])
        conn.execute("DELETE FROM photo_factor WHERE value_id=? AND source='model'", (vid,))
        if g["prompts"]:
            for s in pos:
                conn.execute("INSERT OR REPLACE INTO photo_factor(sha256, value_id, confidence, source) "
                             "VALUES (?, ?, 1.0, 'user')", (s, vid))
        if neg:
            conn.executemany("DELETE FROM photo_factor WHERE value_id=? AND sha256=?", [(vid, s) for s in neg])
        if g["auto"] and p is not None:
            # Screenshots show *pictures of* things (a mountain wallpaper is not a hike): leave them out
            # unless the group is about screenshots.
            skip = set() if "screen" in (g["prompts"] or "").lower() else {
                r[0] for r in conn.execute("SELECT sha256 FROM photos WHERE is_screenshot=1")}
            hits = [(shas[i], vid, round(float(p[i]), 3)) for i in np.nonzero(p >= g["threshold"])[0]
                    if shas[i] not in neg and shas[i] not in pos and shas[i] not in skip]
            conn.executemany("INSERT OR IGNORE INTO photo_factor(sha256, value_id, confidence, source) "
                             "VALUES (?, ?, ?, 'model')", hits)
        return conn.execute("SELECT COUNT(*) FROM photo_factor WHERE value_id=?", (vid,)).fetchone()[0]

    def apply_all(self, conn) -> None:
        for r in conn.execute("SELECT value_id FROM learned_groups").fetchall():
            self.apply(conn, r[0])
        conn.commit()

    def to_ask(self, conn, vid: int) -> tuple[list[str], int]:
        """Most informative photos to ask about, and how many photos are 'in doubt'."""
        g = conn.execute("SELECT * FROM learned_groups WHERE value_id=?", (vid,)).fetchone()
        p = self.score(conn, vid)
        if g is None or p is None:
            return [], 0
        shas, _, _ = self.matrix(conn)
        pos, neg = self.examples(conn, vid, manual=not g["prompts"])
        hidden = {r[0] for r in conn.execute("SELECT sha256 FROM photos WHERE hidden=1")}
        labelled = pos | neg | hidden
        if g["auto"]:
            band = np.nonzero(np.abs(p - g["threshold"]) < ASK_BAND)[0]
            order = band[np.argsort(np.abs(p[band] - g["threshold"]))]
        else:  # manual collection: suggest the most likely additions
            band = np.nonzero(p >= 0.5)[0]
            order = band[np.argsort(-p[band])]
        picks = [shas[i] for i in order if shas[i] not in labelled]
        return picks[:ASK_BATCH], len(picks)

    def teach(self, conn, vid: int, accept: list[str], reject: list[str]) -> None:
        manual = conn.execute("SELECT prompts IS NULL FROM learned_groups WHERE value_id=?", (vid,)).fetchone()
        if manual is None:
            raise ValueError("not a learned group")
        conn.executemany("INSERT OR REPLACE INTO group_examples(value_id, sha256, label) VALUES (?, ?, 1)",
                         [(vid, s) for s in accept])
        conn.executemany("INSERT OR REPLACE INTO group_examples(value_id, sha256, label) VALUES (?, ?, -1)",
                         [(vid, s) for s in reject])
        if manual[0]:  # manual collection: accepted photos join it as the user's own picks
            conn.executemany("INSERT OR REPLACE INTO photo_factor(sha256, value_id, confidence, source) "
                             "VALUES (?, ?, 1.0, 'user')", [(s, vid) for s in accept])
        conn.execute("UPDATE learned_groups SET round = round + 1 WHERE value_id=?", (vid,))
        self.apply(conn, vid)
        conn.commit()

    # ------------------------------------------------------------------ search / near-dups

    def text_search(self, conn, query: str, limit: int = 500) -> list[tuple[str, float]]:
        shas, _, E = self.matrix(conn)
        if not len(shas):
            return []
        with np.errstate(all="ignore"):
            s = E @ self._texts([query])[0]
        order = np.argsort(-s)
        top = float(s[order[0]])
        keep = [(shas[i], float(s[i])) for i in order[:limit] if s[i] >= max(SEARCH_MIN_SCORE, top - SEARCH_WINDOW)]
        return keep

    def similar_pairs(self, conn, new_shas: list[str], window_s: int = 60, min_sim: float = 0.85):
        """Pairs (new photo, neighbour) shot within `window_s` that CLIP sees as near-identical.

        0.85 catches a re-crop of the same scene (measured 0.86) while unrelated photos taken a
        minute apart stay well below 0.80; the time window is the second guard."""
        _, index, E = self.matrix(conn)
        out = []
        for sha in new_shas:
            row = conn.execute("SELECT taken_at, taken_src FROM photos WHERE sha256=?", (sha,)).fetchone()
            if not row or row["taken_src"] == "mtime" or sha not in index:
                continue
            for n in conn.execute(
                "SELECT sha256 FROM photos WHERE sha256 != ? AND taken_src != 'mtime' AND "
                "taken_at BETWEEN strftime('%Y-%m-%dT%H:%M:%S', ?, ?) AND strftime('%Y-%m-%dT%H:%M:%S', ?, ?)",
                (sha, row["taken_at"], f"-{window_s} seconds", row["taken_at"], f"+{window_s} seconds"),
            ):
                j = index.get(n[0])
                if j is not None and float(E[index[sha]] @ E[j]) >= min_sim:
                    out.append((sha, n[0]))
        return out
