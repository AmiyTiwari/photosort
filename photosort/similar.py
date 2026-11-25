"""Near-duplicate detection over perceptual hashes.

A BK-tree over pHash gives sub-linear Hamming-radius lookups, so each new photo
is compared against the whole existing library, not only its own batch.
"""
from datetime import datetime

# Same image re-encoded / resized / lightly compressed.
VARIANT_PHASH = 4
VARIANT_DHASH = 8
# Burst / near-identical shots: looser hash match but taken close together.
BURST_PHASH = 14
BURST_SECONDS = 10


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    def __init__(self):
        self.root = None  # node: [value, key, {distance: child}]

    def add(self, value: int, key) -> None:
        if self.root is None:
            self.root = [value, key, {}]
            return
        node = self.root
        while True:
            d = hamming(value, node[0])
            child = node[2].get(d)
            if child is None:
                node[2][d] = [value, key, {}]
                return
            node = child

    def search(self, value: int, radius: int):
        """Yield (key, distance) for all entries within radius."""
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = hamming(value, node[0])
            if d <= radius:
                yield node[1], d
            for dist, child in node[2].items():
                if d - radius <= dist <= d + radius:
                    stack.append(child)


def _seconds_apart(a: str | None, b: str | None) -> float:
    if not a or not b:
        return float("inf")
    return abs((datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds())


def classify_pair(p: dict, q: dict) -> str | None:
    """Return 'variant', 'burst' or None for two photo rows."""
    dp = hamming(int(p["phash"], 16), int(q["phash"], 16))
    dd = hamming(int(p["dhash"], 16), int(q["dhash"], 16))
    if dp <= VARIANT_PHASH and dd <= VARIANT_DHASH:
        return "variant"
    if dp <= BURST_PHASH and _seconds_apart(p["taken_at"], q["taken_at"]) <= BURST_SECONDS:
        return "burst"
    return None


def best_of(rows: list[dict]) -> str:
    """Pick the copy to keep: most pixels, then real EXIF date, then sharpest, then largest file."""
    return max(
        rows,
        key=lambda r: (
            (r["width"] or 0) * (r["height"] or 0),
            r["taken_src"] == "exif",
            r["quality"] or 0,
            r["bytes"] or 0,
        ),
    )["sha256"]
