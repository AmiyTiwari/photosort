"""The photo library: single-copy store, import pipeline, factor mapping, near-dup groups."""
import functools
import json
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import events as ev
from . import people as pp
from . import questions as qs
from . import rules
from .analyze import IMAGE_EXTS, analyze_file, date_from_filename
from .clip import ClipEngine
from .db import Catalog
from .faces import FaceEngine
from .learning import DEFAULT_GROUPS, Learner
from .places import PlaceIndex
from .similar import BURST_PHASH, BKTree, best_of, classify_pair

log = logging.getLogger("photosort")

BUILTIN_FACTORS = [
    ("date", "Date", "date"),
    ("people", "People", "people"),
    ("events", "Events", "event"),
    ("content", "Content", "content"),
    ("place", "Places", "place"),
    ("camera", "Camera", "camera"),
    ("type", "Type", "type"),
]
FACTOR_ORDER = [k for k, _, _ in BUILTIN_FACTORS]

PHOTO_COLS = (
    "sha256, ext, store_path, bytes, width, height, taken_at, taken_src, lat, lon, "
    "camera_make, camera_model, phash, dhash, quality, is_screenshot, hidden, batch_id, imported_at"
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ImportBusy(RuntimeError):
    pass


def _tx(fn):
    """Commit on success, roll back on any error so a failed request never leaves the DB locked."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        conn = self.catalog.connect()
        try:
            out = fn(self, *args, **kwargs)
            conn.commit()
            return out
        except BaseException:
            conn.rollback()
            raise
    return wrapper


class Library:
    def __init__(self, root: Path, workers: int | None = None, face_engine: FaceEngine | None = None,
                 clip_engine: ClipEngine | None = None, place_index: PlaceIndex | None = None):
        self.root = Path(root).expanduser().resolve()
        self.store_dir = self.root / "store"
        self.thumbs_dir = self.root / "thumbs"
        self.faces_dir = self.root / "faces"
        for d in (self.store_dir, self.thumbs_dir, self.faces_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.face_engine = face_engine or FaceEngine()
        self.clip_engine = clip_engine or ClipEngine()
        self.place_index = place_index or PlaceIndex()
        self.learner = Learner(self)
        self.embedding_version = 0
        self.catalog = Catalog(self.root / "catalog.db")
        self.workers = workers or min(8, (os.cpu_count() or 4))
        self._import_lock = threading.Lock()
        self._seed_factors()

    # ------------------------------------------------------------------ factors

    def _seed_factors(self) -> None:
        with self.catalog.connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO factors(key, name, kind) VALUES (?, ?, ?)", BUILTIN_FACTORS
            )

    def _factor_id(self, conn, key: str) -> int:
        return conn.execute("SELECT id FROM factors WHERE key=?", (key,)).fetchone()["id"]

    def _value_id(self, conn, factor_key: str, key: str, label: str, parent_id: int | None = None) -> int:
        fid = self._factor_id(conn, factor_key)
        conn.execute(
            "INSERT OR IGNORE INTO factor_values(factor_id, key, label, parent_id) VALUES (?, ?, ?, ?)",
            (fid, key, label, parent_id),
        )
        return conn.execute(
            "SELECT id FROM factor_values WHERE factor_id=? AND key=?", (fid, key)
        ).fetchone()["id"]

    def _assign_builtin_factors(self, conn, p: dict) -> None:
        """Rule-based mapping of one photo to the built-in factors."""
        values: list[int] = []

        taken = datetime.fromisoformat(p["taken_at"])
        year = self._value_id(conn, "date", f"{taken.year}", f"{taken.year}")
        values += [year, self._value_id(conn, "date", taken.strftime("%Y-%m"), taken.strftime("%B %Y"), year)]

        values += self._place_values(conn, p["lat"], p["lon"])

        cam = " ".join(x for x in (p["camera_make"], p["camera_model"]) if x)
        if p["camera_make"] and p["camera_model"] and p["camera_model"].startswith(p["camera_make"]):
            cam = p["camera_model"]
        values.append(self._value_id(conn, "camera", cam.lower() or "unknown", cam or "Unknown camera"))

        values.append(
            self._value_id(conn, "type", "screenshot", "Screenshots")
            if p["is_screenshot"]
            else self._value_id(conn, "type", "photo", "Photos")
        )

        conn.executemany(
            "INSERT OR IGNORE INTO photo_factor(sha256, value_id, source) VALUES (?, ?, 'rule')",
            [(p["sha256"], v) for v in values],
        )

    def _place_values(self, conn, lat, lon) -> list[int]:
        """Country › Region › City from the offline index; a 0.1° grid cell if the index is missing."""
        if lat is None:
            return [self._value_id(conn, "place", "none", "No location")]
        hit = self.place_index.lookup(lat, lon) if self.place_index.available() else None
        if hit:
            out, parent = [], None
            for level in ("country", "region", "city"):
                if hit[level]:
                    parent = self._value_id(conn, "place", hit[level][0], hit[level][1], parent)
                    out.append(parent)
            return out
        rlat, rlon = round(lat, 1), round(lon, 1)
        label = f"{abs(rlat):.1f}°{'N' if rlat >= 0 else 'S'}, {abs(rlon):.1f}°{'E' if rlon >= 0 else 'W'}"
        return [self._value_id(conn, "place", f"{rlat},{rlon}", label)]

    def _refresh_places(self, conn) -> None:
        """Once the place index exists, replace grid-cell places with real names."""
        if not self.place_index.available():
            return
        fid = self._factor_id(conn, "place")
        rows = conn.execute(
            "SELECT DISTINCT p.sha256, p.lat, p.lon FROM photos p JOIN photo_factor pf ON pf.sha256=p.sha256 "
            "JOIN factor_values v ON v.id=pf.value_id WHERE v.factor_id=? AND v.key LIKE '%,%' AND p.lat IS NOT NULL",
            (fid,)).fetchall()
        for r in rows:
            conn.execute("DELETE FROM photo_factor WHERE sha256=? AND value_id IN "
                         "(SELECT id FROM factor_values WHERE factor_id=?)", (r["sha256"], fid))
            conn.executemany("INSERT OR IGNORE INTO photo_factor(sha256, value_id, source) VALUES (?, ?, 'rule')",
                             [(r["sha256"], v) for v in self._place_values(conn, r["lat"], r["lon"])])
        conn.execute("DELETE FROM factor_values WHERE factor_id=? AND id NOT IN (SELECT value_id FROM photo_factor) "
                     "AND id NOT IN (SELECT parent_id FROM factor_values WHERE parent_id IS NOT NULL)", (fid,))
        conn.commit()

    # ------------------------------------------------------------------ store

    def _store_rel(self, sha: str, ext: str) -> str:
        return f"store/{sha[:2]}/{sha[2:4]}/{sha}.{ext}"

    def _place_in_store(self, src: Path, rel: str, mode: str) -> None:
        dest = self.root / rel
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(src, dest)
                return
            except OSError:  # cross-device or unsupported: fall back to a copy
                pass
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)

    # ------------------------------------------------------------------ import

    def start_import(self, root: str, mode: str = "copy") -> int:
        """Create a batch and run it on a background thread. Returns the batch id."""
        src = Path(root).expanduser().resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"Not a directory: {src}")
        if src == self.root or self.root in src.parents:
            raise ValueError("Cannot import from inside the library itself")
        if mode not in ("copy", "hardlink"):
            raise ValueError("mode must be 'copy' or 'hardlink'")
        if not self._import_lock.acquire(blocking=False):
            raise ImportBusy("An import is already running")
        try:
            with self.catalog.connect() as conn:
                batch_id = conn.execute(
                    "INSERT INTO import_batches(root, mode, status, started_at) VALUES (?, ?, 'running', ?)",
                    (str(src), mode, _now()),
                ).lastrowid
        except Exception:
            self._import_lock.release()
            raise
        threading.Thread(target=self._run_import, args=(batch_id, src, mode), daemon=True).start()
        return batch_id

    def import_sync(self, root: str, mode: str = "copy") -> dict:
        """Run an import in the calling thread (CLI / tests)."""
        batch_id = self.start_import(root, mode)
        with self._import_lock:  # released by the worker thread when it finishes
            return self.batch(batch_id)

    def batch(self, batch_id: int) -> dict:
        row = self.catalog.connect().execute("SELECT * FROM import_batches WHERE id=?", (batch_id,)).fetchone()
        return dict(row) if row else {}

    def _update_batch(self, conn, batch_id: int, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE import_batches SET {cols} WHERE id=?", (*fields.values(), batch_id))
        conn.commit()

    def _scan(self, src: Path) -> list[Path]:
        files = []
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.endswith(".photoslibrary")]
            for name in filenames:
                if not name.startswith(".") and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    files.append(Path(dirpath) / name)
        return files

    def _build_bktree(self, conn) -> BKTree:
        tree = BKTree()
        for row in conn.execute("SELECT sha256, phash FROM photos WHERE phash IS NOT NULL"):
            tree.add(int(row["phash"], 16), row["sha256"])
        return tree

    def _run_import(self, batch_id: int, src: Path, mode: str) -> None:
        conn = self.catalog.connect()
        try:
            self._update_batch(conn, batch_id, stage="scanning")
            files = self._scan(src)
            self._update_batch(conn, batch_id, total=len(files), stage="importing")

            # Resume: skip files already recorded with identical size + mtime.
            todo, skipped = [], 0
            for f in files:
                st = f.stat()
                row = conn.execute(
                    "SELECT size, mtime FROM sources WHERE original_path=?", (str(f),)
                ).fetchone()
                if row and row["size"] == st.st_size and abs(row["mtime"] - st.st_mtime) < 1e-6:
                    skipped += 1
                else:
                    todo.append(f)
            stats = {"processed": skipped, "skipped": skipped, "new_photos": 0, "exact_dupes": 0, "errors": 0}
            self._update_batch(conn, batch_id, **stats)

            tree = self._build_bktree(conn)
            touched_groups: set[int] = set()

            with ThreadPoolExecutor(self.workers) as pool:
                futures = {pool.submit(analyze_file, f, self.thumbs_dir): f for f in todo}
                for i, fut in enumerate(as_completed(futures), 1):
                    f = futures[fut]
                    try:
                        info = fut.result()
                        self._ingest_one(conn, batch_id, info, mode, tree, touched_groups, stats)
                    except Exception as e:  # unreadable/corrupt file: record and move on
                        log.warning("skip %s: %s", f, e)
                        stats["errors"] += 1
                    stats["processed"] += 1
                    if i % 25 == 0:
                        self._update_batch(conn, batch_id, **stats)

            self._refresh_groups(conn, touched_groups)
            n_groups = len({g for g in touched_groups if self._group_exists(conn, g)})
            self._update_batch(conn, batch_id, **stats, similar_groups=n_groups)
            self._analyse(conn, batch_id)
            self._update_batch(conn, batch_id, status="done", stage="done", finished_at=_now())
        except Exception as e:
            log.exception("import %s failed", batch_id)
            conn.rollback()
            self._update_batch(conn, batch_id, status="failed", message=str(e), finished_at=_now())
        finally:
            self._import_lock.release()

    def _ingest_one(self, conn, batch_id, info, mode, tree, touched_groups, stats) -> None:
        sha = info["sha256"]
        exists = conn.execute("SELECT 1 FROM photos WHERE sha256=?", (sha,)).fetchone()
        if exists:
            stats["exact_dupes"] += 1
        else:
            rel = self._store_rel(sha, info["ext"])
            self._place_in_store(Path(info["path"]), rel, mode)
            photo = {
                **{k: info.get(k) for k in (
                    "sha256", "ext", "width", "height", "taken_at", "taken_src", "lat", "lon",
                    "camera_make", "camera_model", "phash", "dhash", "quality", "is_screenshot",
                )},
                "store_path": rel,
                "bytes": info["size"],
                "hidden": 0,
                "batch_id": batch_id,
                "imported_at": _now(),
            }
            cols = ", ".join(photo)
            conn.execute(
                f"INSERT INTO photos({cols}) VALUES ({', '.join('?' * len(photo))})", tuple(photo.values())
            )
            self._assign_builtin_factors(conn, photo)
            self._link_similar(conn, photo, tree, touched_groups)
            tree.add(int(photo["phash"], 16), sha)
            stats["new_photos"] += 1

        conn.execute(
            "INSERT INTO sources(sha256, original_path, size, mtime, batch_id) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(original_path) DO UPDATE SET sha256=excluded.sha256, size=excluded.size, "
            "mtime=excluded.mtime, batch_id=excluded.batch_id",
            (sha, info["path"], info["size"], info["mtime"], batch_id),
        )
        conn.commit()

    # ------------------------------------------------------------------ analysis (phase 3)

    def start_analysis(self) -> int:
        """Re-run faces/events/rules/questions over the whole library in the background."""
        if not self._import_lock.acquire(blocking=False):
            raise ImportBusy("A job is already running")
        try:
            with self.catalog.connect() as conn:
                batch_id = conn.execute(
                    "INSERT INTO import_batches(root, mode, status, stage, started_at) "
                    "VALUES ('(library)', 'analyze', 'running', 'faces', ?)", (_now(),)).lastrowid
        except Exception:
            self._import_lock.release()
            raise

        def run():
            conn = self.catalog.connect()
            try:
                self._analyse(conn, batch_id)
                self._update_batch(conn, batch_id, status="done", stage="done", finished_at=_now())
            except Exception as e:
                log.exception("analysis failed")
                conn.rollback()
                self._update_batch(conn, batch_id, status="failed", message=str(e), finished_at=_now())
            finally:
                self._import_lock.release()

        threading.Thread(target=run, daemon=True).start()
        return batch_id

    def analyse_sync(self) -> dict:
        batch_id = self.start_analysis()
        with self._import_lock:
            return self.batch(batch_id)

    def _analyse(self, conn, batch_id: int) -> None:
        self._backfill_filename_dates(conn)
        self._update_batch(conn, batch_id, stage="faces")
        found = self._detect_faces(conn, batch_id)
        self._update_batch(conn, batch_id, faces_found=found, stage="content")
        new = self._embed_new(conn)
        self._update_batch(conn, batch_id, stage="people")
        pp.cluster_new_faces(self, conn)
        pp.sync_people_mapping(self, conn)
        self._refresh_places(conn)
        if new:
            touched: set[int] = set()
            for a, b in self.learner.similar_pairs(conn, new):
                self._union(conn, a, b, "burst", touched)
            self._refresh_groups(conn, touched)
        self._update_batch(conn, batch_id, stage="events")
        ev.rebuild_events(self, conn)
        rules.apply_custom_rules(conn)
        if self.clip_engine.available():
            self.learner.apply_all(conn)
        qs.refresh_questions(self, conn)

    def _backfill_filename_dates(self, conn) -> None:
        """Photos imported before filename dates existed (or with only a file mtime) get a better date."""
        fid = self._factor_id(conn, "date")
        for r in conn.execute("SELECT sha256 FROM photos WHERE taken_src='mtime'").fetchall():
            for s in conn.execute("SELECT original_path FROM sources WHERE sha256=?", (r["sha256"],)).fetchall():
                taken = date_from_filename(Path(s["original_path"]).name)
                if taken:
                    conn.execute("UPDATE photos SET taken_at=?, taken_src='filename' WHERE sha256=?", (taken, r["sha256"]))
                    conn.execute("DELETE FROM photo_factor WHERE sha256=? AND value_id IN "
                                 "(SELECT id FROM factor_values WHERE factor_id=?)", (r["sha256"], fid))
                    self._assign_builtin_factors(conn, self._photo(conn, r["sha256"]))
                    break
        conn.commit()

    def _embed_new(self, conn, batch: int = 256) -> list[str]:
        """CLIP-embed photos that have no embedding yet (from their thumbnails). Returns their shas."""
        if not self.clip_engine.available():
            log.info("CLIP not available; skipping content analysis")
            return []
        todo = [r[0] for r in conn.execute("SELECT sha256 FROM photos WHERE clip_done=0")]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            vecs = self.clip_engine.embed_images([self.thumbs_dir / f"{s}.jpg" for s in chunk])
            conn.executemany("INSERT OR REPLACE INTO photo_embeddings(sha256, vec) VALUES (?, ?)",
                             [(s, v.astype("float16").tobytes()) for s, v in zip(chunk, vecs) if v.any()])
            conn.executemany("UPDATE photos SET clip_done=1 WHERE sha256=?", [(s,) for s in chunk])
            conn.commit()
        if todo:
            self.embedding_version += 1
        if self.catalog.setting("content_seeded") == "0":
            for name, prompts in DEFAULT_GROUPS.items():
                self._content_group(conn, name, prompts)
            self.catalog.set_setting("content_seeded", "1")
        return todo

    def _content_group(self, conn, name: str, prompts: list[str], threshold: float = 0.5) -> int:
        vid = self._value_id(conn, "content", name.strip().lower(), name.strip())
        conn.execute(
            "INSERT INTO learned_groups(value_id, prompts, threshold, auto) VALUES (?, ?, ?, 1) "
            "ON CONFLICT(value_id) DO UPDATE SET prompts=excluded.prompts, threshold=excluded.threshold",
            (vid, json.dumps(prompts), threshold))
        return vid

    def _detect_faces(self, conn, batch_id: int) -> int:
        if not self.face_engine.available():
            log.info("face models not available; skipping people detection")
            return 0
        todo = conn.execute("SELECT sha256, store_path FROM photos WHERE faces_done=0").fetchall()
        found = 0
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self.face_engine.detect, self.root / r["store_path"]): r["sha256"] for r in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                sha = futures[fut]
                try:
                    faces = fut.result()
                except Exception as e:
                    log.warning("face detection failed for %s: %s", sha, e)
                    faces = []
                for f in faces:
                    fid = conn.execute(
                        "INSERT INTO faces(sha256, x, y, w, h, score, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sha, f["x"], f["y"], f["w"], f["h"], f["score"], f["embedding"].tobytes())).lastrowid
                    if f["crop"]:
                        (self.faces_dir / f"{fid}.jpg").write_bytes(f["crop"])
                found += len(faces)
                conn.execute("UPDATE photos SET faces_done=1 WHERE sha256=?", (sha,))
                if i % 25 == 0:
                    conn.commit()
                    self._update_batch(conn, batch_id, faces_found=found)
        conn.commit()
        return found

    # People / events / custom factors: thin wrappers used by the API and the agent.

    @_tx
    def name_person(self, value_id: int, name: str) -> int:
        conn = self.catalog.connect()
        vid = pp.name_person(self, conn, value_id, name)
        rules.apply_custom_rules(conn)
        qs.refresh_questions(self, conn)
        return vid

    @_tx
    def merge_people(self, keep: int, drop: int) -> None:
        conn = self.catalog.connect()
        for v in (keep, drop):
            if not conn.execute("SELECT 1 FROM people WHERE value_id=?", (v,)).fetchone():
                raise ValueError(f"unknown person {v}")
        if keep == drop:
            raise ValueError("cannot merge a person with themself")
        if conn.execute("SELECT named FROM people WHERE value_id=?", (drop,)).fetchone()[0]:
            conn.execute("UPDATE people SET named=1 WHERE value_id=?", (keep,))
        pp.merge_people(conn, keep, drop)
        pp.sync_people_mapping(self, conn)
        rules.apply_custom_rules(conn)
        qs.refresh_questions(self, conn)

    @_tx
    def reject_face(self, face_id: int) -> None:
        conn = self.catalog.connect()
        pp.reject_face(self, conn, face_id)
        qs.refresh_questions(self, conn)

    @_tx
    def rename_event(self, value_id: int, name: str) -> None:
        conn = self.catalog.connect()
        ev.rename_event(conn, value_id, name)
        qs.refresh_questions(self, conn)

    @_tx
    def set_event_settings(self, gap_hours: float | None = None, min_photos: int | None = None) -> dict:
        if gap_hours is not None:
            if not 0.25 <= float(gap_hours) <= 24 * 30:
                raise ValueError("event_gap_hours must be between 0.25 and 720")
            self.catalog.set_setting("event_gap_hours", str(float(gap_hours)))
        if min_photos is not None:
            if not 2 <= int(min_photos) <= 1000:
                raise ValueError("event_min_photos must be between 2 and 1000")
            self.catalog.set_setting("event_min_photos", str(int(min_photos)))
        conn = self.catalog.connect()
        out = ev.rebuild_events(self, conn)
        rules.apply_custom_rules(conn)
        qs.refresh_questions(self, conn)
        return out

    @_tx
    def create_custom_value(self, factor_name: str, value_name: str, rule=None, shas: list[str] | None = None) -> dict:
        """Create (or update) a custom factor value. With a rule it is re-applied on every import;
        without one it is a manual collection of the given photos."""
        conn = self.catalog.connect()
        factor_name, value_name = factor_name.strip(), value_name.strip()
        if not factor_name or not value_name:
            raise ValueError("factor and value names are required")
        fkey = "custom:" + factor_name.lower()
        builtin = conn.execute("SELECT kind FROM factors WHERE lower(name)=lower(?)", (factor_name,)).fetchone()
        if builtin and builtin["kind"] != "custom":
            raise ValueError(f"'{factor_name}' is a built-in factor; pick another name")
        conn.execute("INSERT OR IGNORE INTO factors(key, name, kind) VALUES (?, ?, 'custom')", (fkey, factor_name))
        vid = self._value_id(conn, fkey, value_name.lower(), value_name)
        if rule is not None:
            rule = rules.normalize_rule(conn, rule)
            rules.compile_rule(conn, rule)  # validate before saving
            conn.execute("INSERT OR REPLACE INTO custom_rules(value_id, rule) VALUES (?, ?)", (vid, json.dumps(rule)))
            rules.apply_custom_rules(conn)
        else:  # manual collection: learn from its photos and suggest more
            conn.execute("INSERT OR IGNORE INTO learned_groups(value_id, prompts, auto) VALUES (?, NULL, 0)", (vid,))
        if shas:
            conn.executemany("INSERT OR IGNORE INTO photo_factor(sha256, value_id, source) "
                             "SELECT sha256, ?, 'user' FROM photos WHERE sha256=?", [(vid, s) for s in shas])
        qs.refresh_questions(self, conn)  # a collection with photos can start suggesting similar ones
        n = conn.execute("SELECT COUNT(*) FROM photo_factor WHERE value_id=?", (vid,)).fetchone()[0]
        return {"value_id": vid, "factor": factor_name, "value": value_name, "photos": n}

    @_tx
    def delete_custom_value(self, value_id: int) -> None:
        conn = self.catalog.connect()
        row = conn.execute("SELECT f.kind, f.id FROM factor_values v JOIN factors f ON f.id=v.factor_id WHERE v.id=?",
                           (value_id,)).fetchone()
        if not row or row["kind"] not in ("custom", "content"):
            raise ValueError("only custom groups and content groups can be deleted")
        for table in ("photo_factor", "custom_rules", "learned_groups", "group_examples"):
            conn.execute(f"DELETE FROM {table} WHERE value_id=?", (value_id,))
        conn.execute("DELETE FROM factor_values WHERE id=?", (value_id,))
        if row["kind"] == "custom" and not conn.execute(
                "SELECT 1 FROM factor_values WHERE factor_id=?", (row["id"],)).fetchone():
            conn.execute("DELETE FROM factors WHERE id=?", (row["id"],))
        conn.commit()

    @_tx
    def create_content_group(self, name: str, prompts: list[str], threshold: float = 0.5) -> dict:
        """A group defined by what is in the photo, e.g. ("Beach", ["a photo of a beach"])."""
        if not self.clip_engine.available():
            raise ValueError("content analysis is off; run `python -m photosort models` first")
        prompts = [p.strip() for p in prompts if p and p.strip()][:8]
        if not name.strip() or not prompts:
            raise ValueError("a name and at least one description are required")
        if not 0.05 <= float(threshold) <= 0.95:
            raise ValueError("threshold must be between 0.05 and 0.95")
        conn = self.catalog.connect()
        vid = self._content_group(conn, name, prompts, float(threshold))
        n = self.learner.apply(conn, vid)
        qs.refresh_questions(self, conn)
        return {"value_id": vid, "name": name.strip(), "photos": n}

    @_tx
    def teach_group(self, value_id: int, accept: list[str], reject: list[str]) -> dict:
        conn = self.catalog.connect()
        self.learner.teach(conn, value_id, accept or [], reject or [])
        qs.refresh_questions(self, conn)
        n = conn.execute("SELECT COUNT(*) FROM photo_factor WHERE value_id=?", (value_id,)).fetchone()[0]
        return {"value_id": value_id, "photos": n}

    def text_search(self, query: str, rule=None, limit: int = 500) -> list[str]:
        """Plain-language search over CLIP embeddings, optionally narrowed by a rule."""
        if not self.clip_engine.available():
            raise ValueError("content analysis is off; run `python -m photosort models` first")
        conn = self.catalog.connect()
        hits = [s for s, _ in self.learner.text_search(conn, query, limit)]
        hidden = {r[0] for r in conn.execute("SELECT sha256 FROM photos WHERE hidden=1")}
        hits = [s for s in hits if s not in hidden]
        if rule:
            allowed = set(self.search(rule, limit=10**9)["shas"])
            hits = [s for s in hits if s in allowed]
        return hits

    @_tx
    def answer_question(self, qid: int, answer: dict) -> dict:
        return qs.answer(self, self.catalog.connect(), qid, answer)

    def search(self, rule, limit: int = 12) -> dict:
        conn = self.catalog.connect()
        sql, params = rules.compile_rule(conn, rule)
        total = conn.execute(f"SELECT COUNT(*) FROM photos p WHERE p.hidden=0 AND {sql}", params).fetchone()[0]
        rows = conn.execute(f"SELECT p.sha256 FROM photos p WHERE p.hidden=0 AND {sql} "
                            "ORDER BY p.taken_at DESC LIMIT ?", (*params, limit)).fetchall()
        return {"total": total, "shas": [r[0] for r in rows]}

    # ------------------------------------------------------------------ near-duplicates

    def _photo(self, conn, sha: str) -> dict | None:
        row = conn.execute(f"SELECT {PHOTO_COLS} FROM photos WHERE sha256=?", (sha,)).fetchone()
        return dict(row) if row else None

    def _group_of(self, conn, sha: str) -> int | None:
        row = conn.execute("SELECT group_id FROM photo_similar WHERE sha256=?", (sha,)).fetchone()
        return row["group_id"] if row else None

    def _group_exists(self, conn, gid: int) -> bool:
        return conn.execute("SELECT 1 FROM similar_groups WHERE id=?", (gid,)).fetchone() is not None

    def _link_similar(self, conn, photo: dict, tree: BKTree, touched: set[int]) -> None:
        """Union the new photo with every existing near-duplicate in the library."""
        for other_sha, _ in tree.search(int(photo["phash"], 16), BURST_PHASH):
            other = self._photo(conn, other_sha)
            kind = classify_pair(photo, other) if other else None
            if not kind:
                continue
            self._union(conn, photo["sha256"], other_sha, kind, touched)

    def _union(self, conn, a: str, b: str, kind: str, touched: set[int]) -> None:
        """Put photos a and b in the same near-duplicate group (merging groups if needed)."""
        g_new, g_old = self._group_of(conn, a), self._group_of(conn, b)
        if g_new is None and g_old is None:
            gid = conn.execute("INSERT INTO similar_groups(kind) VALUES (?)", (kind,)).lastrowid
            conn.executemany("INSERT INTO photo_similar(sha256, group_id) VALUES (?, ?)", [(a, gid), (b, gid)])
        elif g_new is None or g_old is None:
            gid = g_new if g_new is not None else g_old
            conn.execute("INSERT INTO photo_similar(sha256, group_id) VALUES (?, ?)", (a if g_new is None else b, gid))
        elif g_new != g_old:
            gid = g_old
            conn.execute("UPDATE photo_similar SET group_id=? WHERE group_id=?", (gid, g_new))
            conn.execute("DELETE FROM similar_groups WHERE id=?", (g_new,))
            touched.discard(g_new)
        else:
            touched.add(g_new)
            return  # already grouped together; keep the group's kind
        if kind == "burst":
            conn.execute("UPDATE similar_groups SET kind='burst' WHERE id=?", (gid,))
        touched.add(gid)

    def _refresh_groups(self, conn, gids) -> None:
        """Recompute the best copy (unless the user chose one) and apply the hide policy."""
        hide = self.catalog.setting("dup_policy") == "hide_non_best"
        for gid in gids:
            g = conn.execute("SELECT * FROM similar_groups WHERE id=?", (gid,)).fetchone()
            if not g:
                continue
            members = [
                dict(r) for r in conn.execute(
                    f"SELECT {PHOTO_COLS} FROM photos WHERE sha256 IN "
                    "(SELECT sha256 FROM photo_similar WHERE group_id=?)", (gid,)
                )
            ]
            if len(members) < 2:
                conn.execute("DELETE FROM photo_similar WHERE group_id=?", (gid,))
                conn.execute("DELETE FROM similar_groups WHERE id=?", (gid,))
                conn.executemany("UPDATE photos SET hidden=0 WHERE sha256=?", [(m["sha256"],) for m in members])
                continue
            best = g["best_sha256"] if g["user_picked"] and g["best_sha256"] in {m["sha256"] for m in members} \
                else best_of(members)
            conn.execute("UPDATE similar_groups SET best_sha256=? WHERE id=?", (best, gid))
            for m in members:
                conn.execute("UPDATE photos SET hidden=? WHERE sha256=?",
                             (int(hide and m["sha256"] != best), m["sha256"]))
        conn.commit()

    @_tx
    @_tx
    def rebuild_similar_groups(self) -> int:
        """Re-detect near-duplicates over the whole library from scratch (hash rules, then CLIP pairs).

        Forgets user choices in groups ("keep this one", "not a duplicate"); use it after those
        choices went wrong or after changing the thresholds. Returns the number of groups."""
        conn = self.catalog.connect()
        conn.execute("DELETE FROM photo_similar")
        conn.execute("DELETE FROM similar_groups")
        conn.execute("UPDATE photos SET hidden=0")
        tree, touched = BKTree(), set()
        shas = [r[0] for r in conn.execute("SELECT sha256 FROM photos WHERE phash IS NOT NULL ORDER BY taken_at, sha256")]
        for sha in shas:  # same order-independent union as at import time
            photo = self._photo(conn, sha)
            self._link_similar(conn, photo, tree, touched)
            tree.add(int(photo["phash"], 16), sha)
        if self.clip_engine.available():
            for a, b in self.learner.similar_pairs(conn, shas):
                self._union(conn, a, b, "burst", touched)
        self._refresh_groups(conn, touched)
        return conn.execute("SELECT COUNT(*) FROM similar_groups").fetchone()[0]

    @_tx
    def set_group_best(self, gid: int, sha: str) -> None:
        conn = self.catalog.connect()
        if self._group_of(conn, sha) != gid:
            raise ValueError("photo is not in this group")
        conn.execute("UPDATE similar_groups SET best_sha256=?, user_picked=1 WHERE id=?", (sha, gid))
        self._refresh_groups(conn, [gid])

    @_tx
    def remove_from_group(self, gid: int, sha: str) -> None:
        """User says: this photo is not a duplicate of the others."""
        conn = self.catalog.connect()
        conn.execute("DELETE FROM photo_similar WHERE sha256=? AND group_id=?", (sha, gid))
        conn.execute("UPDATE photos SET hidden=0 WHERE sha256=?", (sha,))
        conn.execute(
            "UPDATE similar_groups SET best_sha256=NULL, user_picked=0 WHERE id=? AND best_sha256=?", (gid, sha)
        )
        self._refresh_groups(conn, [gid])

    @_tx
    def set_dup_policy(self, policy: str) -> None:
        if policy not in ("keep_all", "hide_non_best"):
            raise ValueError("invalid policy")
        self.catalog.set_setting("dup_policy", policy)
        conn = self.catalog.connect()
        self._refresh_groups(conn, [r["id"] for r in conn.execute("SELECT id FROM similar_groups")])

    # ------------------------------------------------------------------ export

    def export_view(self, factor_key: str, dest: str, include_hidden: bool = False) -> dict:
        """Materialise a factor as a folder tree of hardlinks (symlinks across filesystems).

        Every entry points at the single stored copy, so no extra disk space is used.
        """
        conn = self.catalog.connect()
        factor = conn.execute("SELECT * FROM factors WHERE key=?", (factor_key,)).fetchone()
        if not factor:
            raise ValueError(f"unknown factor {factor_key}")
        out = Path(dest).expanduser().resolve() / factor["name"]
        values = {r["id"]: dict(r) for r in conn.execute(
            "SELECT * FROM factor_values WHERE factor_id=?", (factor["id"],))}
        leaf_ids = set(values) - {v["parent_id"] for v in values.values() if v["parent_id"]}

        def folder(vid: int) -> Path:
            parts = []
            while vid:
                parts.append(_safe(values[vid]["label"]))
                vid = values[vid]["parent_id"]
            return out.joinpath(*reversed(parts))

        linked = symlinked = 0
        for vid in leaf_ids:
            rows = conn.execute(
                "SELECT p.sha256, p.store_path, p.ext, p.taken_at FROM photo_factor pf "
                "JOIN photos p ON p.sha256 = pf.sha256 WHERE pf.value_id=?"
                + ("" if include_hidden else " AND p.hidden=0"),
                (vid,),
            ).fetchall()
            if not rows:
                continue
            d = folder(vid)
            d.mkdir(parents=True, exist_ok=True)
            for r in rows:
                name = f"{r['taken_at'].replace(':', '').replace('T', '_')}_{r['sha256'][:8]}.{r['ext']}"
                target, src = d / name, self.root / r["store_path"]
                if target.exists():
                    continue
                try:
                    os.link(src, target)
                    linked += 1
                except OSError:
                    os.symlink(src, target)
                    symlinked += 1
        return {"path": str(out), "hardlinks": linked, "symlinks": symlinked}


def _safe(name: str) -> str:
    return "".join("_" if c in '/\\:\0' else c for c in name).strip() or "_"
