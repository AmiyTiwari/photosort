"""SQLite catalog: schema and connection helpers."""
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    sha256        TEXT PRIMARY KEY,
    ext           TEXT NOT NULL,
    store_path    TEXT NOT NULL,          -- relative to library root
    bytes         INTEGER NOT NULL,
    width         INTEGER,
    height        INTEGER,
    taken_at      TEXT,                   -- ISO-8601, from EXIF or file mtime
    taken_src     TEXT,                   -- 'exif' | 'mtime'
    lat           REAL,
    lon           REAL,
    camera_make   TEXT,
    camera_model  TEXT,
    phash         TEXT,                   -- 64-bit hex
    dhash         TEXT,
    quality       REAL,                   -- sharpness score, higher is better
    is_screenshot INTEGER NOT NULL DEFAULT 0,
    hidden        INTEGER NOT NULL DEFAULT 0,  -- hidden as non-best near-duplicate
    batch_id      INTEGER,
    imported_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at);

-- Every original file that resolved to a stored photo (many -> one).
CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY,
    sha256        TEXT NOT NULL REFERENCES photos(sha256),
    original_path TEXT NOT NULL UNIQUE,
    size          INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    batch_id      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sources_sha ON sources(sha256);

CREATE TABLE IF NOT EXISTS import_batches (
    id           INTEGER PRIMARY KEY,
    root         TEXT NOT NULL,
    mode         TEXT NOT NULL,           -- 'copy' | 'hardlink'
    status       TEXT NOT NULL,           -- running | done | failed
    total        INTEGER NOT NULL DEFAULT 0,
    processed    INTEGER NOT NULL DEFAULT 0,
    new_photos   INTEGER NOT NULL DEFAULT 0,
    exact_dupes  INTEGER NOT NULL DEFAULT 0,
    skipped      INTEGER NOT NULL DEFAULT 0,  -- already imported in an earlier run
    errors       INTEGER NOT NULL DEFAULT 0,
    similar_groups INTEGER NOT NULL DEFAULT 0,
    message      TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS factors (
    id    INTEGER PRIMARY KEY,
    key   TEXT NOT NULL UNIQUE,
    name  TEXT NOT NULL,
    kind  TEXT NOT NULL                   -- date | place | camera | type | custom
);

CREATE TABLE IF NOT EXISTS factor_values (
    id        INTEGER PRIMARY KEY,
    factor_id INTEGER NOT NULL REFERENCES factors(id),
    key       TEXT NOT NULL,
    label     TEXT NOT NULL,
    parent_id INTEGER REFERENCES factor_values(id),
    UNIQUE(factor_id, key)
);

-- The mapping: a photo appears under many factor values, stored once.
CREATE TABLE IF NOT EXISTS photo_factor (
    sha256     TEXT NOT NULL REFERENCES photos(sha256),
    value_id   INTEGER NOT NULL REFERENCES factor_values(id),
    confidence REAL NOT NULL DEFAULT 1.0,
    source     TEXT NOT NULL DEFAULT 'rule',   -- rule | model | user
    PRIMARY KEY (sha256, value_id)
);
CREATE INDEX IF NOT EXISTS idx_pf_value ON photo_factor(value_id);

CREATE TABLE IF NOT EXISTS similar_groups (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,             -- variant (same image re-encoded) | burst
    best_sha256 TEXT REFERENCES photos(sha256),
    user_picked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS photo_similar (
    sha256   TEXT PRIMARY KEY REFERENCES photos(sha256),
    group_id INTEGER NOT NULL REFERENCES similar_groups(id)
);
CREATE INDEX IF NOT EXISTS idx_ps_group ON photo_similar(group_id);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Phase 3: faces -> people clusters (each cluster is a value of the 'people' factor).
CREATE TABLE IF NOT EXISTS faces (
    id         INTEGER PRIMARY KEY,
    sha256     TEXT NOT NULL REFERENCES photos(sha256),
    x REAL, y REAL, w REAL, h REAL,        -- bbox relative to image size (0..1)
    score      REAL,
    embedding  BLOB NOT NULL,              -- float32[128], L2-normalised
    value_id   INTEGER REFERENCES factor_values(id),
    confidence REAL,
    source     TEXT NOT NULL DEFAULT 'model'   -- model | user
);
CREATE INDEX IF NOT EXISTS idx_faces_sha ON faces(sha256);
CREATE INDEX IF NOT EXISTS idx_faces_value ON faces(value_id);

CREATE TABLE IF NOT EXISTS people (
    value_id   INTEGER PRIMARY KEY REFERENCES factor_values(id),
    named      INTEGER NOT NULL DEFAULT 0,
    cover_face INTEGER
);

-- User said "this face is not this person": never auto-assign it there again.
CREATE TABLE IF NOT EXISTS face_negatives (
    face_id  INTEGER NOT NULL,
    value_id INTEGER NOT NULL,
    PRIMARY KEY (face_id, value_id)
);

CREATE TABLE IF NOT EXISTS events (
    value_id INTEGER PRIMARY KEY REFERENCES factor_values(id),
    start_at TEXT NOT NULL,
    end_at   TEXT NOT NULL,
    n        INTEGER NOT NULL,
    named    INTEGER NOT NULL DEFAULT 0
);

-- Custom factors: a value either has a rule (re-applied to every new import) or is a manual collection.
CREATE TABLE IF NOT EXISTS custom_rules (
    value_id INTEGER PRIMARY KEY REFERENCES factor_values(id),
    rule     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,      -- stable identity so a dismissed question is not re-asked
    kind        TEXT NOT NULL,             -- name_person | merge_people | confirm_face | name_event
    priority    REAL NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',   -- open | answered | dismissed | stale
    answer      TEXT,
    created_at  TEXT NOT NULL,
    answered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_q_status ON questions(status, priority);

-- Phase 5: CLIP embeddings and learned groups.
CREATE TABLE IF NOT EXISTS photo_embeddings (
    sha256 TEXT PRIMARY KEY REFERENCES photos(sha256),
    vec    BLOB NOT NULL                   -- float16[512], L2-normalised
);

CREATE TABLE IF NOT EXISTS learned_groups (
    value_id  INTEGER PRIMARY KEY REFERENCES factor_values(id),
    prompts   TEXT,                        -- JSON list for content groups; NULL for manual collections
    threshold REAL NOT NULL DEFAULT 0.5,
    auto      INTEGER NOT NULL DEFAULT 1,  -- 1: assign automatically; 0: only suggest
    round     INTEGER NOT NULL DEFAULT 0   -- bumps after each answer so a fresh batch is asked
);

-- The user's yes/no answers: training data for that group.
CREATE TABLE IF NOT EXISTS group_examples (
    value_id INTEGER NOT NULL,
    sha256   TEXT NOT NULL,
    label    INTEGER NOT NULL,             -- +1 belongs, -1 does not
    PRIMARY KEY (value_id, sha256)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY,
    message    TEXT NOT NULL,              -- provider-neutral message JSON
    created_at TEXT NOT NULL
);
"""

# Columns added after phase 2; applied to existing catalogs on open.
MIGRATIONS = [
    ("photos", "faces_done", "INTEGER NOT NULL DEFAULT 0"),
    ("import_batches", "stage", "TEXT"),
    ("import_batches", "faces_found", "INTEGER NOT NULL DEFAULT 0"),
    ("photos", "clip_done", "INTEGER NOT NULL DEFAULT 0"),
]

DEFAULT_SETTINGS = {
    "dup_policy": "keep_all",   # keep_all | hide_non_best
    "event_gap_hours": "6",     # a gap longer than this starts a new event
    "event_min_photos": "5",    # smaller clusters are not called events
    "llm": "{}",
    "content_seeded": "0",      # default content groups are created once, when CLIP first runs
    "claude_code_session": "",  # Claude Code adapter: conversation to resume; cleared by "New chat"                # provider config, see llm.py (API key kept out of here)
}


class Catalog:
    """Thread-safe wrapper: one connection per thread, WAL mode."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            for table, col, decl in MIGRATIONS:
                cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                if col not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            for k, v in DEFAULT_SETTINGS.items():
                conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (k, v))

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def setting(self, key: str) -> str:
        row = self.connect().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else DEFAULT_SETTINGS[key]

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (key, value))
