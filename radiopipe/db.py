"""SQLite storage. One writer (the runner), many readers (workers, dashboard)."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id              INTEGER PRIMARY KEY,
    path            TEXT UNIQUE NOT NULL,
    station         TEXT,
    recorded_at     TEXT,              -- ISO timestamp from file name or mtime
    size_bytes      INTEGER,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending | done | error
    content         TEXT,              -- verbal | blank
    blank_reason    TEXT,              -- silent | no_speech | music_only | no_words | too_short
    sha256          TEXT,
    duration_s      REAL,
    speech_seconds  REAL,
    speech_ratio    REAL,
    music_bed_ratio REAL,
    separated       INTEGER DEFAULT 0, -- 1 = vocals isolated with Demucs
    language        TEXT,
    language_prob   REAL,
    transcript      TEXT,
    dup_of          INTEGER,           -- id of the recording whose transcript was reused
    dup_kind        TEXT,              -- exact | fingerprint
    asr_run         INTEGER DEFAULT 0, -- 1 = speech recognition actually ran on this file
    n_hashes        INTEGER,
    proc_seconds    REAL,
    processed_at    TEXT,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS ix_rec_status    ON recordings(status);
CREATE INDEX IF NOT EXISTS ix_rec_sha       ON recordings(sha256);
CREATE INDEX IF NOT EXISTS ix_rec_recorded  ON recordings(recorded_at);
CREATE INDEX IF NOT EXISTS ix_rec_processed ON recordings(processed_at);
CREATE INDEX IF NOT EXISTS ix_rec_station   ON recordings(station);

CREATE TABLE IF NOT EXISTS companies (
    key         TEXT PRIMARY KEY,      -- normalised name
    name        TEXT NOT NULL,         -- display name
    mentions    INTEGER NOT NULL DEFAULT 0,   -- recordings that mention it
    first_seen  TEXT,
    last_seen   TEXT,
    verified    INTEGER NOT NULL DEFAULT 0,   -- 1 = confirmed by a person / seed list
    source      TEXT                   -- seed | auto
);
CREATE TABLE IF NOT EXISTS aliases (
    alias_key   TEXT PRIMARY KEY,
    company_key TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mentions (
    rec_id      INTEGER NOT NULL,
    company_key TEXT NOT NULL,
    surface     TEXT,
    method      TEXT,                  -- exact | fuzzy | pattern | ner | llm | inherited
    score       REAL
);
CREATE INDEX IF NOT EXISTS ix_men_rec ON mentions(rec_id);
CREATE INDEX IF NOT EXISTS ix_men_key ON mentions(company_key);

CREATE TABLE IF NOT EXISTS fp_index (h INTEGER NOT NULL, rec_id INTEGER NOT NULL, t INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_fp_h ON fp_index(h);
"""


def connect(path, readonly=False):
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    else:
        conn = sqlite3.connect(path, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init(path):
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
