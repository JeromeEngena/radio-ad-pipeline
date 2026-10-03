"""Scan -> process -> store. The runner is the only process that writes to the DB / CSVs."""
import csv
import datetime as dt
import multiprocessing as mp
import os
import re
import time

from . import db
from .companies import load_seed
from .pipeline import Worker, init_worker, run_one

TRANSCRIPT_COLS = ["recording_id", "file_path", "station", "recorded_at", "content",
                   "blank_reason", "duration_s", "speech_seconds", "language", "language_prob",
                   "reused_from_id", "transcript", "companies", "processed_at"]
COMPANY_COLS = ["company", "key", "recordings_mentioning", "stations", "first_seen",
                "last_seen", "verified", "source"]


def now_iso():
    return dt.datetime.now().replace(microsecond=0).isoformat()


# ---------------------------------------------------------------- scanning
def parse_name(cfg, path, mtime):
    m = re.search(cfg["filename_pattern"], os.path.basename(path))
    station = os.path.basename(os.path.dirname(path)) or "unknown"
    when = dt.datetime.fromtimestamp(mtime)
    if m:
        gd = m.groupdict()
        station = (gd.get("station") or station).strip()
        try:
            when = dt.datetime.strptime(gd["date"] + gd["time"], "%Y%m%d%H%M%S")
        except Exception:
            pass
    return station, when.replace(microsecond=0).isoformat()


def scan(cfg, conn):
    """Register new audio files as 'pending'. Cheap enough to run every few seconds."""
    root = cfg["paths"]["audio_root"]
    exts = tuple(e.lower() for e in cfg["extensions"])
    quiet = cfg["stable_after_seconds"]
    known = {r[0] for r in conn.execute("SELECT path FROM recordings")}
    added, batch = 0, []
    cutoff = time.time() - quiet
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(exts):
                continue
            path = os.path.join(dirpath, name)
            if path in known:
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_mtime > cutoff:       # still being written
                continue
            station, when = parse_name(cfg, path, st.st_mtime)
            batch.append((path, station, when, st.st_size))
            if len(batch) >= 5000:
                conn.executemany("INSERT OR IGNORE INTO recordings(path,station,recorded_at,size_bytes)"
                                 " VALUES(?,?,?,?)", batch)
                conn.commit()
                added += len(batch)
                batch = []
    if batch:
        conn.executemany("INSERT OR IGNORE INTO recordings(path,station,recorded_at,size_bytes)"
                         " VALUES(?,?,?,?)", batch)
        conn.commit()
        added += len(batch)
    return added


# ---------------------------------------------------------------- writing
class CsvSink:
    def __init__(self, cfg):
        self.cfg = cfg
        path = cfg["paths"]["transcripts_csv"]
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.fh = open(path, "a", newline="", encoding="utf-8-sig")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(TRANSCRIPT_COLS)
        self._last_company_write = 0.0

    def close(self):
        self.fh.close()

    def add(self, rec, res):
        self.w.writerow([
            res["id"], rec["path"], rec["station"], rec["recorded_at"],
            res["content"] or res["status"], res["blank_reason"] or "", res["duration_s"],
            res["speech_seconds"], res["language"] or "", res["language_prob"] or "",
            res["dup_of"] or "", (res["transcript"] or "").replace("\n", " "),
            "; ".join(sorted({m["name"] for m in res["mentions"]})), now_iso()])

    def flush(self):
        self.fh.flush()

    def write_companies(self, conn, force=False):
        gap = self.cfg["company_csv_interval_seconds"]
        if not force and time.time() - self._last_company_write < gap:
            return
        rows = conn.execute(
            "SELECT c.name, c.key, c.mentions, c.first_seen, c.last_seen, c.verified, c.source, "
            "(SELECT COUNT(DISTINCT r.station) FROM mentions m JOIN recordings r ON r.id=m.rec_id "
            " WHERE m.company_key=c.key) AS stations "
            "FROM companies c ORDER BY c.mentions DESC, c.name").fetchall()
        path = self.cfg["paths"]["companies_csv"]
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(COMPANY_COLS)
            for r in rows:
                w.writerow([r["name"], r["key"], r["mentions"], r["stations"], r["first_seen"] or "",
                            r["last_seen"] or "", "yes" if r["verified"] else "needs review",
                            r["source"]])
        os.replace(tmp, path)             # atomic: readers never see a half-written file
        self._last_company_write = time.time()


def apply_result(conn, sink, rec, res):
    conn.execute(
        "UPDATE recordings SET status=?, content=?, blank_reason=?, sha256=?, duration_s=?, "
        "speech_seconds=?, speech_ratio=?, music_bed_ratio=?, separated=?, language=?, "
        "language_prob=?, transcript=?, dup_of=?, dup_kind=?, asr_run=?, n_hashes=?, "
        "proc_seconds=?, processed_at=?, error=? WHERE id=?",
        (res["status"], res["content"], res["blank_reason"], res["sha256"], res["duration_s"],
         res["speech_seconds"], res["speech_ratio"], res["music_bed_ratio"], res["separated"],
         res["language"], res["language_prob"], res["transcript"], res["dup_of"], res["dup_kind"],
         res["asr_run"], res["n_hashes"], res["proc_seconds"], now_iso(), res["error"], res["id"]))
    if res["fp"] is not None and len(res["fp"][0]):
        h, t = res["fp"]
        conn.executemany("INSERT INTO fp_index VALUES(?,?,?)",
                         [(int(a), res["id"], int(b)) for a, b in zip(h, t)])
    when = rec["recorded_at"]
    for m in res["mentions"]:
        conn.execute(
            "INSERT INTO companies(key,name,mentions,first_seen,last_seen,verified,source) "
            "VALUES(?,?,1,?,?,0,'auto') ON CONFLICT(key) DO UPDATE SET mentions=mentions+1, "
            "first_seen=MIN(COALESCE(first_seen,excluded.first_seen), excluded.first_seen), "
            "last_seen=MAX(COALESCE(last_seen,excluded.last_seen), excluded.last_seen)",
            (m["key"], m["name"], when, when))
        conn.execute("INSERT INTO mentions VALUES(?,?,?,?,?)",
                     (res["id"], m["key"], m.get("surface"), m["method"], m["score"]))
    sink.add(rec, res)


# ---------------------------------------------------------------- running
class Engine:
    """Holds the worker pool (and the loaded AI models) so `watch` can keep them warm."""

    def __init__(self, cfg):
        self.workers = cfg["processing"]["workers"]
        if self.workers > 1:
            self.pool = mp.get_context("spawn").Pool(
                self.workers, initializer=init_worker, initargs=(cfg,))
            self.local = None
        else:
            self.pool, self.local = None, Worker(cfg)

    def map(self, jobs):
        if self.pool:
            return self.pool.imap_unordered(run_one, jobs)
        return (self.local.process(i, p) for i, p in jobs)

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


def process_pending(cfg, conn, sink, engine, limit=None, log=print):
    batch_size = cfg["processing"]["batch_size"]
    done = 0
    while True:
        take = batch_size if limit is None else min(batch_size, limit - done)
        if take <= 0:
            break
        rows = conn.execute("SELECT id, path, station, recorded_at FROM recordings "
                            "WHERE status='pending' ORDER BY recorded_at, id LIMIT ?",
                            (take,)).fetchall()
        if not rows:
            break
        recs = {r["id"]: r for r in rows}
        t0 = time.time()
        for res in engine.map([(r["id"], r["path"]) for r in rows]):
            apply_result(conn, sink, recs[res["id"]], res)
            conn.commit()          # commit per file so later files can match against it
        sink.flush()
        sink.write_companies(conn)
        done += len(rows)
        log(f"[run] batch of {len(rows)} done ({len(rows) / max(0.001, time.time() - t0):.1f} "
            f"files/s); {done} this run")
    sink.write_companies(conn, force=True)
    return done


def prepare(cfg):
    conn = db.init(cfg["paths"]["db"])
    load_seed(conn, cfg["paths"]["known_companies"])
    return conn


def run_once(cfg, limit=None, log=print):
    conn = prepare(cfg)
    sink = CsvSink(cfg)
    added = scan(cfg, conn)
    log(f"[scan] {added} new recordings registered")
    engine = Engine(cfg)
    try:
        return process_pending(cfg, conn, sink, engine, limit, log)
    finally:
        engine.close()
        sink.close()
        conn.close()


def watch(cfg, log=print):
    """Run forever: pick up new recordings as they land and process them immediately."""
    conn = prepare(cfg)
    sink = CsvSink(cfg)
    engine = Engine(cfg)
    log(f"[watch] watching {cfg['paths']['audio_root']} every {cfg['watch_interval_seconds']}s")
    try:
        while True:
            added = scan(cfg, conn)
            pending = conn.execute(
                "SELECT COUNT(*) FROM recordings WHERE status='pending'").fetchone()[0]
            if pending:
                log(f"[watch] +{added} new, {pending} pending")
                process_pending(cfg, conn, sink, engine, log=log)
            time.sleep(cfg["watch_interval_seconds"])
    finally:
        engine.close()
