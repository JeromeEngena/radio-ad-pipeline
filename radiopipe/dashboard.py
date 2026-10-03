"""Live dashboard: a tiny HTTP server (standard library only) that reads the SQLite DB
read-only, so it never slows down or blocks processing."""
import datetime as dt
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import db

HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")
_CACHE = {"at": 0.0, "data": None}
CACHE_SECONDS = 8


def q1(conn, sql, args=()):
    row = conn.execute(sql, args).fetchone()
    return row[0] if row and row[0] is not None else 0


def build_stats(conn, target_total=None):
    now = dt.datetime.now()
    day30 = (now - dt.timedelta(days=29)).date().isoformat()
    h24 = (now - dt.timedelta(hours=24)).replace(microsecond=0).isoformat()
    h1 = (now - dt.timedelta(hours=1)).replace(microsecond=0).isoformat()

    registered = q1(conn, "SELECT COUNT(*) FROM recordings")
    processed = q1(conn, "SELECT COUNT(*) FROM recordings WHERE status='done'")
    pending = q1(conn, "SELECT COUNT(*) FROM recordings WHERE status='pending'")
    errors = q1(conn, "SELECT COUNT(*) FROM recordings WHERE status='error'")
    verbal = q1(conn, "SELECT COUNT(*) FROM recordings WHERE content='verbal'")
    blank = q1(conn, "SELECT COUNT(*) FROM recordings WHERE content='blank'")
    asr_runs = q1(conn, "SELECT COUNT(*) FROM recordings WHERE asr_run=1")
    reused = q1(conn, "SELECT COUNT(*) FROM recordings WHERE dup_of IS NOT NULL AND status='done'")
    last_hour = q1(conn, "SELECT COUNT(*) FROM recordings WHERE processed_at>=?", (h1,))

    summary = dict(
        registered=registered, processed=processed, pending=pending, errors=errors,
        verbal=verbal, blank=blank, transcriptions_made=asr_runs, transcripts_reused=reused,
        transcripts_reused_verbal=q1(conn, "SELECT COUNT(*) FROM recordings WHERE dup_of IS NOT NULL AND content='verbal'"),
        exact_duplicates=q1(conn, "SELECT COUNT(*) FROM recordings WHERE dup_kind='exact'"),
        fingerprint_duplicates=q1(conn, "SELECT COUNT(*) FROM recordings WHERE dup_kind='fingerprint'"),
        separated=q1(conn, "SELECT COUNT(*) FROM recordings WHERE separated=1 AND dup_of IS NULL"),
        audio_hours=round(q1(conn, "SELECT SUM(duration_s) FROM recordings WHERE status='done'") / 3600, 1),
        speech_hours=round(q1(conn, "SELECT SUM(speech_seconds) FROM recordings WHERE status='done'") / 3600, 1),
        avg_asr_seconds=round(q1(conn, "SELECT AVG(proc_seconds) FROM recordings WHERE asr_run=1"), 2),
        avg_duration=round(q1(conn, "SELECT AVG(duration_s) FROM recordings WHERE status='done'"), 1),
        companies=q1(conn, "SELECT COUNT(*) FROM companies"),
        companies_to_review=q1(conn, "SELECT COUNT(*) FROM companies WHERE verified=0"),
        company_mentions=q1(conn, "SELECT COUNT(*) FROM mentions"),
        stations=q1(conn, "SELECT COUNT(DISTINCT station) FROM recordings"),
        processed_last_hour=last_hour,
        processed_last_24h=q1(conn, "SELECT COUNT(*) FROM recordings WHERE processed_at>=?", (h24,)),
        last_processed_at=conn.execute("SELECT MAX(processed_at) FROM recordings").fetchone()[0],
        target_total=target_total or registered,
    )
    summary["eta_hours"] = round(pending / last_hour, 1) if last_hour and pending else None

    blank_reasons = [dict(reason=r[0] or "unknown", n=r[1]) for r in conn.execute(
        "SELECT blank_reason, COUNT(*) FROM recordings WHERE content='blank' "
        "GROUP BY blank_reason ORDER BY 2 DESC")]

    daily = [dict(day=r[0], total=r[1], verbal=r[2], blank=r[3]) for r in conn.execute(
        "SELECT substr(recorded_at,1,10) d, COUNT(*), "
        "SUM(content='verbal'), SUM(content='blank') FROM recordings "
        "WHERE recorded_at>=? GROUP BY d ORDER BY d", (day30,))]

    hourly = [dict(hour=r[0], n=r[1]) for r in conn.execute(
        "SELECT substr(processed_at,1,13) h, COUNT(*) FROM recordings "
        "WHERE processed_at>=? GROUP BY h ORDER BY h", (h24,))]

    stations = [dict(station=r[0], total=r[1], verbal=r[2] or 0, blank=r[3] or 0) for r in conn.execute(
        "SELECT station, COUNT(*), SUM(content='verbal'), SUM(content='blank') FROM recordings "
        "WHERE status='done' GROUP BY station ORDER BY 2 DESC LIMIT 15")]

    languages = [dict(language=r[0] or "unknown", n=r[1]) for r in conn.execute(
        "SELECT language, COUNT(*) FROM recordings WHERE content='verbal' "
        "GROUP BY language ORDER BY 2 DESC LIMIT 8")]

    top_companies = [dict(name=r[0], mentions=r[1], verified=bool(r[2])) for r in conn.execute(
        "SELECT name, mentions, verified FROM companies WHERE mentions>0 "
        "ORDER BY mentions DESC LIMIT 15")]

    new_companies = [dict(day=r[0], n=r[1]) for r in conn.execute(
        "SELECT substr(first_seen,1,10) d, COUNT(*) FROM companies WHERE first_seen>=? "
        "GROUP BY d ORDER BY d", (day30,))]

    return dict(generated_at=now.replace(microsecond=0).isoformat(), summary=summary,
                blank_reasons=blank_reasons, daily=daily, hourly=hourly, stations=stations,
                languages=languages, top_companies=top_companies, new_companies=new_companies)


def search_recordings(conn, text="", station="", limit=40):
    sql = ("SELECT r.id, r.station, r.recorded_at, r.content, r.blank_reason, r.language, "
           "r.duration_s, r.dup_of, r.status, r.transcript, "
           "(SELECT GROUP_CONCAT(c.name, '; ') FROM mentions m JOIN companies c ON c.key=m.company_key "
           " WHERE m.rec_id=r.id) AS companies FROM recordings r WHERE r.status<>'pending'")
    args = []
    if text:
        sql += " AND (r.transcript LIKE ? OR r.id IN (SELECT m.rec_id FROM mentions m JOIN companies c " \
               "ON c.key=m.company_key WHERE c.name LIKE ?))"
        args += [f"%{text}%", f"%{text}%"]
    if station:
        sql += " AND r.station=?"
        args.append(station)
    sql += " ORDER BY r.processed_at DESC, r.id DESC LIMIT ?"
    args.append(min(int(limit), 200))
    out = []
    for r in conn.execute(sql, args):
        d = dict(r)
        d["transcript"] = (d["transcript"] or "")[:400]
        out.append(d)
    return out


def make_handler(cfg, target_total):
    dbpath = cfg["paths"]["db"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    with open(HTML_PATH, "rb") as fh:
                        return self._send(200, fh.read(), "text/html; charset=utf-8")
                conn = db.connect(dbpath, readonly=True)
                try:
                    if url.path == "/api/dashboard":
                        if time.time() - _CACHE["at"] > CACHE_SECONDS or _CACHE["data"] is None:
                            _CACHE.update(at=time.time(), data=build_stats(conn, target_total))
                        return self._send(200, json.dumps(_CACHE["data"]), "application/json")
                    if url.path == "/api/recordings":
                        qs = parse_qs(url.query)
                        rows = search_recordings(conn, qs.get("q", [""])[0],
                                                 qs.get("station", [""])[0],
                                                 qs.get("limit", ["40"])[0])
                        return self._send(200, json.dumps(rows), "application/json")
                finally:
                    conn.close()
                self._send(404, "not found", "text/plain")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}), "application/json")

    return Handler


def serve(cfg, host="0.0.0.0", port=8050, target_total=None):
    server = ThreadingHTTPServer((host, port), make_handler(cfg, target_total))
    print(f"[dashboard] http://{host}:{port}  (reading {cfg['paths']['db']})")
    server.serve_forever()
