"""Command line:  python -m radiopipe <command> [--config config.yaml]

  init        create the database and load the seed company list
  scan        register new audio files (no processing)
  run         process everything pending (use --limit N to try a few)
  watch       keep running: pick up and process new recordings as they arrive
  export      rewrite transcripts.csv and company_names.csv from the database
  dashboard   serve the live dashboard (default http://0.0.0.0:8050)
  stats       print headline numbers
"""
import argparse
import csv

from . import config, dashboard, db, runner


def export_all(cfg):
    conn = runner.prepare(cfg)
    path = cfg["paths"]["transcripts_csv"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(runner.TRANSCRIPT_COLS)
        for r in conn.execute(
                "SELECT r.*, (SELECT GROUP_CONCAT(c.name,'; ') FROM mentions m JOIN companies c "
                "ON c.key=m.company_key WHERE m.rec_id=r.id) AS companies FROM recordings r "
                "WHERE r.status<>'pending' ORDER BY r.id"):
            w.writerow([r["id"], r["path"], r["station"], r["recorded_at"],
                        r["content"] or r["status"], r["blank_reason"] or "", r["duration_s"],
                        r["speech_seconds"], r["language"] or "", r["language_prob"] or "",
                        r["dup_of"] or "", (r["transcript"] or "").replace("\n", " "),
                        r["companies"] or "", r["processed_at"]])
    sink = runner.CsvSink.__new__(runner.CsvSink)
    sink.cfg, sink._last_company_write = cfg, 0.0
    sink.write_companies(conn, force=True)
    print("exported", path, "and", cfg["paths"]["companies_csv"])


def main():
    ap = argparse.ArgumentParser(prog="radiopipe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "scan", "run", "watch", "export", "dashboard", "stats"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="run: stop after N recordings")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--target", type=int, default=None,
                    help="dashboard: expected total (e.g. 850000) so progress shows the full backlog")
    a = ap.parse_args()
    cfg = config.load(a.config)

    if a.command == "init":
        runner.prepare(cfg)
        print("database ready:", cfg["paths"]["db"])
    elif a.command == "scan":
        conn = runner.prepare(cfg)
        print(runner.scan(cfg, conn), "new recordings registered")
    elif a.command == "run":
        runner.run_once(cfg, a.limit)
    elif a.command == "watch":
        runner.watch(cfg)
    elif a.command == "export":
        export_all(cfg)
    elif a.command == "dashboard":
        runner.prepare(cfg)
        dashboard.serve(cfg, a.host, a.port, a.target)
    elif a.command == "stats":
        conn = db.connect(cfg["paths"]["db"], readonly=True)
        for k, v in dashboard.build_stats(conn)["summary"].items():
            print(f"{k:28} {v}")


if __name__ == "__main__":
    main()
