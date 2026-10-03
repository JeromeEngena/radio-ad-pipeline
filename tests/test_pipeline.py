"""End-to-end test with synthetic audio. Run:  python -m unittest discover -s tests"""
import os
import shutil
import sqlite3
import tempfile
import unittest

from radiopipe import config, runner
from tests import make_demo_data


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.audio = os.path.join(self.tmp, "audio")
        self.made = make_demo_data.build(self.audio, n_files=60, days=3)
        cfg = config.load(None)
        cfg["paths"].update(audio_root=self.audio, work_dir=os.path.join(self.tmp, "data"),
                            known_companies=os.path.join(os.path.dirname(__file__), "..", "known_companies.txt"))
        os.makedirs(cfg["paths"]["work_dir"], exist_ok=True)
        cfg["paths"]["db"] = os.path.join(cfg["paths"]["work_dir"], "radio.db")
        cfg["paths"]["transcripts_csv"] = os.path.join(cfg["paths"]["work_dir"], "transcripts.csv")
        cfg["paths"]["companies_csv"] = os.path.join(cfg["paths"]["work_dir"], "company_names.csv")
        cfg["processing"].update(workers=1, vad_backend="energy", separation="never")
        cfg["asr"]["backend"] = "sidecar"
        cfg["companies"]["use_spacy"] = False
        cfg["stable_after_seconds"] = 0
        self.cfg = cfg

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_end_to_end(self):
        quiet = lambda *_: None
        runner.run_once(self.cfg, log=quiet)
        conn = sqlite3.connect(self.cfg["paths"]["db"])
        total, = conn.execute("SELECT COUNT(*) FROM recordings").fetchone()
        done, = conn.execute("SELECT COUNT(*) FROM recordings WHERE status='done'").fetchone()
        self.assertEqual(total, len(self.made))
        self.assertEqual(done, total, "every recording must be handled")

        # blanks are blanks, ads are verbal
        for path, kind in self.made:
            content, = conn.execute("SELECT content FROM recordings WHERE path=?", (path,)).fetchone()
            self.assertEqual(content, "verbal" if kind.startswith("ad") else "blank", (path, kind))

        # repeat airings reuse the transcript instead of re-running speech recognition
        asr_runs, = conn.execute("SELECT SUM(asr_run) FROM recordings").fetchone()
        reused, = conn.execute("SELECT COUNT(*) FROM recordings WHERE dup_of IS NOT NULL").fetchone()
        ads = sum(1 for _, k in self.made if k.startswith("ad"))
        distinct_ads = len({k for _, k in self.made if k.startswith("ad")})
        self.assertLessEqual(asr_runs, distinct_ads + 2)
        self.assertGreaterEqual(reused, ads - distinct_ads - 2)

        # companies found, including a misspelled one (fuzzy) and a brand-new one
        names = {r[0] for r in conn.execute("SELECT name FROM companies WHERE mentions>0")}
        self.assertIn("Stanbic Bank Uganda", names)
        self.assertIn("Airtel Uganda", names)
        self.assertTrue(any("Kampala Dairies" in n for n in names), names)

        # CSV outputs exist and are populated
        self.assertGreater(os.path.getsize(self.cfg["paths"]["transcripts_csv"]), 200)
        self.assertGreater(os.path.getsize(self.cfg["paths"]["companies_csv"]), 100)

        # a second run only picks up NEW files
        extra = make_demo_data.build(self.audio, n_files=5, days=1, seed=99)
        runner.run_once(self.cfg, log=quiet)
        total2, = conn.execute("SELECT COUNT(*) FROM recordings").fetchone()
        self.assertGreaterEqual(total2, total + 1)
        pending, = conn.execute("SELECT COUNT(*) FROM recordings WHERE status='pending'").fetchone()
        self.assertEqual(pending, 0)


if __name__ == "__main__":
    unittest.main()
