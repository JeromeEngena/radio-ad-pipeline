"""Process ONE recording end to end. Runs inside worker processes; never writes to the DB
(the runner is the single writer), so there are no write conflicts."""
import hashlib
import time

from . import audio as A
from . import db
from .asr import make_asr
from .companies import CompanyExtractor
from .fingerprint import fingerprint, find_match


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(pref):
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Worker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.p = cfg["processing"]
        self.device = resolve_device(self.p["device"])
        self.conn = db.connect(cfg["paths"]["db"], readonly=True)
        self.vad = A.make_vad(self.p["vad_backend"])
        self.asr = make_asr(cfg["asr"], self.device)
        self.extractor = CompanyExtractor(cfg["companies"], self.conn)
        self._separator = None

    @property
    def separator(self):
        if self._separator is None:
            self._separator = A.VocalSeparator(self.device)
        return self._separator

    # ------------------------------------------------------------------
    def process(self, rec_id, path):
        t0 = time.time()
        out = dict(id=rec_id, status="done", content=None, blank_reason=None, sha256=None,
                   duration_s=None, speech_seconds=0.0, speech_ratio=0.0, music_bed_ratio=None,
                   separated=0, language=None, language_prob=None, transcript=None,
                   dup_of=None, dup_kind=None, asr_run=0, fp=None, n_hashes=None,
                   mentions=[], error=None)
        try:
            self._run(out, path)
        except Exception as exc:  # one bad file must never stop the run
            out.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        out["proc_seconds"] = round(time.time() - t0, 3)
        return out

    def _run(self, out, path):
        sha = sha256_file(path)
        out["sha256"] = sha
        row = self.conn.execute(
            "SELECT id FROM recordings WHERE sha256=? AND status='done' AND id<>? "
            "AND dup_of IS NULL ORDER BY id LIMIT 1", (sha, out["id"])).fetchone()
        if row:
            return self._inherit(out, row["id"], "exact")

        audio = A.decode(path)
        dur = len(audio) / A.SR
        out["duration_s"] = round(dur, 2)
        if dur < 0.5:
            return self._blank(out, "too_short")
        if A.rms_db(audio) < self.p["silence_rms_db"]:
            return self._blank(out, "silent")

        segs = self.vad(audio)
        speech = A.speech_seconds(segs)
        out["speech_seconds"] = round(speech, 2)
        out["speech_ratio"] = round(speech / dur, 3)
        if speech < self.p["min_speech_seconds"]:
            return self._blank(out, "no_speech")
        ratio = A.music_bed_ratio(audio, segs)
        out["music_bed_ratio"] = round(ratio, 3)

        # Compare with everything already processed
        fcfg = self.cfg["fingerprint"]
        if fcfg["enabled"]:
            h, t = fingerprint(audio, fcfg["max_seconds"])
            out["n_hashes"] = int(len(h))
            match = find_match(self.conn, h, t, dur, fcfg)
            if match:
                return self._inherit(out, match[0], "fingerprint")
            out["fp"] = (h, t)

        # Isolate the voice from music / noise
        speech_audio = audio
        mode = self.p["separation"]
        if mode == "always" or (mode == "auto" and ratio >= self.p["music_bed_ratio_threshold"]):
            vocals = self.separator.vocals_16k(path)
            out["separated"] = 1
            vsegs = self.vad(vocals)
            vspeech = A.speech_seconds(vsegs)
            if vspeech < self.p["min_speech_seconds"]:
                out["speech_seconds"] = round(vspeech, 2)
                out["speech_ratio"] = round(vspeech / dur, 3)
                return self._blank(out, "music_only")
            speech_audio = vocals

        prompt = None
        if self.cfg["asr"]["prompt_with_known_companies"]:
            names = self.extractor.top_names()
            if names:
                prompt = ("Advertisers: " + ", ".join(names))[:500]
        text, lang, prob = self.asr.transcribe(speech_audio, path=path, prompt=prompt)
        out["asr_run"] = 1
        out.update(language=lang, language_prob=prob)
        if not text:
            return self._blank(out, "no_words")
        out.update(content="verbal", transcript=text, mentions=self.extractor.extract(text))

    # ------------------------------------------------------------------
    def _blank(self, out, reason):
        out.update(content="blank", blank_reason=reason)

    def _inherit(self, out, orig_id, kind):
        """Same audio as an earlier recording: reuse its result, skip the heavy work."""
        o = self.conn.execute("SELECT * FROM recordings WHERE id=?", (orig_id,)).fetchone()
        out.update(content=o["content"], blank_reason=o["blank_reason"], dup_of=orig_id,
                   dup_kind=kind, language=o["language"], language_prob=o["language_prob"],
                   transcript=o["transcript"], duration_s=o["duration_s"],
                   speech_seconds=o["speech_seconds"], speech_ratio=o["speech_ratio"],
                   music_bed_ratio=o["music_bed_ratio"], separated=o["separated"])
        rows = self.conn.execute(
            "SELECT m.company_key, c.name, m.surface FROM mentions m "
            "JOIN companies c ON c.key=m.company_key WHERE m.rec_id=?", (orig_id,)).fetchall()
        out["mentions"] = [dict(key=r["company_key"], name=r["name"], method="inherited",
                                score=100.0, surface=r["surface"]) for r in rows]
        out["fp"] = None


_WORKER = None


def init_worker(cfg):
    global _WORKER
    _WORKER = Worker(cfg)


def run_one(args):
    rec_id, path = args
    return _WORKER.process(rec_id, path)
