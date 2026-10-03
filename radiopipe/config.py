"""Configuration: defaults merged with config.yaml."""
import copy
import os

import yaml

DEFAULTS = {
    "paths": {
        "audio_root": "./audio",                    # where station recordings land
        "work_dir": "./data",                       # DB + CSV outputs
        "known_companies": "./known_companies.txt", # optional seed list: Name|alias|alias
    },
    # Optional regex to read station / timestamp from file names. Named groups:
    # station, date (YYYYMMDD), time (HHMMSS). Falls back to parent folder + file mtime.
    "filename_pattern": r"(?P<station>[A-Za-z0-9\- ]+?)_(?P<date>\d{8})_(?P<time>\d{6})",
    "extensions": [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"],
    "processing": {
        "workers": 2,
        "batch_size": 50,
        "min_speech_seconds": 1.0,     # below this the recording is "blank"
        "silence_rms_db": -55.0,       # whole-file loudness below this = silent
        "vad_backend": "silero",       # silero | energy (energy is a crude fallback)
        "separation": "auto",          # always | auto | never  (Demucs vocal isolation)
        "music_bed_ratio_threshold": 0.25,
        "device": "auto",              # auto | cpu | cuda
    },
    "asr": {
        "backend": "faster-whisper",   # faster-whisper | sidecar (tests: reads <file>.txt)
        "model": "large-v3",
        "compute_type": "auto",        # auto -> float16 on GPU, int8 on CPU
        "language": None,              # None = auto-detect per recording
        "beam_size": 5,
        "prompt_with_known_companies": True,
    },
    "fingerprint": {
        "enabled": True,
        "min_votes": 12,
        "min_ratio": 0.20,             # votes / min(query hashes, reference hashes)
        "duration_tolerance_s": 1.5,
        "max_seconds": 120,
    },
    "companies": {
        "fuzzy_threshold": 90,
        "use_spacy": True,             # English NER for discovering new names (optional)
        "use_llm": False,              # Claude-based extraction (needs ANTHROPIC_API_KEY)
        "llm_model": "claude-haiku-4-5-20251001",
    },
    "watch_interval_seconds": 30,
    "stable_after_seconds": 10,        # ignore files modified more recently than this
    "company_csv_interval_seconds": 60,
}


def _merge(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    work = cfg["paths"]["work_dir"]
    os.makedirs(work, exist_ok=True)
    cfg["paths"]["db"] = os.path.join(work, "radio.db")
    cfg["paths"]["transcripts_csv"] = os.path.join(work, "transcripts.csv")
    cfg["paths"]["companies_csv"] = os.path.join(work, "company_names.csv")
    return cfg
