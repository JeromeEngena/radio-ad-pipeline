"""Create synthetic radio recordings for tests and dashboard demos (no real speech:
syllable-like noise bursts stand in for voices, and a '.txt' sidecar holds the
transcript the test ASR backend will return)."""
import datetime as dt
import os
import random

import numpy as np
from scipy.io import wavfile

SR = 16000

ADS = [  # (transcript, language)
    ("Brought to you by Stanbic Bank. Open an account today and save for tomorrow.", "en"),
    ("Top up with Airtel Ugandaa and get double data this weekend.", "en"),
    ("Roofings Group, quality iron sheets for every home. Visit our showroom in Namanve.", "en"),
    ("Sponsored by Kampala Dairies Ltd. Fresh milk every morning.", "en"),
    ("Centenary Bank, the bank that cares. Apply for a school fees loan now.", "en"),
    ("Drink Coca Cola, taste the feeling.", "en"),
    ("MTN Uganda MoMo makes sending money easy. Dial star one five six hash.", "en"),
    ("Visit Mukwano Industries Limited for the best soap in town.", "en"),
    ("Equity Bank Uganda, we are growing with you.", "en"),
    ("Nile Breweries Ltd reminds you to drink responsibly.", "en"),
]
STATIONS = ["CBS FM", "Capital FM", "Radio Simba", "Bukedde FM", "Radio One", "Dembe FM",
            "Sanyu FM", "Radio West", "Mega FM", "Radio Pacis", "Voice of Teso", "Beat FM"]


def speechlike(rng, dur):
    out = np.zeros(int(dur * SR), np.float32)
    i = 0
    while i < len(out):
        n = int(rng.uniform(0.12, 0.30) * SR)
        t = np.arange(n) / SR
        f0 = rng.uniform(110, 240)
        sig = sum(rng.uniform(0.3, 1) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) for k in range(1, 12)
                  if f0 * k < 3800)
        sig = sig * np.hanning(n) * rng.uniform(0.15, 0.3) / 4
        seg = out[i:i + n]
        seg += sig.astype(np.float32)[:len(seg)]
        i += n + int(rng.uniform(0.04, 0.16) * SR)
    return out


def steady_music(rng, dur):
    t = np.arange(int(dur * SR)) / SR
    return (0.2 * (np.sin(2 * np.pi * 262 * t) + np.sin(2 * np.pi * 330 * t) + np.sin(2 * np.pi * 392 * t)) / 3
            ).astype(np.float32)


def save(path, x):
    x = np.clip(x, -1, 1)
    wavfile.write(path, SR, (x * 32767).astype(np.int16))


def build(out_dir, n_files=60, days=7, seed=7):
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    bases = [speechlike(rng, rnd.uniform(9, 15)) for _ in ADS]
    now = dt.datetime.now().replace(microsecond=0) - dt.timedelta(minutes=5)
    made = []
    for i in range(n_files):
        station = rnd.choice(STATIONS)
        when = now - dt.timedelta(days=rnd.uniform(0, days), seconds=rnd.uniform(0, 80000))
        sdir = os.path.join(out_dir, station.replace(" ", "_"))
        os.makedirs(sdir, exist_ok=True)
        name = f"{station.replace(' ', '-')}_{when:%Y%m%d_%H%M%S}.wav"
        path = os.path.join(sdir, name)
        roll = rnd.random()
        if roll < 0.12:
            x = (rng.normal(0, 1e-5, SR * 8)).astype(np.float32); kind = "silent"
        elif roll < 0.25:
            x = steady_music(rng, rnd.uniform(8, 20)) + rng.normal(0, 0.002, 1).astype(np.float32); kind = "music"
        else:
            k = rnd.randrange(len(ADS))
            base = bases[k]
            cut = int(rnd.uniform(0, 0.3) * SR)
            x = base[cut:] + rng.normal(0, 0.004, len(base) - cut).astype(np.float32)
            x *= rnd.uniform(0.7, 1.3)
            kind = f"ad{k}"
            with open(path + ".txt", "w", encoding="utf-8") as fh:
                fh.write(ADS[k][0])
        save(path, x)
        os.utime(path, (when.timestamp(), when.timestamp()))
        made.append((path, kind))
    return made


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_audio"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print(len(build(out, n)), "files written to", out)
