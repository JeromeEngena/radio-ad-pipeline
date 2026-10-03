"""Landmark fingerprinting (Shazam/audfprint style).

Each recording is reduced to spectrogram peaks; pairs of nearby peaks become
hashes (f1, f2, dt). Two copies of the same ad share many hashes at one
consistent time offset, even with noise, a different start point or a different
loudness. Only *original* (non-duplicate) recordings are indexed, so the index
stays small even when millions of airings arrive.
"""
import collections

import numpy as np
from scipy.ndimage import maximum_filter

SR = 16000
N_FFT = 1024
HOP = 256
FAN_OUT = 3
MAX_DT = 120            # frames (~1.9 s)
PEAKS_PER_SECOND = 6
F_LO, F_HI = 4, 420     # FFT bins kept (~60 Hz - 6.5 kHz)


def _spectrogram(audio):
    n = 1 + (len(audio) - N_FFT) // HOP
    if n <= 0:
        return np.zeros((0, N_FFT // 2 + 1), dtype=np.float32)
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((n, N_FFT // 2 + 1), dtype=np.float32)
    step = 512
    for i in range(0, n, step):                      # chunked to bound memory
        j = min(n, i + step)
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(i, j)[:, None]
        out[i:j] = np.abs(np.fft.rfft(audio[idx] * win, axis=1))
    return np.log1p(out * 50.0)


def fingerprint(audio, max_seconds=120):
    """Return (hashes int64[], times int32[])."""
    audio = audio[: int(max_seconds * SR)]
    S = _spectrogram(audio)
    if S.shape[0] < 10:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    band = S[:, F_LO:F_HI]
    local_max = maximum_filter(band, size=(15, 21), mode="constant")
    thresh = band.mean() + 0.8 * band.std()
    ti, fi = np.nonzero((band == local_max) & (band > thresh))
    if len(ti) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    amp = band[ti, fi]
    keep = int(PEAKS_PER_SECOND * len(audio) / SR) + 1
    if len(ti) > keep:
        top = np.argsort(-amp)[:keep]
        ti, fi = ti[top], fi[top]
    order = np.lexsort((fi, ti))
    ti, fi = ti[order], fi[order] + F_LO

    hashes, times = [], []
    n = len(ti)
    for a in range(n):
        made = 0
        for b in range(a + 1, n):
            dt = int(ti[b] - ti[a])
            if dt > MAX_DT:
                break
            if dt < 1:
                continue
            hashes.append((int(fi[a]) << 16) | (int(fi[b]) << 7) | dt)
            times.append(int(ti[a]))
            made += 1
            if made >= FAN_OUT:
                break
    return np.asarray(hashes, np.int64), np.asarray(times, np.int32)


def find_match(conn, hashes, times, duration_s, cfg):
    """Look for an already-indexed recording that is the same audio.
    Returns (rec_id, votes, ratio) or None."""
    if len(hashes) < cfg["min_votes"]:
        return None
    by_hash = collections.defaultdict(list)
    for h, t in zip(hashes.tolist(), times.tolist()):
        by_hash[h].append(t)
    uniq = list(by_hash)

    offsets = collections.defaultdict(collections.Counter)  # rec_id -> offset -> votes
    for i in range(0, len(uniq), 500):
        chunk = uniq[i:i + 500]
        q = "SELECT h, rec_id, t FROM fp_index WHERE h IN (%s)" % ",".join("?" * len(chunk))
        for h, rec_id, t_ref in conn.execute(q, chunk):
            for t_q in by_hash[h]:
                offsets[rec_id][t_ref - t_q] += 1

    scored = []
    for rec_id, counter in offsets.items():
        best = max(c + counter.get(o + 1, 0) for o, c in counter.items())
        scored.append((best, rec_id))
    scored.sort(reverse=True)

    for votes, rec_id in scored[:5]:
        if votes < cfg["min_votes"]:
            break
        ref = conn.execute("SELECT n_hashes, duration_s FROM recordings WHERE id=?",
                           (rec_id,)).fetchone()
        if not ref or not ref["n_hashes"]:
            continue
        ratio = votes / max(1, min(len(hashes), ref["n_hashes"]))
        close = abs((ref["duration_s"] or 0) - duration_s) <= cfg["duration_tolerance_s"]
        if ratio >= cfg["min_ratio"] and close:
            return rec_id, votes, ratio
    return None
