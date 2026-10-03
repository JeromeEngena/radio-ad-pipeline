"""Audio helpers: decode, detect speech, estimate background music, isolate vocals."""
import subprocess

import numpy as np

SR = 16000


def decode(path, sr=SR, channels=1):
    """Decode any ffmpeg-readable file to float32 PCM. Returns a flat array
    (interleaved if channels > 1)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path,
           "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + proc.stderr.decode("utf-8", "ignore")[:300])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def rms_db(x):
    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def _frame_rms(x, frame=480):
    n = len(x) // frame
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1) + 1e-12)


class EnergyVAD:
    """Crude fallback: loud AND syllable-like modulation. Music beds are steadier
    than speech, but this WILL misjudge some music. Use Silero for production."""

    def __call__(self, audio):
        r = _frame_rms(audio)                    # 30 ms frames
        if len(r) == 0:
            return []
        floor = np.percentile(r, 10)
        loud = r > max(floor * 3.0, 10 ** (-50 / 20))
        w = 33                                   # ~1 s window
        mask = np.zeros(len(r), dtype=bool)
        for i in range(0, len(r), w // 2 or 1):
            seg = r[i:i + w]
            if len(seg) < 8:
                continue
            cv = seg.std() / (seg.mean() + 1e-9)
            if cv > 0.45:
                mask[i:i + w] = True
        mask &= loud
        return _mask_to_segments(mask, 0.03)


class SileroVAD:
    def __init__(self):
        import torch
        from silero_vad import load_silero_vad
        self.torch = torch
        self.model = load_silero_vad()

    def __call__(self, audio):
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(self.torch.from_numpy(audio), self.model,
                                   sampling_rate=SR, min_speech_duration_ms=250)
        return [(t["start"] / SR, t["end"] / SR) for t in ts]


def _mask_to_segments(mask, step):
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start * step, i * step))
            start = None
    if start is not None:
        segs.append((start * step, len(mask) * step))
    return segs


def make_vad(name):
    if name == "silero":
        try:
            return SileroVAD()
        except Exception as exc:  # silero/torch missing
            print(f"[warn] Silero VAD unavailable ({exc}); using energy VAD")
    return EnergyVAD()


def speech_seconds(segments):
    return float(sum(e - s for s, e in segments))


def music_bed_ratio(audio, segments):
    """Loudness of the non-speech parts relative to the speech parts (0-1+).
    High = a bed of music/noise under or around the voice. If speech covers
    nearly the whole clip there is nothing to compare, so assume a bed (1.0):
    broadcast ads usually have one."""
    r = _frame_rms(audio)
    if len(r) == 0:
        return 0.0
    mask = np.zeros(len(r), dtype=bool)
    for s, e in segments:
        mask[int(s / 0.03): int(e / 0.03) + 1] = True
    if (~mask).sum() < 0.05 * len(r) or mask.sum() == 0:
        return 1.0
    return float(r[~mask].mean() / (r[mask].mean() + 1e-9))


class VocalSeparator:
    """Demucs (htdemucs) vocal stem. Heavy: use a GPU for volume."""

    def __init__(self, device="cpu", model_name="htdemucs"):
        import torch
        from demucs.pretrained import get_model
        self.torch = torch
        self.device = device
        self.model = get_model(model_name)
        self.model.to(device).eval()
        self.vocals_idx = self.model.sources.index("vocals")

    def vocals_16k(self, path):
        import torchaudio
        from demucs.apply import apply_model
        torch = self.torch
        sr = self.model.samplerate
        flat = decode(path, sr=sr, channels=2)
        wav = torch.from_numpy(flat.reshape(-1, 2).T.copy())          # (2, n)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            out = apply_model(self.model, wav[None].to(self.device), split=True,
                              overlap=0.25, progress=False)[0]
        voc = out[self.vocals_idx].mean(0).cpu()
        voc = torchaudio.functional.resample(voc, sr, SR)
        return voc.numpy().astype(np.float32)
