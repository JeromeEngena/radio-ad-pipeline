"""Speech-to-text backends."""
import os


class FasterWhisperASR:
    def __init__(self, cfg, device):
        from faster_whisper import WhisperModel
        compute = cfg["compute_type"]
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(cfg["model"], device=device, compute_type=compute)
        self.language = cfg["language"]
        self.beam = cfg["beam_size"]

    def transcribe(self, audio, path=None, prompt=None):
        segments, info = self.model.transcribe(
            audio, language=self.language, beam_size=self.beam, vad_filter=True,
            initial_prompt=prompt, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language, float(info.language_probability)


class SidecarASR:
    """Test backend: reads '<audio file>.txt' next to the recording."""

    def transcribe(self, audio, path=None, prompt=None):
        side = (path or "") + ".txt"
        if os.path.exists(side):
            with open(side, "r", encoding="utf-8") as fh:
                return fh.read().strip(), "en", 1.0
        return "", None, 0.0


def make_asr(cfg, device):
    if cfg["backend"] == "sidecar":
        return SidecarASR()
    return FasterWhisperASR(cfg, device)
