import io
from dataclasses import dataclass
from math import ceil, gcd, isfinite
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .constants import SR

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "audio.yaml"
MIN_SR, MAX_SR = 8000, 192000


class AudioError(ValueError):
    """The audio (or a time range) is unusable."""


class VADError(RuntimeError):
    """The VAD model could not be loaded or run."""


def load_config(path=CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start


def _check_sr(sr):
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or not MIN_SR <= int(sr) <= MAX_SR:
        raise AudioError(f"invalid sample rate {sr!r}")
    return int(sr)


def _check_samples(y, what="audio"):
    if not isinstance(y, np.ndarray) or y.ndim not in (1, 2) or not np.issubdtype(y.dtype, np.number):
        raise AudioError(f"{what} must be a numeric 1-D or 2-D array")
    if y.size == 0:
        raise AudioError(f"{what} is empty")
    if not np.all(np.isfinite(y)):
        raise AudioError(f"{what} contains NaN or infinite values")


# 1. load ------------------------------------------------------------------------------
def load_audio(source) -> tuple[np.ndarray, int]:
    """Path or bytes -> (float32 [samples, channels], sample rate). Raises AudioError for anything unusable."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise AudioError("file not found")
        source = str(p)
    elif isinstance(source, (bytes, bytearray)):
        if not source:
            raise AudioError("no audio data")
        source = io.BytesIO(source)
    else:
        raise AudioError("source must be a path or bytes")
    try:
        y, sr = sf.read(source, dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioError("unreadable or corrupted audio") from e
    _check_samples(y, "audio")
    return y, _check_sr(sr)


# 2. preprocess ------------------------------------------------------------------------
def to_mono(y: np.ndarray) -> np.ndarray:
    """Average the channels, unless that cancels the sound (channels in opposite phase). Then use the loudest channel."""
    if y.ndim == 1 or y.shape[1] == 1:
        return y.reshape(-1)
    rms = np.sqrt(np.mean(y.astype("float64") ** 2, axis=0))
    mix = y.mean(axis=1)
    if np.sqrt(np.mean(mix.astype("float64") ** 2)) < 0.7 * float(np.mean(rms)):
        return y[:, int(np.argmax(rms))]
    return mix


def preprocess(y: np.ndarray, sr: int, cfg: dict | None = None) -> np.ndarray:
    """Mono, 16 kHz, DC removed, peak-normalised with a gain cap. A silent track stays silent. Returns float32 [samples]."""
    a = (cfg or load_config())["audio"]
    _check_samples(y)
    sr = _check_sr(sr)
    y = to_mono(np.asarray(y, dtype="float32"))
    target = a["sample_rate"]
    if sr != target:
        from scipy.signal import resample_poly
        g = gcd(target, sr)
        y = resample_poly(y, target // g, sr // g)
    if y.size == 0:
        raise AudioError("audio too short after resampling")
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    if peak < a["silence_peak"]:
        return np.zeros(y.shape, dtype="float32")
    y = y * min(a["normalize_peak"] / peak, 10 ** (a["max_gain_db"] / 20))
    return np.ascontiguousarray(y, dtype="float32")


def load_track(source, cfg: dict | None = None) -> np.ndarray:
    """Steps 1 and 2 together: file or bytes -> 16 kHz mono float32."""
    y, sr = load_audio(source)
    return preprocess(y, sr, cfg)


# 3. VAD -------------------------------------------------------------------------------
class SileroVAD:
    """Silero VAD (MIT). The model is loaded on first use."""
    MIN_SAMPLES = 512                      # Silero cannot score less than one frame

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.v, self.sr, self.max_s = cfg["vad"], cfg["audio"]["sample_rate"], cfg["segments"]["max_seconds"]
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from silero_vad import load_silero_vad
                self._model = load_silero_vad()
            except Exception as e:
                raise VADError(f"Silero VAD could not be loaded ({type(e).__name__}); pip install silero-vad torch") from e
        return self._model

    def detect(self, audio: np.ndarray, sr: int = SR) -> list[Segment]:
        """Speech regions only; silence is skipped. Silero also cuts regions longer than max_seconds at pauses."""
        if _check_sr(sr) != self.sr:
            raise AudioError(f"VAD needs {self.sr} Hz audio, got {sr}")
        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.dtype != np.float32:
            raise AudioError("VAD input must be a 1-D float32 array")
        _check_samples(audio)
        if len(audio) < self.MIN_SAMPLES:
            return []
        model = self._load()
        try:
            import torch
            from silero_vad import get_speech_timestamps
            ts = get_speech_timestamps(
                torch.from_numpy(audio), model, threshold=self.v["threshold"], sampling_rate=self.sr,
                min_speech_duration_ms=self.v["min_speech_ms"], min_silence_duration_ms=self.v["min_silence_ms"],
                speech_pad_ms=self.v["speech_pad_ms"], max_speech_duration_s=self.max_s, return_seconds=True)
        except Exception as e:
            raise VADError(f"VAD failed ({type(e).__name__})") from e
        dur = len(audio) / self.sr
        return [Segment(max(0.0, float(t["start"])), min(dur, float(t["end"]))) for t in ts]


# 4. segments --------------------------------------------------------------------------
def _quietest_point(audio: np.ndarray, lo: float, hi: float, sr: int) -> float:
    """Time of the lowest-energy 50 ms window between lo and hi (seconds): the safest place to cut speech."""
    i0, i1 = int(lo * sr), min(len(audio), int(hi * sr))
    win = int(0.05 * sr)
    if i1 - i0 <= win:
        return hi
    e = np.convolve(audio[i0:i1].astype("float64") ** 2, np.ones(win), mode="valid")
    return lo + (int(np.argmin(e)) + win / 2) / sr


def _split(seg: Segment, mx: float, audio, sr: int, frac: float) -> list[Segment]:
    if seg.duration <= mx + 1e-9:
        return [seg]
    if audio is None:                                          # no audio to look at: equal pieces
        n = ceil(seg.duration / mx - 1e-9); step = seg.duration / n
        return [Segment(seg.start + i * step, seg.start + (i + 1) * step) for i in range(n)]
    out, start = [], seg.start
    while seg.end - start > mx + 1e-9:
        cut = _quietest_point(audio, start + mx * (1 - frac), start + mx, sr)
        cut = min(max(cut, start + mx * (1 - frac)), start + mx)
        out.append(Segment(start, cut)); start = cut
    out.append(Segment(start, seg.end))
    return out


def make_segments(regions: list[Segment], cfg: dict | None = None, total_seconds: float | None = None,
                  audio: np.ndarray | None = None) -> list[Segment]:
    """Regions -> pieces of at most max_seconds inside the recording.
    Invalid times (NaN, end before start) raise AudioError; out-of-range times are clipped; overlaps are joined;
    regions closer than merge_gap are joined while they fit; long regions are cut at the quietest point
    (or equally when no audio is given)."""
    s = (cfg or load_config())["segments"]
    mx, mn, gap, frac = s["max_seconds"], s["min_seconds"], s["merge_gap_seconds"], s["split_search_fraction"]
    clean = []
    for r in regions:
        if not (isfinite(r.start) and isfinite(r.end)) or r.end < r.start:
            raise AudioError(f"invalid segment times {r.start!r}..{r.end!r}")
        a, b = max(0.0, r.start), r.end if total_seconds is None else min(r.end, total_seconds)
        if b > a:                                              # zero-length and fully out-of-range regions vanish
            clean.append(Segment(a, b))
    clean.sort(key=lambda r: r.start)
    joined: list[Segment] = []
    for r in clean:                                            # overlaps first, whatever the length
        if joined and r.start <= joined[-1].end:
            joined[-1].end = max(joined[-1].end, r.end)
        else:
            joined.append(Segment(r.start, r.end))
    merged: list[Segment] = []
    for r in joined:                                           # then near neighbours, while the result fits
        if merged and r.start - merged[-1].end <= gap and r.end - merged[-1].start <= mx:
            merged[-1].end = r.end
        else:
            merged.append(Segment(r.start, r.end))
    out: list[Segment] = []
    for m in merged:
        if m.duration >= mn:
            out += _split(m, mx, audio, SR, frac)
    return out


def fixed_segments(total_seconds: float, max_seconds: float, min_seconds: float = 0.1) -> list[Segment]:
    """Plain consecutive pieces covering the whole recording, including the last partial one (no VAD)."""
    if not isfinite(total_seconds) or total_seconds <= 0:
        return []
    n = ceil(total_seconds / max_seconds - 1e-9)
    segs = [Segment(i * max_seconds, min((i + 1) * max_seconds, total_seconds)) for i in range(n)]
    return [x for x in segs if x.duration >= min_seconds]


def cut(audio: np.ndarray, seg: Segment, sr: int = SR) -> np.ndarray:
    """The samples of a segment. Raises AudioError rather than return an empty or invalid slice."""
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        raise AudioError("audio must be a 1-D array")
    if not (isfinite(seg.start) and isfinite(seg.end)):
        raise AudioError("invalid segment times")
    i0, i1 = max(0, int(round(seg.start * sr))), min(len(audio), int(round(seg.end * sr)))
    if i1 <= i0:
        raise AudioError(f"segment {seg.start:.2f}-{seg.end:.2f}s is empty or outside the recording")
    return audio[i0:i1]


def speech_segments(audio: np.ndarray, cfg: dict | None = None, vad: SileroVAD | None = None) -> list[Segment]:
    """Steps 3 and 4 together: audio -> speech pieces of at most max_seconds, silence skipped."""
    cfg = cfg or load_config()
    vad = vad or SileroVAD(cfg)
    return make_segments(vad.detect(audio), cfg, total_seconds=len(audio) / cfg["audio"]["sample_rate"], audio=audio)