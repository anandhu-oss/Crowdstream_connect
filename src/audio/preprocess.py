import io
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from math import gcd, log10
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from ..utils.constants import SR


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "audio.yaml"

MIN_SR, MAX_SR = 8000, 192000

DEFAULTS = {
    "audio": {
        "sample_rate": 16000, "normalize": "rms", "target_rms_dbfs": -20.0, "normalize_tolerance_db": 2.0,
        "normalize_peak": 0.95, "max_gain_db": 20.0, "highpass_hz": 80, "highpass_order": 2, "denoise": False,
        "clip_level": 0.999, "clip_warn_ratio": 0.001, "clip_severe_ratio": 0.02, "silence_peak": 0.0001,
        "silence_dbfs": -65.0, "quiet_speech_dbfs": -45.0, "min_snr_db": 10.0, "min_duration_s": 0.05,
        "allowed_input_rates": [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000],
    },
    "vad": {
        "backend": "auto", "threshold": 0.5, "min_speech_ms": 250, "min_silence_ms": 300, "speech_pad_ms": 100,
        "noisy_floor_dbfs": -40.0, "noisy_threshold_bump": 0.1, "energy_margin_db": 9.0, "energy_floor_dbfs": -52.0,
        "energy_start_frames": 3, "energy_adapt_rate": 0.03,
    },
    "segments": {"max_seconds": 25, "min_seconds": 0.2, "merge_gap_seconds": 0.6, "split_search_fraction": 0.4},
    "gate": {
        "action": "flag", "min_segment_dbfs": -55, "bleed": True, "bleed_margin_db": 15, "bleed_history": 20,
        "cross_track_margin_db": 12.0, "cross_track_soft_margin_db": 6.0, "cross_track_correlation": 0.6,
        "cross_track_overlap_s": 0.3,
    },
    "streaming": {
        "input_format": "pcm_s16le", "frame_samples": 512, "chunk_ms": 200, "preroll_ms": 300,
        "endpoint_silence_ms": 450, "min_utterance_seconds": 0.25, "max_utterance_seconds": 12, "overlap_ms": 500,
        "forced_cut_search_fraction": 0.35, "max_buffer_seconds": 40, "partial_interval_s": 1.5,
        "min_partial_seconds": 0.8, "min_partial_growth_s": 0.4, "partial_load_factor": 1.0,
        "max_inflight_chunks": 200, "max_chunk_bytes": 4194304, "max_message_bytes": 12582912,
        "drain_timeout_s": 45, "session_resume_ttl_s": 60, "keep_captions": 300,
    },
}

PCM_FORMATS = {"pcm_s16le": np.dtype("<i2"), "pcm_f32le": np.dtype("<f4")}


class AudioError(ValueError):
    pass


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return _merge(deepcopy(DEFAULTS), user)


def section(cfg, name):
    given = (cfg if cfg is not None else load_config()).get(name) or {}
    return {**DEFAULTS.get(name, {}), **given}


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


def sniff_container(data):
    head = bytes(data[:12])
    if head[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    return None


def decode_container(data, sample_rate=SR, timeout=30):
    if not data:
        raise AudioError("no audio data")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioError("FFmpeg was not found. Install FFmpeg and add it to PATH.")
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-vn", "-ac", "1",
               "-ar", str(int(sample_rate)), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    try:
        result = subprocess.run(command, input=bytes(data), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioError("FFmpeg audio decoding timed out.") from exc
    except OSError as exc:
        raise AudioError("Could not execute FFmpeg.") from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError("FFmpeg could not decode the audio" + (f": {detail[-300:]}" if detail else "."))
    y = np.frombuffer(result.stdout[: len(result.stdout) // 4 * 4], dtype="<f4").copy()
    _check_samples(y, "decoded audio")
    return y


def _decode_with_ffmpeg(source):
    return decode_container(source, SR, 60), SR


def pcm_to_float(data, fmt="pcm_s16le"):
    dt = PCM_FORMATS.get(fmt)
    if dt is None:
        raise AudioError(f"unsupported PCM format {fmt!r}")
    if len(data) % dt.itemsize:
        raise AudioError("PCM data ends in the middle of a sample")
    a = np.frombuffer(data, dtype=dt)
    y = a.astype("float32") / 32768.0 if fmt == "pcm_s16le" else a.astype("float32", copy=True)
    if y.size and not np.all(np.isfinite(y)):
        raise AudioError("PCM data contains NaN or infinite values")
    return y


def load_audio(source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise AudioError("file not found")
        try:
            y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            try:
                return _decode_with_ffmpeg(path.read_bytes())[0].reshape(-1, 1), SR
            except OSError as exc:
                raise AudioError("Unreadable or corrupted audio file.") from exc
    elif isinstance(source, (bytes, bytearray)):
        if not source:
            raise AudioError("no audio data")
        raw = bytes(source)
        try:
            y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        except Exception:
            y, sr = _decode_with_ffmpeg(raw)
            return y.reshape(-1, 1), sr
    else:
        raise AudioError("source must be a path or bytes")
    _check_samples(y, "audio")
    return y, _check_sr(sr)


def to_mono(y):
    if y.ndim == 1 or y.shape[1] == 1:
        return y.reshape(-1)
    rms = np.sqrt(np.mean(y.astype("float64") ** 2, axis=0))
    mix = y.mean(axis=1)
    mix_rms = np.sqrt(np.mean(mix.astype("float64") ** 2))
    if mix_rms < 0.7 * float(np.mean(rms)):
        return y[:, int(np.argmax(rms))]
    return mix


def denoise(y, sr):
    try:
        import noisereduce as nr
    except ImportError as exc:
        raise AudioError("audio.denoise is enabled but noisereduce is not installed. pip install noisereduce") from exc
    return np.asarray(nr.reduce_noise(y=y, sr=sr, stationary=True), dtype="float32")


def _frame_rms(y, sr, ms=20):
    n = max(1, int(sr * ms / 1000))
    k = len(y) // n
    if k == 0:
        return np.array([float(np.sqrt(np.mean(y.astype("float64") ** 2)))])
    return np.sqrt(np.mean(y[: k * n].astype("float64").reshape(k, n) ** 2, axis=1))


def _db(x):
    return 20 * log10(max(float(x), 1e-9))


def speech_level(y, sr):
    fr = _frame_rms(y, sr)
    active = fr >= 0.1 * np.percentile(fr, 95)
    if active.any():
        return float(np.sqrt(np.mean(fr[active] ** 2)))
    return 0.0


def highpass(y, sr, hz, order=2):
    if not hz or hz <= 0 or hz >= sr / 2 or len(y) <= 24:
        return y
    from scipy.signal import butter, sosfiltfilt
    sos = butter(int(order), hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype("float32")


@dataclass
class TrackReport:
    duration_s: float
    clipped_ratio: float
    speech_dbfs: float
    noise_floor_dbfs: float
    gain_db: float
    silent: bool
    warnings: list
    snr_db: float = 0.0
    peak_dbfs: float = -180.0
    input_rate: int = 16000
    quality: str = "good"


def normalize_level(y, cfg=None):
    a = section(cfg, "audio")
    mode = a.get("normalize", "rms")
    if mode is True:
        mode = "rms"
    if mode in (False, None, "none", "off") or y.size == 0:
        return y, 0.0
    peak = float(np.max(np.abs(y)))
    if peak < a["silence_peak"]:
        return y, 0.0
    ceiling = float(a["normalize_peak"])
    max_gain = float(a["max_gain_db"])
    if mode == "peak":
        gain = min(20 * log10(ceiling / peak), max_gain)
    else:
        level = speech_level(y, a["sample_rate"])
        if level < 1e-8:
            return y, 0.0
        gain = float(a["target_rms_dbfs"]) - _db(level)
        gain = max(-max_gain, min(gain, max_gain))
        if abs(gain) <= float(a["normalize_tolerance_db"]):
            return y, 0.0
        body = float(np.percentile(np.abs(y), 99.9))
        if body > 0:
            gain = min(gain, 20 * log10(ceiling / body))
    if abs(gain) < 0.05:
        return y, 0.0
    out = np.clip(y * (10 ** (gain / 20)), -ceiling, ceiling).astype("float32")
    return out, round(gain, 2)


def prepare_for_asr(y, cfg=None):
    y = np.asarray(y, dtype="float32")
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise AudioError("audio is empty or invalid")
    return normalize_level(y, cfg)


def measure(y, sr, cfg=None):
    a = section(cfg, "audio")
    fr = _frame_rms(y, sr)
    level = speech_level(y, sr)
    floor = float(np.percentile(fr, 10)) if len(fr) else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    silent = peak < a["silence_peak"] or level < 1e-8 or _db(level) < a["silence_dbfs"]
    return {"speech_dbfs": _db(level), "noise_floor_dbfs": _db(floor), "peak_dbfs": _db(peak),
            "snr_db": _db(level) - _db(floor), "silent": bool(silent)}


def quality_warnings(clipped, m, cfg=None):
    a = section(cfg, "audio")
    if m["silent"]:
        return ["silent track"], "poor"
    warnings = []
    quality = "good"
    if clipped > a["clip_warn_ratio"]:
        warnings.append(f"clipping: {clipped * 100:.2f}% of samples at full scale")
        quality = "poor" if clipped > a["clip_severe_ratio"] else "fair"
    if m["snr_db"] < a["min_snr_db"]:
        warnings.append(f"low signal-to-noise ratio ({m['snr_db']:.1f} dB)")
        quality = "fair" if quality == "good" else quality
    if m["speech_dbfs"] < a["quiet_speech_dbfs"]:
        warnings.append(f"very quiet speech ({m['speech_dbfs']:.1f} dBFS)")
        quality = "fair" if quality == "good" else quality
    return warnings, quality


def preprocess(y, sr, cfg=None, return_report=False):
    cfg = cfg if cfg is not None else load_config()
    a = section(cfg, "audio")
    _check_samples(y)
    sr = _check_sr(sr)
    raw = np.asarray(y, dtype="float32")
    clipped = float(np.mean(np.abs(raw) >= a["clip_level"]))
    y = to_mono(raw)
    input_rate = sr
    target = a["sample_rate"]
    if sr != target:
        from scipy.signal import resample_poly
        g = gcd(target, sr)
        y = resample_poly(y, target // g, sr // g).astype("float32")
        sr = target
    if len(y) == 0:
        raise AudioError("audio is empty after resampling")
    y = (y - np.mean(y)).astype("float32")
    y = highpass(y, sr, a.get("highpass_hz", 80), a.get("highpass_order", 2))
    m = measure(y, sr, cfg)
    warnings, quality = quality_warnings(clipped, m, cfg)
    if a.get("denoise", False) and not m["silent"]:
        y = denoise(y, sr)
    gain_db = 0.0
    if not m["silent"]:
        y, gain_db = normalize_level(y, cfg)
    y = np.clip(y, -1.0, 1.0).astype("float32")
    if not return_report:
        return y
    report = TrackReport(round(len(y) / sr, 3), clipped, m["speech_dbfs"], m["noise_floor_dbfs"], gain_db, m["silent"],
                         warnings, m["snr_db"], m["peak_dbfs"], input_rate, quality)
    return y, report


def load_track(source, cfg=None, return_report=False):
    y, sr = load_audio(source)
    return preprocess(y, sr, cfg=cfg, return_report=return_report)


class StreamResampler:
    PAD = 96

    def __init__(self, in_rate, out_rate=SR):
        g = gcd(int(in_rate), int(out_rate))
        self.up, self.down = int(out_rate) // g, int(in_rate) // g
        self.passthrough = int(in_rate) == int(out_rate)
        self.tail = np.zeros(0, dtype="float32")
        self.in_total = 0
        self.out_total = 0

    def process(self, x):
        if self.passthrough:
            return x
        from scipy.signal import resample_poly
        joined = np.concatenate([self.tail, x])
        y = resample_poly(joined, self.up, self.down).astype("float32")
        drop = int(round(len(self.tail) * self.up / self.down))
        want = (self.in_total + len(x)) * self.up // self.down - self.out_total
        y = y[drop: drop + want]
        if len(y) < want:
            y = np.pad(y, (0, want - len(y)), mode="edge") if len(y) else np.zeros(want, dtype="float32")
        self.tail = joined[-self.PAD:]
        self.in_total += len(x)
        self.out_total += want
        return y


class StreamPreprocessor:
    def __init__(self, cfg=None, input_rate=SR):
        self.a = section(cfg, "audio")
        if int(input_rate) not in [int(r) for r in self.a["allowed_input_rates"]]:
            raise AudioError(f"unsupported input sample rate {input_rate!r}")
        self.input_rate = int(input_rate)
        self.target = int(self.a["sample_rate"])
        self.clip_level = float(self.a["clip_level"])
        hz = self.a.get("highpass_hz", 80)
        self._sos = None
        if hz and 0 < hz < self.target / 2:
            from scipy.signal import butter
            self._sos = butter(int(self.a.get("highpass_order", 2)), hz, btype="highpass", fs=self.target, output="sos")
        self.reset()

    def reset(self):
        self._resampler = StreamResampler(self.input_rate, self.target)
        self._zi = None
        self.samples = 0
        self.clipped = 0
        self.peak = 0.0
        self._recent = []

    def process(self, chunk):
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 1 or not np.issubdtype(chunk.dtype, np.number):
            raise AudioError("a chunk must be a 1-D numeric array")
        if chunk.size == 0:
            return np.zeros(0, dtype="float32")
        if not np.all(np.isfinite(chunk)):
            raise AudioError("chunk contains NaN or infinite values")
        chunk = chunk.astype("float32", copy=False)
        clipped = int(np.count_nonzero(np.abs(chunk) >= self.clip_level))
        self._recent.append((clipped, len(chunk)))
        if len(self._recent) > 25:
            self._recent.pop(0)
        y = self._resampler.process(chunk)
        if len(y) == 0:
            return y
        if self._sos is not None:
            from scipy.signal import sosfilt, sosfilt_zi
            if self._zi is None:
                self._zi = sosfilt_zi(self._sos) * float(y[0])
            y, self._zi = sosfilt(self._sos, y, zi=self._zi)
        y = y.astype("float32")
        self.samples += len(y)
        self.clipped += clipped
        self.peak = max(self.peak, float(np.max(np.abs(y))))
        return y

    def clipping_ratio(self):
        total = sum(n for _, n in self._recent)
        return sum(c for c, _ in self._recent) / total if total else 0.0

    def stats(self):
        return {"samples": self.samples, "clipped_ratio": self.clipped / max(1, self.samples),
                "peak_dbfs": _db(self.peak), "recent_clipping": self.clipping_ratio()}
