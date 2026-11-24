from collections import deque
from dataclasses import dataclass, field
from math import ceil, isfinite

import numpy as np

from ..utils.constants import SR
from ..utils.logging import get_logger
from .preprocess import AudioError, _check_sr, _check_samples, _db, _frame_rms, load_config, section

log = get_logger()


class VADError(RuntimeError):
    pass


@dataclass
class Segment:
    start: float
    end: float
    flags: list = field(default_factory=list)

    @property
    def duration(self):
        return self.end - self.start


def noise_floor_dbfs(audio, sr=SR):
    fr = _frame_rms(audio, sr)
    return _db(float(np.percentile(fr, 10))) if len(fr) else -180.0


class EnergyStep:
    def __init__(self, cfg=None, sr=SR, frame=512):
        cfg = cfg if cfg is not None else load_config()
        v, s = section(cfg, "vad"), section(cfg, "streaming")
        self.sr, self.frame = int(sr), int(frame)
        self.margin = float(v["energy_margin_db"])
        self.floor_min = float(v["energy_floor_dbfs"])
        self.start_frames = max(1, int(v["energy_start_frames"]))
        self.rate = float(v["energy_adapt_rate"])
        self.pad = int(v["speech_pad_ms"] * self.sr / 1000)
        self.end_frames = max(1, int(round(s["endpoint_silence_ms"] * self.sr / 1000 / self.frame)))
        self.reset()

    def reset(self):
        self.noise = -60.0
        self.warm_left = 4
        self.warm_levels = []
        self.pos = 0
        self.speaking = False
        self.run = 0
        self.quiet = 0
        self.first = 0
        self.last_loud = 0

    def _rms_db(self, frame):
        return _db(float(np.sqrt(np.mean(frame.astype("float64") ** 2))))

    def __call__(self, frame):
        level = self._rms_db(frame)
        begin = self.pos
        self.pos += len(frame)
        if self.warm_left > 0:
            self.warm_levels.append(level)
            self.warm_left -= 1
            if self.warm_left == 0:
                self.noise = min(max(min(self.warm_levels), -80.0), -25.0)
            return None
        on = max(self.noise + self.margin, self.floor_min)
        off = on - 3.0
        loud = level >= (off if self.speaking else on)
        if not self.speaking:
            if level < self.noise:
                self.noise = self.noise + 0.3 * (level - self.noise)
            else:
                self.noise = self.noise + self.rate * (level - self.noise)
            self.noise = min(self.noise, -25.0)
            if loud:
                if self.run == 0:
                    self.first = begin
                self.run += 1
                if self.run >= self.start_frames:
                    self.speaking = True
                    self.quiet = 0
                    self.last_loud = self.pos
                    return {"start": max(0, self.first - self.pad) / self.sr}
            else:
                self.run = 0
            return None
        if loud:
            self.quiet = 0
            self.last_loud = self.pos
            return None
        self.quiet += 1
        if self.quiet >= self.end_frames:
            self.speaking = False
            self.run = 0
            return {"end": min(self.pos, self.last_loud + self.pad) / self.sr}
        return None


class EnergyVAD:
    MIN_SAMPLES = 512

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.v = section(self.cfg, "vad")
        self.sr = section(self.cfg, "audio")["sample_rate"]
        self.min_ms = self.v["min_speech_ms"]
        self.gap_ms = self.v["min_silence_ms"]

    def detect(self, audio, sr=SR):
        if _check_sr(sr) != self.sr:
            raise AudioError(f"VAD needs {self.sr} Hz audio, got {sr}")
        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.dtype != np.float32:
            raise AudioError("VAD input must be a 1-D float32 array")
        _check_samples(audio)
        if len(audio) < self.MIN_SAMPLES:
            return []
        cfg = {**self.cfg, "streaming": {**section(self.cfg, "streaming"), "endpoint_silence_ms": self.gap_ms}}
        step = EnergyStep(cfg, self.sr)
        total = len(audio) / self.sr
        regions, begin = [], None
        for i in range(0, len(audio) - 511, 512):
            ev = step(audio[i: i + 512])
            if ev and "start" in ev and begin is None:
                begin = ev["start"]
            elif ev and "end" in ev and begin is not None:
                regions.append((begin, min(total, ev["end"])))
                begin = None
        if begin is not None:
            regions.append((begin, total))
        return [Segment(a, b) for a, b in regions if (b - a) * 1000 >= self.min_ms]


class SileroVAD:
    MIN_SAMPLES = 512

    def __init__(self, cfg=None):
        cfg = cfg if cfg is not None else load_config()
        self.v = section(cfg, "vad")
        self.sr = section(cfg, "audio")["sample_rate"]
        self.max_s = section(cfg, "segments")["max_seconds"]
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from silero_vad import load_silero_vad
                self._model = load_silero_vad()
            except Exception as e:
                raise VADError(f"Silero VAD could not be loaded ({type(e).__name__}); pip install silero-vad torch") from e
        return self._model

    def threshold_for(self, audio):
        t = float(self.v["threshold"])
        floor = noise_floor_dbfs(audio, self.sr)
        if floor > float(self.v["noisy_floor_dbfs"]):
            t = min(0.9, t + float(self.v["noisy_threshold_bump"]))
        return t

    def detect(self, audio, sr=SR):
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
                torch.from_numpy(audio), model, threshold=self.threshold_for(audio), sampling_rate=self.sr,
                min_speech_duration_ms=self.v["min_speech_ms"], min_silence_duration_ms=self.v["min_silence_ms"],
                speech_pad_ms=self.v["speech_pad_ms"], max_speech_duration_s=self.max_s, return_seconds=True)
        except Exception as e:
            raise VADError(f"VAD failed ({type(e).__name__})") from e
        dur = len(audio) / self.sr
        return [Segment(max(0.0, float(t["start"])), min(dur, float(t["end"]))) for t in ts]


class FallbackVAD:
    def __init__(self, cfg=None):
        self.primary = SileroVAD(cfg)
        self.backup = EnergyVAD(cfg)
        self.using_backup = False

    def detect(self, audio, sr=SR):
        if not self.using_backup:
            try:
                return self.primary.detect(audio, sr)
            except VADError as e:
                log.warning("silero vad unavailable, using the energy detector: %s", e)
                self.using_backup = True
        return self.backup.detect(audio, sr)


def make_vad(cfg=None):
    cfg = cfg if cfg is not None else load_config()
    backend = section(cfg, "vad").get("backend", "auto")
    if backend == "silero":
        return SileroVAD(cfg)
    if backend == "energy":
        return EnergyVAD(cfg)
    if backend == "auto":
        return FallbackVAD(cfg)
    raise AudioError(f"vad.backend must be silero, energy or auto, not {backend!r}")


def quietest_point(audio, lo, hi, sr=SR):
    i0, i1 = max(0, int(lo * sr)), min(len(audio), int(hi * sr))
    win = int(0.05 * sr)
    if i1 - i0 <= win:
        return hi
    e = np.convolve(audio[i0:i1].astype("float64") ** 2, np.ones(win), mode="valid")
    return lo + (int(np.argmin(e)) + win / 2) / sr


def _split(seg, mx, audio, sr, frac):
    if seg.duration <= mx + 1e-9:
        return [seg]
    if audio is None:
        n = ceil(seg.duration / mx - 1e-9)
        step = seg.duration / n
        return [Segment(seg.start + i * step, seg.start + (i + 1) * step) for i in range(n)]
    out, start = [], seg.start
    while seg.end - start > mx + 1e-9:
        cut = quietest_point(audio, start + mx * (1 - frac), start + mx, sr)
        cut = min(max(cut, start + mx * (1 - frac)), start + mx)
        out.append(Segment(start, cut))
        start = cut
    out.append(Segment(start, seg.end))
    return out


def make_segments(regions, cfg=None, total_seconds=None, audio=None):
    s = section(cfg, "segments")
    mx, mn, gap, frac = s["max_seconds"], s["min_seconds"], s["merge_gap_seconds"], s["split_search_fraction"]
    clean = []
    for r in regions:
        if not (isfinite(r.start) and isfinite(r.end)) or r.end < r.start:
            raise AudioError(f"invalid segment times {r.start!r}..{r.end!r}")
        a, b = max(0.0, r.start), r.end if total_seconds is None else min(r.end, total_seconds)
        if b > a:
            clean.append(Segment(a, b))
    clean.sort(key=lambda r: r.start)
    joined = []
    for r in clean:
        if joined and r.start <= joined[-1].end:
            joined[-1].end = max(joined[-1].end, r.end)
        else:
            joined.append(Segment(r.start, r.end))
    merged = []
    for r in joined:
        if merged and r.start - merged[-1].end <= gap and r.end - merged[-1].start <= mx:
            merged[-1].end = r.end
        else:
            merged.append(Segment(r.start, r.end))
    out = []
    for m in merged:
        if m.duration >= mn:
            out += _split(m, mx, audio, SR, frac)
    return out


def fixed_segments(total_seconds, max_seconds, min_seconds=0.1):
    if not isfinite(total_seconds) or total_seconds <= 0:
        return []
    n = ceil(total_seconds / max_seconds - 1e-9)
    segs = [Segment(i * max_seconds, min((i + 1) * max_seconds, total_seconds)) for i in range(n)]
    return [x for x in segs if x.duration >= min_seconds]


def cut(audio, seg, sr=SR):
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        raise AudioError("audio must be a 1-D array")
    if not (isfinite(seg.start) and isfinite(seg.end)):
        raise AudioError("invalid segment times")
    i0, i1 = max(0, int(round(seg.start * sr))), min(len(audio), int(round(seg.end * sr)))
    if i1 <= i0:
        raise AudioError(f"segment {seg.start:.2f}-{seg.end:.2f}s is empty or outside the recording")
    return audio[i0:i1]


def segment_level_db(audio, start, end, sr=SR):
    i0, i1 = max(0, int(round(start * sr))), min(len(audio), int(round(end * sr)))
    if i1 <= i0:
        return -180.0
    return _db(float(np.sqrt(np.mean(audio[i0:i1].astype("float64") ** 2))))


def _flag(x, name):
    if name not in x.flags:
        x.flags.append(name)


def flag_segments(segs, audio, cfg=None):
    g = section(cfg, "gate")
    min_db, margin, bleed_on = g["min_segment_dbfs"], g["bleed_margin_db"], g["bleed"]
    levels = [segment_level_db(audio, x.start, x.end) for x in segs]
    ref = float(np.percentile(levels, 90)) if len(levels) >= 3 else None
    for x, lv in zip(segs, levels):
        if lv < min_db:
            _flag(x, "near_silent")
        elif bleed_on and ref is not None and lv < ref - margin:
            _flag(x, "probable_bleed")
    return segs


def _envelope(audio, start, end, sr=SR, ms=20):
    i0, i1 = max(0, int(round(start * sr))), min(len(audio), int(round(end * sr)))
    if i1 - i0 < int(sr * ms / 1000) * 2:
        return None
    return _frame_rms(audio[i0:i1], sr, ms)


def _envelope_correlation(a, b):
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    x, y = np.log10(a[:n] + 1e-6), np.log10(b[:n] + 1e-6)
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _covered(seg, others):
    if seg.duration <= 0:
        return 0.0
    got = sum(max(0.0, min(seg.end, o.end) - max(seg.start, o.start)) for o in others)
    return min(1.0, got / seg.duration)


def flag_bleed_cross(tracks, margin_db=12.0, gains_db=None, soft_margin_db=6.0, min_correlation=0.6):
    gains_db = gains_db or {}
    for name, (audio, segs) in tracks.items():
        for x in segs:
            mine = segment_level_db(audio, x.start, x.end) - gains_db.get(name, 0.0)
            mine_env = _envelope(audio, x.start, x.end)
            for n2, (a2, s2) in tracks.items():
                if n2 == name:
                    continue
                theirs = segment_level_db(a2, x.start, x.end) - gains_db.get(n2, 0.0)
                if theirs >= mine + margin_db:
                    _flag(x, "probable_bleed")
                    break
                if theirs >= mine + soft_margin_db and _covered(x, s2) >= 0.5 and mine_env is not None:
                    other_env = _envelope(a2, x.start, x.end)
                    if other_env is not None and _envelope_correlation(mine_env, other_env) >= min_correlation:
                        _flag(x, "probable_bleed")
                        break
    return {name: segs for name, (_, segs) in tracks.items()}


def flag_cross_track_overlap(tracks, min_overlap_s=0.3):
    names = list(tracks)
    found = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for x in tracks[a][1]:
                for y in tracks[b][1]:
                    ov = min(x.end, y.end) - max(x.start, y.start)
                    if ov >= min_overlap_s:
                        _flag(x, "overlapping_speech")
                        _flag(y, "overlapping_speech")
                        found.append({"tracks": (a, b), "start": max(x.start, y.start), "end": min(x.end, y.end)})
    return found


def drop_flagged(segs, flags=("near_silent", "probable_bleed")):
    return [x for x in segs if not any(f in x.flags for f in flags)]


def speech_segments(audio, cfg=None, vad=None):
    cfg = cfg if cfg is not None else load_config()
    vad = vad or make_vad(cfg)
    total = len(audio) / section(cfg, "audio")["sample_rate"]
    segs = make_segments(vad.detect(audio), cfg, total_seconds=total, audio=audio)
    flag_segments(segs, audio, cfg)
    return drop_flagged(segs) if section(cfg, "gate")["action"] == "drop" else segs


class UtteranceGate:
    def __init__(self, cfg=None):
        g = section(cfg, "gate")
        self.min_db = float(g["min_segment_dbfs"])
        self.margin = float(g["bleed_margin_db"])
        self.bleed_on = bool(g["bleed"])
        self.drop = g["action"] == "drop"
        self.history = deque(maxlen=int(g["bleed_history"]))

    def check(self, audio):
        if audio is None or len(audio) == 0:
            return ["near_silent"]
        level = _db(float(np.sqrt(np.mean(np.asarray(audio, dtype="float64") ** 2))))
        flags = []
        if level < self.min_db:
            flags.append("near_silent")
        elif self.bleed_on and len(self.history) >= 3:
            ref = float(np.percentile(list(self.history), 90))
            if level < ref - self.margin:
                flags.append("probable_bleed")
        if "near_silent" not in flags:
            self.history.append(level)
        return flags

    def should_drop(self, flags):
        return self.drop and any(f in flags for f in ("near_silent", "probable_bleed"))
