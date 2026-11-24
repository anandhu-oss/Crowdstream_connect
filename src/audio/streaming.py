from dataclasses import dataclass

import numpy as np

from ..utils.constants import SR
from ..utils.logging import get_logger
from .preprocess import PCM_FORMATS, AudioError, load_config, pcm_to_float, section
from .vad import EnergyStep, VADError, quietest_point

log = get_logger()


@dataclass
class Utterance:
    start: float
    end: float
    audio: np.ndarray
    forced: bool = False
    context_s: float = 0.0

    @property
    def duration(self):
        return self.end - self.start


class AudioBuffer:
    def __init__(self, capacity=SR * 8):
        self._initial = max(int(capacity), 1024)
        self._a = np.zeros(self._initial, dtype="float32")
        self._head = 0
        self._n = 0
        self._off = 0

    @property
    def start(self):
        return self._off + self._head

    @property
    def end(self):
        return self._off + self._n

    @property
    def size(self):
        return self._n - self._head

    @property
    def capacity(self):
        return len(self._a)

    def append(self, x):
        need = len(x)
        if need == 0:
            return
        if self._n + need > len(self._a):
            if self._head > 0:
                live = self._n - self._head
                self._a[:live] = self._a[self._head:self._n]
                self._off += self._head
                self._n, self._head = live, 0
            if self._n + need > len(self._a):
                bigger = np.zeros(max(len(self._a) * 2, self._n + need), dtype="float32")
                bigger[:self._n] = self._a[:self._n]
                self._a = bigger
        self._a[self._n:self._n + need] = x
        self._n += need

    def view(self, a, b):
        i0 = max(self._head, a - self._off)
        i1 = min(self._n, b - self._off)
        if i1 <= i0:
            return np.zeros(0, dtype="float32")
        return self._a[i0:i1].copy()

    def discard_before(self, idx):
        h = min(max(self._head, int(idx) - self._off), self._n)
        self._head = h
        live = self._n - self._head
        if len(self._a) > self._initial * 4 and live < len(self._a) // 8:
            smaller = np.zeros(max(self._initial, live * 2), dtype="float32")
            smaller[:live] = self._a[self._head:self._n]
            self._off += self._head
            self._a, self._n, self._head = smaller, live, 0

    def clear(self):
        self._off += self._n
        self._head = self._n = 0
        if len(self._a) > self._initial:
            self._a = np.zeros(self._initial, dtype="float32")


class PcmAssembler:
    def __init__(self, fmt="pcm_s16le", channels=1):
        if fmt not in PCM_FORMATS:
            raise AudioError(f"unsupported PCM format {fmt!r}")
        if int(channels) not in (1, 2):
            raise AudioError("only mono or stereo PCM is supported")
        self.fmt = fmt
        self.channels = int(channels)
        self.frame_bytes = PCM_FORMATS[fmt].itemsize * self.channels
        self._left = b""
        self.dropped_bytes = 0

    @property
    def pending_bytes(self):
        return len(self._left)

    def push(self, data):
        if not data:
            return np.zeros(0, dtype="float32")
        buf = self._left + bytes(data)
        n = len(buf) // self.frame_bytes * self.frame_bytes
        self._left = buf[n:]
        if n == 0:
            return np.zeros(0, dtype="float32")
        y = pcm_to_float(buf[:n], self.fmt)
        if self.channels > 1:
            y = y.reshape(-1, self.channels).mean(axis=1).astype("float32")
        return y

    def flush(self):
        self.dropped_bytes += len(self._left)
        n = len(self._left)
        self._left = b""
        return n


class SileroStep:
    def __init__(self, cfg):
        try:
            from silero_vad import VADIterator, load_silero_vad
            v, s = section(cfg, "vad"), section(cfg, "streaming")
            self._it = VADIterator(load_silero_vad(), threshold=v["threshold"], sampling_rate=section(cfg, "audio")["sample_rate"],
                                   min_silence_duration_ms=s["endpoint_silence_ms"], speech_pad_ms=v["speech_pad_ms"])
        except Exception as e:
            raise VADError(f"streaming VAD could not be loaded ({type(e).__name__}); pip install silero-vad torch") from e

    def __call__(self, frame):
        import torch
        return self._it(torch.from_numpy(frame), return_seconds=True)


def make_step(cfg=None):
    cfg = cfg if cfg is not None else load_config()
    backend = section(cfg, "vad").get("backend", "auto")
    if backend == "energy":
        return EnergyStep(cfg)
    if backend == "silero":
        return SileroStep(cfg)
    if backend != "auto":
        raise AudioError(f"vad.backend must be silero, energy or auto, not {backend!r}")
    try:
        return SileroStep(cfg)
    except VADError as e:
        log.warning("silero streaming vad unavailable, using the energy detector: %s", e)
        return EnergyStep(cfg)


class UtteranceStream:
    def __init__(self, cfg=None, step=None):
        self.cfg = cfg if cfg is not None else load_config()
        s = section(self.cfg, "streaming")
        self.frame = int(s["frame_samples"])
        self.preroll = int(s["preroll_ms"] * SR / 1000)
        self.overlap = int(s["overlap_ms"] * SR / 1000)
        self.max_samples = int(s["max_utterance_seconds"] * SR)
        self.min_samples = int(s["min_utterance_seconds"] * SR)
        self.cut_fraction = float(s["forced_cut_search_fraction"])
        self.max_buffer = int(s["max_buffer_seconds"] * SR)
        self._step = step or make_step(self.cfg)
        self.buffer = AudioBuffer(self.max_samples + self.overlap + SR * 2)
        self._pending = np.zeros(0, dtype="float32")
        self._pos = 0
        self._start = None
        self._lead = 0
        self.samples_fed = 0

    @property
    def in_speech(self):
        return self._start is not None

    def speech_start(self):
        return None if self._start is None else self._start / SR

    def speech_seconds(self):
        if self._start is None:
            return 0.0
        return (self.buffer.end - max(self.buffer.start, self._start - self._lead)) / SR

    def current_info(self):
        if self._start is None:
            return None
        a0 = max(self.buffer.start, self._start - self._lead)
        audio = self.buffer.view(a0, self.buffer.end)
        return self._start / SR, audio, (self._start - a0) / SR

    def current(self):
        info = self.current_info()
        if info is None:
            return None
        return info[0], info[1]

    def _emit(self, end, forced=False, keep_short=False):
        a0 = max(self.buffer.start, self._start - self._lead)
        needed = int(0.1 * SR) if self._lead else self.min_samples
        if end - self._start < needed and not keep_short and not forced:
            return None
        audio = self.buffer.view(a0, end)
        if len(audio) == 0:
            return None
        return Utterance(self._start / SR, end / SR, audio, forced, (self._start - a0) / SR)

    def _forced_cut(self):
        lo = self._start + int(self.max_samples * (1 - self.cut_fraction))
        span = self.buffer.view(self._start, self._pos)
        t = quietest_point(span, (lo - self._start) / SR, (self._pos - self._start) / SR)
        cut = min(max(self._start + int(t * SR), lo), self._pos)
        u = self._emit(cut, forced=True)
        self._start = cut
        self._lead = self.overlap
        self.buffer.discard_before(cut - self.overlap)
        return u

    def push(self, chunk):
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 1 or not np.issubdtype(chunk.dtype, np.number):
            raise AudioError("a chunk must be a 1-D numeric array")
        if chunk.size == 0:
            return []
        if not np.all(np.isfinite(chunk)):
            raise AudioError("chunk contains NaN or infinite values")
        chunk = chunk.astype("float32", copy=False)
        self.buffer.append(chunk)
        self.samples_fed += len(chunk)
        data = np.concatenate([self._pending, chunk]) if len(self._pending) else chunk
        out = []
        i = 0
        while len(data) - i >= self.frame:
            ev = self._step(data[i: i + self.frame])
            i += self.frame
            self._pos += self.frame
            if ev and "start" in ev and self._start is None:
                self._start = max(self.buffer.start, int(ev["start"] * SR))
                self._lead = 0
            elif ev and "end" in ev and self._start is not None:
                end = min(self._pos, max(self._start + 1, int(ev["end"] * SR)))
                u = self._emit(end)
                if u is not None:
                    out.append(u)
                self._start = None
                self._lead = 0
            if self._start is not None and self._pos - self._start >= self.max_samples:
                u = self._forced_cut()
                if u is not None:
                    out.append(u)
            if self._start is None:
                self.buffer.discard_before(self._pos - self.preroll)
        self._pending = data[i:].copy()
        if self.buffer.size > self.max_buffer:
            self.buffer.discard_before(self.buffer.end - self.max_buffer)
            if self._start is not None and self._start < self.buffer.start:
                self._start = self.buffer.start
        return out

    def flush(self):
        u = None
        if self._start is not None:
            end = self.buffer.end
            u = self._emit(end, keep_short=True) if end > self._start else None
        self._start = None
        self._lead = 0
        self._pending = np.zeros(0, dtype="float32")
        self.buffer.clear()
        return u

    def release(self):
        self._start = None
        self._lead = 0
        self._pending = np.zeros(0, dtype="float32")
        self.buffer.clear()
        self._step = None
