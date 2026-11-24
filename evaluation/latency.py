"""Timing: real-time factor, per-step times, caption delay, GPU memory."""
import time
from collections import defaultdict


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0


def rtf(processing_s: float, audio_s: float):
    """Processing time divided by audio time (below 1 is faster than real time). None when there was no audio."""
    return round(processing_s / audio_s, 4) if audio_s and audio_s > 0 else None


def gpu_memory_mb():
    try:
        import torch
        return round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else None
    except Exception:
        return None


def percentiles(values, points=(50, 95)) -> dict:
    v = sorted(values)
    if not v:
        return {}
    out = {f"p{p}": round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 3) for p in points}
    out.update(mean=round(sum(v) / len(v), 3), max=round(v[-1], 3), n=len(v))
    return out


class LatencyTracker:
    def __init__(self):
        self._v = defaultdict(list)

    def add(self, name: str, seconds: float):
        self._v[name].append(float(seconds))

    def summary(self) -> dict:
        return {k: percentiles(v) for k, v in self._v.items()}


def summarize_captions(captions) -> dict:
    """Delay numbers from LiveCaptioner output: ASR time, translation time, endpoint wait, and the total estimated delay."""
    finals = [c for c in captions if c.kind == "final"]
    t = LatencyTracker()
    for c in finals:
        t.add("asr_s", c.asr_s); t.add("endpoint_wait_s", c.endpoint_wait_s); t.add("estimated_delay_s", c.estimated_delay_s)
        if c.translation is not None:
            t.add("translation_s", c.translation_s)
        if c.wall_delay_s is not None:
            t.add("wall_delay_s", c.wall_delay_s)
    return {"finals": len(finals), "partials": sum(c.kind == "partial" for c in captions),
            "rejected_or_failed": sum(c.status in ("rejected", "failed") for c in finals), **t.summary()}
