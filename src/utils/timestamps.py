"""Time helpers. All times in the project are seconds (float) on the original track timeline."""
import re
from math import isfinite

_HMS = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)(?:[.,](\d{1,6}))?$")


def parse_time(s: str) -> float:
    """'HH:MM:SS.mmm' (also 'HH:MM:SS,mmm') -> seconds. Raises ValueError for anything else."""
    m = _HMS.match(str(s).strip())
    if not m:
        raise ValueError(f"not a HH:MM:SS.mmm time: {s!r}")
    h, mi, sec, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(sec) + (int(frac.ljust(3, "0")[:6]) / 10 ** len(frac.ljust(3, "0")[:6]) if frac else 0.0)


def _split(seconds: float):
    if not isfinite(seconds) or seconds < 0:
        raise ValueError(f"invalid time {seconds!r}")
    ms = int(round(seconds * 1000))
    return ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000


def format_hms(seconds: float, sep: str = ".") -> str:
    h, m, s, ms = _split(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def srt_time(seconds: float) -> str:
    return format_hms(seconds, ",")


def vtt_time(seconds: float) -> str:
    return format_hms(seconds, ".")


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length in seconds of the overlap of two time ranges (0 when apart)."""
    return max(0.0, min(a1, b1) - max(a0, b0))
