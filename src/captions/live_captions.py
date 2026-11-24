import re
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass, field

from ..audio.preprocess import load_config
from ..audio.streaming import UtteranceStream
from ..utils.constants import SR
from ..utils.logging import get_logger, log_event

log = get_logger()

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text):
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def _units(text):
    return text.split() if " " in text.strip() else list(text.strip())


def trim_overlap(prev, new, max_words=6):
    if not prev or not new:
        return new
    a, b = _units(prev), _units(new)
    spaced = " " in new.strip()
    limit = max_words if spaced else max_words * 3
    na = [normalize_text(x) for x in a]
    nb = [normalize_text(x) for x in b]
    for k in range(min(limit, len(na), len(nb)), 0, -1):
        if na[-k:] == nb[:k] and any(nb[:k]):
            rest = b[k:]
            return (" " if spaced else "").join(rest).strip()
    return new


@dataclass
class Caption:
    kind: str
    utterance: float
    start: float
    end: float
    text: str
    language: str | None
    status: str = "ok"
    translation: str | None = None
    translation_language: str | None = None
    uncertain: bool = False
    reason: str | None = None
    flags: list = field(default_factory=list)
    asr_s: float = 0.0
    translation_s: float = 0.0
    endpoint_wait_s: float = 0.0
    estimated_delay_s: float = 0.0
    wall_delay_s: float | None = None


@dataclass
class LiveCaption:
    segment_id: str
    start: float
    end: float
    text: str = ""
    language: str | None = None
    status: str = "interim"
    version: int = 0
    speaker_id: str | None = None
    target_language: str | None = None
    translation: str | None = None
    translation_status: str = "none"
    translation_error: str | None = None
    uncertain: bool = False
    reason: str | None = None
    flags: list = field(default_factory=list)
    model: str | None = None
    detection: dict | None = None
    forced: bool = False
    sent: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    final_at: float | None = None


class CaptionManager:
    def __init__(self, target_language=None, speaker_id=None, translation_enabled=True, keep=300, first_number=1,
                 clock=time.time, duplicate_window_s=3.0):
        self.target = target_language
        self.speaker_id = speaker_id
        self.translation_enabled = bool(translation_enabled)
        self.keep = int(keep)
        self.clock = clock
        self.duplicate_window_s = float(duplicate_window_s)
        self._next = int(first_number)
        self._caps = {}
        self._order = []
        self._last_final = None
        self._lock = threading.RLock()

    @property
    def next_number(self):
        return self._next

    def open(self, start, end=None):
        with self._lock:
            sid = str(self._next)
            self._next += 1
            now = self.clock()
            cap = LiveCaption(sid, float(start), float(end if end is not None else start), speaker_id=self.speaker_id,
                              target_language=self.target if self.translation_enabled else None, created_at=now, updated_at=now)
            self._caps[sid] = cap
            self._order.append(sid)
            return sid

    def get(self, sid):
        with self._lock:
            return self._caps.get(sid)

    def message(self, cap):
        return {
            "type": "segment", "segment_id": cap.segment_id, "speaker_id": cap.speaker_id,
            "status": "partial" if cap.status == "interim" else "completed", "transcription_status": cap.status,
            "start_time": round(cap.start, 3), "end_time": round(cap.end, 3), "source_language": cap.language,
            "original_text": cap.text, "target_language": cap.target_language, "translated_text": cap.translation,
            "translation_status": cap.translation_status, "translation_error": cap.translation_error,
            "asr_model": cap.model, "detected_language": cap.language, "detection": cap.detection,
            "version": cap.version, "uncertain": cap.uncertain, "reason": cap.reason, "flags": list(cap.flags),
            "server_time": round(self.clock(), 3),
        }

    def _emit(self, cap):
        cap.version += 1
        cap.updated_at = self.clock()
        cap.sent = True
        return self.message(cap)

    def interim(self, sid, text, language, start, end, flags=(), uncertain=False, reason=None, model=None, detection=None):
        with self._lock:
            cap = self._caps.get(sid)
            text = (text or "").strip()
            if cap is None or cap.status != "interim" or not text:
                return None
            if text == cap.text and abs(end - cap.end) < 0.05:
                return None
            if self._last_final and normalize_text(text) == normalize_text(self._last_final[0]) \
                    and start - self._last_final[1] < self.duplicate_window_s:
                return None
            cap.text, cap.language, cap.start, cap.end = text, language or cap.language, float(start), float(end)
            cap.flags, cap.uncertain, cap.reason, cap.model = list(flags), bool(uncertain), reason, model or cap.model
            cap.detection = detection or cap.detection
            return self._emit(cap)

    def finalize(self, sid, text, language, start, end, flags=(), uncertain=False, reason=None, model=None, forced=False,
                 context_s=0.0, detection=None):
        with self._lock:
            cap = self._caps.get(sid)
            if cap is None or cap.status == "final" or cap.status == "removed":
                return None
            text = (text or "").strip()
            fl = list(flags)
            if text and context_s > 0 and self._last_final:
                trimmed = trim_overlap(self._last_final[0], text)
                if trimmed != text:
                    fl.append("overlap_trimmed")
                    text = trimmed
            if text and self._last_final and normalize_text(text) == normalize_text(self._last_final[0]) \
                    and start - self._last_final[1] < self.duplicate_window_s:
                fl.append("duplicate")
                text = ""
            if not text:
                return self.remove(sid)
            cap.text, cap.language, cap.start, cap.end = text, language or cap.language, float(start), float(end)
            cap.flags, cap.uncertain, cap.reason, cap.model, cap.forced = fl, bool(uncertain), reason, model or cap.model, forced
            cap.detection = detection or cap.detection
            cap.status = "final"
            cap.final_at = self.clock()
            self._last_final = (text, float(end))
            if not self.translation_enabled:
                cap.translation_status, cap.target_language = "disabled", None
            elif not self.target:
                cap.translation_status = "none"
            elif cap.language and cap.language == self.target:
                cap.translation, cap.translation_status = text, "same_language"
            else:
                cap.translation_status = "pending"
            return self._emit(cap)

    def remove(self, sid):
        with self._lock:
            cap = self._caps.get(sid)
            if cap is None or cap.status == "removed":
                return None
            was_sent = cap.sent
            cap.status = "removed"
            self._prune()
            return {"type": "remove", "segment_id": sid} if was_sent else None

    def needs_translation(self, sid):
        with self._lock:
            cap = self._caps.get(sid)
            return bool(cap and cap.status == "final" and cap.translation_status == "pending")

    def set_translation(self, sid, text, source_text, final, error=None):
        with self._lock:
            cap = self._caps.get(sid)
            if cap is None or cap.status == "removed":
                return None
            if final:
                if cap.status != "final" or normalize_text(source_text) != normalize_text(cap.text):
                    return None
                if error:
                    cap.translation_status, cap.translation_error = "failed", "Translation failed"
                    return self._emit(cap)
                cap.translation, cap.translation_status, cap.translation_error = text, "completed", None
                return self._emit(cap)
            if error or cap.status != "interim" or not text:
                return None
            if not normalize_text(cap.text).startswith(normalize_text(source_text)):
                return None
            if text == cap.translation:
                return None
            cap.translation, cap.translation_status, cap.translation_error = text, "interim", None
            return self._emit(cap)

    def snapshot(self):
        with self._lock:
            return [self.message(self._caps[s]) for s in self._order if s in self._caps and self._caps[s].status != "removed"]

    def finals(self):
        with self._lock:
            return sum(1 for c in self._caps.values() if c.status == "final")

    def _prune(self):
        extra = len(self._order) - self.keep
        if extra <= 0:
            return
        for sid in self._order[:extra]:
            cap = self._caps.get(sid)
            if cap is not None and cap.status in ("final", "removed") and cap.translation_status != "pending":
                self._caps.pop(sid, None)
        self._order = [s for s in self._order if s in self._caps]

    def prune(self):
        with self._lock:
            self._prune()

    def clear(self):
        with self._lock:
            self._caps.clear()
            self._order.clear()
            self._last_final = None


class LiveCaptioner:
    def __init__(self, transcriber, translator=None, target=None, cfg=None, step=None, clock=time.monotonic):
        self.cfg = cfg or load_config()
        s = self.cfg["streaming"]
        self.transcriber, self.translator, self.target, self.clock = transcriber, translator, target, clock
        self.stream = UtteranceStream(self.cfg, step)
        self.partial_every, self.min_partial = float(s["partial_interval_s"]), float(s["min_partial_seconds"])
        self._fed = []
        self._cum = 0
        self._last_partial = None

    def feed(self, chunk):
        now = self.clock()
        out = []
        utts = self.stream.push(chunk)
        self._cum += len(chunk)
        self._fed.append((self._cum, now))
        if len(self._fed) > 4000:
            del self._fed[:2000]
        for u in utts:
            out.append(self._final(u))
            self._last_partial = None
        if self.partial_every > 0 and not utts:
            out += self._maybe_partial(now)
        return out

    def flush(self):
        u = self.stream.flush()
        return [self._final(u)] if u is not None else []

    def _arrived_at(self, sample):
        i = bisect_left([c for c, _ in self._fed], sample)
        return self._fed[min(i, len(self._fed) - 1)][1] if self._fed else self.clock()

    def _maybe_partial(self, now):
        cur = self.stream.current()
        if cur is None or len(cur[1]) < self.min_partial * SR:
            return []
        if self._last_partial is not None and now - self._last_partial < self.partial_every:
            return []
        self._last_partial = now
        start, audio = cur
        t0 = time.perf_counter()
        p = self.transcriber.transcribe(audio)
        asr = time.perf_counter() - t0
        return [Caption("partial", round(start, 3), start, start + len(audio) / SR, p.text, p.language, p.status,
                        uncertain=p.uncertain, reason=p.reason, flags=list(p.flags), asr_s=asr)]

    def _final(self, u):
        t0 = time.perf_counter()
        p = self.transcriber.transcribe(u.audio)
        asr = time.perf_counter() - t0
        cap = Caption("final", round(u.start, 3), u.start, u.end, p.text, p.language, p.status, uncertain=p.uncertain,
                      reason=p.reason, flags=list(p.flags) + (["cut_at_max_length"] if u.forced else []), asr_s=asr)
        if self.target and self.translator and p.status == "ok" and p.text and p.language != self.target:
            t1 = time.perf_counter()
            try:
                tr = self.translator.translate(p.text, p.language, self.target)
                cap.translation, cap.translation_language = tr.text, self.target
                cap.flags += [f for f in tr.flags if f != "same_language"]
            except (RuntimeError, ValueError):
                cap.flags.append("translation_failed")
            cap.translation_s = time.perf_counter() - t1
        cap.endpoint_wait_s = max(0.0, (self.stream.samples_fed - int(u.end * SR)) / SR)
        cap.estimated_delay_s = cap.endpoint_wait_s + cap.asr_s + cap.translation_s
        cap.wall_delay_s = self.clock() - self._arrived_at(int(u.end * SR))
        log_event(log, "caption_final", utterance=cap.utterance, status=cap.status, asr_s=round(asr, 3),
                  translation_s=round(cap.translation_s, 3), delay_s=round(cap.estimated_delay_s, 3))
        return cap
