import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..audio import (AudioError, StreamPreprocessor, UtteranceGate, UtteranceStream, PcmAssembler, cut,
                     load_config as load_audio_config, load_track, prepare_for_asr, section, speech_segments)
from ..captions.live_captions import CaptionManager
from ..language import (ConfigError, DetectorError, LangResult, LanguageDetector, LanguageTracker, is_code_mixed,
                        load_models_config, uncertain_reason)
from ..translation.translator import (InterimTranslationPlanner, TranslationError, check_target, translation_enabled)
from ..utils.constants import SR
from ..utils.logging import get_logger, log_event
from ..utils.timestamps import format_hms, srt_time, vtt_time
from .base import (ASRModel, InferenceError, InvalidAudio, LanguageRejected, ModelBusy, ModelLoadError, ModelNotConfigured,
                   UnsupportedLanguage, check_audio, finish_result)
from .indic_asr import IndicASR
from .qwen_asr import QwenASR

log = get_logger()


def _build(key, spec):
    if key == "qwen3_asr":
        return QwenASR.from_config(spec)
    if key == "indic":
        return IndicASR.from_config(spec)
    raise ConfigError(f"no adapter for model {key!r}")


class ModelSet:
    def __init__(self, cfg=None, overrides=None):
        self.cfg = cfg or load_models_config()
        self._specs = self.cfg["asr"]["models"]
        self._m = dict(overrides or {})
        self._lock = threading.Lock()

    def get(self, key):
        if key not in self._specs and key not in self._m:
            raise ConfigError(f"unknown model {key!r}")
        with self._lock:
            if key not in self._m:
                self._m[key] = _build(key, self._specs[key])
            return self._m[key]

    def keys(self):
        return list(self._specs)

    def unload_all(self):
        for m in self._m.values():
            m.unload()

    def close(self):
        for m in self._m.values():
            m.close()


def get_model(key, cfg=None):
    return ModelSet(cfg).get(key)


@dataclass
class Route:
    model: str
    language: str | None
    uncertain: bool = False
    reason: str | None = None
    fallbacks: tuple = ()


class Router:
    def __init__(self, cfg=None, models=None):
        self.cfg = cfg or load_models_config()
        self.models = models or ModelSet(self.cfg)
        self.l, self.asr = self.cfg["language"], self.cfg["asr"]
        self._validate()

    def _validate(self):
        for k in ("on_uncertain", "on_unsupported"):
            if self.l.get(k) not in ("reject", "best_guess"):
                raise ConfigError(f"language.{k} must be reject or best_guess, not {self.l.get(k)!r}")
        bg = self.l.get("best_guess_model")
        if "best_guess" in (self.l["on_uncertain"], self.l["on_unsupported"]) and not bg:
            raise ConfigError("best_guess needs language.best_guess_model: name the model that gets uncertain audio")
        if bg and bg not in self.asr["models"]:
            raise ConfigError(f"language.best_guess_model {bg!r} is not in asr.models")
        if self.asr["default_model"] not in self.asr["models"]:
            raise ConfigError(f"asr.default_model {self.asr['default_model']!r} is not in asr.models")
        for code, key in self.asr["routing"].items():
            if key not in self.asr["models"]:
                raise ConfigError(f"routing {code}: unknown model {key!r}")
            if not self.models.get(key).supports(code):
                raise ConfigError(f"routing {code} -> {key}, but {key} does not support {code}")
        for code, keys in (self.asr.get("fallback") or {}).items():
            for key in keys if isinstance(keys, (list, tuple)) else [keys]:
                if key not in self.asr["models"]:
                    raise ConfigError(f"fallback {code}: unknown model {key!r}")

    def _model_for(self, key):
        m = self.models.get(key)
        if not m.available:
            raise ModelNotConfigured(f"{key} is not configured yet")
        return m

    def fallbacks_for(self, code, primary):
        keys = (self.asr.get("fallback") or {}).get(code) or []
        keys = keys if isinstance(keys, (list, tuple)) else [keys]
        out = []
        for key in keys:
            if key == primary:
                continue
            m = self.models.get(key)
            if m.available and m.supports(code):
                out.append(key)
        return tuple(out)

    def coverage(self):
        out = {}
        for code, key in self.asr["routing"].items():
            m = self.models.get(key)
            out[code] = {"model": key, "ready": bool(m.available)}
        return out

    def route(self, r):
        reason = uncertain_reason(r, self.l["min_confidence"])
        if reason and (r.code is None or r.suspect):
            if self.l["on_uncertain"] == "reject":
                raise LanguageRejected(reason)
            self._model_for(self.l["best_guess_model"])
            return Route(self.l["best_guess_model"], None, True, reason)
        key = self.asr["routing"].get(r.code)
        if key is None:
            if self.l["on_unsupported"] == "reject":
                raise UnsupportedLanguage(r.code)
            self._model_for(self.l["best_guess_model"])
            return Route(self.l["best_guess_model"], None, True, f"unsupported language {r.code}")
        if reason and self.l["on_uncertain"] == "reject":
            raise LanguageRejected(reason)
        self._model_for(key)
        if r.weak and not reason:
            reason = r.note or "weak language detection"
        return Route(key, r.code, bool(reason), reason, self.fallbacks_for(r.code, key))


@dataclass
class PieceResult:
    text: str
    language: str | None
    raw_language: str | None
    model: str | None
    status: str
    uncertain: bool = False
    reason: str | None = None
    flags: tuple = ()
    seconds: float = 0.0
    detection: dict | None = None      # both detectors' answers, agreement, selected language (see LangResult.describe)
    weak_language: bool = False        # language is the detectors' best guess, not a confident call
    language_unresolved: bool = False  # rejected only because the language could not be settled; neighbours may settle it
    mode: str | None = None            # Indic-Transcribe-Flex output mode actually used


class SegmentTranscriber:
    def __init__(self, cfg=None, models=None, detector=None, router=None):
        self.cfg = cfg or load_models_config()
        self.models = models or ModelSet(self.cfg)
        self.router = router or Router(self.cfg, self.models)
        self.fixed = self.cfg["language"].get("fixed")
        self._detector = detector

    @property
    def detector(self):
        if self._detector is None:
            mode = self.cfg["language"]["detector"]
            asr = self.models.get("qwen3_asr") if mode in ("qwen", "compare") else None
            self._detector = LanguageDetector(self.cfg, asr=asr)
        return self._detector

    def _call(self, key, audio, code, interim, context):
        m = self.models.get(key)
        if getattr(m, "accepts_window_args", False):
            return m.transcribe(audio, code, interim=interim, context=context)
        return m.transcribe(audio, code)

    def transcribe(self, audio, language=None, interim=False, context=None):
        t0 = time.perf_counter()
        det, unresolved, weak = None, False, False
        try:
            audio = check_audio(audio)
            code = language or self.fixed
            if code:
                r = LangResult(code, code, None, "given", detail={"given": {"code": code, "raw": code, "confidence": None}})
            else:
                r = self.detector.detect(audio)
            det = r.describe()
            try:
                route = self.router.route(r)
            except LanguageRejected:
                unresolved = True
                raise
            det["selected"] = route.language or r.code
            weak = bool(r.weak)
            window = self.cfg["language"]["window_seconds"] * SR
            # Qwen's detection request already produced a transcript. Reuse it only when Qwen is the chosen ASR AND the final
            # language is the one Qwen itself transcribed in; otherwise (VoxLingua overrode Qwen, or Indic is chosen) it is dropped.
            reuse = (r.text is not None and route.model == "qwen3_asr" and (route.language is None or route.language == r.code)
                     and (r.qwen_code is None or r.qwen_code == r.code)
                     and len(audio) <= window)
            used = route.model
            if reuse:
                res = finish_result(r.text, r.code, r.raw, audio, expected_language=r.code, model="qwen3_asr")
                res.language_confidence = (r.detail.get("qwen") or {}).get("confidence")
            else:
                try:
                    res = self._call(route.model, audio, route.language, interim, context)
                except (InferenceError, ModelLoadError, ModelNotConfigured) as e:
                    if isinstance(e, ModelBusy) and interim:
                        raise
                    res = None
                    for key in route.fallbacks:
                        try:
                            res = self._call(key, audio, route.language, interim, context)
                            used = key
                            break
                        except (InferenceError, ModelLoadError, ModelNotConfigured):
                            continue
                    if res is None:
                        raise e
                    det["fallback_from"] = route.model
        except ModelBusy as e:
            return PieceResult("", None, None, None, "busy", reason=str(e), seconds=time.perf_counter() - t0, detection=det)
        except (LanguageRejected, UnsupportedLanguage, ModelNotConfigured, InvalidAudio) as e:
            return PieceResult("", None, None, None, "rejected", reason=str(e) or type(e).__name__,
                               seconds=time.perf_counter() - t0, detection=det, language_unresolved=unresolved)
        except (InferenceError, ModelLoadError, DetectorError) as e:
            return PieceResult("", None, None, None, "failed", reason=str(e), seconds=time.perf_counter() - t0, detection=det)
        det["asr_model"] = used
        return PieceResult(res.text, res.language or route.language, res.raw_language, used, "empty" if res.empty else "ok",
                           route.uncertain or res.suspicious, route.reason, res.flags, time.perf_counter() - t0, det, weak, False,
                           res.mode)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str | None
    status: str = "ok"
    speaker: str | None = None
    uncertain: bool = False
    reason: str | None = None
    flags: list = field(default_factory=list)
    model: str | None = None
    translation: str | None = None
    translation_language: str | None = None
    detection: dict | None = None
    mode: str | None = None


@dataclass
class Transcript:
    track: str | None
    speaker: str | None
    duration_s: float
    segments: list
    warnings: list = field(default_factory=list)
    speech_s: float = 0.0
    processing_s: float = 0.0

    @property
    def rtf(self):
        return self.processing_s / self.duration_s if self.duration_s else None

    def to_dict(self):
        d = asdict(self)
        d["rtf"] = None if self.rtf is None else round(self.rtf, 4)
        return d

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_txt(self):
        out = []
        for s in self.segments:
            who = f"{s.speaker}: " if s.speaker else ""
            body = f"[not transcribed: {s.reason}]" if s.status in ("rejected", "failed") else (s.text or "[no text]")
            notes = ([f"uncertain: {s.reason}"] if s.uncertain else []) + ([f"check: {', '.join(s.flags)}"] if s.flags else [])
            line = f"[{format_hms(s.start)} - {format_hms(s.end)}] ({s.language or '?'}) {who}{body}"
            out.append(line + (f"  [{'; '.join(notes)}]" if notes else ""))
            if s.translation:
                out.append(f"    -> ({s.translation_language}) {s.translation}")
        return "\n".join(out) + ("\n" if out else "")

    def _captions(self, fmt):
        blocks, n = [], 0
        t = srt_time if fmt == "srt" else vtt_time
        for s in self.segments:
            if not s.text or s.status != "ok":
                continue
            n += 1
            body = s.text + (f"\n{s.translation}" if s.translation else "")
            blocks.append(f"{n}\n" if fmt == "srt" else "")
            blocks[-1] += f"{t(s.start)} --> {t(s.end)}\n{body}\n"
        return "\n".join(blocks)

    def to_srt(self):
        return self._captions("srt")

    def to_vtt(self):
        return "WEBVTT\n\n" + self._captions("vtt")

    def save(self, directory, stem, formats=("json", "txt", "srt")):
        render = {"json": self.to_json, "txt": self.to_txt, "srt": self.to_srt, "vtt": self.to_vtt}
        bad = [f for f in formats if f not in render]
        if bad:
            raise ValueError(f"unknown formats {bad}")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in formats:
            p = d / f"{stem}.{f}"
            p.write_text(render[f](), encoding="utf-8")
            paths.append(str(p))
        return paths


def merge_transcripts(transcripts):
    segs = []
    for t in transcripts:
        for s in t.segments:
            c = TranscriptSegment(**asdict(s))
            c.speaker = c.speaker or t.speaker
            segs.append(c)
    segs.sort(key=lambda s: (s.start, s.end))
    return Transcript(None, None, max((t.duration_s for t in transcripts), default=0.0), segs,
                      [f"{t.speaker or t.track}: {w}" for t in transcripts for w in t.warnings],
                      sum(t.speech_s for t in transcripts), sum(t.processing_s for t in transcripts))


class TranscriptionPipeline:
    def __init__(self, audio_cfg=None, models_cfg=None, transcriber=None, vad=None, translator=None, translate=None):
        self.audio_cfg = audio_cfg or load_audio_config()
        self.models_cfg = models_cfg or load_models_config()
        self.transcriber = transcriber or SegmentTranscriber(self.models_cfg)
        self.vad, self.translator = vad, translator
        self.translate_enabled = (translation_enabled(self.models_cfg) if translate is None else bool(translate)) and translator is not None

    def transcribe_track(self, source, speaker=None, language=None, translate_to=None):
        audio, report = load_track(source, self.audio_cfg, return_report=True)
        return self.transcribe_audio(audio, report, str(source) if isinstance(source, (str, Path)) else None,
                                     speaker, language, translate_to)

    def _settle_from_neighbours(self, audio, segs, pieces, sr):
        gap_max = float(self.models_cfg["language"].get("neighbor_max_gap_s", 2.0))
        for i, p in enumerate(pieces):
            if not (p.status == "rejected" and p.language_unresolved):
                continue
            cands = (p.detection or {}).get("candidates") or []
            prev = next((j for j in range(i - 1, -1, -1) if pieces[j].status == "ok" and pieces[j].language), None)
            nxt = next((j for j in range(i + 1, len(pieces)) if pieces[j].status == "ok" and pieces[j].language), None)
            pick = None
            for j in (prev, nxt):
                if j is None:
                    continue
                lang = pieces[j].language
                gap = abs(segs[i].start - segs[j].end) if j < i else abs(segs[j].start - segs[i].end)
                # with candidate languages the neighbour must be one of them; with none (too short / no label) it must be adjacent
                if (cands and lang in cands) or (not cands and abs(i - j) == 1 and gap <= gap_max):
                    pick = (lang, j)
                    break
            if pick is None:
                continue
            q = self.transcriber.transcribe(cut(audio, segs[i], sr), pick[0])
            if q.status != "ok":
                continue
            q.uncertain = True
            q.reason = f"language unresolved ({p.reason}); taken from neighbouring chunk: {pick[0]}"
            q.detection = {**(p.detection or {}), **(q.detection or {}), "selected": pick[0], "resolution": "neighbouring chunk"}
            q.detection["agreed"] = (p.detection or {}).get("agreed")
            pieces[i] = q

    def transcribe_audio(self, audio, report=None, track=None, speaker=None, language=None, translate_to=None):
        t0 = time.perf_counter()
        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
            raise AudioError("audio must be a non-empty 1-D array")
        if translate_to:
            check_target(translate_to)
        sr = self.audio_cfg["audio"]["sample_rate"]
        warnings = list(report.warnings) if report else []
        segs = [] if (report and report.silent) else speech_segments(audio, self.audio_cfg, self.vad)
        # Pass 1: detect + route + transcribe every chunk on its own, so a language change mid-recording is followed.
        pieces = [self.transcriber.transcribe(cut(audio, seg, sr), language) for seg in segs]
        # Pass 2: a chunk whose language could not be settled (detectors disagreed, low confidence) is settled from the chunks
        # around it, but only to a language one of the detectors actually proposed for that chunk.
        if language is None and not self.transcriber.fixed:
            self._settle_from_neighbours(audio, segs, pieces, sr)
        out, speech = [], 0.0
        for seg, p in zip(segs, pieces):
            ts = TranscriptSegment(round(seg.start, 3), round(seg.end, 3), p.text, p.language, p.status, speaker, p.uncertain,
                                   p.reason, list(seg.flags) + list(p.flags), p.model, detection=p.detection, mode=p.mode)
            if translate_to and self.translate_enabled and p.status == "ok" and p.text and p.language and p.language != translate_to:
                try:
                    tr = self.translator.translate(p.text, p.language, translate_to)
                    ts.translation, ts.translation_language = tr.text, translate_to
                except (RuntimeError, ValueError, TranslationError):
                    ts.flags.append("translation_failed")
            out.append(ts)
            speech += seg.duration
        t = Transcript(track, speaker, round(len(audio) / sr, 3), out, warnings, round(speech, 3),
                       round(time.perf_counter() - t0, 3))
        log_event(log, "track_done", pieces=len(out), speech_s=t.speech_s, audio_s=t.duration_s, processing_s=t.processing_s,
                  rejected=sum(s.status == "rejected" for s in out), failed=sum(s.status == "failed" for s in out))
        return t


@dataclass
class LiveTask:
    kind: str
    segment_id: str
    audio: object = None
    start: float = 0.0
    end: float = 0.0
    forced: bool = False
    context_s: float = 0.0
    text: str | None = None
    language: str | None = None
    final: bool = False
    gate_flags: tuple = ()


@dataclass
class TaskOutcome:
    messages: list = field(default_factory=list)
    followups: list = field(default_factory=list)


class LiveSession:
    def __init__(self, pipeline, source_language="auto", target_language=None, speaker_id=None, session_id=None,
                 input_rate=SR, input_format="pcm_s16le", channels=1, step=None, state=None, clock=time.monotonic):
        self.pipeline = pipeline
        self.cfg = pipeline.audio_cfg
        self.models_cfg = pipeline.models_cfg
        self.s = section(self.cfg, "streaming")
        self.session_id = session_id
        self.clock = clock
        state = state or {}
        self.fixed = None if source_language in (None, "", "auto") else source_language
        self.target = target_language or None
        self.translate_on = bool(self.target) and pipeline.translate_enabled
        self.pre = StreamPreprocessor(self.cfg, input_rate)
        self.asm = PcmAssembler(input_format, channels)
        self.stream = UtteranceStream(self.cfg, step)
        self.gate = UtteranceGate(self.cfg)
        self.tracker = LanguageTracker(self.models_cfg, current=state.get("language"))
        tr = self.models_cfg.get("translation") or {}
        self.interim_translation = bool(tr.get("interim", True))
        self.planner = InterimTranslationPlanner(tr.get("interim_min_words", 5), tr.get("interim_min_interval_s", 1.5),
                                                 tr.get("interim_min_growth_words", 3), clock)
        self.captions = CaptionManager(self.target, speaker_id, pipeline.translate_enabled, int(self.s["keep_captions"]),
                                       int(state.get("next_number", 1)))
        self.offset = float(state.get("time_offset", 0.0))
        live = (self.models_cfg["language"].get("live") or {})
        self.recheck_every = max(1, int(live.get("recheck_every", 1)))
        self._finals_since_check = self.recheck_every
        self._lock = threading.RLock()
        self._open = None
        self._inflight = set()
        self._last_interim = {}
        self._latency = None
        self._last_text = ""
        self._warned = {}
        self.finals = 0
        self.interims = 0
        self.closed = False

    @property
    def seconds_received(self):
        return self.stream.samples_fed / SR

    def _warn(self, key, text, every=10.0):
        now = self.clock()
        if now - self._warned.get(key, -1e9) < every:
            return []
        self._warned[key] = now
        return [{"type": "warning", "message": text}]

    def feed_bytes(self, data):
        return self.feed_samples(self.asm.push(data))

    def feed_samples(self, samples):
        with self._lock:
            if self.closed:
                return [], []
            messages = []
            y = self.pre.process(samples)
            if self.pre.clipping_ratio() > float(section(self.cfg, "audio")["clip_warn_ratio"]) * 20:
                messages += self._warn("clip", "Microphone input is clipping; lower the input volume.")
            tasks = []
            for u in self.stream.push(y):
                t, m = self._on_utterance(u)
                tasks += t
                messages += m
            t, m = self._interim_task()
            return tasks + t, messages + m

    def _on_utterance(self, u):
        sid = self._open
        self._open = None
        if sid is None:
            sid = self.captions.open(u.start + self.offset, u.end + self.offset)
        flags = self.gate.check(u.audio)
        self._last_interim.pop(sid, None)
        if "near_silent" in flags or self.gate.should_drop(flags):
            m = self.captions.remove(sid)
            return [], [m] if m else []
        task = LiveTask("final", sid, u.audio, u.start + self.offset, u.end + self.offset, u.forced, u.context_s,
                        gate_flags=tuple(flags))
        return [task], []

    def _interim_due(self, sid, length_s):
        interval = float(self.s["partial_interval_s"])
        if interval <= 0 or length_s < float(self.s["min_partial_seconds"]) or sid in self._inflight:
            return False
        last = self._last_interim.get(sid)
        if last is None:
            return True
        wait = max(interval, (self._latency or 0.0) * float(self.s["partial_load_factor"]))
        return self.clock() - last[0] >= wait and length_s - last[1] >= float(self.s["min_partial_growth_s"])

    def _interim_task(self):
        if not self.stream.in_speech:
            return [], []
        start = self.stream.speech_start()
        if self._open is None:
            self._open = self.captions.open(start + self.offset)
        sid = self._open
        length = self.stream.speech_seconds()
        if not self._interim_due(sid, length):
            return [], []
        info = self.stream.current_info()
        if info is None:
            return [], []
        s0, audio, ctx = info
        self._inflight.add(sid)
        self._last_interim[sid] = (self.clock(), length)
        end = s0 + (len(audio) / SR - ctx)
        return [LiveTask("interim", sid, audio, s0 + self.offset, end + self.offset, context_s=ctx)], []

    def flush(self):
        with self._lock:
            tasks, messages = [], []
            u = self.stream.flush()
            if u is not None:
                tasks, messages = self._on_utterance(u)
            elif self._open is not None:
                m = self.captions.remove(self._open)
                messages = [m] if m else []
            self._open = None
            leftover = self.asm.flush()
            if leftover:
                log.info("dropped %d trailing bytes of an incomplete audio frame", leftover)
            return tasks, messages

    def run(self, task):
        if task.kind == "interim":
            return self._run_interim(task)
        if task.kind == "final":
            return self._run_final(task)
        if task.kind == "translate":
            return self._run_translate(task)
        return TaskOutcome()

    def _language_hint(self):
        return self.fixed or self.tracker.current

    def _run_interim(self, task):
        sid = task.segment_id
        t0 = time.perf_counter()
        try:
            cap = self.captions.get(sid)
            if cap is None or cap.status != "interim" or self.closed:
                return TaskOutcome()
            audio, _ = prepare_for_asr(task.audio, self.cfg)
            lang = self._language_hint()
            piece = self.pipeline.transcriber.transcribe(audio, lang, interim=True)
        except (AudioError, InvalidAudio):
            return TaskOutcome()
        finally:
            with self._lock:
                self._inflight.discard(sid)
        if piece.status == "ok":
            with self._lock:
                self._latency = piece.seconds if self._latency is None else 0.6 * self._latency + 0.4 * piece.seconds
        if piece.status != "ok" or not piece.text:
            return TaskOutcome()
        flags = list(piece.flags) + (["code_mixed"] if is_code_mixed(piece.text) else [])
        msg = self.captions.interim(sid, piece.text, piece.language or lang, task.start, task.end, flags, piece.uncertain,
                                    piece.reason, piece.model, detection=piece.detection)
        if msg is None:
            return TaskOutcome()
        self.interims += 1
        out = TaskOutcome([msg])
        if self.translate_on and self.interim_translation and piece.language and piece.language != self.target:
            stable = self.planner.update(sid, piece.text)
            if stable:
                out.followups.append(LiveTask("translate", sid, text=stable, language=piece.language, final=False))
        return out

    def _transcribe_final(self, audio, duration):
        T = self.pipeline.transcriber
        ctx = self._last_text[-200:] or None
        if self.fixed:
            return T.transcribe(audio, self.fixed, context=ctx)
        locked = self.tracker.current
        if locked and duration < self.tracker.switch_min:
            return T.transcribe(audio, locked, context=ctx)
        if locked and self._finals_since_check < self.recheck_every - 1:
            self._finals_since_check += 1
            return T.transcribe(audio, locked, context=ctx)
        self._finals_since_check = 0
        piece = T.transcribe(audio, None, context=ctx)
        if piece.status in ("rejected", "failed", "empty") and locked:
            why = piece.reason if piece.language_unresolved else None
            return self._keep_locked(T, audio, locked, ctx, why, piece.detection)
        if piece.status != "ok":
            return piece
        decision = self.tracker.observe(piece.language, duration, weak=piece.weak_language)
        if decision.accepted and decision.code:
            piece.language = decision.code
            if piece.detection is not None:
                piece.detection["selected"] = decision.code
                if decision.switched and decision.reason:
                    piece.detection["resolution"] = decision.reason
            return piece
        if locked:
            return self._keep_locked(T, audio, locked, ctx, decision.reason, piece.detection)
        return piece

    def _keep_locked(self, T, audio, locked, ctx, why, detection):
        """Chunk-level detection was not settled: stay on the language this speaker is already locked to, and say so."""
        q = T.transcribe(audio, locked, context=ctx)
        if q.status == "ok" and why:
            q.uncertain = True
            q.reason = f"language unresolved ({why}); kept {locked}"
            q.detection = {**(detection or {}), **(q.detection or {}), "selected": locked, "resolution": "kept locked language"}
        return q

    def _run_final(self, task):
        sid = task.segment_id
        try:
            audio, _ = prepare_for_asr(task.audio, self.cfg)
            duration = max(0.0, len(task.audio) / SR - task.context_s)
            piece = self._transcribe_final(audio, duration)
        except (AudioError, InvalidAudio) as e:
            piece = None
            reason = str(e)
        messages = []
        if piece is None or piece.status != "ok" or not piece.text:
            m = self.captions.remove(sid)
            if m:
                messages.append(m)
            if piece is not None and piece.status in ("rejected", "failed"):
                messages += self._warn(f"skip:{piece.reason}", f"A segment was skipped: {piece.reason}", 8.0)
            elif piece is None:
                messages += self._warn("audio", f"A segment was skipped: {reason}", 8.0)
            self.planner.forget(sid)
            return TaskOutcome(messages)
        flags = list(piece.flags) + list(task.gate_flags)
        if task.forced:
            flags.append("cut_at_max_length")
        if is_code_mixed(piece.text):
            flags.append("code_mixed")
        msg = self.captions.finalize(sid, piece.text, piece.language, task.start, task.end, flags, piece.uncertain,
                                     piece.reason, piece.model, task.forced, task.context_s, detection=piece.detection)
        self.planner.forget(sid)
        out = TaskOutcome(messages)
        if msg is None:
            return out
        self.finals += 1
        cap = self.captions.get(sid)
        if cap is not None:
            self._last_text = cap.text
        out.messages.append(msg)
        if self.captions.needs_translation(sid):
            out.followups.append(LiveTask("translate", sid, text=cap.text, language=cap.language, final=True))
        return out

    def _run_translate(self, task):
        tr = self.pipeline.translator
        if tr is None or not self.translate_on:
            return TaskOutcome()
        try:
            fn = getattr(tr, "translate_interim", tr.translate) if not task.final else tr.translate
            res = fn(task.text, task.language, self.target)
        except (TranslationError, RuntimeError, ValueError) as e:
            log.warning("translation failed: %s", type(e).__name__)
            msg = self.captions.set_translation(task.segment_id, None, task.text, task.final, error=str(e))
        else:
            msg = self.captions.set_translation(task.segment_id, res.text, task.text, task.final)
        return TaskOutcome([msg] if msg else [])

    def update_languages(self, source=None, target=None):
        with self._lock:
            if source is not None:
                self.fixed = None if source in ("", "auto") else source
                if self.fixed is None:
                    self.tracker.reset()
            if target is not None:
                self.target = target or None
                self.captions.target = self.target if self.pipeline.translate_enabled else None
                self.translate_on = bool(self.target) and self.pipeline.translate_enabled

    def abandon(self, task):
        if task.kind == "interim":
            with self._lock:
                self._inflight.discard(task.segment_id)
                self._last_interim.pop(task.segment_id, None)

    def export_state(self):
        return {"language": self.tracker.current, "next_number": self.captions.next_number,
                "time_offset": self.offset + self.seconds_received}

    def stats(self):
        return {"seconds_received": round(self.seconds_received, 2), "finals": self.finals, "interims": self.interims,
                "language": self.tracker.current, "audio": self.pre.stats()}

    def close(self):
        with self._lock:
            self.closed = True
            self.stream.release()
            self.captions.clear()
            self.planner.reset()
            self._inflight.clear()
            self._last_interim.clear()
