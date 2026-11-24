import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

import numpy as np
import yaml

from ..utils.constants import CONFUSABLE, QWEN_LANGUAGES, SR
from ..utils.logging import get_logger

from dotenv import load_dotenv

load_dotenv()
log = get_logger()

MODELS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"
MIN_DETECT_SECONDS = 0.5

MODEL_DEFAULTS = {
    "language": {
        "detector": "qwen", "primary": "qwen", "window_seconds": 20, "min_confidence": 0.5, "on_uncertain": "reject",
        "on_unsupported": "reject", "best_guess_model": None, "verify_suspect": False,
        "override_confidence": 0.85, "neighbor_max_gap_s": 2.0,
        "accept_confidence": 0.6, "blind_min_confidence": 0.35, "floor_confidence": 0.3, "min_supported_mass": 0.2,
        "vox_fold": {"yue": "zh", "ur": "hi"},
        "live": {"switch_min_seconds": 2.0, "strong_seconds": 4.0, "switch_confirmations": 2, "lock_min_seconds": 1.2,
                 "default_language": None},
    },
    "asr": {"fallback": {}},
    "translation": {"enabled": True},
    "concurrency": {"asr": 2, "uploads": 1, "translation": 2, "max_live_sessions": 8, "shed_waiting": 4},
}

SCRIPT_RANGES = {
    "latin": ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)),
    "devanagari": ((0x900, 0x97F),),
    "bengali": ((0x980, 0x9FF),),
    "gurmukhi": ((0xA00, 0xA7F),),
    "gujarati": ((0xA80, 0xAFF),),
    "oriya": ((0xB00, 0xB7F),),
    "tamil": ((0xB80, 0xBFF),),
    "telugu": ((0xC00, 0xC7F),),
    "kannada": ((0xC80, 0xCFF),),
    "malayalam": ((0xD00, 0xD7F),),
    "arabic": ((0x600, 0x6FF), (0x750, 0x77F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "cyrillic": ((0x400, 0x52F),),
    "greek": ((0x370, 0x3FF),),
    "han": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),
    "hiragana": ((0x3040, 0x309F),),
    "katakana": ((0x30A0, 0x30FF),),
    "hangul": ((0xAC00, 0xD7AF), (0x1100, 0x11FF)),
    "thai": ((0xE00, 0xE7F),),
}

LATIN_LANGUAGES = ("en", "de", "fr", "es", "it", "pt", "id", "vi", "tr", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "hu", "ro")

LANGUAGE_SCRIPTS = {code: {"latin"} for code in LATIN_LANGUAGES}
LANGUAGE_SCRIPTS.update({
    "hi": {"devanagari"}, "mr": {"devanagari"}, "ne": {"devanagari"}, "bn": {"bengali"}, "pa": {"gurmukhi"},
    "gu": {"gujarati"}, "or": {"oriya"}, "ta": {"tamil"}, "te": {"telugu"}, "kn": {"kannada"}, "ml": {"malayalam"},
    "ar": {"arabic"}, "fa": {"arabic"}, "ur": {"arabic"}, "ru": {"cyrillic"}, "mk": {"cyrillic"}, "el": {"greek"},
    "zh": {"han"}, "yue": {"han"}, "ja": {"hiragana", "katakana", "han"}, "ko": {"hangul"}, "th": {"thai"},
})


class ConfigError(ValueError):
    pass


class DetectorError(RuntimeError):
    pass


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_models_config(path=MODELS_PATH):
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return _merge(deepcopy(MODEL_DEFAULTS), user)


def _script_of(ch):
    o = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= o <= hi:
                return name
    return None


def script_profile(text):
    counts = {}
    total = 0
    for ch in text or "":
        if ch.isalpha():
            total += 1
            s = _script_of(ch)
            if s:
                counts[s] = counts.get(s, 0) + 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def expected_scripts(code):
    return LANGUAGE_SCRIPTS.get(code)


def script_mismatch(text, code):
    exp = expected_scripts(code)
    if not exp or not text:
        return None
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 3:
        return None
    prof = script_profile(text)
    share = sum(prof.get(s, 0.0) for s in exp)
    if exp == {"latin"}:
        return f"output is not in the expected script for {code}" if share < 0.7 else None
    return f"output is not in the expected script for {code}" if share < 0.3 else None


def is_code_mixed(text, minimum=0.15):
    prof = script_profile(text)
    return sum(1 for v in prof.values() if v >= minimum) >= 2


def normalize_language_code(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    r = raw.strip()
    if r in QWEN_LANGUAGES:
        return QWEN_LANGUAGES[r]
    t = r.title()
    if t in QWEN_LANGUAGES:
        return QWEN_LANGUAGES[t]
    low = r.lower().replace("_", "-")
    if low in QWEN_LANGUAGES.values():
        return low
    base = low.split("-")[0]
    return base if base in QWEN_LANGUAGES.values() else None


@dataclass
class LangResult:
    code: str | None
    raw: str | None
    confidence: float | None
    backend: str
    text: str | None = None
    suspect: tuple = ()
    other: "LangResult | None" = None
    agree: bool | None = None
    note: str | None = None
    qwen_code: str | None = None      # what Qwen itself said; its transcript is reusable only if this equals the final code
    resolved: bool = False            # a disagreement that the fusion rules settled (not left uncertain)
    detail: dict = field(default_factory=dict)   # per-detector {code, raw, confidence} for the response / logs
    probs: dict = field(default_factory=dict)    # VoxLingua posterior renormalised over the languages we can route
    weak: bool = False                           # best available answer, but not convincing: transcribe it, flag it uncertain

    def candidates(self):
        """Languages any detector proposed, for resolving an uncertain chunk from its neighbours."""
        out = []
        for d in (self.detail or {}).values():
            c = d.get("code")
            if c and c not in out:
                out.append(c)
        for c in self.suspect:
            if c not in out:
                out.append(c)
        return out

    def describe(self, selected=None):
        top = sorted((self.probs or {}).items(), key=lambda kv: -kv[1])[:3]
        return {**{k: dict(v) for k, v in (self.detail or {}).items()}, "agreed": self.agree, "selected": selected or self.code,
                "resolution": self.note, "weak": self.weak, "candidates": self.candidates(),
                "supported_probs": {k: round(v, 3) for k, v in top}}


def uncertain_reason(r, min_confidence):
    if r.code is None:
        return r.note or f"no_language ({r.raw!r})"
    if r.suspect:
        return f"label {r.raw!r} may really be {'/'.join(r.suspect)}"
    if r.confidence is not None and r.confidence < min_confidence and not (r.weak or r.resolved):
        return f"low_confidence ({r.confidence:.2f})"
    if r.agree is False and not r.resolved:
        return f"detectors disagree ({r.code} vs {r.other.code if r.other else None})"
    return None


def _check_audio(audio):
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
        raise ValueError("audio must be a non-empty 1-D array")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or infinite values")


class QwenDetector:
    name = "qwen"

    def __init__(self, asr):
        self.asr = asr

    def detect(self, audio):
        _check_audio(audio)
        r = self.asr.transcribe(audio, None)
        code, conf = r.language, getattr(r, "language_confidence", None)
        return LangResult(code, r.raw_language, conf, self.name, text=r.text, suspect=CONFUSABLE.get(code, ()), qwen_code=code,
                          detail={"qwen": {"code": code, "raw": r.raw_language, "confidence": conf}})


class VoxLinguaDetector:
    """VoxLingua107 (107 languages). The posterior is renormalised over the languages the system can actually route
    (closed set), because the raw top-1 over 107 languages is spread thin on short or noisy chunks and often sits below any sensible
    threshold even when the answer is right. Audio that clearly belongs to an unsupported language still comes back as that language."""
    name = "voxlingua"

    def __init__(self, savedir="pretrained/voxlingua107", supported=None, fold=None, min_supported_mass=0.2):
        self.savedir, self._m = savedir, None
        self.supported = set(supported) if supported else None
        self.fold = dict(fold if fold is not None else MODEL_DEFAULTS["language"]["vox_fold"])
        self.min_mass = float(min_supported_mass)
        self._lock = threading.Lock()
        self._labels = None

    @property
    def loaded(self):
        return self._m is not None

    def _load(self):
        if self._m is None:
            with self._lock:
                if self._m is None:
                    try:
                        from speechbrain.inference.classifiers import EncoderClassifier
                        m = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=self.savedir)
                        self._labels = {int(i): parse_voxlingua(l) for i, l in m.hparams.label_encoder.ind2lab.items()}
                        self._m = m
                    except Exception as e:
                        raise DetectorError(f"VoxLingua could not be loaded ({type(e).__name__}); pip install speechbrain torch") from e
        return self._m

    def warm_up(self):
        self._load()
        self.detect(np.zeros(SR, dtype="float32") + 1e-3)

    def posterior(self, out_prob):
        """{code: probability} over all 107 labels, folding near-identical labels (ur->hi, yue->zh) into routed ones."""
        v = out_prob.detach().cpu().numpy() if hasattr(out_prob, "detach") else out_prob
        v = np.asarray(v, dtype="float64").reshape(-1)
        p = np.exp(v) if bool((v <= 1e-6).all()) else v           # speechbrain returns log-probabilities
        total = float(p.sum())
        p = (p / total) if total > 0 else p
        out = {}
        for i, pr in enumerate(p.tolist()):
            code = self._labels.get(i)
            if code is None:
                continue
            code = self.fold.get(code, code) if (self.supported is None or self.fold.get(code, code) in self.supported) else code
            out[code] = out.get(code, 0.0) + pr
        return out

    def classify(self, post):
        """Closed-set decision from the full posterior -> (code, confidence, probs_over_supported, open_top)."""
        open_code, open_p = max(post.items(), key=lambda kv: kv[1])
        if not self.supported:
            return open_code, open_p, {}, (open_code, open_p)
        closed = {c: post.get(c, 0.0) for c in self.supported}
        mass = sum(closed.values())
        if mass < self.min_mass and open_code not in self.supported and open_p >= 0.5:
            return open_code, open_p, {}, (open_code, open_p)     # clearly some other language: let the router reject it
        probs = {c: v / mass for c, v in closed.items()} if mass > 0 else {}
        code = max(probs, key=probs.get) if probs else None
        return code, (probs[code] if code else None), probs, (open_code, open_p)

    def detect(self, audio):
        _check_audio(audio)
        if len(audio) < MIN_DETECT_SECONDS * SR:
            return LangResult(None, None, None, self.name, note="too_short")
        try:
            import torch
            m = self._load()
            with self._lock:
                out_prob, _, _, label = m.classify_batch(torch.from_numpy(audio).unsqueeze(0))
            post = self.posterior(out_prob)
        except DetectorError:
            raise
        except Exception as e:
            raise DetectorError(f"VoxLingua failed ({type(e).__name__})") from e
        code, conf, probs, (ocode, op) = self.classify(post)
        raw = label[0] if isinstance(label[0], str) else None
        return LangResult(code, raw, conf, self.name, probs=probs,
                          detail={"voxlingua": {"code": code, "raw": raw, "confidence": conf, "open_set": {"code": ocode, "p": round(op, 3)}}})


class RemoteVoxLinguaDetector(VoxLinguaDetector):
    """VoxLingua107 hosted on the GPU notebook (POST /identify-language). Same closed-set decision as the local detector.

    Reads `probabilities` ({label: p}, all 107 labels) when the server sends it; otherwise falls back to `top_candidates`
    (top 5 only; confidences may be log-probabilities and are exponentiated)."""

    def __init__(self, api_url, api_key, supported=None, fold=None, min_supported_mass=0.2, timeout=15, connect_timeout=5,
                 max_concurrent=2, retries=1, session=None):
        super().__init__(None, supported, fold, min_supported_mass)
        from ..transcription.qwen_asr import RemoteClient    # imported here: qwen_asr itself imports this module
        self.client = RemoteClient(api_url, api_key, timeout, connect_timeout, timeout, max_concurrent, retries, 0.3, 30, session)

    @property
    def loaded(self):
        return self.client.configured

    def warm_up(self):
        self.client.ping()

    @staticmethod
    def parse(payload):
        probs = payload.get("probabilities")
        if isinstance(probs, dict) and probs:
            items = [(k, v) for k, v in probs.items() if isinstance(v, (int, float))]
        else:
            items = [(c.get("language"), c.get("confidence")) for c in payload.get("top_candidates") or []
                     if isinstance(c, dict) and isinstance(c.get("confidence"), (int, float))]
        vals = [float(v) for _, v in items]
        logs = bool(vals) and all(v <= 1e-6 for v in vals)
        post = {}
        for (label, _), v in zip(items, vals):
            code = parse_voxlingua(label if isinstance(label, str) and ":" in label else f"{label}:")
            if code:
                post[code] = post.get(code, 0.0) + (float(np.exp(v)) if logs else v)
        return post       # true probabilities (top-5 replies cover only part of the mass, which classify() accounts for)

    def detect(self, audio):
        _check_audio(audio)
        if len(audio) < MIN_DETECT_SECONDS * SR:
            return LangResult(None, None, None, self.name, note="too_short")
        from ..transcription.qwen_asr import to_wav_bytes
        try:
            payload = self.client.post_audio("/identify-language", to_wav_bytes(audio), {})
        except RuntimeError as e:
            raise DetectorError(f"remote VoxLingua failed: {e}") from e
        post = self.parse(payload)
        if not post:
            raise DetectorError("remote VoxLingua returned no language scores")
        fold = {k: v for k, v in self.fold.items() if self.supported is None or v in self.supported}
        folded = {}
        for k, v in post.items():
            folded[fold.get(k, k)] = folded.get(fold.get(k, k), 0.0) + v
        code, conf, probs, (ocode, op) = self.classify(folded)
        raw = payload.get("predicted_language") if isinstance(payload.get("predicted_language"), str) else None
        return LangResult(code, raw, conf, self.name, probs=probs,
                          detail={"voxlingua": {"code": code, "raw": raw, "confidence": conf, "open_set": {"code": ocode, "p": round(op, 3)}}})


# VoxLingua107 uses a few legacy ISO codes; map them to the codes the rest of the system uses.
VOX_ALIASES = {"iw": "he", "jw": "jv", "tl": "fil", "nn": "no", "mo": "ro"}


def parse_voxlingua(label):
    if not isinstance(label, str) or ":" not in label:
        return None
    code = VOX_ALIASES.get(label.split(":")[0].strip().lower(), label.split(":")[0].strip().lower())
    return code if 2 <= len(code) <= 3 and code.isalpha() else None


def parse_confidence(v):
    if not isfinite(v):
        return None
    if v <= 0:
        v = float(np.exp(v))
    return v if 0.0 <= v <= 1.0 + 1e-6 else None


@dataclass
class LanguageDecision:
    code: str | None
    accepted: bool
    switched: bool = False
    reason: str | None = None


class LanguageTracker:
    def __init__(self, cfg=None, current=None):
        c = ((cfg or load_models_config())["language"]).get("live", {})
        d = MODEL_DEFAULTS["language"]["live"]
        self.switch_min = float(c.get("switch_min_seconds", d["switch_min_seconds"]))
        self.strong = float(c.get("strong_seconds", d["strong_seconds"]))
        self.confirmations = max(1, int(c.get("switch_confirmations", d["switch_confirmations"])))
        self.lock_min = float(c.get("lock_min_seconds", d["lock_min_seconds"]))
        self.current = current or c.get("default_language")
        self.candidate = None
        self.candidate_count = 0
        self.history = []

    def reset(self):
        self.current = None
        self.candidate = None
        self.candidate_count = 0

    def _clear_candidate(self):
        self.candidate = None
        self.candidate_count = 0

    def observe(self, code, seconds, suspect=(), weak=False):
        self.history.append((code, round(float(seconds), 2)))
        if len(self.history) > 20:
            self.history.pop(0)
        if code is None or suspect:
            return LanguageDecision(self.current, False, False, "unreliable detection")
        if weak:
            # a best-guess detection never changes an established language; with no language yet, the same guess twice in a row locks it
            if self.current is not None:
                return LanguageDecision(self.current, code == self.current, False, None if code == self.current else "weak detection ignored")
            if self.candidate == code:
                self.current = code
                self._clear_candidate()
                return LanguageDecision(code, True, True, "language locked (repeated weak detection)")
            self.candidate, self.candidate_count = code, 1
            return LanguageDecision(code, True, False, "tentative")
        if self.current is None:
            if seconds >= self.lock_min or self.candidate == code:
                self.current = code
                self._clear_candidate()
                return LanguageDecision(code, True, True, "language locked")
            self.candidate, self.candidate_count = code, 1
            return LanguageDecision(code, True, False, "tentative")
        if code == self.current:
            self._clear_candidate()
            return LanguageDecision(code, True, False, None)
        if seconds < self.switch_min:
            return LanguageDecision(self.current, False, False, "segment too short to change language")
        if seconds >= self.strong:
            self.current = code
            self._clear_candidate()
            return LanguageDecision(code, True, True, "language changed")
        if self.candidate == code:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = code, 1
        if self.candidate_count >= self.confirmations:
            self.current = code
            self._clear_candidate()
            return LanguageDecision(code, True, True, "language changed")
        return LanguageDecision(self.current, False, False, "change not yet confirmed")


class LanguageDetector:
    def __init__(self, cfg=None, asr=None, qwen=None, voxlingua=None):
        full = cfg or load_models_config()
        c = full["language"]
        if c["detector"] not in ("qwen", "voxlingua", "compare"):
            raise ConfigError(f"language.detector must be qwen, voxlingua or compare, not {c['detector']!r}")
        if c["primary"] not in ("qwen", "voxlingua"):
            raise ConfigError(f"language.primary must be qwen or voxlingua, not {c['primary']!r}")
        if not (isinstance(c["window_seconds"], (int, float)) and c["window_seconds"] >= MIN_DETECT_SECONDS):
            raise ConfigError("language.window_seconds is too small")
        if not (isinstance(c["min_confidence"], (int, float)) and 0 <= c["min_confidence"] <= 1):
            raise ConfigError("language.min_confidence must be between 0 and 1")
        self.cfg, self.mode = c, c["detector"]
        if qwen is None and self.mode in ("qwen", "compare"):
            if asr is None:
                raise ValueError("the qwen detector needs the shared Qwen ASR model (asr=...)")
            qwen = QwenDetector(asr)
        self._qwen = qwen
        self.supported = set((full.get("asr") or {}).get("routing") or {}) or None
        self._vox = voxlingua or (self._new_vox(c) if self.mode in ("voxlingua", "compare") else None)
        self.verify_suspect = bool(c.get("verify_suspect", False))
        self._executor = None
        self.degraded = None

    def _new_vox(self, c):
        import os
        v = c.get("voxlingua") or {}
        kw = dict(supported=self.supported, fold=c.get("vox_fold"), min_supported_mass=c.get("min_supported_mass", 0.2))
        url = v.get("api_url") or os.getenv(v.get("api_url_env", "VOXLINGUA_API_URL"), "")
        key = os.getenv(v.get("api_key_env", "VOXLINGUA_API_KEY"), "")
        backend = v.get("backend", "auto")
        if backend == "remote" or (backend == "auto" and url and key):
            return RemoteVoxLinguaDetector(url, key, timeout=v.get("timeout", 15), connect_timeout=v.get("connect_timeout", 5),
                                           max_concurrent=v.get("max_concurrent", 2), retries=v.get("retries", 1), **kw)
        return VoxLinguaDetector(**kw)

    def warm_up(self):
        """Load VoxLingua now (first load downloads/initialises the model and would otherwise stall the first live chunk)."""
        if self._vox is not None and hasattr(self._vox, "warm_up"):
            self._vox.warm_up()

    def status(self):
        return {"mode": self.mode, "voxlingua_backend": type(self._vox).__name__ if self._vox is not None else None,
                "voxlingua_loaded": bool(getattr(self._vox, "loaded", False)), "degraded": self.degraded}

    def detect(self, audio):
        _check_audio(audio)
        audio = audio[: int(self.cfg["window_seconds"] * SR)]
        if self.mode == "qwen":
            return self._verify(self._qwen.detect(audio), audio)
        if self.mode == "voxlingua":
            return self._vox.detect(audio)
        return self._compare(audio)

    def detect_safe(self, audio, fallback=None):
        try:
            return self.detect(audio)
        except (DetectorError, RuntimeError) as e:
            log.warning("language detection failed: %s", type(e).__name__)
            return LangResult(fallback, fallback, None, "fallback", note=f"detector_failed ({type(e).__name__})")

    def _verify(self, r, audio):
        if not (self.verify_suspect and r.suspect):
            return r
        if self._vox is None:
            self._vox = self._new_vox(self.cfg)
        try:
            v = self._vox.detect(audio)
        except (DetectorError, RuntimeError) as e:
            r.note = f"verification unavailable ({type(e).__name__})"
            return r
        if v.code and (v.code in r.suspect or v.code == r.code):
            return LangResult(v.code, v.raw, v.confidence, "qwen+voxlingua", text=None, note="resolved by voxlingua", other=r)
        return r

    def _compare(self, audio):
        res, errs = {}, {}
        jobs = (("qwen", self._qwen), ("voxlingua", self._vox))
        # Qwen is a network call and VoxLingua runs locally, so run them side by side to keep live latency down.
        futures = {name: self._pool().submit(d.detect, audio) for name, d in jobs}
        for name, fut in futures.items():
            try:
                res[name] = fut.result()
            except (DetectorError, RuntimeError) as e:
                errs[name] = str(e)
        if not res:
            raise DetectorError("both detectors failed: " + "; ".join(errs.values()))
        if len(res) == 1:
            name, only = next(iter(res.items()))
            failed = next(iter(errs))
            only.note = f"{failed} failed, single detector ({name})"
            only.agree = None
            if self.degraded != failed:
                log.warning("language ID degraded: %s failed (%s); using %s alone", failed, errs[failed], name)
            self.degraded = failed
            return only
        self.degraded = None
        return fuse(res["qwen"], res["voxlingua"], self.cfg, self.supported)

    def _pool(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="langid")
        return self._executor

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


def fuse(qwen, vox, cfg, supported=None):
    """Combine the Qwen3-ASR and VoxLingua predictions for one chunk. Neither is trusted blindly.

    VoxLingua supplies a closed-set posterior p over the routable languages; Qwen supplies a label (rarely a score).
    Qwen3-ASR cannot identify Malayalam or Tamil at all and is known to answer Hindi (or Malay) for them, so for those two VoxLingua
    leads; for every other language Qwen's label is a strong second opinion.

      A. Qwen's language is VoxLingua's top choice            -> accept (agree).
      B. VoxLingua's top is a language Qwen cannot identify    -> accept if p >= blind_min_confidence.
      C. Qwen reports a confidence >= override_confidence      -> accept Qwen.
      D. VoxLingua p >= accept_confidence                      -> accept VoxLingua (Qwen disagrees).
      E. Ambiguous: Qwen's language has real mass in p         -> accept Qwen's (flagged weak); else VoxLingua's top if p >= floor
                                                                  (flagged weak). Below the floor the chunk is left unresolved
                                                                  for neighbouring chunks / the live lock to settle.
    "weak" results are still transcribed and carry uncertain=True; nothing is silently dropped for being unsure.
    """
    min_c = float(cfg["min_confidence"])
    over = float(cfg.get("override_confidence", 0.85))
    accept = float(cfg.get("accept_confidence", 0.6))
    blind_min = float(cfg.get("blind_min_confidence", 0.35))
    floor = float(cfg.get("floor_confidence", 0.3))
    qcodes = set(QWEN_LANGUAGES.values())
    ok = lambda c: c is not None and (supported is None or c in supported)

    qc, vc = qwen.code, vox.code
    qs, vs = qwen.confidence, vox.confidence
    probs = dict(vox.probs) if vox.probs else ({vc: (vs if vs is not None else min_c)} if vc else {})
    p = lambda c: probs.get(c, 0.0)
    # Qwen's confusable label (Malay) is evidence for the real candidate, not for itself
    qeff = qc
    if qc is not None and qwen.suspect and vc in qwen.suspect:
        qeff = vc
    detail = {"qwen": {"code": qc, "raw": qwen.raw, "confidence": qs}, "voxlingua": {"code": vc, "raw": vox.raw, "confidence": vs}}
    if vox.detail.get("voxlingua", {}).get("open_set"):
        detail["voxlingua"]["open_set"] = vox.detail["voxlingua"]["open_set"]
    base = dict(qwen_code=qc, detail=detail, text=qwen.text, other=vox, probs=probs)

    def out(code, conf, agree, note, resolved=False, weak=False, suspect=(), raw=None):
        return LangResult(code, raw if raw is not None else (vox.raw if code == vc else qwen.raw), conf, "compare", suspect=suspect,
                          agree=agree, note=note, resolved=resolved, weak=weak, **base)

    if qc is None and vc is None:
        return out(None, None, None, qwen.note or vox.note or "no_language")
    # VoxLingua's answer is a language we cannot route (it only gets here when almost none of its mass is on routable languages)
    if vc is not None and supported is not None and vc not in supported:
        if qeff == vc or (vs is not None and vs >= accept):
            return out(vc, best_of(qs, vs), qeff == vc, f"{vc} is not a routable language", suspect=())
        # weak evidence for an unroutable language (e.g. Hindi heard as Marathi): fall back on Qwen's label, flagged weak
        if ok(qeff):
            return out(qeff, qs, False, f"detectors disagree ({qc} vs {vc}); settled to {qeff}, weak", resolved=True, weak=True)
        return out(None, vs, None, f"no routable language (voxlingua: {vc}, qwen: {qc})")
    if vc is None:
        return out(qc, qs, None, "voxlingua gave no language", suspect=qwen.suspect, weak=qs is None and qc is not None)
    if qeff == vc:                                                                   # A
        return out(vc, best_of(qs, vs), True, "detectors agree")
    if vc not in qcodes:                                                             # B
        if vs is None or vs >= blind_min:
            return out(vc, vs, None, f"qwen3-asr cannot identify {vc}; voxlingua decides", resolved=True, weak=(vs or 0) < min_c)
    elif qs is not None and qs >= over and ok(qeff) and (vs is None or vs < accept):   # C
        return out(qeff, qs, False, f"detectors disagree ({qc} vs {vc}); qwen confident ({qs:.2f})", resolved=True)
    if vs is not None and vs >= accept and ok(vc) and (qs is None or qs < over):      # D
        return out(vc, vs, False, f"detectors disagree ({qc} vs {vc}); voxlingua confident ({vs:.2f})", resolved=True)
    if qs is not None and qs >= over and vs is not None and vs >= accept:            # both confident and different
        first = qwen if cfg.get("primary", "qwen") == "qwen" else vox
        return out(first.code, first.confidence, False, f"detectors disagree ({qc} vs {vc}); both confident", suspect=first.suspect,
                   raw=first.raw)
    if qeff is not None and ok(qeff) and p(qeff) >= 0.15 and qeff in qcodes:         # E: Qwen's language is plausible to VoxLingua
        return out(qeff, p(qeff), False, f"detectors disagree ({qc} vs {vc}); settled to {qeff}, weak", resolved=True, weak=True)
    if vs is not None and vs >= floor and ok(vc):
        return out(vc, vs, False, f"detectors disagree ({qc} vs {vc}); best guess {vc}, weak", resolved=True, weak=True)
    first = qwen if cfg.get("primary", "qwen") == "qwen" else vox
    return out(first.code, first.confidence, False, f"detectors disagree ({qc} vs {vc}); unresolved", suspect=first.suspect, raw=first.raw)


def best_of(*xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None
