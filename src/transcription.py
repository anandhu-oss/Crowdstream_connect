"""ASR adapters behind one interface, the model set, and the language router (roadmap steps 3 and 4).
transcribe(audio, language) -> AsrResult, always the same shape, even when the model cannot name the language."""
import logging
from dataclasses import dataclass

import numpy as np

from .constants import CODE_TO_QWEN, QWEN_LANGUAGES, SR
from .language import ConfigError, LangResult, load_models_config, uncertain_reason

log = logging.getLogger("connect_ai")


class UnsupportedLanguage(ValueError):
    """The model cannot transcribe this language. Never guessed around."""


class LanguageRejected(ValueError):
    """The language is unknown or uncertain and the config says to reject (on_uncertain / on_unsupported)."""


class ModelNotConfigured(RuntimeError):
    """A model is listed in models.yaml but has no checkpoint yet."""


class ModelLoadError(RuntimeError):
    """A model could not be loaded (missing package, no memory, bad checkpoint)."""


class InferenceError(RuntimeError):
    """A model failed on one piece of audio. The message never contains audio or text."""


class InvalidAudio(ValueError):
    """Empty, non-numeric, wrong-shaped or non-finite audio."""


def check_audio(audio) -> np.ndarray:
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or not np.issubdtype(audio.dtype, np.number):
        raise InvalidAudio("audio must be a numeric 1-D array")
    if audio.size == 0:
        raise InvalidAudio("audio is empty")
    if not np.all(np.isfinite(audio)):
        raise InvalidAudio("audio contains NaN or infinite values")
    return audio.astype("float32", copy=False)


@dataclass
class AsrResult:
    text: str                   # "" when the model returned nothing
    language: str | None        # ISO code; None when the model's label is not a known language
    raw_language: str | None    # what the model reported, unchanged
    empty: bool = False         # True when no text came back


class ASRModel:
    name = "base"
    languages: set[str] | None = None        # None = no restriction known

    def load(self): ...
    def unload(self): ...

    @property
    def available(self) -> bool:
        return True

    def supports(self, code: str | None) -> bool:
        return code is None or self.languages is None or code in self.languages

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> AsrResult:
        raise NotImplementedError

    def info(self) -> dict:
        return {"model": self.name}


class QwenASR(ASRModel):
    """Qwen3-ASR 1.7B (Apache 2.0). 30 languages (src/constants.py); no Tamil or Malayalam. Streaming needs vLLM, not used here."""
    name = "qwen3_asr"
    languages = set(QWEN_LANGUAGES.values())

    def __init__(self, checkpoint="Qwen/Qwen3-ASR-1.7B", model=None, max_new_tokens=512):
        self.checkpoint, self.max_new_tokens, self.model, self.device = checkpoint, max_new_tokens, model, None

    def load(self):
        if self.model is not None:
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
            if torch.cuda.is_available():
                try:
                    cap = torch.cuda.get_device_capability()[0]
                except Exception:
                    cap = 0
                dtype, self.device = (torch.bfloat16 if cap >= 8 else torch.float16), "cuda:0"   # T4 has no bf16
            else:
                dtype, self.device = torch.float32, "cpu"
                log.warning("no CUDA device: Qwen3-ASR will run on the CPU, very slowly")
            self.model = Qwen3ASRModel.from_pretrained(self.checkpoint, dtype=dtype, device_map=self.device,
                                                       max_inference_batch_size=1, max_new_tokens=self.max_new_tokens)
        except ImportError as e:
            raise ModelLoadError("qwen-asr or torch is not installed") from e
        except Exception as e:
            raise ModelLoadError(f"Qwen3-ASR could not be loaded ({type(e).__name__})") from e

    def unload(self):
        self.model = None
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def transcribe(self, audio, language=None) -> AsrResult:
        audio = check_audio(audio)
        if language is not None and language not in CODE_TO_QWEN:
            raise UnsupportedLanguage(language)
        if self.model is None:
            self.load()
        try:
            resp = self.model.transcribe(audio=[(audio, SR)], language=[CODE_TO_QWEN[language]] if language else None)
        except Exception as e:
            raise InferenceError(f"Qwen3-ASR failed ({type(e).__name__})") from e
        if not resp:
            return AsrResult("", language, None, empty=True)
        r = resp[0]
        text = getattr(r, "text", None)
        text = text.strip() if isinstance(text, str) else ""
        raw = getattr(r, "language", None)
        raw = raw if isinstance(raw, str) and raw else None
        code = QWEN_LANGUAGES.get(raw) if raw else None
        return AsrResult(text, code or language, raw, empty=not text)    # a forced language is kept when the label is unknown

    def info(self):
        out = {"model": self.name, "device": self.device}
        try:
            import torch
            if torch.cuda.is_available():
                out["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / 2**20)
        except Exception:
            pass
        return out


class IndicASR(ASRModel):
    """The Indian-language model. NOT READY: the checkpoint is not chosen (Indic-Transcribe Flex is under test).
    `languages` comes from models.yaml and is a claim to verify, not proven coverage. Once a model is chosen, implement
    load() (gated checkpoint, token), transcribe() (mode, 45 s limit), output normalisation and memory reporting here."""
    name = "indic"

    def __init__(self, checkpoint=None, languages=()):
        self.checkpoint, self.languages = checkpoint, set(languages or ())

    @property
    def available(self) -> bool:
        return bool(self.checkpoint)

    def supports(self, code):
        return code is not None and code in self.languages        # automatic language is not offered by this slot

    def load(self):
        if not self.checkpoint:
            raise ModelNotConfigured("indic: set asr.models.indic.checkpoint in configs/models.yaml")
        raise NotImplementedError("Indian ASR loading is implemented once the model is chosen")

    def transcribe(self, audio, language=None) -> AsrResult:
        check_audio(audio)
        if not self.supports(language):
            raise UnsupportedLanguage(language)
        self.load()


class FakeASR(ASRModel):
    """For tests and for running the pipeline without a GPU."""
    name = "fake"

    def __init__(self, languages=None, text=None):
        self.languages, self.text, self.calls = languages, text, []

    def transcribe(self, audio, language=None) -> AsrResult:
        audio = check_audio(audio)
        if not self.supports(language):
            raise UnsupportedLanguage(language)
        self.calls.append((len(audio), language))
        t = self.text or f"[{len(audio) / SR:.1f}s]"
        return AsrResult(t, language, language)


def _build(key: str, spec: dict) -> ASRModel:
    if key == "qwen3_asr":
        return QwenASR(spec.get("checkpoint") or "Qwen/Qwen3-ASR-1.7B")
    if key == "indic":
        return IndicASR(spec.get("checkpoint"), spec.get("languages") or ())
    raise ConfigError(f"no adapter for model {key!r}")


class ModelSet:
    """Each model is built once, so the detector and the transcriber share one copy in memory."""

    def __init__(self, cfg: dict | None = None, overrides: dict | None = None):
        self.cfg = cfg or load_models_config()
        self._specs = self.cfg["asr"]["models"]
        self._m = dict(overrides or {})            # tests inject fakes here

    def get(self, key: str) -> ASRModel:
        if key not in self._specs and key not in self._m:
            raise ConfigError(f"unknown model {key!r}")
        if key not in self._m:
            self._m[key] = _build(key, self._specs[key])
        return self._m[key]

    def unload_all(self):
        for m in self._m.values():
            m.unload()


def get_model(key: str, cfg: dict | None = None) -> ASRModel:
    return ModelSet(cfg).get(key)


@dataclass
class Route:
    model: str
    language: str | None        # None = let the model detect
    uncertain: bool = False
    reason: str | None = None


class Router:
    """Language -> model, with the configured policy for unsure and unsupported cases. Checked at start-up so a
    wrong config fails with a clear message instead of sending audio to a model that cannot read it."""

    def __init__(self, cfg: dict | None = None, models: ModelSet | None = None):
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

    def _model_for(self, key: str) -> ASRModel:
        m = self.models.get(key)
        if not m.available:
            raise ModelNotConfigured(f"{key} has no checkpoint yet")
        return m

    def route(self, r: LangResult) -> Route:
        reason = uncertain_reason(r, self.l["min_confidence"])
        if reason and (r.code is None or r.suspect):             # the code itself cannot be trusted
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
        if reason and self.l["on_uncertain"] == "reject":        # known code, but low confidence or detectors disagree
            raise LanguageRejected(reason)
        self._model_for(key)
        return Route(key, r.code, bool(reason), reason)