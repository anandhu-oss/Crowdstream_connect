"""Language detection with a selectable backend: qwen | voxlingua | compare.
Both return ISO codes. Qwen3-ASR names the language while transcribing, so the Qwen detector runs the shared ASR model
ONCE and keeps the transcript in the result; the pipeline reuses it instead of transcribing again.
Detection is per segment: a meeting can mix languages (Malayalam-English, Hindi-English), so one window is never
taken as the language of the whole recording."""
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
import yaml

from .constants import CONFUSABLE, SR

MODELS_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
MIN_DETECT_SECONDS = 0.5


class ConfigError(ValueError):
    """models.yaml has an invalid or inconsistent value."""


class DetectorError(RuntimeError):
    """A detector model could not be loaded or run."""


def load_models_config(path=MODELS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class LangResult:
    code: str | None                  # ISO code, or None when the label is not a known language
    raw: str | None                   # exactly what the backend said
    confidence: float | None          # 0..1, None when the backend gives no score
    backend: str
    text: str | None = None           # transcript from the same inference (qwen only), for reuse
    suspect: tuple = ()               # languages the label may really be (see CONFUSABLE)
    other: "LangResult | None" = None # compare mode: the second backend's answer
    agree: bool | None = None         # compare mode: None when either side has no code
    note: str | None = None


def uncertain_reason(r: LangResult, min_confidence: float) -> str | None:
    """Why this result should not be trusted, or None."""
    if r.code is None:
        return r.note or f"no_language ({r.raw!r})"
    if r.suspect:
        return f"label {r.raw!r} may really be {'/'.join(r.suspect)}"
    if r.confidence is not None and r.confidence < min_confidence:
        return f"low_confidence ({r.confidence:.2f})"
    if r.agree is False:
        return f"detectors disagree ({r.code} vs {r.other.code if r.other else None})"
    return None


def _check_audio(audio):
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
        raise ValueError("audio must be a non-empty 1-D array")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or infinite values")


class QwenDetector:
    """Uses the SHARED Qwen ASR adapter (anything with transcribe(audio, None) -> result with text/language/raw_language),
    so there is one copy of the model in GPU memory and no second inference for the transcript."""
    name = "qwen"

    def __init__(self, asr):
        self.asr = asr

    def detect(self, audio: np.ndarray) -> LangResult:
        _check_audio(audio)
        r = self.asr.transcribe(audio, None)
        code = r.language
        return LangResult(code, r.raw_language, None, self.name, text=r.text, suspect=CONFUSABLE.get(code, ()))


class VoxLinguaDetector:
    """SpeechBrain VoxLingua107 ECAPA (107 languages, includes Tamil and Malayalam)."""
    name = "voxlingua"

    def __init__(self, savedir="pretrained/voxlingua107"):
        self.savedir, self._m = savedir, None

    def _load(self):
        if self._m is None:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                self._m = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=self.savedir)
            except Exception as e:
                raise DetectorError(f"VoxLingua could not be loaded ({type(e).__name__}); pip install speechbrain torch") from e
        return self._m

    def detect(self, audio: np.ndarray) -> LangResult:
        _check_audio(audio)
        if len(audio) < MIN_DETECT_SECONDS * SR:
            return LangResult(None, None, None, self.name, note="too_short")
        try:
            import torch
            _, score, _, label = self._load().classify_batch(torch.from_numpy(audio).unsqueeze(0))
            raw, conf = label[0], parse_confidence(float(score.reshape(-1)[0]))
        except DetectorError:
            raise
        except Exception as e:
            raise DetectorError(f"VoxLingua failed ({type(e).__name__})") from e
        return LangResult(parse_voxlingua(raw), raw if isinstance(raw, str) else None, conf, self.name)


def parse_voxlingua(label) -> str | None:
    """'ml: Malayalam' -> 'ml'. Anything that does not look like a language code gives None."""
    if not isinstance(label, str) or ":" not in label:
        return None
    code = label.split(":")[0].strip().lower()
    return code if 2 <= len(code) <= 3 and code.isalpha() else None


def parse_confidence(v: float) -> float | None:
    """Accept a probability (0..1) or a log-probability (<= 0); anything else is treated as no score."""
    if not isfinite(v):
        return None
    if v <= 0:
        v = float(np.exp(v))
    return v if 0.0 <= v <= 1.0 + 1e-6 else None


class LanguageDetector:
    def __init__(self, cfg: dict | None = None, asr=None, qwen=None, voxlingua=None):
        c = (cfg or load_models_config())["language"]
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
        self._vox = voxlingua or (VoxLinguaDetector() if self.mode in ("voxlingua", "compare") else None)

    def detect(self, audio: np.ndarray) -> LangResult:
        _check_audio(audio)
        audio = audio[: int(self.cfg["window_seconds"] * SR)]
        if self.mode == "qwen":
            return self._qwen.detect(audio)
        if self.mode == "voxlingua":
            return self._vox.detect(audio)
        return self._compare(audio)

    def _compare(self, audio) -> LangResult:
        res, errs = {}, {}
        for name, d in (("qwen", self._qwen), ("voxlingua", self._vox)):
            try:
                res[name] = d.detect(audio)
            except (DetectorError, RuntimeError) as e:
                errs[name] = str(e)
        if not res:
            raise DetectorError("both detectors failed: " + "; ".join(errs.values()))
        if len(res) == 1:                                       # one backend failed: use the other, marked
            only = next(iter(res.values())); only.note = f"{[n for n in errs][0]} failed, single detector"
            return only
        first = res[self.cfg["primary"]]
        second = res["voxlingua" if self.cfg["primary"] == "qwen" else "qwen"]
        agree = None if first.code is None or second.code is None else first.code == second.code
        return LangResult(first.code, first.raw, first.confidence, "compare", text=first.text, suspect=first.suspect,
                          other=second, agree=agree, note=first.note)