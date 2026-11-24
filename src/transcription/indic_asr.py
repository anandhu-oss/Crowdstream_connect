
import importlib.util
import os
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

from ..audio.vad import quietest_point
from ..utils.constants import SR
from ..utils.logging import get_logger
from .base import (
    ASRModel,
    InferenceError,
    ModelBusy,
    ModelLoadError,
    ModelNotConfigured,
    UnsupportedLanguage,
    check_audio,
    finish_result,
)
from .qwen_asr import RemoteClient, read_text, to_wav_bytes

log = get_logger()

MODES = ("native", "mixed", "romanized")

# Languages listed on the Indic-Transcribe-Flex model card (27). `lang` is the argument the model's own
# `asr(path, lang=..., mode=...)` call takes; the model refuses audio longer than 45 s, so pieces stay <= 30 s.
FLEX_LANGUAGES = frozenset(
    "as bn brx doi gu hi kn ks kok mai ml mni mr ne or pa sa sat sd ta te ur en bho hne bgc bhb".split()
)
FLEX_MAX_SECONDS = 45.0


class IndicASR(ASRModel):
    name = "indic"
    accepts_window_args = True

    def __init__(
        self,
        checkpoint=None,
        languages=(),
        backend="auto",
        api_url=None,
        api_key=None,
        mode="mixed",
        device=None,
        hf_token_env="HF_TOKEN",
        timeout=120,
        connect_timeout=5,
        interim_timeout=20,
        max_concurrent=1,
        retries=2,
        backoff_s=0.4,
        queue_timeout_s=60,
        max_audio_seconds=30.0,
        min_audio_seconds=0.5,
        session=None,
    ):
        if backend not in ("auto", "remote", "local"):
            raise ValueError("indic backend must be auto, remote or local")
        if mode not in MODES:
            raise ValueError(f"indic mode must be one of {MODES}")

        self.languages = set(languages or ())
        unknown = sorted(self.languages - FLEX_LANGUAGES)
        if unknown:
            raise ValueError(f"Indic-Transcribe-Flex does not support: {', '.join(unknown)}")
        if float(max_audio_seconds) > FLEX_MAX_SECONDS:
            raise ValueError(f"indic max_audio_seconds must be <= {FLEX_MAX_SECONDS:.0f} (model limit)")

        self.checkpoint = checkpoint
        self.mode = mode
        self.device = device
        self.hf_token_env = hf_token_env
        self.max_audio_s = float(max_audio_seconds)
        self.min_samples = int(float(min_audio_seconds) * SR)

        self.client = RemoteClient(
            api_url or "",
            api_key or "",
            timeout,
            connect_timeout,
            interim_timeout,
            max_concurrent,
            retries,
            backoff_s,
            queue_timeout_s,
            session,
        )

        if backend == "auto":
            backend = "remote" if self.client.api_url else "local"

        self.backend = backend
        self._model = None
        self._load_lock = threading.Lock()
        self._gpu = threading.BoundedSemaphore(max(1, int(max_concurrent)))
        self._queue_timeout = float(queue_timeout_s)
        self.latency = None

    @classmethod
    def from_config(cls, spec):
        keys = (
            "mode",
            "device",
            "hf_token_env",
            "timeout",
            "connect_timeout",
            "interim_timeout",
            "max_concurrent",
            "retries",
            "backoff_s",
            "queue_timeout_s",
            "max_audio_seconds",
            "min_audio_seconds",
        )

        extra = {k: spec[k] for k in keys if k in spec}

        url = spec.get("api_url") or os.getenv(
            spec.get("api_url_env", "INDIC_API_URL"), ""
        )
        key = os.getenv(spec.get("api_key_env", "INDIC_API_KEY"), "")

        return cls(
            spec.get("checkpoint"),
            spec.get("languages") or (),
            spec.get("backend", "auto"),
            url,
            key,
            **extra,
        )

    @property
    def available(self):
        if self.backend == "remote":
            return self.client.configured

        return (
            bool(self.checkpoint)
            and importlib.util.find_spec("torch") is not None
        )

    def supports(self, code):
        return code is not None and code in self.languages

    def _token(self):
        return (
            os.getenv(self.hf_token_env)
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or None
        )

    def load(self):
        if self.backend == "remote":
            if not self.client.api_url:
                raise ModelNotConfigured(
                    "indic: set INDIC_API_URL (or asr.models.indic.api_url) "
                    "in configs/models.yaml"
                )

            if not self.client.api_key:
                raise ModelLoadError("INDIC_API_KEY is not configured")

            return

        if not self.checkpoint:
            raise ModelNotConfigured(
                "indic: set asr.models.indic.checkpoint in configs/models.yaml"
            )

        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            try:
                from huggingface_hub import snapshot_download

                path = snapshot_download(
                    self.checkpoint,
                    token=self._token(),
                )

                if path not in sys.path:
                    sys.path.insert(0, path)

                from indic_transcribe import IndicTranscribe

                self._model = IndicTranscribe.from_pretrained(path)

            except Exception as e:
                raise ModelLoadError(
                    f"Indic-Transcribe could not be loaded "
                    f"({type(e).__name__}): {e}"
                ) from e

            log.info("indic model loaded from %s", self.checkpoint)

    def unload(self):
        self._model = None

    def close(self):
        self.unload()
        self.client.close()

    def _pieces(self, audio):
        mx = int(self.max_audio_s * SR)

        if len(audio) <= mx:
            return [audio]

        out, start = [], 0

        while len(audio) - start > mx:
            hi = start + mx

            cut = int(
                quietest_point(
                    audio,
                    max(start, hi - 3 * SR) / SR,
                    hi / SR,
                ) * SR
            )

            cut = min(max(cut, start + SR), hi)

            out.append(audio[start:cut])
            start = cut

        out.append(audio[start:])

        return out

    def _pad(self, piece):
        if len(piece) < self.min_samples:
            return np.concatenate(
                [
                    piece,
                    np.zeros(
                        self.min_samples - len(piece),
                        dtype="float32",
                    ),
                ]
            )

        return piece

    def _run_remote(self, piece, language, interim, mode):
        # Request contract for POST /transcribe-indic: multipart `file` + form fields `language` and `mode`
        # (no `decoder`), Bearer auth. Response: {"text": ..., "language": ..., "mode": ...}.
        payload = self.client.post_audio(
            "/transcribe-indic",
            to_wav_bytes(piece),
            {"language": language, "mode": mode},
            interim=interim,
        )

        text = read_text(payload)
        echoed_lang = payload.get("language")
        echoed_mode = payload.get("mode")
        flags = []

        if isinstance(echoed_lang, str) and echoed_lang.strip().lower() not in ("", language):
            flags.append("indic_language_mismatch")

        if isinstance(echoed_mode, str) and echoed_mode.strip().lower() not in ("", mode):
            flags.append("indic_mode_mismatch")

        return text, flags

    def _run_local(self, piece, language, interim, mode):
        self.load()

        if interim:
            if not self._gpu.acquire(blocking=False):
                raise ModelBusy("indic model is busy")

        elif not self._gpu.acquire(timeout=self._queue_timeout):
            raise ModelBusy("timed out waiting for the indic model")

        path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as tmp:
                path = tmp.name

            sf.write(path, piece, SR, subtype="PCM_16")

            out = self._model(
                path,
                lang=language,
                mode=mode,
            )

        except (ModelBusy, ModelLoadError):
            raise

        except Exception as e:
            raise InferenceError(
                f"indic inference failed ({type(e).__name__})"
            ) from e

        finally:
            self._gpu.release()

            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

        if isinstance(out, dict):
            out = out.get("text", "")

        if isinstance(out, (list, tuple)):
            out = " ".join(str(x) for x in out)

        if not isinstance(out, str):
            raise InferenceError(
                "indic model returned a non-text result"
            )

        return out.strip(), []

    def transcribe(self, audio, language=None, interim=False, context=None, mode=None):
        audio = check_audio(audio)

        if not self.supports(language):
            raise UnsupportedLanguage(language)

        mode = mode or self.mode

        if mode not in MODES:
            raise ValueError(f"indic mode must be one of {MODES}")

        if self.backend == "remote":
            self.load()

        t0 = time.perf_counter()
        texts, extra = [], []

        for piece in self._pieces(audio):
            piece = self._pad(piece)

            text, flags = (
                self._run_remote(piece, language, interim, mode)
                if self.backend == "remote"
                else self._run_local(piece, language, interim, mode)
            )
            extra += [f for f in flags if f not in extra]

            if text:
                texts.append(text)

        seconds = time.perf_counter() - t0

        self.latency = (
            seconds
            if self.latency is None
            else 0.7 * self.latency + 0.3 * seconds
        )

        res = finish_result(
            " ".join(texts),
            language,
            language,
            audio,
            expected_language=language,
            model=self.name,
            seconds=seconds,
        )

        res.mode = mode
        res.flags = tuple(res.flags) + tuple(extra)

        return res

    def health(self):
        out = {
            "model": self.name,
            "available": self.available,
            "backend": self.backend,
            "languages": sorted(self.languages),
        }

        if self.backend == "remote":
            out.update(self.client.ping())
        else:
            out["loaded"] = self._model is not None

        return out

    def info(self):
        return {
            "model": self.name,
            "checkpoint": self.checkpoint,
            "backend": self.backend,
            "mode": self.mode,
            "languages": sorted(self.languages),
            "available": self.available,
            **self.client.stats(),
        }
