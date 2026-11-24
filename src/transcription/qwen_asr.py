import hashlib
import io
import os
import threading
import time
import wave
from collections import OrderedDict

import numpy as np
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

from ..audio.vad import quietest_point
from ..language.detector import normalize_language_code
from ..utils.constants import CODE_TO_QWEN, QWEN_LANGUAGES, SR
from ..utils.logging import get_logger
from .base import (ASRModel, InferenceError, ModelBusy, ModelLoadError, UnsupportedLanguage, check_audio, finish_result)

load_dotenv()

log = get_logger()

RETRY_STATUS = (429, 502, 503, 504)


def to_wav_bytes(audio, sr=SR):
    pcm = np.round(np.clip(np.asarray(audio, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class RemoteClient:
    def __init__(self, api_url, api_key, timeout=120, connect_timeout=5, interim_timeout=20, max_concurrent=2, retries=2,
                 backoff_s=0.4, queue_timeout_s=60, session=None):
        self.api_url = (api_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.interim_timeout = float(interim_timeout)
        self.retries = max(0, int(retries))
        self.backoff_s = float(backoff_s)
        self.queue_timeout_s = float(queue_timeout_s)
        self.max_concurrent = max(1, int(max_concurrent))
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self._lock = threading.Lock()
        self._session = session
        self.latency = None
        self.requests = 0
        self.failures = 0
        self.busy_rejections = 0

    @property
    def configured(self):
        return bool(self.api_url and self.api_key)

    def _http(self):
        if self._session is None:
            s = requests.Session()
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=self.max_concurrent * 2, max_retries=0)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            self._session = s
        return self._session

    def _note(self, seconds, ok):
        with self._lock:
            self.requests += 1
            if ok:
                self.latency = seconds if self.latency is None else 0.7 * self.latency + 0.3 * seconds
            else:
                self.failures += 1

    def post_audio(self, path, wav, data, interim=False):
        if not self.configured:
            raise ModelLoadError("remote API URL or key is not configured")
        if interim:
            if not self._sem.acquire(blocking=False):
                with self._lock:
                    self.busy_rejections += 1
                raise ModelBusy("remote model is busy")
        elif not self._sem.acquire(timeout=self.queue_timeout_s):
            with self._lock:
                self.busy_rejections += 1
            raise ModelBusy("timed out waiting for a free remote model slot")
        try:
            return self._post(path, wav, data, interim)
        finally:
            self._sem.release()

    def _post(self, path, wav, data, interim):
        timeout = self.interim_timeout if interim else self.timeout
        attempts = 1 if interim else self.retries + 1
        failure = None
        for attempt in range(attempts):
            t0 = time.perf_counter()
            delay = self.backoff_s * (2 ** attempt)
            try:
                r = self._http().post(f"{self.api_url}{path}", headers={"Authorization": f"Bearer {self.api_key}"}, data=data,
                                      files={"file": ("audio.wav", io.BytesIO(wav), "audio/wav")},
                                      timeout=(self.connect_timeout, timeout))
            except (requests.ConnectionError, requests.Timeout) as e:
                self._note(time.perf_counter() - t0, False)
                failure = InferenceError(f"remote request failed: {type(e).__name__}")
            except requests.RequestException as e:
                self._note(time.perf_counter() - t0, False)
                raise InferenceError(f"remote request failed: {type(e).__name__}") from e
            else:
                if r.status_code in RETRY_STATUS:
                    self._note(time.perf_counter() - t0, False)
                    failure = InferenceError(f"remote model returned HTTP {r.status_code}")
                    try:
                        delay = max(delay, min(5.0, float(r.headers.get("Retry-After", 0))))
                    except ValueError:
                        pass
                elif r.status_code in (401, 403):
                    self._note(time.perf_counter() - t0, False)
                    raise ModelLoadError(f"remote API rejected the API key (HTTP {r.status_code})")
                elif r.status_code >= 400:
                    self._note(time.perf_counter() - t0, False)
                    raise InferenceError(f"remote model returned HTTP {r.status_code}")
                else:
                    try:
                        payload = r.json()
                    except ValueError as e:
                        self._note(time.perf_counter() - t0, False)
                        raise InferenceError("remote model returned invalid JSON") from e
                    if not isinstance(payload, dict):
                        self._note(time.perf_counter() - t0, False)
                        raise InferenceError("remote model returned an unexpected response")
                    self._note(time.perf_counter() - t0, True)
                    return payload
            if attempt + 1 < attempts:
                time.sleep(delay)
        raise failure

    def ping(self, timeout=3.0):
        if not self.api_url:
            return {"ok": False, "reason": "not configured"}
        try:
            r = self._http().get(f"{self.api_url}/health", timeout=timeout)
            return {"ok": r.ok, "status": r.status_code}
        except requests.RequestException as e:
            return {"ok": False, "reason": type(e).__name__}

    def stats(self):
        with self._lock:
            return {"requests": self.requests, "failures": self.failures, "busy_rejections": self.busy_rejections,
                    "latency_s": None if self.latency is None else round(self.latency, 3),
                    "max_concurrent": self.max_concurrent}

    def close(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None


def read_confidence(payload):
    """Optional language-ID score from the remote Qwen API; None when the server does not report one."""
    for key in ("language_confidence", "detected_language_confidence", "lang_confidence"):
        v = payload.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
            return float(v)
    return None


def read_text(payload):
    for key in ("text", "transcription", "transcript"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()
        if value is None and key in payload:
            return ""
    nested = payload.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("text"), str):
        return nested["text"].strip()
    raise InferenceError("remote model response has no text field")


class QwenASR(ASRModel):
    name = "qwen3_asr"
    languages = set(QWEN_LANGUAGES.values())
    accepts_window_args = True

    def __init__(self, checkpoint="Qwen/Qwen3-ASR-1.7B", model=None, max_new_tokens=512, api_url=None, api_key=None,
                 timeout=120, connect_timeout=5, interim_timeout=20, max_concurrent=2, retries=2, backoff_s=0.4,
                 queue_timeout_s=60, min_audio_seconds=0.5, max_audio_seconds=30.0, cache_size=64, send_context=False,
                 session=None):
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.device = "remote GPU"
        self.client = RemoteClient(api_url or os.getenv("QWEN_API_URL", ""), api_key or os.getenv("QWEN_API_KEY", ""), timeout,
                                   connect_timeout, interim_timeout, max_concurrent, retries, backoff_s, queue_timeout_s, session)
        self.min_samples = int(float(min_audio_seconds) * SR)
        self.max_audio_s = float(max_audio_seconds)
        self.send_context = bool(send_context)
        self._cache = OrderedDict()
        self._cache_size = int(cache_size)
        self._cache_lock = threading.Lock()
        self.cache_hits = 0

    @classmethod
    def from_config(cls, spec):
        keys = ("timeout", "connect_timeout", "interim_timeout", "max_concurrent", "retries", "backoff_s", "queue_timeout_s",
                "min_audio_seconds", "max_audio_seconds", "cache_size", "send_context", "max_new_tokens")
        extra = {k: spec[k] for k in keys if k in spec}
        env_url, env_key = spec.get("api_url_env", "QWEN_API_URL"), spec.get("api_key_env", "QWEN_API_KEY")
        return cls(spec.get("checkpoint") or "Qwen/Qwen3-ASR-1.7B", api_url=spec.get("api_url") or os.getenv(env_url, ""),
                   api_key=os.getenv(env_key, ""), **extra)

    @property
    def api_url(self):
        return self.client.api_url

    @property
    def api_key(self):
        return self.client.api_key

    @property
    def available(self):
        return self.client.configured

    @property
    def latency(self):
        return self.client.latency

    def load(self):
        if not self.client.api_url:
            raise ModelLoadError("QWEN_API_URL is not configured")
        if not self.client.api_key:
            raise ModelLoadError("QWEN_API_KEY is not configured")

    def unload(self):
        self.model = None
        with self._cache_lock:
            self._cache.clear()

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
            cut = int(quietest_point(audio, max(start, hi - 3 * SR) / SR, hi / SR) * SR)
            cut = min(max(cut, start + SR), hi)
            out.append(audio[start:cut])
            start = cut
        out.append(audio[start:])
        return out

    def _cache_get(self, key):
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return self._cache[key]
        return None

    def _cache_put(self, key, value):
        if self._cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = value
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _request(self, piece, language, interim, context):
        if len(piece) < self.min_samples:
            piece = np.concatenate([piece, np.zeros(self.min_samples - len(piece), dtype="float32")])
        wav = to_wav_bytes(piece)
        use_context = context if (self.send_context and context) else None
        key = (hashlib.blake2b(wav, digest_size=16).hexdigest(), language, use_context)
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        data = {"source_language": language or "auto"}
        if use_context:
            data["context"] = use_context
        payload = self.client.post_audio("/transcribe", wav, data, interim=interim)
        text = read_text(payload)
        raw = payload.get("detected_language", payload.get("language"))
        raw = raw.strip() if isinstance(raw, str) and raw.strip() else None
        value = (text, raw, read_confidence(payload))
        self._cache_put(key, value)
        return value

    def transcribe(self, audio, language=None, interim=False, context=None):
        audio = check_audio(audio)
        if language is not None and language not in CODE_TO_QWEN:
            raise UnsupportedLanguage(language)
        if not self.client.configured:
            self.load()
        t0 = time.perf_counter()
        texts, raw_language, confs = [], None, []
        for piece in self._pieces(audio):
            text, raw, conf = self._request(piece, language, interim, context)
            if text:
                texts.append(text)
            raw_language = raw_language or raw
            if conf is not None:
                confs.append(conf)
        code = normalize_language_code(raw_language) if raw_language else None
        if code is not None and code not in self.languages:
            code = None
        joined = " ".join(texts)
        res = finish_result(joined, code or language, raw_language, audio, expected_language=language, model=self.name,
                            seconds=time.perf_counter() - t0)
        res.language_confidence = min(confs) if confs and code else None
        return res

    def health(self):
        out = {"model": self.name, "available": self.available, "remote": True}
        out.update(self.client.ping())
        return out

    def info(self):
        return {"model": self.name, "checkpoint": self.checkpoint, "device": self.device, "api_url": self.client.api_url,
                "remote": True, "cache_hits": self.cache_hits, **self.client.stats()}
