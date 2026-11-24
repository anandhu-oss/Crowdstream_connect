import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

from ..language import load_models_config
from ..utils.logging import get_logger

load_dotenv()

log = get_logger()

_SENTENCE = re.compile(r"(?<=[.!?।؟。！？])\s+")
_CLAUSE = re.compile(r"(?<=[,;:،、，；：—])\s+")
_TARGET = re.compile(r"^[a-z]{2,3}$")
_END = re.compile(r"[.!?।؟。！？]$")


class TranslationError(RuntimeError):
    pass


@dataclass
class TranslationResult:
    text: str
    source: str | None
    target: str
    seconds: float = 0.0
    flags: list = field(default_factory=list)
    cached: bool = False


def split_sentences(text):
    return [s for s in _SENTENCE.split(text.strip()) if s]


def chunk_text(text, max_chars=400):
    out = []
    for sentence in split_sentences(text) or [text.strip()]:
        if len(sentence) <= max_chars:
            out.append(sentence)
            continue
        buf = ""
        for part in _CLAUSE.split(sentence):
            if buf and len(buf) + 1 + len(part) > max_chars:
                out.append(buf)
                buf = ""
            if len(part) > max_chars:
                words, cur = (part.split(" ") if " " in part else list(part)), ""
                glue = " " if " " in part else ""
                for w in words:
                    if cur and len(cur) + len(glue) + len(w) > max_chars:
                        out.append(cur)
                        cur = ""
                    cur = f"{cur}{glue}{w}" if cur else w
                buf = f"{buf} {cur}".strip() if buf else cur
            else:
                buf = f"{buf} {part}".strip() if buf else part
        if buf:
            out.append(buf)
    return [c for c in out if c.strip()]


def check_target(target):
    if not isinstance(target, str) or not _TARGET.match(target):
        raise ValueError(f"target language must be a 2-3 letter code, got {target!r}")
    return target


def check_translation(source_text, out):
    flags = []
    if not isinstance(out, str) or not out.strip():
        return ["empty_translation"]
    if out.strip() == source_text.strip() and len(source_text.split()) > 2:
        flags.append("unchanged")
    ratio = len(out) / max(1, len(source_text))
    if ratio > 4 or ratio < 0.25:
        flags.append("length_mismatch")
    return flags


def translation_enabled(cfg=None):
    value = os.getenv("TRANSLATION_ENABLED")
    if value is not None and value.strip():
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(((cfg or load_models_config()).get("translation") or {}).get("enabled", True))


class Translator:
    name = "base"

    def __init__(self, cache_size=512, max_chars=400):
        self._cache, self._cache_size, self.max_chars = OrderedDict(), cache_size, max_chars
        self._cache_lock = threading.Lock()

    def load(self):
        pass

    def unload(self):
        pass

    def close(self):
        self.unload()

    def _translate_sentence(self, sentence, target):
        raise NotImplementedError

    def _cached(self, key):
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _remember(self, key, value):
        if self._cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = value
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def translate(self, text, source, target):
        check_target(target)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("nothing to translate")
        if source == target:
            return TranslationResult(text, source, target, 0.0, ["same_language"])
        t0 = time.perf_counter()
        parts, all_cached = [], True
        for chunk in chunk_text(text, self.max_chars):
            key = (chunk, target)
            hit = self._cached(key)
            if hit is None:
                all_cached = False
                hit = self._translate_sentence(chunk, target)
                self._remember(key, hit)
            parts.append(hit)
        out = " ".join(parts).strip()
        return TranslationResult(out, source, target, 0.0 if all_cached else time.perf_counter() - t0,
                                 check_translation(text, out), all_cached)

    def translate_interim(self, text, source, target):
        return self.translate(text, source, target)

    def info(self):
        return {"translator": self.name}


class MadladTranslator(Translator):
    name = "madlad400_3b"

    def __init__(self, checkpoint="google/madlad400-3b-mt", dtype="auto", max_new_tokens=256, num_beams=1, cache_size=512,
                 model=None, tokenizer=None, max_chars=400):
        super().__init__(cache_size, max_chars)
        self.checkpoint, self.dtype, self.max_new_tokens, self.num_beams = checkpoint, dtype, max_new_tokens, num_beams
        self.model, self.tokenizer, self.device = model, tokenizer, None
        self._gpu = threading.Lock()

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            if torch.cuda.is_available():
                self.device = "cuda:0"
                try:
                    cap = torch.cuda.get_device_capability()[0]
                except Exception:
                    cap = 0
                dt = {"auto": torch.bfloat16 if cap >= 8 else torch.float16, "bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}.get(self.dtype)
                if dt is None:
                    raise ValueError(f"dtype must be auto, bf16, fp16 or fp32, not {self.dtype!r}")
                if dt == torch.float16:
                    log.warning("translation in fp16: T5-style models can overflow; check the output")
            else:
                self.device, dt = "cpu", torch.float32
                log.warning("no CUDA device: translation will run on the CPU, very slowly")
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint, torch_dtype=dt).to(self.device).eval()
        except ValueError:
            raise
        except ImportError as e:
            raise TranslationError("transformers or torch is not installed") from e
        except Exception as e:
            raise TranslationError(f"translation model could not be loaded ({type(e).__name__})") from e

    def unload(self):
        self.model = self.tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _translate_sentence(self, sentence, target):
        if self.model is None:
            self.load()
        try:
            import torch
            with self._gpu:
                enc = self.tokenizer(f"<2{target}> {sentence}", return_tensors="pt").to(self.device or "cpu")
                with torch.no_grad():
                    out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            raise TranslationError(f"translation failed ({type(e).__name__})") from e

    def info(self):
        out = {"translator": self.name, "device": self.device}
        try:
            import torch
            if torch.cuda.is_available():
                out["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20)
        except Exception:
            pass
        return out


class RemoteMadladTranslator(Translator):
    name = "madlad400_3b_remote"

    def __init__(self, api_url=None, api_key=None, cache_size=512, timeout=60, connect_timeout=5, interim_timeout=15,
                 max_concurrent=2, retries=1, backoff_s=0.4, queue_timeout_s=30, max_chars=400, breaker_failures=4,
                 breaker_cooldown_s=15, session=None):
        super().__init__(cache_size, max_chars)
        self.api_url = (api_url or os.getenv("MADLAD_API_URL") or os.getenv("QWEN_API_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("MADLAD_API_KEY") or os.getenv("QWEN_API_KEY") or ""
        self.timeout, self.connect_timeout, self.interim_timeout = float(timeout), float(connect_timeout), float(interim_timeout)
        self.retries, self.backoff_s, self.queue_timeout_s = max(0, int(retries)), float(backoff_s), float(queue_timeout_s)
        self.max_concurrent = max(1, int(max_concurrent))
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self._session = session
        self._lock = threading.Lock()
        self.breaker_failures, self.breaker_cooldown_s = int(breaker_failures), float(breaker_cooldown_s)
        self._fails = 0
        self._open_until = 0.0
        self._tl = threading.local()
        self.latency = None
        self.requests = 0
        self.failures = 0

    @property
    def available(self):
        return bool(self.api_url and self.api_key)

    def load(self):
        if not self.api_url or not self.api_key:
            raise TranslationError("MADLAD API URL or API key is missing")

    def unload(self):
        pass

    def close(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _http(self):
        if self._session is None:
            s = requests.Session()
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=self.max_concurrent * 2, max_retries=0)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            self._session = s
        return self._session

    def _breaker_open(self):
        with self._lock:
            return time.monotonic() < self._open_until

    def _record(self, ok, seconds=0.0):
        with self._lock:
            self.requests += 1
            if ok:
                self._fails = 0
                self.latency = seconds if self.latency is None else 0.7 * self.latency + 0.3 * seconds
            else:
                self.failures += 1
                self._fails += 1
                if self._fails >= self.breaker_failures:
                    self._open_until = time.monotonic() + self.breaker_cooldown_s
                    self._fails = 0

    def _translate_sentence(self, sentence, target):
        self.load()
        if self._breaker_open():
            raise TranslationError("translation service is cooling down after repeated failures")
        interim = self.interim
        if interim:
            if not self._sem.acquire(blocking=False):
                raise TranslationError("translation service is busy")
        elif not self._sem.acquire(timeout=self.queue_timeout_s):
            raise TranslationError("timed out waiting for the translation service")
        try:
            return self._post(sentence, target, interim)
        finally:
            self._sem.release()

    def _post(self, sentence, target, interim):
        attempts = 1 if interim else self.retries + 1
        timeout = self.interim_timeout if interim else self.timeout
        failure = None
        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                r = self._http().post(f"{self.api_url}/translate", headers={"Authorization": f"Bearer {self.api_key}"},
                                      json={"text": sentence, "target_language": target},
                                      timeout=(self.connect_timeout, timeout))
            except (requests.ConnectionError, requests.Timeout) as e:
                self._record(False)
                failure = TranslationError(f"remote translation failed ({type(e).__name__})")
            except requests.RequestException as e:
                self._record(False)
                raise TranslationError(f"remote translation failed ({type(e).__name__})") from e
            else:
                if r.status_code in (429, 502, 503, 504):
                    self._record(False)
                    failure = TranslationError(f"remote translation returned HTTP {r.status_code}")
                elif r.status_code >= 400:
                    self._record(False)
                    raise TranslationError(f"remote translation returned HTTP {r.status_code}")
                else:
                    try:
                        payload = r.json()
                    except ValueError as e:
                        self._record(False)
                        raise TranslationError("remote translation returned invalid JSON") from e
                    translated = payload.get("translated_text") if isinstance(payload, dict) else None
                    if not isinstance(translated, str) or not translated.strip():
                        self._record(False)
                        raise TranslationError("remote translation returned an empty or invalid result")
                    self._record(True, time.perf_counter() - t0)
                    return translated.strip()
            if attempt + 1 < attempts:
                time.sleep(self.backoff_s * (2 ** attempt))
        raise failure

    @property
    def interim(self):
        return getattr(self._tl, "interim", False)

    def translate_interim(self, text, source, target):
        self._tl.interim = True
        try:
            return self.translate(text, source, target)
        finally:
            self._tl.interim = False

    def info(self):
        return {"translator": self.name, "device": "remote GPU", "checkpoint": "google/madlad400-3b-mt",
                "requests": self.requests, "failures": self.failures,
                "latency_s": None if self.latency is None else round(self.latency, 3), "max_concurrent": self.max_concurrent}


class FakeTranslator(Translator):
    name = "fake"

    def __init__(self, delay_s=0.0, **k):
        super().__init__(**k)
        self.delay_s, self.calls = delay_s, []

    def _translate_sentence(self, sentence, target):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.calls.append((sentence, target))
        return f"[{target}] {sentence}"


def _words(text):
    return text.split() if " " in text.strip() else list(text.strip())


class InterimTranslationPlanner:
    def __init__(self, min_words=5, min_interval_s=1.5, min_growth_words=3, clock=time.monotonic):
        self.min_words, self.min_interval_s, self.min_growth = int(min_words), float(min_interval_s), int(min_growth_words)
        self.clock = clock
        self._state = {}

    def forget(self, key):
        self._state.pop(key, None)

    def reset(self):
        self._state.clear()

    def update(self, key, text):
        st = self._state.setdefault(key, {"prev": [], "sent": 0, "at": -1e9})
        words = _words(text)
        prev = st["prev"]
        st["prev"] = words
        common = 0
        for a, b in zip(prev, words):
            if a != b:
                break
            common += 1
        stable = words[:max(0, common - 1)]
        joined = (" " if " " in text.strip() else "").join(stable)
        sentence_end = None
        for i in range(len(stable) - 1, -1, -1):
            if _END.search(stable[i]):
                sentence_end = i + 1
                break
        if sentence_end:
            stable = stable[:sentence_end]
            joined = (" " if " " in text.strip() else "").join(stable)
        if len(stable) < self.min_words:
            return None
        if len(stable) - st["sent"] < self.min_growth and not sentence_end:
            return None
        if len(stable) <= st["sent"]:
            return None
        now = self.clock()
        if now - st["at"] < self.min_interval_s:
            return None
        st["sent"], st["at"] = len(stable), now
        return joined


def get_translator(cfg=None):
    cfg = cfg or load_models_config()
    t = cfg.get("translation") or {}
    backend = t.get("backend", "remote")
    common = {"cache_size": t.get("cache_size", 512), "max_chars": t.get("max_chars", 400)}
    if backend == "fake":
        return FakeTranslator(**common)
    if backend == "local":
        return MadladTranslator(checkpoint=t.get("checkpoint", "google/madlad400-3b-mt"), dtype=t.get("dtype", "auto"),
                                max_new_tokens=t.get("max_new_tokens", 256), num_beams=t.get("num_beams", 1), **common)
    keys = ("timeout", "connect_timeout", "interim_timeout", "max_concurrent", "retries", "backoff_s", "queue_timeout_s",
            "breaker_failures", "breaker_cooldown_s")
    extra = {k: t[k] for k in keys if k in t}
    tr = RemoteMadladTranslator(api_url=t.get("api_url") or os.getenv(t.get("api_url_env", "MADLAD_API_URL")) or None,
                                **common, **extra)
    return tr
