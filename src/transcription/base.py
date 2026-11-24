import re
from dataclasses import dataclass

import numpy as np

from ..language.detector import script_mismatch
from ..utils.constants import SR


class UnsupportedLanguage(ValueError):
    pass


class LanguageRejected(ValueError):
    pass


class ModelNotConfigured(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


class ModelBusy(InferenceError):
    pass


class InvalidAudio(ValueError):
    pass


def check_audio(audio):
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or not np.issubdtype(audio.dtype, np.number):
        raise InvalidAudio("audio must be a numeric 1-D array")
    if audio.size == 0:
        raise InvalidAudio("audio is empty")
    if not np.all(np.isfinite(audio)):
        raise InvalidAudio("audio contains NaN or infinite values")
    return audio.astype("float32", copy=False)


@dataclass
class AsrResult:
    text: str
    language: str | None
    raw_language: str | None
    empty: bool = False
    flags: tuple = ()
    raw_text: str | None = None
    model: str | None = None
    seconds: float = 0.0
    suspicious: bool = False
    language_confidence: float | None = None
    mode: str | None = None


GUARD = {
    "silence_dbfs": -55.0,
    "max_words_per_second": 8.0,
    "max_chars_per_second": 15.0,
    "min_repeats": 3,
    "min_repeated_tokens": 6,
    "min_repeated_chars": 12,
    "min_seconds": 0.5,
    "max_char_run": 20,
    "min_letters_for_script_check": 3,
    "known_phrases": (
        "thanks for watching", "thank you for watching", "please subscribe", "like and subscribe",
        "subscribe to my channel", "subtitles by the amara.org community", "transcribed by", "www", "music", "applause",
    ),
}

DROP_FLAGS = ("text_in_silence", "known_hallucination", "symbols_only", "invalid_output")

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�​-‍﻿]")
_TAG_ONLY = re.compile(r"^\s*[\[\(\{<][^\]\)\}>]*[\]\)\}>]\s*$|^\s*[♪♫\s]+$")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def configure_guard(overrides):
    for k, v in (overrides or {}).items():
        if k in GUARD:
            GUARD[k] = tuple(v) if isinstance(GUARD[k], tuple) else v


def _tokens(text):
    spaced = " " in text.strip()
    return (text.split() if spaced else list(text.strip())), spaced


def _longest_repeat(tokens, max_n=6):
    best = (1, 1)
    for n in range(1, max_n + 1):
        i = 0
        while i + n <= len(tokens):
            reps = 1
            while tokens[i + reps * n: i + (reps + 1) * n] == tokens[i: i + n]:
                reps += 1
            if reps * n > best[0] * best[1] and reps > 1:
                best = (reps, n)
            i += max(1, reps * n if reps > 1 else 1)
    return best


def collapse_repeats(text, max_n=6):
    toks, spaced = _tokens(text)
    need = GUARD["min_repeats"]
    out, i = [], 0
    while i < len(toks):
        done = False
        for n in range(1, max_n + 1):
            reps = 1
            while toks[i + reps * n: i + (reps + 1) * n] == toks[i: i + n] and i + (reps + 1) * n <= len(toks):
                reps += 1
            floor = GUARD["min_repeated_tokens"] if spaced else GUARD["min_repeated_chars"]
            if reps >= need and reps * n >= floor:
                out += toks[i: i + n]
                i += reps * n
                done = True
                break
        if not done:
            out.append(toks[i])
            i += 1
    return (" " if spaced else "").join(out)


def _normalized(text):
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def text_flags(text, audio=None, language=None):
    if not isinstance(text, str) or not text.strip():
        return ()
    flags = []
    secs = len(audio) / SR if audio is not None and len(audio) else None
    if audio is not None and len(audio):
        rms = float(np.sqrt(np.mean(np.asarray(audio, dtype="float64") ** 2)))
        if 20 * np.log10(max(rms, 1e-9)) < GUARD["silence_dbfs"]:
            flags.append("text_in_silence")
    toks, spaced = _tokens(text)
    reps, n = _longest_repeat(toks)
    need = GUARD["min_repeated_tokens"] if spaced else GUARD["min_repeated_chars"]
    if reps >= GUARD["min_repeats"] and reps * n >= need:
        flags.append("repeated_phrase")
    if secs and secs >= GUARD["min_seconds"]:
        limit = GUARD["max_words_per_second"] if spaced else GUARD["max_chars_per_second"]
        if len(toks) / secs > limit:
            flags.append("implausibly_dense")
    if not any(ch.isalnum() for ch in text):
        flags.append("symbols_only")
    else:
        norm = _normalized(text)
        if _TAG_ONLY.match(text) or norm in GUARD["known_phrases"]:
            flags.append("known_hallucination")
        if re.search(r"(\w)\1{%d,}" % (GUARD["max_char_run"] - 1), text):
            flags.append("character_loop")
    if language:
        why = script_mismatch(text, language)
        if why:
            flags.append("unexpected_script")
    return tuple(flags)


@dataclass
class OutputCheck:
    text: str
    flags: tuple
    drop: bool
    reason: str | None = None


def inspect_output(text, audio=None, language=None):
    if not isinstance(text, str):
        return OutputCheck("", ("invalid_output",), True, "model returned a non-text result")
    flags = []
    clean = _CONTROL.sub("", text)
    if clean != text:
        flags.append("invalid_characters")
    clean = " ".join(clean.split()) if "\n" in clean or "\t" in clean else clean.strip()
    if not clean:
        return OutputCheck("", tuple(flags), False, None)
    found = list(text_flags(clean, audio, language))
    if "repeated_phrase" in found:
        collapsed = collapse_repeats(clean)
        if collapsed and collapsed != clean:
            clean = collapsed
            found.append("repeats_collapsed")
    flags += found
    bad = [f for f in flags if f in DROP_FLAGS]
    if bad:
        return OutputCheck("", tuple(flags), True, bad[0])
    return OutputCheck(clean, tuple(flags), False, None)


def finish_result(raw_text, language, raw_language, audio=None, expected_language=None, model=None, seconds=0.0):
    check = inspect_output(raw_text, audio, expected_language or language)
    text = check.text
    flags = check.flags
    if check.drop:
        flags = tuple(flags) + ("dropped_suspicious_output",)
    suspicious = bool(check.drop or any(f in ("repeated_phrase", "implausibly_dense", "unexpected_script") for f in flags))
    return AsrResult(text, language, raw_language, empty=not text, flags=tuple(flags),
                     raw_text=raw_text if isinstance(raw_text, str) else None, model=model, seconds=seconds,
                     suspicious=suspicious)


class ASRModel:
    name = "base"
    languages = None
    accepts_window_args = False

    def load(self):
        pass

    def unload(self):
        pass

    def close(self):
        self.unload()

    @property
    def available(self):
        return True

    def supports(self, code):
        return code is None or self.languages is None or code in self.languages

    def transcribe(self, audio, language=None):
        raise NotImplementedError

    def health(self):
        return {"model": self.name, "available": self.available}

    def info(self):
        return {"model": self.name}


class FakeASR(ASRModel):
    name = "fake"
    accepts_window_args = True

    def __init__(self, languages=None, text=None):
        self.languages, self.text, self.calls = languages, text, []

    def transcribe(self, audio, language=None, interim=False, context=None):
        audio = check_audio(audio)
        if not self.supports(language):
            raise UnsupportedLanguage(language)
        self.calls.append((len(audio), language))
        t = self.text or f"[{len(audio) / SR:.1f}s]"
        return AsrResult(t, language, language)
