"""Transcript cleaning, normalisation, validation and chunking.

Design rule: be conservative. We only remove things that are certainly not
speech (ASR special tokens, runaway repetition loops). Ordinary repetitions and
self-corrections stay because they can carry decisions ("no wait, make it Friday").
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .errors import EmptyTranscriptError, InvalidInputError
from .schemas import SummarizeRequest, TranscriptSegment

# --------------------------------------------------------------------------- helpers

# NOTE: ZWJ (U+200D) and ZWNJ (U+200C) are intentionally NOT stripped: they change
# how Malayalam / Hindi / Tamil conjuncts render and are part of the spelling.
_STRIP_CHARS = dict.fromkeys(map(ord, "​⁠﻿‎‏"), None)
_ASR_TOKENS = re.compile(
    r"<\|[^|>]*\|>|<unk>|<s>|</s>|\[(?:BLANK_AUDIO|MUSIC|NOISE)\]|\((?:silence)\)", re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥。？！])\s+")
_TS = re.compile(r"^\s*\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\]?\s*")
_SPEAKER = re.compile(r"^(?P<spk>[^\W_][^:\n]{0,30}?):\s+(?P<text>.+)$")


def estimate_tokens(text: str) -> int:
    """Cheap, tokenizer-free, deliberately conservative token estimate.

    ASCII text averages ~3.5 chars/token; Indic / CJK / Arabic scripts often cost
    about one token per character on Qwen-style tokenizers, so we count them 1:1.
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return int(ascii_chars / 3.5 + (len(text) - ascii_chars)) + 1


def fmt_ts(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def parse_ts(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    nums = [float(p) for p in parts]
    if len(nums) == 2:
        nums = [0.0] + nums
    h, m, s = nums
    return h * 3600 + m * 60 + s


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_STRIP_CHARS)
    text = _ASR_TOKENS.sub(" ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return _collapse_runs(text).strip()


def _collapse_runs(text: str, max_run: int = 3) -> str:
    """Collapse ASR loops like 'the the the the the' (>=4 identical tokens) to 2."""
    tokens = text.split(" ")
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        key = tokens[i].strip(".,!?;:").casefold()
        while j < len(tokens) and tokens[j].strip(".,!?;:").casefold() == key and key:
            j += 1
        run = j - i
        if run > max_run:
            out.extend(tokens[i : i + 2])
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


# --------------------------------------------------------------------------- model

@dataclass
class Turn:
    speaker: Optional[str]
    start: Optional[float]
    end: Optional[float]
    text: str
    language: Optional[str] = None

    def render(self) -> str:
        ts = fmt_ts(self.start)
        head = f"[{ts}] " if ts else ""
        who = f"{self.speaker}: " if self.speaker else ""
        return f"{head}{who}{self.text}"


@dataclass
class Chunk:
    index: int
    turns: list[Turn]
    context: list[Turn] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(t.render() for t in self.turns)

    def context_text(self) -> str:
        return "\n".join(t.render() for t in self.context)


@dataclass
class PreparedTranscript:
    turns: list[Turn]
    languages: list[str]
    speakers: list[str]
    duration: Optional[float]
    total_tokens: int
    warnings: list[str]

    def text(self) -> str:
        return "\n".join(t.render() for t in self.turns)


# --------------------------------------------------------------------------- parsing


def parse_plain_text(raw: str) -> list[TranscriptSegment]:
    """Parse '[00:01:23] Speaker 1: text', 'Speaker: text' or bare lines."""
    segments: list[TranscriptSegment] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        start: Optional[float] = None
        m = _TS.match(line)
        if m:
            try:
                start = parse_ts(m.group("ts"))
                line = line[m.end():].strip()
            except ValueError:
                start = None
        speaker = None
        sm = _SPEAKER.match(line)
        if sm and len(sm.group("spk").split()) <= 4:
            speaker, line = sm.group("spk").strip(), sm.group("text")
        if line:
            segments.append(TranscriptSegment(text=line, speaker=speaker, start=start))
    return segments


# --------------------------------------------------------------------------- prepare


def prepare(request: SummarizeRequest, settings: Settings) -> PreparedTranscript:
    if request.segments is not None:
        raw_len = sum(len(s.text) for s in request.segments)
        segments = list(request.segments)
    else:
        raw = request.transcript or ""
        raw_len = len(raw)
        segments = parse_plain_text(raw)

    if raw_len > settings.max_input_chars:
        raise InvalidInputError(
            f"Transcript too large ({raw_len} chars; limit {settings.max_input_chars})"
        )

    if segments and all(s.start is not None for s in segments):
        segments = sorted(segments, key=lambda s: s.start)  # stable

    languages: list[str] = []
    turns: list[Turn] = []
    for seg in segments:
        text = normalize_text(seg.text)
        if not text:
            continue
        if seg.language and seg.language not in languages:
            languages.append(seg.language)
        turns.append(Turn(seg.speaker, seg.start, seg.end, text, seg.language))

    if not turns:
        raise EmptyTranscriptError("Transcript is empty after cleaning")

    turns = _merge_turns(turns, settings.merge_gap_seconds)

    joined = " ".join(t.text for t in turns)
    if len(joined) < settings.min_transcript_chars:
        raise InvalidInputError(
            f"Transcript too short to summarise ({len(joined)} chars; "
            f"minimum {settings.min_transcript_chars})"
        )
    if not any(c.isalpha() for c in joined):
        raise InvalidInputError("Transcript contains no readable text")

    speakers = list(dict.fromkeys(t.speaker for t in turns if t.speaker))
    starts = [t.start for t in turns if t.start is not None]
    ends = [t.end if t.end is not None else t.start for t in turns if (t.end or t.start) is not None]
    duration = (max(ends) - min(starts)) if starts and ends else None

    warnings: list[str] = []
    if not speakers:
        warnings.append("No speaker labels found; action-item owners can only come from names spoken in the text.")
    if not starts:
        warnings.append("No timestamps found; items will not carry source timestamps.")

    total_tokens = sum(estimate_tokens(t.render()) + 1 for t in turns)
    return PreparedTranscript(turns, languages, speakers, duration, total_tokens, warnings)


def _merge_turns(turns: list[Turn], max_gap: float) -> list[Turn]:
    """Merge consecutive segments of the same speaker into one speaker turn."""
    merged: list[Turn] = []
    for t in turns:
        prev = merged[-1] if merged else None
        if (
            prev
            and t.speaker
            and prev.speaker == t.speaker
            and (
                prev.end is None
                or t.start is None
                or (t.start - prev.end) <= max_gap
            )
        ):
            prev.text = f"{prev.text} {t.text}"
            prev.end = t.end if t.end is not None else prev.end
        else:
            merged.append(Turn(t.speaker, t.start, t.end, t.text, t.language))
    return merged


# --------------------------------------------------------------------------- chunking


def _split_turn(turn: Turn, max_tokens: int) -> list[Turn]:
    """Split an oversized turn at sentence boundaries (then words as last resort)."""
    limit = max(max_tokens - 30, 50)  # room for '[hh:mm:ss] Speaker: ' prefix
    if estimate_tokens(turn.text) <= limit:
        return [turn]
    pieces: list[str] = []
    cur = ""
    for sentence in _SENTENCE_SPLIT.split(turn.text):
        if estimate_tokens(sentence) > limit:  # no punctuation (common in ASR output)
            if cur:
                pieces.append(cur)
                cur = ""
            words, buf = sentence.split(), ""
            for w in words:
                if buf and estimate_tokens(buf + " " + w) > limit:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf:
                pieces.append(buf)
            continue
        if cur and estimate_tokens(cur + " " + sentence) > limit:
            pieces.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        pieces.append(cur)
    return [Turn(turn.speaker, turn.start, turn.end, p, turn.language) for p in pieces]


def _tail_turn(turn: Turn, budget_tokens: int) -> Turn:
    """Keep only the end of a long turn (whole words) so it fits the overlap budget."""
    words = turn.text.split()
    kept: list[str] = []
    used = 20  # prefix allowance
    for w in reversed(words):
        c = estimate_tokens(w) + 1
        if used + c > budget_tokens:
            break
        kept.insert(0, w)
        used += c
    text = ("... " + " ".join(kept)) if kept and len(kept) < len(words) else " ".join(kept)
    return Turn(turn.speaker, turn.start, turn.end, text or turn.text[-50:], turn.language)


def chunk_turns(
    turns: list[Turn],
    max_tokens: int,
    overlap_tokens: int = 0,
    soft_gap_seconds: Optional[float] = None,
) -> list[Chunk]:
    """Greedy chunking aligned to speaker turns, preferring breaks at long pauses."""
    pieces: list[Turn] = []
    for t in turns:
        pieces.extend(_split_turn(t, max_tokens))

    groups: list[list[Turn]] = []
    cur: list[Turn] = []
    cur_tokens = 0
    prev: Optional[Turn] = None
    for p in pieces:
        cost = estimate_tokens(p.render()) + 1
        hard_break = bool(cur) and cur_tokens + cost > max_tokens
        soft_break = (
            bool(cur)
            and soft_gap_seconds is not None
            and cur_tokens >= 0.75 * max_tokens
            and prev is not None
            and prev.end is not None
            and p.start is not None
            and (p.start - prev.end) >= soft_gap_seconds
        )
        if hard_break or soft_break:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += cost
        prev = p
    if cur:
        groups.append(cur)

    chunks: list[Chunk] = []
    for i, g in enumerate(groups):
        context: list[Turn] = []
        if i > 0 and overlap_tokens > 0:
            used = 0
            for t in reversed(groups[i - 1]):
                c = estimate_tokens(t.render()) + 1
                if used + c > overlap_tokens:
                    if not context:  # last turn alone is too big: keep its tail
                        context.append(_tail_turn(t, overlap_tokens))
                    break
                context.insert(0, t)
                used += c
        chunks.append(Chunk(index=i, turns=g, context=context))
    return chunks
