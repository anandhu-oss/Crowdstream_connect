"""Deterministic date / time mention finder.

Why it exists: a small (quantised) LLM sometimes skips a date. This module finds every
date-like mention in the transcript with plain regexes - independent of the model - so
the pipeline can (1) hand the model a checklist, and (2) verify afterwards that each
mention shows up in the summary, re-asking once if something is missing.

Coverage: English month names, numeric dates (dd/mm/yyyy, ISO), ordinal days ("the 28th"),
weekdays (English + Hindi / Malayalam / Tamil names), relative words (today, tomorrow,
next week, end of month ...) and clock times. Other languages' dates are still handled by
the LLM itself; they just aren't part of the automatic checklist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .preprocessing import Turn, fmt_ts

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4,
    "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11,
    "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"
_F = re.IGNORECASE

_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_WEEKDAY_NATIVE = {
    # Hindi
    "सोमवार": "monday", "मंगलवार": "tuesday", "बुधवार": "wednesday", "गुरुवार": "thursday",
    "बृहस्पतिवार": "thursday", "शुक्रवार": "friday", "शनिवार": "saturday", "रविवार": "sunday",
    # Malayalam
    "തിങ്കളാഴ്ച": "monday", "ചൊവ്വാഴ്ച": "tuesday", "ബുധനാഴ്ച": "wednesday",
    "വ്യാഴാഴ്ച": "thursday", "വെള്ളിയാഴ്ച": "friday", "ശനിയാഴ്ച": "saturday",
    "ഞായറാഴ്ച": "sunday",
    # Tamil
    "திங்கள்": "monday", "செவ்வாய்": "tuesday", "புதன்": "wednesday", "வியாழன்": "thursday",
    "வெள்ளி": "friday", "சனி": "saturday", "ஞாயிறு": "sunday",
}

_P_MD = re.compile(rf"\b(?P<m>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}}){_ORD}\b", _F)
_P_DM = re.compile(rf"\b(?P<d>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?P<m>{_MONTH_ALT})\b", _F)
_P_ISO = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_P_NUM = re.compile(r"\b(?P<a>\d{1,2})[/-](?P<b>\d{1,2})[/-](?P<y>\d{2,4})\b")
_P_ORDDAY = re.compile(r"\b(?:the|on|by|until|till|before|after)\s+(?P<d>\d{1,2})(?:st|nd|rd|th)\b", _F)
_P_WD = re.compile(rf"\b(?P<w>{_WEEKDAYS})\b", _F)
_P_WD_NATIVE = re.compile("|".join(map(re.escape, sorted(_WEEKDAY_NATIVE, key=len, reverse=True))))
_P_REL = re.compile(
    r"\b(?P<r>day after tomorrow|tomorrow|today|tonight|next week|this week|next month|"
    r"this month|end of (?:the )?(?:day|week|month|quarter|year)|eod|eow|eom)\b",
    _F,
)
_P_TIME12 = re.compile(
    r"\b(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?\s*(?P<ap>a\.m\.|p\.m\.|am|pm)(?![a-z])", _F
)
_P_TIME24 = re.compile(r"\bat\s+(?P<h>\d{1,2}):(?P<mi>\d{2})\b(?!\s*[ap]\.?m)", _F)

# "today" etc. only matter when the sentence is about a commitment / deadline, not small talk
_CUE = re.compile(
    r"\b(?:by|till|until|before|due|deadline|will|shall|going to|need|needs|please|remind|send|"
    r"finish|complete|submit|deliver|share|follow up|ready|\w+[\u2019']ll)\b",
    _F,
)
_MD_KEY = re.compile(r"\d\d-\d\d")
_KIND_PRIORITY = {"date": 0, "weekday": 1, "time": 2, "relative": 3}
_SENT_BOUNDARY = re.compile(r"(?<=[.!?।॥。？！])\s+")


@dataclass
class DateMention:
    key: str            # normalised identity, e.g. "12-05", "d28", "wd:friday", "rel:today"
    kind: str           # date | weekday | relative | time
    text: str           # as spoken
    snippet: str        # the sentence it appeared in (truncated)
    timestamp: Optional[str] = None
    speaker: Optional[str] = None


def _scan(text: str) -> list[tuple[int, int, str, str, str]]:
    found: list[tuple[int, int, str, str, str]] = []

    def add(m: re.Match, kind: str, key: str) -> None:
        found.append((m.start(), m.end(), kind, key, m.group(0)))

    for pat in (_P_MD, _P_DM):
        for m in pat.finditer(text):
            mon, day = _MONTHS[m.group("m").lower()], int(m.group("d"))
            if 1 <= day <= 31:
                add(m, "date", f"{mon:02d}-{day:02d}")
    for m in _P_ISO.finditer(text):
        mon, day = int(m.group("m")), int(m.group("d"))
        if 1 <= mon <= 12 and 1 <= day <= 31:
            add(m, "date", f"{mon:02d}-{day:02d}")
    for m in _P_NUM.finditer(text):
        a, b = int(m.group("a")), int(m.group("b"))
        if a > 12:
            day, mon = a, b
        elif b > 12:
            mon, day = a, b
        else:  # ambiguous: day-first (Indian / European convention)
            day, mon = a, b
        if 1 <= mon <= 12 and 1 <= day <= 31:
            add(m, "date", f"{mon:02d}-{day:02d}")
    for m in _P_ORDDAY.finditer(text):
        day = int(m.group("d"))
        if 1 <= day <= 31:
            add(m, "date", f"d{day:02d}")
    for m in _P_WD.finditer(text):
        add(m, "weekday", f"wd:{m.group('w').lower()}")
    for m in _P_WD_NATIVE.finditer(text):
        add(m, "weekday", f"wd:{_WEEKDAY_NATIVE[m.group(0)]}")
    for m in _P_REL.finditer(text):
        add(m, "relative", "rel:" + " ".join(m.group("r").lower().split()))
    for m in _P_TIME12.finditer(text):
        h, mi = int(m.group("h")), int(m.group("mi") or 0)
        ap = m.group("ap").lower().replace(".", "")
        if not (1 <= h <= 12 and mi < 60):
            continue
        if ap.startswith("p") and h != 12:
            h += 12
        elif ap.startswith("a") and h == 12:
            h = 0
        add(m, "time", f"t:{h:02d}:{mi:02d}")
    for m in _P_TIME24.finditer(text):
        h, mi = int(m.group("h")), int(m.group("mi"))
        if h < 24 and mi < 60:
            add(m, "time", f"t:{h:02d}:{mi:02d}")

    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: list[tuple[int, int, str, str, str]] = []
    last_end = -1
    for item in found:  # drop overlapping matches, keep the longest at each position
        if item[0] >= last_end:
            kept.append(item)
            last_end = item[1]
    return kept


def mention_keys(text: str) -> set[str]:
    """All normalised date/time keys found in arbitrary text."""
    return {k for _, _, _, k, _ in _scan(text)}


def _sentence_around(text: str, pos: int, limit: int = 110) -> str:
    start, end = 0, len(text)
    for m in _SENT_BOUNDARY.finditer(text):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            end = m.start()
            break
    s = text[start:end].strip()
    if len(s) <= limit:
        return s
    rel = pos - start
    a = max(0, min(rel - limit // 2, len(s) - limit))
    return ("..." if a > 0 else "") + s[a : a + limit] + ("..." if a + limit < len(s) else "")


def find_date_mentions(turns: list[Turn]) -> list[DateMention]:
    out: list[DateMention] = []
    for t in turns:
        for start, _end, kind, key, spoken in _scan(t.text):
            sentence = _sentence_around(t.text, start)
            if kind == "relative" and not _CUE.search(sentence):
                continue
            out.append(DateMention(key, kind, spoken, sentence, fmt_ts(t.start), t.speaker))
    return out


def dedupe_mentions(mentions: list[DateMention], cap: int) -> list[DateMention]:
    """First occurrence per key; if over `cap`, keep concrete dates before weekdays/relative words."""
    seen: set[str] = set()
    unique: list[tuple[int, DateMention]] = []
    for i, m in enumerate(mentions):
        if m.key not in seen:
            seen.add(m.key)
            unique.append((i, m))
    if len(unique) > cap:
        unique = sorted(unique, key=lambda p: (_KIND_PRIORITY.get(p[1].kind, 9), p[0]))[:cap]
        unique.sort(key=lambda p: p[0])
    return [m for _, m in unique]


def format_checklist(mentions: list[DateMention]) -> str:
    lines = []
    for m in mentions:
        where = " ".join(x for x in (m.timestamp, m.speaker and f"{m.speaker}:") if x)
        lines.append(f'- "{m.text}"' + (f" [{where}]" if where else "") + f" - {m.snippet}")
    return "\n".join(lines)


def uncovered(mentions: list[DateMention], summary_text: str, english: bool = True) -> list[DateMention]:
    """Mentions that do not appear in the summary text.

    Concrete dates are matched on digits/month so they work in any output language.
    Weekdays, relative words and times depend on English words, so they are only
    checked when the summary is in English.
    """
    keys = mention_keys(summary_text)
    days = {k[3:] for k in keys if _MD_KEY.fullmatch(k)} | {k[1:] for k in keys if k.startswith("d")}
    missing: list[DateMention] = []
    for m in mentions:
        k = m.key
        if m.kind == "date":
            if _MD_KEY.fullmatch(k):
                ok = k in keys or f"d{k[3:]}" in keys
            else:
                ok = k in keys or k[1:] in days
        elif not english:
            ok = True
        else:
            ok = k in keys
        if not ok:
            missing.append(m)
    return missing
