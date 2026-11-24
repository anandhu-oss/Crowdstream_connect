"""The ONE place that knows about the transcription module's output shape.

The summarization pipeline only understands ``SummarizeRequest``. When the module
is integrated, the transcription result is mapped to it here - and nowhere else.
If the transcription output changes, edit the alias tables below (or the function)
and nothing else in the package needs to change.

!! The alias tables are a best guess of common shapes. Replace/extend them once the
!! real final-transcript JSON from the transcription service is known.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .errors import InvalidInputError
from .preprocessing import parse_ts
from .schemas import SummarizeRequest, TranscriptSegment

# keys that may hold the list of segments
SEGMENT_LIST_KEYS = ("segments", "utterances", "transcript_segments", "results", "chunks")
# keys that may hold a whole-transcript string
FULL_TEXT_KEYS = ("transcript", "full_text", "text", "final_transcript")
# per-segment field aliases
TEXT_KEYS = ("text", "transcript", "transcription", "content", "original_text")
SPEAKER_KEYS = ("speaker", "speaker_id", "speaker_label", "spk", "participant")
START_KEYS = ("start", "start_time", "start_s", "begin", "timestamp", "t0")
END_KEYS = ("end", "end_time", "end_s", "t1")
LANG_KEYS = ("language", "lang", "detected_language", "original_language", "source_lang")
MEETING_ID_KEYS = ("meeting_id", "session_id", "room_id", "id")


def _first(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _seconds(value: Any) -> Optional[float]:
    """Accept 12.5, '12.5', '00:01:12'. Assumes SECONDS for numbers (change here if ms)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return parse_ts(str(value)) if ":" in str(value) else float(value)
    except ValueError:
        return None


def _segment(d: dict) -> Optional[TranscriptSegment]:
    text = _first(d, TEXT_KEYS)
    if not isinstance(text, str) or not text.strip():
        return None
    return TranscriptSegment(
        text=text,
        speaker=_first(d, SPEAKER_KEYS),
        start=_seconds(_first(d, START_KEYS)),
        end=_seconds(_first(d, END_KEYS)),
        language=_first(d, LANG_KEYS),
    )


def from_transcription_output(
    data: Any,
    meeting_id: Optional[str] = None,
    output_language: str = "English",
    default_meeting_id: Optional[str] = None,
) -> SummarizeRequest:
    """Map whatever the transcription module produced to a SummarizeRequest.

    Accepts: a plain string, a list of segment dicts, or a dict containing a
    segment list and/or full text (plus an optional meeting id).
    """
    mid = meeting_id
    segments: Optional[list[TranscriptSegment]] = None
    transcript: Optional[str] = None

    if isinstance(data, str):
        transcript = data
    elif isinstance(data, list):
        segments = [s for s in (_segment(x) for x in data if isinstance(x, dict)) if s]
    elif isinstance(data, dict):
        mid = mid or _first(data, MEETING_ID_KEYS)
        seg_list = next(
            (data[k] for k in SEGMENT_LIST_KEYS if isinstance(data.get(k), list)), None
        )
        if seg_list is not None:
            segments = [s for s in (_segment(x) for x in seg_list if isinstance(x, dict)) if s]
        else:
            text = _first(data, FULL_TEXT_KEYS)
            if isinstance(text, str):
                transcript = text
    else:
        raise InvalidInputError(f"Unsupported transcription output type: {type(data).__name__}")

    if segments is None and transcript is None:
        raise InvalidInputError("Could not find segments or text in the transcription output")
    mid = mid or default_meeting_id
    if mid is None:
        raise InvalidInputError("meeting_id is required (pass it explicitly or include it in the data)")

    return SummarizeRequest(
        meeting_id=str(mid),
        segments=segments,
        transcript=transcript,
        output_language=output_language,
    )


def request_from_file(
    path: str | Path, meeting_id: Optional[str] = None, output_language: str = "English"
) -> SummarizeRequest:
    """Load a .json (transcription output) or plain-text transcript file as a request.

    Used by the CLI and the evaluation script; the file name is the fallback meeting id.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise InvalidInputError(f"{p.name}: invalid JSON ({exc})") from exc
        return from_transcription_output(
            data, meeting_id, output_language, default_meeting_id=p.stem
        )
    return SummarizeRequest(
        meeting_id=meeting_id or p.stem, transcript=raw, output_language=output_language
    )
