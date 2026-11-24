"""Input / output data models.

The ``Lenient*`` coercion helpers exist because small LLMs frequently return
``"null"`` strings, a bare string instead of a list, or objects inside lists.
We normalise those harmless deviations instead of failing the whole request.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

_NULLISH = {"", "null", "none", "n/a", "na", "unspecified", "not specified", "unknown", "-"}

Status = Literal["open", "in_progress", "done", "unspecified"]


def _clean_optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _NULLISH else s


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (str, dict)):
        v = [v]
    out: list[str] = []
    seen: set[str] = set()
    for item in v:
        if isinstance(item, dict):
            item = " - ".join(str(x).strip() for x in item.values() if x not in (None, ""))
        s = str(item).strip()
        if not s or s.lower() in _NULLISH:
            continue
        key = s.casefold()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# --------------------------------------------------------------------------- input


class TranscriptSegment(BaseModel):
    text: str
    speaker: Optional[str] = None
    start: Optional[float] = Field(default=None, ge=0, description="seconds from meeting start")
    end: Optional[float] = Field(default=None, ge=0)
    language: Optional[str] = None

    @field_validator("speaker", "language", mode="before")
    @classmethod
    def _opt(cls, v):
        return _clean_optional_str(v)


class SummaryPreferences(BaseModel):
    max_key_points: int = Field(default=10, ge=3, le=30)
    style: Literal["concise", "detailed"] = "concise"
    focus: Optional[str] = Field(default=None, max_length=300, description="e.g. 'hiring decisions'")


class SummarizeRequest(BaseModel):
    meeting_id: str = Field(min_length=1, max_length=128)
    transcript: Optional[str] = Field(default=None, description="plain-text transcript")
    segments: Optional[list[TranscriptSegment]] = Field(
        default=None, description="structured transcript; takes precedence over `transcript`"
    )
    output_language: str = Field(default="English", min_length=2, max_length=40)
    preferences: SummaryPreferences = Field(default_factory=SummaryPreferences)

    @model_validator(mode="after")
    def _need_input(self):
        if self.transcript is None and self.segments is None:
            raise ValueError("Provide either `transcript` or `segments`")
        return self


# --------------------------------------------------------------------------- LLM output


class ActionItem(BaseModel):
    task: str = Field(min_length=1)
    assigned_to: Optional[str] = None
    deadline: Optional[str] = None
    status: Status = "unspecified"
    source_timestamp: Optional[str] = None

    @field_validator("assigned_to", "deadline", "source_timestamp", mode="before")
    @classmethod
    def _opt(cls, v):
        return _clean_optional_str(v)

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        s = str(v or "").strip().lower().replace("-", "_").replace(" ", "_")
        if s in {"open", "pending", "todo", "to_do", "not_started", "new"}:
            return "open"
        if s in {"in_progress", "ongoing", "started", "working"}:
            return "in_progress"
        if s in {"done", "completed", "complete", "closed", "finished"}:
            return "done"
        return "unspecified"


class KeyDate(BaseModel):
    """Any date / time / deadline / milestone mentioned in the meeting."""

    date: str = Field(min_length=1)
    what: str = ""
    owner: Optional[str] = None
    timestamp: Optional[str] = None

    @field_validator("owner", "timestamp", mode="before")
    @classmethod
    def _opt(cls, v):
        return _clean_optional_str(v)

    @field_validator("what", mode="before")
    @classmethod
    def _what(cls, v):
        return "" if v is None else str(v).strip()


def _coerce_key_dates(v: Any) -> list[dict]:
    if v is None:
        return []
    if isinstance(v, (str, dict)):
        v = [v]
    out: list[dict] = []
    for item in v:
        if isinstance(item, str):
            item = {"date": item}
        if not isinstance(item, dict):
            continue
        d = dict(item)
        raw = d.get("date") or d.get("when") or d.get("deadline") or d.get("due")
        date = _clean_optional_str(raw)
        if date is None:
            continue
        d["date"] = date
        out.append(d)
    return out


class MeetingSummary(BaseModel):
    """What the final LLM call must produce."""

    meeting_overview: str = ""
    executive_summary: str = ""
    key_discussion_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    key_dates: list[KeyDate] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)

    @field_validator(
        "key_discussion_points", "decisions", "unresolved_issues", "next_steps", mode="before"
    )
    @classmethod
    def _lists(cls, v):
        return _as_str_list(v)

    @field_validator("action_items", mode="before")
    @classmethod
    def _items(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, dict)):
            v = [v]
        return [{"task": i} if isinstance(i, str) else i for i in v]

    @field_validator("key_dates", mode="before")
    @classmethod
    def _dates(cls, v):
        return _coerce_key_dates(v)

    @model_validator(mode="after")
    def _non_empty(self):
        if not self.executive_summary.strip():
            raise ValueError("executive_summary is empty")
        return self


class ChunkTopic(BaseModel):
    title: str = ""
    summary: str = ""
    timestamp: Optional[str] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _opt(cls, v):
        return _clean_optional_str(v)


class ChunkDecision(BaseModel):
    decision: str = Field(min_length=1)
    timestamp: Optional[str] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _opt(cls, v):
        return _clean_optional_str(v)


class ChunkSummary(BaseModel):
    """Intermediate notes for one chunk (also the shape of merged notes)."""

    topics: list[ChunkTopic] = Field(default_factory=list)
    decisions: list[ChunkDecision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    dates: list[KeyDate] = Field(default_factory=list)

    @field_validator("dates", mode="before")
    @classmethod
    def _dates(cls, v):
        return _coerce_key_dates(v)

    @field_validator("topics", mode="before")
    @classmethod
    def _topics(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, dict)):
            v = [v]
        return [{"summary": t} if isinstance(t, str) else t for t in v]

    @field_validator("decisions", mode="before")
    @classmethod
    def _decisions(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, dict)):
            v = [v]
        return [{"decision": d} if isinstance(d, str) else d for d in v]

    @field_validator("action_items", mode="before")
    @classmethod
    def _items(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, dict)):
            v = [v]
        return [{"task": i} if isinstance(i, str) else i for i in v]

    @field_validator("unresolved", "facts", mode="before")
    @classmethod
    def _lists(cls, v):
        return _as_str_list(v)


# --------------------------------------------------------------------------- API output


class TranscriptStats(BaseModel):
    segments: int
    speakers: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    estimated_tokens: int = 0


class SummarizeResponse(BaseModel):
    meeting_id: str
    status: Literal["completed"] = "completed"
    summary_language: str
    meeting_overview: str
    executive_summary: str
    key_discussion_points: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    key_dates: list[KeyDate] = Field(default_factory=list)
    unresolved_issues: list[str]
    next_steps: list[str]
    processing_time_seconds: float
    model: str
    chunks_processed: int = 1
    transcript_stats: Optional[TranscriptStats] = None
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    status: Literal["failed"] = "failed"
    error_code: str
    detail: str
    meeting_id: Optional[str] = None
