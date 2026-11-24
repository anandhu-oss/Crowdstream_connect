"""The complete summarization workflow.

    request -> prepare (clean/merge/validate)
            -> short?  yes: one final call straight from the transcript
                       no : chunk -> per-chunk notes -> (recursive) merge -> final call
            -> date check: every date found in the transcript must be in the summary,
               otherwise one revision call with the missing ones listed
            -> validated SummarizeResponse

Pure Python, no web framework and no import of the transcription code, so the main
app can simply ``await summarize(request)``.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings
from .dates import dedupe_mentions, find_date_mentions, format_checklist, uncovered
from .errors import (
    InvalidModelOutputError,
    SummarizationError,
    SummarizationTimeoutError,
)
from .llm_engine import BaseEngine, get_engine
from .preprocessing import PreparedTranscript, chunk_turns, estimate_tokens, prepare
from .prompts import (
    SYSTEM_PROMPT,
    build_chunk_prompt,
    build_final_prompt,
    build_merge_prompt,
    build_repair_prompt,
    resolve_language,
)
from .schemas import (
    ChunkSummary,
    MeetingSummary,
    SummarizeRequest,
    SummarizeResponse,
    TranscriptStats,
)

T = TypeVar("T", bound=BaseModel)

CHECKLIST_CAP = 20   # date mentions listed in a prompt
COVERAGE_CAP = 60    # date mentions verified after the final summary


# --------------------------------------------------------------------------- JSON helpers


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating fences and stray prose."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    try:
        value = json.loads(s)
    except ValueError:
        a, b = s.find("{"), s.rfind("}")
        if a == -1 or b <= a:
            raise ValueError("no JSON object found in model output")
        value = json.loads(s[a : b + 1])  # raises ValueError if still invalid
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


async def _call_json(
    engine: BaseEngine,
    model_cls: Type[T],
    user_prompt: str,
    *,
    max_tokens: int,
    repair_attempts: int,
    stage: str,
) -> T:
    raw = await engine.generate(SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)
    attempt = 0
    while True:
        try:
            return model_cls.model_validate(parse_json(raw))
        except (ValueError, ValidationError) as exc:  # ValidationError subclasses ValueError
            if attempt >= repair_attempts:
                raise InvalidModelOutputError(
                    f"{stage}: model returned invalid output ({str(exc)[:300]})"
                ) from exc
            attempt += 1
            raw = await engine.generate(
                SYSTEM_PROMPT, build_repair_prompt(raw, str(exc)), max_tokens=max_tokens
            )


def _dump(notes: ChunkSummary) -> str:
    # ensure_ascii=False keeps Indic scripts compact (escaped \uXXXX would cost many tokens)
    return json.dumps(notes.model_dump(), ensure_ascii=False, separators=(",", ":"))


def _dump_list(notes: list[ChunkSummary]) -> str:
    return "[" + ",".join(_dump(n) for n in notes) + "]"


# --------------------------------------------------------------------------- stages


async def _summarize_chunks(
    prepared: PreparedTranscript, lang: str, engine: BaseEngine, s: Settings
) -> list[ChunkSummary]:
    chunks = chunk_turns(
        prepared.turns, s.chunk_tokens, s.chunk_overlap_tokens, s.chunk_gap_seconds
    )
    total = len(chunks)

    async def one(c) -> ChunkSummary:
        checklist = format_checklist(
            dedupe_mentions(find_date_mentions(c.turns), CHECKLIST_CAP)
        )
        prompt = build_chunk_prompt(
            c.text(), c.context_text(), c.index, total, prepared.languages, lang, checklist
        )
        return await _call_json(
            engine, ChunkSummary, prompt,
            max_tokens=s.max_output_tokens,
            repair_attempts=s.json_repair_attempts,
            stage=f"chunk {c.index + 1}/{total}",
        )

    # The engine's semaphore enforces SUMMARY_MAX_CONCURRENCY; gather keeps order.
    return list(await asyncio.gather(*(one(c) for c in chunks)))


def _group(notes: list[ChunkSummary], budget: int) -> list[list[ChunkSummary]]:
    groups: list[list[ChunkSummary]] = []
    cur: list[ChunkSummary] = []
    used = 0
    for n in notes:
        t = estimate_tokens(_dump(n))
        if cur and used + t > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(n)
        used += t
    if cur:
        groups.append(cur)
    return groups


async def _reduce(
    notes: list[ChunkSummary], lang: str, engine: BaseEngine, s: Settings
) -> list[ChunkSummary]:
    """Merge notes in order until they fit the final call's input budget."""
    rounds = 0
    while estimate_tokens(_dump_list(notes)) > s.single_pass_tokens:
        if rounds >= s.max_reduce_rounds:
            raise SummarizationError(
                "Meeting is too long for the current chunk/context settings "
                f"(still too large after {rounds} merge rounds)."
            )
        groups = _group(notes, s.merge_group_tokens)
        if all(len(g) == 1 for g in groups):
            raise SummarizationError(
                "Per-chunk notes are too large to merge; lower SUMMARY_CHUNK_TOKENS."
            )

        async def merge(g: list[ChunkSummary]) -> ChunkSummary:
            if len(g) == 1:
                return g[0]
            return await _call_json(
                engine, ChunkSummary, build_merge_prompt(_dump_list(g), lang),
                max_tokens=s.final_max_output_tokens,
                repair_attempts=s.json_repair_attempts,
                stage="merge",
            )

        notes = list(await asyncio.gather(*(merge(g) for g in groups)))
        rounds += 1
    return notes


# --------------------------------------------------------------------------- public API


async def summarize(
    request: SummarizeRequest,
    *,
    settings: Optional[Settings] = None,
    engine: Optional[BaseEngine] = None,
) -> SummarizeResponse:
    s = settings or get_settings()
    eng = engine or get_engine(s)
    started = time.perf_counter()
    try:
        return await asyncio.wait_for(_run(request, s, eng, started), timeout=s.total_timeout)
    except asyncio.TimeoutError as exc:
        raise SummarizationTimeoutError(
            f"Summarization exceeded {s.total_timeout:.0f}s (SUMMARY_TOTAL_TIMEOUT)"
        ) from exc


async def _run(
    request: SummarizeRequest, s: Settings, engine: BaseEngine, started: float
) -> SummarizeResponse:
    prepared = prepare(request, s)
    lang = resolve_language(request.output_language, s.default_language)
    warnings = list(prepared.warnings)
    prefs = request.preferences

    # Deterministic scan of every date/time mention (independent of the LLM).
    all_mentions = dedupe_mentions(find_date_mentions(prepared.turns), COVERAGE_CAP)
    checklist = format_checklist(dedupe_mentions(all_mentions, CHECKLIST_CAP))

    if prepared.total_tokens <= s.single_pass_tokens:
        body, source, n_chunks = prepared.text(), "transcript", 1
    else:
        notes = await _summarize_chunks(prepared, lang, engine, s)
        n_chunks = len(notes)
        notes = await _reduce(notes, lang, engine, s)
        body, source = _dump_list(notes), "notes"
        warnings.append(f"Long meeting: summarised hierarchically from {n_chunks} chunks.")

    async def final(must_include: str = "") -> MeetingSummary:
        prompt = build_final_prompt(
            body, source, prepared.languages, lang, prefs, prepared.speakers,
            date_checklist=checklist, must_include=must_include,
        )
        return await _call_json(
            engine, MeetingSummary, prompt,
            max_tokens=s.final_max_output_tokens,
            repair_attempts=s.json_repair_attempts,
            stage="final summary",
        )

    summary = await final()

    # Date check: nothing date-like that was said may silently go missing.
    if s.coverage_check and all_mentions:
        english = lang.lower() == "english"
        missing = uncovered(all_mentions, summary.model_dump_json(), english)
        if missing:
            try:
                revised = await final(format_checklist(missing[:CHECKLIST_CAP]))
                still = uncovered(all_mentions, revised.model_dump_json(), english)
                if len(still) <= len(missing):  # only accept a revision that is not worse
                    summary, missing = revised, still
            except SummarizationError:
                pass  # keep the first summary; the warning below tells the caller
            if missing:
                shown = "; ".join(
                    f'"{m.text}"' + (f" at {m.timestamp}" if m.timestamp else "")
                    for m in missing[:10]
                )
                warnings.append(f"Date mentions possibly missing from the summary: {shown}")

    summary.key_discussion_points = summary.key_discussion_points[: prefs.max_key_points]

    return SummarizeResponse(
        meeting_id=request.meeting_id,
        summary_language=lang,
        meeting_overview=summary.meeting_overview,
        executive_summary=summary.executive_summary,
        key_discussion_points=summary.key_discussion_points,
        decisions=summary.decisions,
        action_items=summary.action_items,
        key_dates=summary.key_dates,
        unresolved_issues=summary.unresolved_issues,
        next_steps=summary.next_steps,
        processing_time_seconds=round(time.perf_counter() - started, 3),
        model=engine.name,
        chunks_processed=n_chunks,
        transcript_stats=TranscriptStats(
            segments=len(prepared.turns),
            speakers=prepared.speakers,
            languages=prepared.languages,
            duration_seconds=prepared.duration,
            estimated_tokens=prepared.total_tokens,
        ),
        warnings=warnings,
    )
