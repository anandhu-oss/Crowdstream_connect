"""FastAPI router. Mount it in any app (``app.include_router(router)``) or run it
standalone via ``summarization_app.py`` in the project root.

    POST /summarize        structured request (meeting_id, transcript | segments, ...)
    POST /summarize/raw    raw transcription-module output, mapped by adapters.py
    GET  /summarize/health LLM backend reachability
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .adapters import from_transcription_output
from .errors import SummarizationError
from .llm_engine import BaseEngine, get_engine
from .pipeline import summarize
from .schemas import ErrorResponse, SummarizeRequest, SummarizeResponse

router = APIRouter(tags=["summarization"])


def get_engine_dep() -> BaseEngine:
    """Dependency so tests (or the host app) can swap the engine."""
    return get_engine()


class RawSummarizeRequest(BaseModel):
    meeting_id: Optional[str] = None
    transcription: Any = Field(description="Output of the transcription module, any supported shape")
    output_language: str = "English"


def _error(exc: SummarizationError, meeting_id: Optional[str]) -> JSONResponse:
    body = ErrorResponse(error_code=exc.code, detail=str(exc), meeting_id=meeting_id)
    return JSONResponse(status_code=exc.http_status, content=body.model_dump())


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
               503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def summarize_endpoint(request: SummarizeRequest, engine: BaseEngine = Depends(get_engine_dep)):
    try:
        return await summarize(request, engine=engine)
    except SummarizationError as exc:
        return _error(exc, request.meeting_id)


@router.post("/summarize/raw", response_model=SummarizeResponse)
async def summarize_raw_endpoint(
    request: RawSummarizeRequest, engine: BaseEngine = Depends(get_engine_dep)
):
    try:
        mapped = from_transcription_output(
            request.transcription, request.meeting_id, request.output_language
        )
        return await summarize(mapped, engine=engine)
    except SummarizationError as exc:
        return _error(exc, request.meeting_id)


@router.get("/summarize/health")
async def health(engine: BaseEngine = Depends(get_engine_dep)):
    return await engine.healthcheck()
