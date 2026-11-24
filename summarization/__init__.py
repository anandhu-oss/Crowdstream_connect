"""Standalone meeting-summarization module.

Public surface for the host application::

    from summarization import summarize, SummarizeRequest, from_transcription_output
    from summarization.api import router        # app.include_router(router)
"""
from .adapters import from_transcription_output
from .errors import SummarizationError
from .pipeline import summarize
from .schemas import SummarizeRequest, SummarizeResponse

__all__ = [
    "summarize",
    "SummarizeRequest",
    "SummarizeResponse",
    "SummarizationError",
    "from_transcription_output",
]
