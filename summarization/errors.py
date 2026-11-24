"""Typed errors for the summarization module.

Each error carries a stable machine-readable ``code`` and the HTTP status the
API layer should answer with, so callers never have to parse messages.
"""
from __future__ import annotations


class SummarizationError(Exception):
    code = "summarization_error"
    http_status = 500


class InvalidInputError(SummarizationError):
    code = "invalid_input"
    http_status = 422


class EmptyTranscriptError(InvalidInputError):
    code = "empty_transcript"


class ModelUnavailableError(SummarizationError):
    """LLM server unreachable / overloaded (after retries)."""

    code = "model_unavailable"
    http_status = 503


class ModelTimeoutError(SummarizationError):
    """A single LLM request timed out (after retries)."""

    code = "model_timeout"
    http_status = 504


class ModelError(SummarizationError):
    """LLM server answered with a non-retryable error (bad request, auth, ...)."""

    code = "model_error"
    http_status = 502


class InvalidModelOutputError(SummarizationError):
    """Model answered, but not with valid / schema-conforming JSON."""

    code = "invalid_model_output"
    http_status = 502


class SummarizationTimeoutError(SummarizationError):
    """Whole pipeline exceeded SUMMARY_TOTAL_TIMEOUT."""

    code = "summarization_timeout"
    http_status = 504
