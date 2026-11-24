"""Configuration. Everything is driven by environment variables (prefix SUMMARY_).

A ``.env`` file next to this module (or in the project root) is loaded
automatically if ``python-dotenv`` is installed. Nothing here imports the
transcription code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # dotenv is optional
        return
    here = Path(__file__).resolve().parent
    for candidate in (here / ".env", here.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _get(name: str, default, cast=str):
    raw = os.getenv(f"SUMMARY_{name}")
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    try:
        if cast is bool:
            return raw.lower() in {"1", "true", "yes", "on"}
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid value for SUMMARY_{name}: {raw!r}") from exc


# Rough allowance (tokens) for system prompt + field guide + worked example + JSON skeleton
# + date checklist + context lines.
PROMPT_OVERHEAD_TOKENS = 2400


@dataclass(frozen=True)
class Settings:
    # --- LLM server (OpenAI-compatible: vLLM, Ollama /v1, LM Studio, llama.cpp server...) ---
    llm_backend: str = "openai_compat"  # "openai_compat" | "mock"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    json_mode: str = "json_object"  # "json_object" | "off"
    # Qwen3-8B / 1.7B are hybrid "thinking" models; this switches thinking off via the
    # chat template (vLLM `chat_template_kwargs`). Set false if your server rejects the field.
    disable_thinking: bool = True

    # --- generation ---
    temperature: float = 0.1
    max_output_tokens: int = 2048        # chunk + merge steps
    final_max_output_tokens: int = 3072  # final structured summary

    # --- sizing (all in estimated tokens) ---
    context_tokens: int = 8192   # must match the server's max model length
    chunk_tokens: int = 3000     # transcript tokens per chunk
    chunk_overlap_tokens: int = 300  # tail of previous chunk shown as context only
    chunk_gap_seconds: float = 45.0  # prefer chunk breaks at pauses at least this long

    # --- preprocessing ---
    merge_gap_seconds: float = 8.0   # merge same-speaker segments closer than this
    min_transcript_chars: int = 20
    max_input_chars: int = 3_000_000

    # --- robustness / performance ---
    request_timeout: float = 180.0   # per LLM call
    total_timeout: float = 1200.0    # whole pipeline
    max_retries: int = 2
    max_concurrency: int = 2         # parallel LLM calls
    json_repair_attempts: int = 1
    max_reduce_rounds: int = 6
    # After the final summary: verify every date found in the transcript appears in it,
    # and ask the model once more (with the missing ones listed) if not.
    coverage_check: bool = True

    default_language: str = "English"

    def __post_init__(self) -> None:
        if self.llm_backend not in {"openai_compat", "mock"}:
            raise ValueError("SUMMARY_LLM_BACKEND must be 'openai_compat' or 'mock'")
        if self.json_mode not in {"json_object", "off"}:
            raise ValueError("SUMMARY_JSON_MODE must be 'json_object' or 'off'")
        if self.chunk_tokens + self.max_output_tokens + PROMPT_OVERHEAD_TOKENS > self.context_tokens:
            raise ValueError(
                f"chunk_tokens + max_output_tokens + ~{PROMPT_OVERHEAD_TOKENS} prompt overhead exceeds context_tokens; "
                "lower SUMMARY_CHUNK_TOKENS or raise SUMMARY_CONTEXT_TOKENS"
            )
        if self.single_pass_tokens < 500:
            raise ValueError("final_max_output_tokens leaves too little room in context_tokens")
        if self.max_concurrency < 1:
            raise ValueError("SUMMARY_MAX_CONCURRENCY must be >= 1")

    @property
    def single_pass_tokens(self) -> int:
        """Max input tokens (transcript or merged notes) the final call can take."""
        return self.context_tokens - self.final_max_output_tokens - PROMPT_OVERHEAD_TOKENS

    @property
    def merge_group_tokens(self) -> int:
        """Max input per merge call. Merge calls use the larger final output cap, so
        input + output + overhead still fits the context window (same maths as the
        final call), and notes can always be merged at least pairwise."""
        return self.single_pass_tokens

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file()
        d = cls()  # defaults
        return cls(
            llm_backend=_get("LLM_BACKEND", d.llm_backend),
            llm_base_url=_get("LLM_BASE_URL", d.llm_base_url),
            llm_api_key=_get("LLM_API_KEY", d.llm_api_key),
            llm_model=_get("LLM_MODEL", d.llm_model),
            json_mode=_get("JSON_MODE", d.json_mode),
            disable_thinking=_get("DISABLE_THINKING", d.disable_thinking, bool),
            temperature=_get("TEMPERATURE", d.temperature, float),
            max_output_tokens=_get("MAX_OUTPUT_TOKENS", d.max_output_tokens, int),
            final_max_output_tokens=_get("FINAL_MAX_OUTPUT_TOKENS", d.final_max_output_tokens, int),
            context_tokens=_get("CONTEXT_TOKENS", d.context_tokens, int),
            chunk_tokens=_get("CHUNK_TOKENS", d.chunk_tokens, int),
            chunk_overlap_tokens=_get("CHUNK_OVERLAP_TOKENS", d.chunk_overlap_tokens, int),
            chunk_gap_seconds=_get("CHUNK_GAP_SECONDS", d.chunk_gap_seconds, float),
            merge_gap_seconds=_get("MERGE_GAP_SECONDS", d.merge_gap_seconds, float),
            min_transcript_chars=_get("MIN_TRANSCRIPT_CHARS", d.min_transcript_chars, int),
            max_input_chars=_get("MAX_INPUT_CHARS", d.max_input_chars, int),
            request_timeout=_get("REQUEST_TIMEOUT", d.request_timeout, float),
            total_timeout=_get("TOTAL_TIMEOUT", d.total_timeout, float),
            max_retries=_get("MAX_RETRIES", d.max_retries, int),
            max_concurrency=_get("MAX_CONCURRENCY", d.max_concurrency, int),
            json_repair_attempts=_get("JSON_REPAIR_ATTEMPTS", d.json_repair_attempts, int),
            max_reduce_rounds=_get("MAX_REDUCE_ROUNDS", d.max_reduce_rounds, int),
            coverage_check=_get("COVERAGE_CHECK", d.coverage_check, bool),
            default_language=_get("DEFAULT_LANGUAGE", d.default_language),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (useful in tests / CLI)."""
    get_settings.cache_clear()
    return get_settings()
