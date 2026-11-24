"""LLM access.

The module never loads model weights itself. It talks to a self-hosted,
OpenAI-compatible server (vLLM on your RunPod GPU, Ollama, llama.cpp, ...) whose
URL / model name come from the environment - the same "drop an endpoint in .env"
pattern as the transcription service. A ``mock`` backend exists for tests.
"""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

import httpx

from .config import Settings, get_settings
from .errors import (
    InvalidModelOutputError,
    ModelError,
    ModelTimeoutError,
    ModelUnavailableError,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:  # opening tag was swallowed by the chat template
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


class BaseEngine(ABC):
    name: str = "unknown"

    @abstractmethod
    async def generate(self, system: str, user: str, *, max_tokens: Optional[int] = None) -> str:
        """Return the raw text answer for one chat completion."""

    async def healthcheck(self) -> dict:
        return {"backend": type(self).__name__, "model": self.name, "reachable": True}


class OpenAICompatEngine(BaseEngine):
    def __init__(self, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.s = settings
        self.name = settings.llm_model
        self._transport = transport  # tests inject httpx.MockTransport here
        self._sem = asyncio.Semaphore(settings.max_concurrency)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.s.llm_api_key}", "Content-Type": "application/json"}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.s.request_timeout, transport=self._transport)

    def _payload(self, system: str, user: str, max_tokens: Optional[int]) -> dict:
        payload: dict = {
            "model": self.s.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.s.temperature,
            "max_tokens": max_tokens or self.s.max_output_tokens,
        }
        if self.s.json_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.s.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    async def generate(self, system: str, user: str, *, max_tokens: Optional[int] = None) -> str:
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        payload = self._payload(system, user, max_tokens)
        attempt = 0
        async with self._sem:
            while True:
                try:
                    async with self._client() as client:
                        resp = await client.post(url, json=payload, headers=self._headers())
                except httpx.TimeoutException as exc:
                    err: Exception = ModelTimeoutError(
                        f"LLM request timed out after {self.s.request_timeout:.0f}s"
                    )
                    err.__cause__ = exc
                except httpx.TransportError as exc:
                    err = ModelUnavailableError(f"Cannot reach LLM server at {url}: {exc}")
                    err.__cause__ = exc
                else:
                    if resp.status_code == 200:
                        return self._extract(resp)
                    if resp.status_code in (429, 502, 503, 504):
                        err = ModelUnavailableError(
                            f"LLM server returned HTTP {resp.status_code}: {resp.text[:300]}"
                        )
                    else:  # includes 500: a generation failure is deterministic, don't retry it
                        raise ModelError(
                            f"LLM server returned HTTP {resp.status_code}: {resp.text[:300]}"
                        )
                if attempt >= self.s.max_retries:
                    raise err
                attempt += 1
                await asyncio.sleep(min(2**attempt, 10))

    @staticmethod
    def _extract(resp: httpx.Response) -> str:
        try:
            choice = resp.json()["choices"][0]
            content = (choice.get("message") or {}).get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelError("Unexpected response format from LLM server") from exc
        if choice.get("finish_reason") == "length":
            head = " ".join(content.split())[:160]
            hint = (
                " The output starts like reasoning text, so the server is probably still in "
                "thinking mode: disable it in the server's chat template call."
                if "</think>" not in content and not content.lstrip().startswith(("{", "```"))
                else ""
            )
            raise InvalidModelOutputError(
                "Model output was cut off (hit max tokens). Raise SUMMARY_MAX_OUTPUT_TOKENS / "
                f"SUMMARY_FINAL_MAX_OUTPUT_TOKENS or lower SUMMARY_CHUNK_TOKENS.{hint} "
                f"Output began: {head!r}"
            )
        return strip_thinking(content)

    async def healthcheck(self) -> dict:
        url = self.s.llm_base_url.rstrip("/") + "/models"
        info = {"backend": "openai_compat", "base_url": self.s.llm_base_url, "model": self.s.llm_model}
        try:
            async with httpx.AsyncClient(timeout=10, transport=self._transport) as client:
                resp = await client.get(url, headers=self._headers())
            info["reachable"] = resp.status_code == 200
            if resp.status_code == 200:
                ids = [m.get("id") for m in resp.json().get("data", [])]
                info["model_served"] = self.s.llm_model in ids if ids else None
                info["served_models"] = ids[:10]
            else:
                info["detail"] = f"HTTP {resp.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            info["reachable"] = False
            info["detail"] = str(exc)[:200]
        return info


class MockEngine(BaseEngine):
    """Deterministic fake model so the whole pipeline can be tested without a GPU.

    It does no real understanding: chunk notes are synthetic, merge/final steps combine
    whatever notes they receive. That is enough to test chunking, hierarchical reduction,
    JSON handling and the API plumbing.
    """

    name = "mock"

    async def generate(self, system: str, user: str, *, max_tokens: Optional[int] = None) -> str:
        if user.startswith("TASK:CHUNK"):
            m = re.search(r"part (\d+) of (\d+)", user)
            i = int(m.group(1)) if m else 1
            body = _between(user, "<<<TRANSCRIPT", "TRANSCRIPT>>>")
            return json.dumps(
                {
                    "topics": [{"title": f"Topic {i}", "summary": body[:60], "timestamp": None}],
                    "decisions": [{"decision": f"decision from chunk {i}", "timestamp": None}],
                    "action_items": [
                        {"task": f"task from chunk {i}", "assigned_to": None,
                         "deadline": None, "status": "open", "source_timestamp": None}
                    ],
                    "unresolved": [f"question from chunk {i}"],
                    "facts": [],
                    "dates": [],
                }
            )
        if user.startswith("TASK:MERGE"):
            merged = _merge_notes(json.loads(_between(user, "<<<NOTES", "NOTES>>>")))
            merged["topics"] = []  # a real model condenses topics; simulate that
            merged["facts"] = []
            return json.dumps(merged)
        if user.startswith("TASK:FINAL"):
            if "<<<NOTES" in user:
                notes = _merge_notes(json.loads(_between(user, "<<<NOTES", "NOTES>>>")))
                return json.dumps(
                    {
                        "meeting_overview": "Mock meeting overview.",
                        "executive_summary": f"Mock summary built from {len(notes['topics'])} topics.",
                        "key_discussion_points": [t["summary"] or t["title"] for t in notes["topics"]],
                        "decisions": [d["decision"] for d in notes["decisions"]],
                        "action_items": notes["action_items"],
                        "key_dates": notes["dates"],
                        "unresolved_issues": notes["unresolved"],
                        "next_steps": [],
                    }
                )
            body = _between(user, "<<<TRANSCRIPT", "TRANSCRIPT>>>")
            return json.dumps(
                {
                    "meeting_overview": "Mock meeting overview.",
                    "executive_summary": f"Mock summary of a {len(body)}-character transcript.",
                    "key_discussion_points": ["Mock discussion point"],
                    "decisions": [],
                    "action_items": [],
                    "unresolved_issues": [],
                    "next_steps": [],
                }
            )
        return "{}"


def _between(text: str, start: str, end: str) -> str:
    a = text.index(start) + len(start)
    return text[a : text.rindex(end)].strip()


def _merge_notes(notes: list[dict]) -> dict:
    merged = {"topics": [], "decisions": [], "action_items": [], "unresolved": [], "facts": [], "dates": []}
    for n in notes:
        for k in merged:
            merged[k].extend(n.get(k, []))
    return merged


@lru_cache(maxsize=4)
def _build(settings: Settings) -> BaseEngine:
    if settings.llm_backend == "mock":
        return MockEngine()
    return OpenAICompatEngine(settings)


def get_engine(settings: Optional[Settings] = None) -> BaseEngine:
    return _build(settings or get_settings())
