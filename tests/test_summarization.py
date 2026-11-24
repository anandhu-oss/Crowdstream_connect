import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from summarization.adapters import from_transcription_output
from summarization.api import get_engine_dep, router
from summarization.config import Settings
from summarization.errors import (
    EmptyTranscriptError,
    InvalidInputError,
    InvalidModelOutputError,
    ModelUnavailableError,
)
from summarization.llm_engine import BaseEngine, MockEngine, OpenAICompatEngine, strip_thinking
from summarization.pipeline import parse_json, summarize
from summarization.preprocessing import (
    chunk_turns,
    estimate_tokens,
    normalize_text,
    parse_plain_text,
    prepare,
)
from summarization.schemas import ActionItem, SummarizeRequest

SAMPLES = Path(__file__).parent / "sample_transcripts"

# small context so multi-chunk paths are exercised with short test data
SMALL = replace(
    Settings(), llm_backend="mock", context_tokens=4096, final_max_output_tokens=1024,
    max_output_tokens=512, chunk_tokens=300, chunk_overlap_tokens=60,
)


def run(coro):
    return asyncio.run(coro)


def words(n, seed=0):
    """n distinct-ish words (identical runs would be collapsed as ASR loops)."""
    return " ".join(f"w{(seed * 7 + i) % 997}x" for i in range(n))


def req(text, **kw):
    return SummarizeRequest(meeting_id="t1", transcript=text, **kw)


# ----------------------------------------------------------------- preprocessing


def test_parse_plain_text_speakers_and_timestamps():
    segs = parse_plain_text("[00:01:05] Speaker 1: hello there\nSpeaker 2: hi\nMeeting at 10:30 is fine")
    assert segs[0].speaker == "Speaker 1" and segs[0].start == 65
    assert segs[1].speaker == "Speaker 2" and segs[1].start is None
    assert segs[2].speaker is None  # "10:30 is" must not be read as a speaker


def test_normalize_keeps_indic_joiners_and_collapses_loops():
    ml = "ക്‍ഷ"  # contains ZWJ - must survive
    assert "‍" in normalize_text(ml)
    assert normalize_text("no no no no no no wait") == "no no wait"
    assert normalize_text("really really  good") == "really really good"  # normal repetition kept
    assert normalize_text("hello <unk> world <|en|>") == "hello world"


def test_empty_and_invalid_transcripts():
    with pytest.raises(EmptyTranscriptError):
        prepare(req("   \n  "), SMALL)
    with pytest.raises(InvalidInputError):
        prepare(req("ok"), SMALL)
    with pytest.raises(InvalidInputError):
        prepare(req("1234 5678 9012 3456 !!!"), SMALL)


def test_merge_same_speaker_and_stats():
    p = prepare(req((SAMPLES / "en_standup.txt").read_text()), SMALL)
    assert p.speakers == ["Speaker 1", "Speaker 2", "Speaker 3"]
    assert p.duration and p.duration > 0


def test_chunking_respects_turns_and_budget():
    text = "\n".join(f"[00:{i // 60:02d}:{i % 60:02d}] Speaker {i % 2 + 1}: " + words(40, i) for i in range(80))
    p = prepare(req(text), SMALL)
    chunks = chunk_turns(p.turns, 300, 60)
    assert len(chunks) > 3
    for c in chunks:
        assert sum(estimate_tokens(t.render()) for t in c.turns) <= 300 + 30
    assert chunks[0].context == [] and chunks[1].context  # overlap context only after first
    assert sum(len(c.turns) for c in chunks) >= len(p.turns)


def test_oversized_single_turn_is_split_even_without_punctuation():
    text = "Speaker 1: " + " ".join(f"പദം{i}" for i in range(600))
    p = prepare(req(text), SMALL)
    chunks = chunk_turns(p.turns, 300, 0)
    assert len(chunks) > 1
    assert all(sum(estimate_tokens(t.render()) for t in c.turns) <= 330 for c in chunks)


# ----------------------------------------------------------------- pipeline (mock LLM)


def test_short_meeting_single_pass():
    res = run(summarize(req((SAMPLES / "en_standup.txt").read_text()), settings=SMALL, engine=MockEngine()))
    assert res.status == "completed" and res.chunks_processed == 1
    assert res.summary_language == "English" and res.processing_time_seconds >= 0


BIG = replace(Settings(), llm_backend="mock", chunk_tokens=300, chunk_overlap_tokens=60)


def long_meeting(lines):
    return "\n".join(
        f"[{i * 3 // 3600:02d}:{(i * 3 // 60) % 60:02d}:{i * 3 % 60:02d}] Speaker {i % 3 + 1}: " + words(90, i)
        for i in range(lines)
    )


def test_long_meeting_keeps_every_chunk_decision_through_reduction():
    res = run(summarize(req(long_meeting(40), output_language="ml"), settings=BIG, engine=MockEngine()))
    assert res.chunks_processed > 30
    assert res.summary_language == "Malayalam"
    assert len(res.decisions) == res.chunks_processed          # nothing lost in the merge round(s)
    assert len(res.action_items) == res.chunks_processed
    assert any("hierarchically" in w for w in res.warnings)


def test_absurdly_long_meeting_fails_cleanly():
    from summarization.errors import SummarizationError
    tiny = replace(SMALL, max_reduce_rounds=1)
    with pytest.raises(SummarizationError):
        run(summarize(req(long_meeting(400)), settings=tiny, engine=MockEngine()))


def test_structured_segments_input_and_adapter():
    data = json.loads((SAMPLES / "hi_en_codemixed.json").read_text())
    r = from_transcription_output(data)
    assert r.meeting_id == "demo-hi-en-001" and r.segments[0].language == "hi"
    res = run(summarize(r, settings=SMALL, engine=MockEngine()))
    assert res.transcript_stats.languages == ["hi", "en"]


def test_adapter_shapes():
    assert from_transcription_output("hello world text here", "m1").transcript
    r = from_transcription_output([{"text": "hi there everyone", "speaker": "A", "start": "00:00:10"}], "m1")
    assert r.segments[0].start == 10
    with pytest.raises(InvalidInputError):
        from_transcription_output({"foo": 1}, "m1")
    with pytest.raises(InvalidInputError):
        from_transcription_output("some text here ok", None)


# ----------------------------------------------------------------- JSON robustness


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_strip_thinking():
    assert strip_thinking("<think>blah</think>{\"a\":1}") == '{"a":1}'
    assert strip_thinking("blah</think>{\"a\":1}") == '{"a":1}'


def test_action_item_coerces_small_model_noise():
    a = ActionItem(task="x", assigned_to="null", deadline="N/A", status="In Progress")
    assert a.assigned_to is None and a.deadline is None and a.status == "in_progress"


class ScriptedEngine(BaseEngine):
    name = "scripted"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    async def generate(self, system, user, *, max_tokens=None):
        self.calls += 1
        return self.outputs.pop(0)


GOOD = json.dumps({"meeting_overview": "o", "executive_summary": "e", "key_discussion_points": "single string",
                   "decisions": [], "action_items": ["just a string task"], "unresolved_issues": [], "next_steps": []})


def test_repair_attempt_recovers_and_coerces():
    eng = ScriptedEngine(["not json at all", GOOD])
    res = run(summarize(req("This is a reasonably long enough transcript."), settings=SMALL, engine=eng))
    assert eng.calls == 2
    assert res.key_discussion_points == ["single string"] and res.action_items[0].task == "just a string task"


def test_invalid_output_after_repair_raises():
    eng = ScriptedEngine(["garbage", "still garbage"])
    with pytest.raises(InvalidModelOutputError):
        run(summarize(req("This is a reasonably long enough transcript."), settings=SMALL, engine=eng))


# ----------------------------------------------------------------- real engine (fake HTTP server)


def make_engine(handler, **kw):
    s = replace(SMALL, llm_backend="openai_compat", llm_base_url="http://llm/v1", llm_model="m", max_retries=1, **kw)
    return OpenAICompatEngine(s, transport=httpx.MockTransport(handler))


def test_openai_engine_payload_and_thinking_strip():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think>{\"ok\":1}"},
                                                      "finish_reason": "stop"}]})

    out = run(make_engine(handler).generate("sys", "usr"))
    assert out == '{"ok":1}'
    b = seen["body"]
    assert b["model"] == "m" and b["response_format"] == {"type": "json_object"}
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["auth"] == "Bearer EMPTY"


def test_openai_engine_retries_then_fails_503(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("summarization.llm_engine.asyncio.sleep", no_sleep)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(ModelUnavailableError):
        run(make_engine(handler).generate("s", "u"))
    assert calls["n"] == 2  # 1 try + 1 retry


def test_openai_engine_truncated_output_is_flagged():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"a\":"}, "finish_reason": "length"}]})

    with pytest.raises(InvalidModelOutputError):
        run(make_engine(handler).generate("s", "u"))


# ----------------------------------------------------------------- API


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_engine_dep] = lambda: MockEngine()
    return TestClient(app)


def test_api_summarize_ok(client):
    r = client.post("/summarize", json={"meeting_id": "m1", "transcript": (SAMPLES / "en_standup.txt").read_text()})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("meeting_id", "summary_language", "executive_summary", "key_discussion_points", "decisions",
                "action_items", "unresolved_issues", "next_steps", "processing_time_seconds", "status"):
        assert key in body
    assert body["status"] == "completed"


def test_api_raw_endpoint(client):
    data = json.loads((SAMPLES / "hi_en_codemixed.json").read_text())
    r = client.post("/summarize/raw", json={"transcription": data, "output_language": "hi"})
    assert r.status_code == 200 and r.json()["summary_language"] == "Hindi"


def test_api_validation_errors(client):
    assert client.post("/summarize", json={"meeting_id": "m1"}).status_code == 422          # no input
    r = client.post("/summarize", json={"meeting_id": "m1", "transcript": "   "})
    assert r.status_code == 422 and r.json()["error_code"] == "empty_transcript"
    r = client.post("/summarize/raw", json={"transcription": {"nope": 1}, "meeting_id": "m1"})
    assert r.status_code == 422 and r.json()["status"] == "failed"


def test_api_model_failure_maps_to_502():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_engine_dep] = lambda: ScriptedEngine(["x", "y"])
    r = TestClient(app).post("/summarize", json={"meeting_id": "m1", "transcript": "A long enough transcript line."})
    assert r.status_code == 502 and r.json()["error_code"] == "invalid_model_output"


def test_health_mock(client):
    assert client.get("/summarize/health").json()["reachable"] is True
