import numpy as np
import pytest

from evaluation.latency import LatencyTracker, Timer, percentiles, rtf, summarize_captions
from src.audio import AudioError, UtteranceStream, load_config
from src.captions import LiveCaptioner
from src.translation import FakeTranslator
from src.transcription import PieceResult

SR, FRAME = 16000, 512
CFG = load_config()


class EnergyStep:
    """Stands in for Silero in tests: loud frame = speech; 13 quiet frames (~416 ms) end an utterance."""
    def __init__(self, quiet_frames=13): self.n, self.speech, self.quiet, self.last_end, self.q = 0, False, 0, 0.0, quiet_frames
    def __call__(self, frame):
        t0 = self.n * FRAME / SR; self.n += 1; t1 = self.n * FRAME / SR
        if np.sqrt(np.mean(frame ** 2)) > 0.02:
            self.quiet, self.last_end = 0, t1
            if not self.speech: self.speech = True; return {"start": t0}
        elif self.speech:
            self.quiet += 1
            if self.quiet >= self.q: self.speech = False; return {"end": self.last_end}


def tone(sec, amp=0.3): return (amp * np.sin(2 * np.pi * 300 * np.arange(int(sec * SR)) / SR)).astype("float32")
def quiet(sec): return np.zeros(int(sec * SR), dtype="float32")
def feed_all(obj, audio, chunk=1600):
    out = []
    for i in range(0, len(audio), chunk): out += obj.push(audio[i:i + chunk]) if isinstance(obj, UtteranceStream) else obj.feed(audio[i:i + chunk])
    return out


# --- utterance stream --------------------------------------------------------------------
def test_one_utterance_comes_out_after_the_pause_with_correct_times():
    s = UtteranceStream(CFG, EnergyStep())
    out = feed_all(s, np.concatenate([quiet(1), tone(2), quiet(1.5)]))
    assert len(out) == 1 and out[0].start == pytest.approx(1.0, abs=0.05) and out[0].end == pytest.approx(3.0, abs=0.05)
    assert abs(len(out[0].audio) / SR - out[0].duration) < 0.01 and not out[0].forced


def test_chunk_size_does_not_change_the_result():
    audio = np.concatenate([quiet(0.5), tone(1.2), quiet(1), tone(0.8), quiet(1)])
    ref = feed_all(UtteranceStream(CFG, EnergyStep()), audio, 1600)
    for size in (1, 100, 513, 4000, 99999):
        got = feed_all(UtteranceStream(CFG, EnergyStep()), audio, size)
        assert [(round(u.start, 2), round(u.end, 2)) for u in got] == [(round(u.start, 2), round(u.end, 2)) for u in ref] and len(got) == 2


def test_nothing_in_silence_and_memory_stays_bounded():
    s = UtteranceStream(CFG, EnergyStep())
    assert feed_all(s, quiet(60)) == [] and len(s._buf) < SR * 1.0 and s.current() is None


def test_current_shows_the_growing_utterance():
    s = UtteranceStream(CFG, EnergyStep()); feed_all(s, np.concatenate([quiet(0.5), tone(1.5)]))
    start, audio = s.current(); assert s.in_speech and start == pytest.approx(0.5, abs=0.05) and len(audio) / SR == pytest.approx(1.5, abs=0.1)


def test_long_speech_is_cut_at_the_maximum_and_continues():
    s = UtteranceStream(CFG, EnergyStep()); out = feed_all(s, np.concatenate([tone(60), quiet(1)]))
    assert len(out) >= 3 and all(u.duration <= 25.05 for u in out) and out[0].forced
    assert out[0].end == pytest.approx(out[1].start, abs=0.05)                      # nothing lost between the pieces


def test_flush_finishes_an_utterance_in_progress():
    s = UtteranceStream(CFG, EnergyStep()); feed_all(s, np.concatenate([quiet(0.5), tone(2)]))
    u = s.flush(); assert u is not None and u.duration == pytest.approx(2.0, abs=0.1) and s.flush() is None


@pytest.mark.parametrize("bad", [np.zeros((2, 5), dtype="float32"), np.array([np.nan] * 600, dtype="float32"), [0.1] * 600, "x"])
def test_bad_chunks_are_rejected(bad):
    with pytest.raises(AudioError): UtteranceStream(CFG, EnergyStep()).push(bad)
    assert UtteranceStream(CFG, EnergyStep()).push(np.zeros(0, dtype="float32")) == []


def test_missing_silero_is_a_clear_error(monkeypatch):
    import sys
    from src.audio import VADError
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    with pytest.raises(VADError): UtteranceStream(CFG)


# --- live captions -----------------------------------------------------------------------
class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


class T:
    """Transcriber stand-in: language from a list, text 'utt N'."""
    def __init__(self, langs=("hi",), status="ok", flags=()): self.langs, self.status, self.flags, self.n, self.sizes = list(langs), status, flags, 0, []
    def transcribe(self, audio, language=None):
        self.sizes.append(len(audio)); lang = self.langs[self.n % len(self.langs)]; self.n += 1
        return PieceResult(f"utt {self.n}" if self.status == "ok" else "", lang, lang, "qwen3_asr", self.status, flags=self.flags)


def run_live(cap, audio, clock, chunk=1600):
    out = []
    for i in range(0, len(audio), chunk):
        clock.t += chunk / SR; out += cap.feed(audio[i:i + chunk])
    return out + cap.flush()


def test_final_captions_with_translation_and_languages():
    clock = Clock(); cap = LiveCaptioner(T(("hi", "en")), FakeTranslator(), "en", CFG, EnergyStep(), clock)
    out = run_live(cap, np.concatenate([tone(2), quiet(1), tone(1.5), quiet(1)]), clock)
    assert [c.kind for c in out] == ["final", "final"] and out[0].translation == "[en] utt 1" and out[0].translation_language == "en"
    assert out[1].translation is None and out[1].language == "en"               # already English


def test_delay_numbers_are_measured():
    clock = Clock(); cap = LiveCaptioner(T(), FakeTranslator(delay_s=0.02), "en", CFG, EnergyStep(), clock)
    out = run_live(cap, np.concatenate([tone(2), quiet(1.5)]), clock)
    c = out[0]
    assert 0.35 <= c.endpoint_wait_s <= 0.6                                       # the ~416 ms of silence that had to pass
    assert c.translation_s >= 0.02 and c.estimated_delay_s == pytest.approx(c.endpoint_wait_s + c.asr_s + c.translation_s)
    assert c.wall_delay_s == pytest.approx(c.endpoint_wait_s, abs=0.2)             # clock advanced in step with the audio


def test_no_translation_when_not_asked_and_failures_are_flagged():
    clock = Clock(); out = run_live(LiveCaptioner(T(), FakeTranslator(), None, CFG, EnergyStep(), clock), np.concatenate([tone(1), quiet(1)]), clock)
    assert out[0].translation is None and out[0].translation_s == 0
    class Bad(FakeTranslator):
        def translate(self, *a, **k): raise RuntimeError("x")
    clock = Clock(); out = run_live(LiveCaptioner(T(), Bad(), "en", CFG, EnergyStep(), clock), np.concatenate([tone(1), quiet(1)]), clock)
    assert "translation_failed" in out[0].flags and out[0].text


def test_rejected_utterance_is_reported_not_translated():
    clock = Clock(); out = run_live(LiveCaptioner(T(status="rejected"), FakeTranslator(), "en", CFG, EnergyStep(), clock),
                                    np.concatenate([tone(1), quiet(1)]), clock)
    assert out[0].status == "rejected" and out[0].translation is None


def test_forced_cut_is_flagged():
    clock = Clock(); out = run_live(LiveCaptioner(T(), None, None, CFG, EnergyStep(), clock), np.concatenate([tone(30), quiet(1)]), clock)
    assert "cut_at_max_length" in out[0].flags


def test_partial_captions_grow_and_the_final_follows():
    cfg = {**CFG, "streaming": {**CFG["streaming"], "partial_interval_s": 0.5, "min_partial_seconds": 0.5}}
    clock = Clock(); t = T(); out = run_live(LiveCaptioner(t, None, None, cfg, EnergyStep(), clock), np.concatenate([tone(3), quiet(1)]), clock)
    parts = [c for c in out if c.kind == "partial"]; finals = [c for c in out if c.kind == "final"]
    assert len(parts) >= 3 and len(finals) == 1 and all(p.utterance == finals[0].utterance for p in parts)
    assert parts[0].end < parts[-1].end <= finals[0].end + 0.1
    assert t.sizes[0] < t.sizes[-1]                                                # partials were shorter than the final


def test_partials_are_off_by_default():
    assert CFG["streaming"]["partial_interval_s"] == 0
    clock = Clock(); out = run_live(LiveCaptioner(T(), None, None, CFG, EnergyStep(), clock), np.concatenate([tone(3), quiet(1)]), clock)
    assert all(c.kind == "final" for c in out)


# --- latency helpers ---------------------------------------------------------------------
def test_latency_helpers():
    assert rtf(5, 10) == 0.5 and rtf(1, 0) is None and rtf(1, None) is None
    p = percentiles([1, 2, 3, 4, 100]); assert p["p50"] == 3 and p["max"] == 100 and p["n"] == 5 and percentiles([]) == {}
    tr = LatencyTracker(); tr.add("a", 1); tr.add("a", 3); assert tr.summary()["a"]["mean"] == 2
    with Timer() as t: pass
    assert t.elapsed >= 0


def test_summarize_captions():
    clock = Clock(); out = run_live(LiveCaptioner(T(), FakeTranslator(), "en", CFG, EnergyStep(), clock),
                                    np.concatenate([tone(1), quiet(1), tone(1), quiet(1)]), clock)
    s = summarize_captions(out); assert s["finals"] == 2 and s["partials"] == 0 and "estimated_delay_s" in s and s["translation_s"]["n"] == 2
