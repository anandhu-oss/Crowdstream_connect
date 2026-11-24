import json
import types
import numpy as np
import pytest

from src.audio import AudioError, Segment, TrackReport
from src.language import LangResult, load_models_config
from src.translation import FakeTranslator
from src.transcription import (AsrResult, FakeASR, ModelSet, PieceResult, SegmentTranscriber, Transcript,
                               TranscriptionPipeline, merge_transcripts)
from src.utils.constants import QWEN_LANGUAGES

SR = 16000


def tone(sec, amp=0.2, f=300): return (amp * np.sin(2 * np.pi * f * np.arange(int(sec * SR)) / SR)).astype("float32")


class StubVAD:
    def __init__(self, regions): self.regions = regions
    def detect(self, audio): return [Segment(a, b) for a, b in self.regions]


class FakeDetector:
    """Returns the language of the next piece from a script; code None = unknown."""
    def __init__(self, codes, text=None): self.codes, self.text, self.i, self.seen = list(codes), text, 0, []
    def detect(self, audio):
        c = self.codes[self.i % len(self.codes)]; self.i += 1; self.seen.append(len(audio))
        return LangResult(c, c, None, "fake", text=self.text)


class CountingQwen(FakeASR):
    name = "qwen3_asr"
    def __init__(self): super().__init__(set(QWEN_LANGUAGES.values()), "from the model")


class Indic(FakeASR):
    name = "indic"
    def __init__(self): super().__init__({"ta", "ml"}, "indic text")


def make(codes, detector_text=None, vad=None, translator=None, models_cfg=None):
    cfg = models_cfg or load_models_config()
    qwen, indic = CountingQwen(), Indic()
    ms = ModelSet(cfg, overrides={"qwen3_asr": qwen, "indic": indic})
    st = SegmentTranscriber(cfg, ms, detector=FakeDetector(codes, detector_text))
    return TranscriptionPipeline(transcriber=st, vad=vad, translator=translator), qwen, indic, st


def run(pipe, audio, **kw): return pipe.transcribe_audio(audio, **kw)


def test_silence_is_skipped_and_timestamps_are_on_the_track_timeline():
    audio = np.concatenate([np.zeros(SR * 2, dtype="float32"), tone(3), np.zeros(SR * 5, dtype="float32"), tone(2)])
    pipe, *_ = make(["en"], vad=StubVAD([(2.0, 5.0), (10.0, 12.0)]))
    t = run(pipe, audio)
    assert [(s.start, s.end) for s in t.segments] == [(2.0, 5.0), (10.0, 12.0)] and t.duration_s == 12.0 and t.speech_s == 5.0


def test_language_is_detected_per_piece_and_routed():
    audio = tone(12)
    pipe, qwen, indic, _ = make(["en", "ml", "hi"], vad=StubVAD([(0, 3), (4, 7), (8, 11)]))
    t = run(pipe, audio)
    assert [s.language for s in t.segments] == ["en", "ml", "hi"] and [s.model for s in t.segments] == ["qwen3_asr", "indic", "qwen3_asr"]
    assert len(qwen.calls) == 2 and len(indic.calls) == 1 and indic.calls[0][1] == "ml"


def test_a_rejected_piece_does_not_stop_the_track():
    pipe, *_ = make(["en", "bn", "en"], vad=StubVAD([(0, 2), (3, 5), (6, 8)]))        # bn: no model supports it
    t = run(pipe, tone(9))
    assert [s.status for s in t.segments] == ["ok", "rejected", "ok"] and "bn" in t.segments[1].reason and t.segments[1].text == ""
    assert "[not transcribed:" in t.to_txt()


def test_unknown_language_is_rejected_by_default_and_best_guess_is_marked_uncertain():
    pipe, *_ = make([None], vad=StubVAD([(0, 2)])); assert run(pipe, tone(3)).segments[0].status == "rejected"
    cfg = load_models_config(); cfg["language"].update(on_uncertain="best_guess", best_guess_model="qwen3_asr")
    pipe, *_ = make([None], vad=StubVAD([(0, 2)]), models_cfg=cfg)
    s = run(pipe, tone(3)).segments[0]
    assert s.status == "ok" and s.uncertain and s.reason and s.language is None


def test_no_second_inference_when_the_detector_already_transcribed():
    pipe, qwen, _, _ = make(["en"], detector_text="already done", vad=StubVAD([(0, 2)]))
    t = run(pipe, tone(3))
    assert t.segments[0].text == "already done" and qwen.calls == []
    pipe, qwen, _, _ = make(["ml"], detector_text="already done", vad=StubVAD([(0, 2)]))      # routed elsewhere: the text is not reused
    assert run(pipe, tone(3)).segments[0].text == "indic text"


def test_forced_language_skips_detection():
    pipe, qwen, _, st = make(["en"], vad=StubVAD([(0, 2)]))
    t = run(pipe, tone(3), language="hi")
    assert st.detector.seen == [] and t.segments[0].language == "hi" and qwen.calls[0][1] == "hi"


def test_flags_from_the_audio_check_and_from_the_output_guard_are_combined():
    from src.transcription import QwenASR
    class M:                                                       # same call shape as qwen_asr
        def transcribe(self, audio, language=None):
            return [types.SimpleNamespace(text="thank you thank you thank you thank you", language="English")]
    cfg = load_models_config()
    st = SegmentTranscriber(cfg, ModelSet(cfg, overrides={"qwen3_asr": QwenASR(model=M()), "indic": Indic()}), detector=FakeDetector(["en"]))
    audio = np.concatenate([tone(2, 0.3), np.zeros(SR), tone(2, 0.3), np.zeros(SR), tone(2, 0.3), np.zeros(SR), tone(2, 0.0001)])
    regions = [(0, 2), (3, 5), (6, 8), (9, 11)]
    t = TranscriptionPipeline(transcriber=st, vad=StubVAD(regions)).transcribe_audio(audio)
    assert t.segments[0].flags == ["repeated_phrase"]
    assert t.segments[3].flags == ["near_silent", "text_in_silence", "repeated_phrase"]
    assert t.segments[3].text and "check: near_silent" in t.to_txt()          # flagged, never removed


def test_silent_track_gives_an_empty_transcript_with_the_warning():
    rep = TrackReport(5.0, 0.0, -180, -180, 0.0, True, ["silent track"])
    pipe, *_ = make(["en"], vad=StubVAD([(0, 2)]))
    t = run(pipe, np.zeros(SR * 5, dtype="float32"), report=rep)
    assert t.segments == [] and t.warnings == ["silent track"] and t.speech_s == 0 and t.to_txt() == ""


def test_translation_added_when_language_differs():
    pipe, *_ = make(["hi", "en"], vad=StubVAD([(0, 2), (3, 5)]), translator=FakeTranslator())
    t = run(pipe, tone(6), translate_to="en")
    assert t.segments[0].translation.startswith("[en]") and t.segments[0].translation_language == "en"
    assert t.segments[1].translation is None                          # already English


def test_translation_failure_is_flagged_not_fatal():
    class Bad(FakeTranslator):
        def translate(self, *a, **k): raise RuntimeError("down")
    pipe, *_ = make(["hi"], vad=StubVAD([(0, 2)]), translator=Bad())
    s = run(pipe, tone(3), translate_to="en").segments[0]
    assert s.text and s.translation is None and "translation_failed" in s.flags


def test_writers_and_save(tmp_path):
    pipe, *_ = make(["hi", "en"], vad=StubVAD([(1.5, 3.25), (65.0, 70.0)]), translator=FakeTranslator())
    t = run(pipe, tone(71), speaker="Asha", translate_to="en")
    txt = t.to_txt(); assert "[00:00:01.500 - 00:00:03.250] (hi) Asha:" in txt and "-> (en)" in txt and "[00:01:05.000" in txt
    srt = t.to_srt(); assert srt.startswith("1\n00:00:01,500 --> 00:00:03,250\n") and "\n2\n" in srt
    assert t.to_vtt().startswith("WEBVTT\n\n00:00:01.500 --> 00:00:03.250")
    d = json.loads(t.to_json()); assert d["speaker"] == "Asha" and d["segments"][0]["language"] == "hi" and d["rtf"] is not None
    paths = t.save(str(tmp_path / "o"), "track", ("json", "txt", "srt", "vtt")); assert len(paths) == 4 and all(__import__("os").path.exists(p) for p in paths)
    with pytest.raises(ValueError): t.save(str(tmp_path), "x", ("doc",))


def test_rejected_segments_are_left_out_of_subtitles():
    pipe, *_ = make(["en", "bn"], vad=StubVAD([(0, 2), (3, 5)]))
    srt = run(pipe, tone(6)).to_srt(); assert srt.count("-->") == 1


def test_merge_tracks_orders_by_start_and_names_speakers():
    a, *_ = make(["en"], vad=StubVAD([(0, 2), (10, 12)])); b, *_ = make(["hi"], vad=StubVAD([(5, 7)]))
    m = merge_transcripts([run(a, tone(13), speaker="Asha"), run(b, tone(13), speaker="Ben")])
    assert [(s.speaker, s.start) for s in m.segments] == [("Asha", 0.0), ("Ben", 5.0), ("Asha", 10.0)] and m.duration_s == 13.0
    assert merge_transcripts([]).segments == []


def test_real_files_roundtrip_through_load_track(tmp_path):
    import soundfile as sf
    p = tmp_path / "t.wav"; sf.write(p, tone(4), SR)
    pipe, *_ = make(["en"], vad=StubVAD([(0.5, 3.5)]))
    t = pipe.transcribe_track(p, speaker="x"); assert t.track == str(p) and len(t.segments) == 1
    with pytest.raises(AudioError): pipe.transcribe_track(tmp_path / "missing.wav")
