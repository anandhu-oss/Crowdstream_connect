"""Language-ID fusion -> routing -> ASR -> translation -> upload response, with fake remote servers (no GPU, no network)."""
import io
import types

import numpy as np
import pytest
import soundfile as sf

from src.audio import Segment
from src.language import LangResult, LanguageDetector, load_models_config
from src.translation import FakeTranslator
from src.transcription import (FakeASR, IndicASR, LiveSession, ModelSet, QwenASR, SegmentTranscriber, TranscriptionPipeline,
                               UnsupportedLanguage)
from src.transcription.indic_asr import FLEX_LANGUAGES
from src.utils.constants import QWEN_LANGUAGES

SR = 16000


def tone(sec, amp=0.2, f=300): return (amp * np.sin(2 * np.pi * f * np.arange(int(sec * SR)) / SR)).astype("float32")


class StubVAD:
    def __init__(self, regions): self.regions = regions
    def detect(self, audio): return [Segment(a, b) for a, b in self.regions]


class Resp:
    def __init__(self, payload, status=200): self.payload, self.status_code, self.headers, self.ok = payload, status, {}, status < 400
    def json(self): return self.payload


class Session:
    """Stands in for requests.Session; records what the backend sends."""
    def __init__(self, reply): self.reply, self.calls = reply, []
    def post(self, url, headers=None, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data, "files": files})
        r = self.reply(url, data) if callable(self.reply) else self.reply
        return r if isinstance(r, Resp) else Resp(r)
    def close(self): pass


def indic(reply, **kw):
    s = Session(reply)
    return IndicASR("bodhan-ai/Indic-Transcribe-Flex", ["ta", "ml", "hi"], "remote", "http://kaggle.test", "secret", session=s, **kw), s


class Det:
    """Detector double that returns scripted LangResults (a fused result, as the real compare detector would)."""
    def __init__(self, results): self.results, self.i, self.seen = list(results), 0, []
    def detect(self, audio):
        r = self.results[self.i % len(self.results)]; self.i += 1; self.seen.append(len(audio)); return r


def fused(qwen=None, vox=None, vconf=0.9, qtext=None, qconf=None):
    from src.language import fuse
    q = LangResult(qwen, qwen, qconf, "qwen", text=qtext, qwen_code=qwen,
                   suspect={"ms": ("ml",)}.get(qwen, ()))
    v = LangResult(vox, vox, vconf, "voxlingua")
    return fuse(q, v, load_models_config()["language"] | {"min_confidence": 0.5}, {"en", "hi", "ml", "ta", "de", "fr"})


# ---- Indic-Transcribe-Flex: request format, auth, validation, response parsing -------------------------------------------
def test_indic_remote_sends_language_and_mode_only_with_bearer_auth():
    m, s = indic({"text": " നമസ്കാരം ", "language": "ml", "mode": "mixed"})
    r = m.transcribe(tone(2), "ml")
    c = s.calls[0]
    assert c["url"] == "http://kaggle.test/transcribe-indic" and c["headers"] == {"Authorization": "Bearer secret"}
    assert c["data"] == {"language": "ml", "mode": "mixed"} and "decoder" not in c["data"]
    assert r.text == "നമസ്കാരം" and r.model == "indic" and r.language == "ml" and r.mode == "mixed"


@pytest.mark.parametrize("mode", ["native", "mixed", "romanized"])
def test_indic_mode_is_per_call_and_validated(mode):
    m, s = indic({"text": "x y"})
    assert m.transcribe(tone(2), "ta", mode=mode).mode == mode and s.calls[0]["data"]["mode"] == mode


def test_indic_rejects_bad_mode_unsupported_language_and_flex_unknown_languages():
    m, _ = indic({"text": "x"})
    with pytest.raises(ValueError): m.transcribe(tone(2), "ml", mode="ctc")
    with pytest.raises(UnsupportedLanguage): m.transcribe(tone(2), "en")
    with pytest.raises(ValueError): IndicASR("x", ["ml", "xx"])                       # not a Flex language
    with pytest.raises(ValueError): IndicASR("x", ["ml"], mode="rnnt")
    assert {"ml", "ta", "hi", "en"} <= FLEX_LANGUAGES and len(FLEX_LANGUAGES) == 27


def test_indic_flags_when_server_answers_for_a_different_language_or_mode():
    m, _ = indic({"text": "hello there", "language": "hi", "mode": "native"})
    assert {"indic_language_mismatch", "indic_mode_mismatch"} <= set(m.transcribe(tone(2), "ml").flags)


def test_indic_auth_failure_is_a_clear_error_that_allows_fallback():
    from src.transcription import ModelLoadError
    m, _ = indic(Resp({"detail": "no"}, 401))
    with pytest.raises(ModelLoadError, match="API key"): m.transcribe(tone(2), "ml")


def test_indic_long_audio_is_cut_below_the_model_limit():
    m, s = indic({"text": "word"})
    m.transcribe(tone(70), "ml")
    assert len(s.calls) >= 3
    for c in s.calls:
        data, _ = sf.read(io.BytesIO(c["files"]["file"][1].getvalue())); assert len(data) / SR <= 30.0


# ---- Qwen: confidence and detected language come back in the same structure ---------------------------------------------------
def test_qwen_remote_returns_language_confidence_when_server_reports_it():
    s = Session({"text": "hello world", "detected_language": "English", "language_confidence": 0.93})
    q = QwenASR(api_url="http://q.test", api_key="k", session=s)
    r = q.transcribe(tone(2), None)
    assert r.language == "en" and r.language_confidence == 0.93 and r.model == "qwen3_asr"
    assert s.calls[0]["data"] == {"source_language": "auto"}
    q2 = QwenASR(api_url="http://q.test", api_key="k", session=Session({"text": "hi", "detected_language": "Hindi"}))
    assert q2.transcribe(tone(2), None).language_confidence is None


# ---- routing + transcript reuse -----------------------------------------------------------------------------------------------
def pipeline_with(results, vad, qwen=None, ind=None, translator=None):
    cfg = load_models_config()
    qwen = qwen or FakeASR(set(QWEN_LANGUAGES.values()), "qwen text"); qwen.name = "qwen3_asr"
    ind = ind or FakeASR({"ta", "ml", "hi"}, "indic text"); ind.name = "indic"
    ms = ModelSet(cfg, overrides={"qwen3_asr": qwen, "indic": ind})
    st = SegmentTranscriber(cfg, ms, detector=Det(results))
    return TranscriptionPipeline(transcriber=st, vad=StubVAD(vad), translator=translator, translate=True if translator else None), qwen, ind


def test_agreement_routes_to_qwen_and_reuses_the_detection_transcript_without_a_second_call():
    pipe, qwen, ind = pipeline_with([fused("en", "en", qtext="from detection")], [(0, 2)])
    seg = pipe.transcribe_audio(tone(3)).segments[0]
    assert seg.text == "from detection" and seg.model == "qwen3_asr" and qwen.calls == [] and ind.calls == []
    assert seg.detection["agreed"] is True and seg.detection["voxlingua"]["code"] == "en"


def test_voxlingua_overruling_qwen_drops_the_qwen_transcript_and_runs_the_right_model():
    # Qwen heard Hindi, VoxLingua says Malayalam (which Qwen cannot recognise): Indic must transcribe, not Qwen's Hindi text
    pipe, qwen, ind = pipeline_with([fused("hi", "ml", qtext="wrong hindi text")], [(0, 2)])
    seg = pipe.transcribe_audio(tone(3)).segments[0]
    assert seg.language == "ml" and seg.model == "indic" and seg.text == "indic text" and qwen.calls == []
    assert ind.calls[0][1] == "ml"


def test_qwen_selected_but_voxlingua_overrides_language_reruns_qwen_with_forced_language():
    pipe, qwen, ind = pipeline_with([fused("de", "en", vconf=0.95, qtext="deutsch?")], [(0, 2)])
    seg = pipe.transcribe_audio(tone(3)).segments[0]
    assert seg.language == "en" and seg.text == "qwen text" and qwen.calls[0][1] == "en"


def test_hindi_falls_back_to_indic_when_qwen_fails():
    from src.transcription import InferenceError
    class Broken(FakeASR):
        def transcribe(self, *a, **k): raise InferenceError("down")
    pipe, qwen, ind = pipeline_with([fused("hi", "hi")], [(0, 2)], qwen=Broken(set(QWEN_LANGUAGES.values())))
    seg = pipe.transcribe_audio(tone(3)).segments[0]
    assert seg.model == "indic" and seg.text == "indic text" and seg.detection["fallback_from"] == "qwen3_asr"


# ---- code-mixed recording: language is decided per chunk, never for the whole file ---------------------------------------
def test_code_mixed_malayalam_english_recording_follows_each_chunk():
    seq = [fused("hi", "ml"), fused("en", "en", qtext="so the budget"), fused("ms", "ml"), fused("en", "en", qtext="next item")]
    pipe, qwen, ind = pipeline_with(seq, [(0, 2), (3, 5), (6, 8), (9, 11)])
    t = pipe.transcribe_audio(tone(12))
    assert [s.language for s in t.segments] == ["ml", "en", "ml", "en"]
    assert [s.model for s in t.segments] == ["indic", "qwen3_asr", "indic", "qwen3_asr"]


# ---- unresolved language: settled from neighbours, but only to a language a detector proposed ------------------------------
def test_unresolved_chunk_takes_the_neighbouring_language_when_it_was_a_candidate():
    seq = [fused("hi", "hi"), fused("hi", "en", vconf=0.2), fused("hi", "hi")]       # VoxLingua nearly clueless and Qwen disagrees -> unresolved
    pipe, qwen, ind = pipeline_with(seq, [(0, 2), (3, 5), (6, 8)])
    t = pipe.transcribe_audio(tone(9))
    mid = t.segments[1]
    assert mid.status == "ok" and mid.language == "hi" and mid.uncertain and "neighbouring chunk" in mid.reason
    assert mid.detection["resolution"] == "neighbouring chunk"


def test_unresolved_chunk_is_not_forced_into_a_language_nobody_proposed():
    seq = [fused("hi", "hi"), fused("de", "en", vconf=0.2), fused("hi", "hi")]       # de vs en; neighbours are Hindi -> stay rejected
    pipe, *_ = pipeline_with(seq, [(0, 2), (3, 5), (6, 8)])
    mid = pipe.transcribe_audio(tone(9)).segments[1]
    assert mid.status == "rejected" and "disagree" in mid.reason


def test_translation_is_separate_from_the_original_text():
    pipe, *_ = pipeline_with([fused("hi", "hi", qtext="नमस्ते"), fused("en", "en", qtext="hello")], [(0, 2), (3, 5)], translator=FakeTranslator())
    t = pipe.transcribe_audio(tone(6), translate_to="en")
    assert t.segments[0].text == "नमस्ते" and t.segments[0].translation.startswith("[en]") and t.segments[0].translation_language == "en"
    assert t.segments[1].translation is None


# ---- live: language switching and detection info on captions -----------------------------------------------------------------
def make_live(results):
    pipe, qwen, ind = pipeline_with(results, [])
    return pipe, LiveSession(pipe, "auto", "en", clock=lambda: 0.0)


def test_live_final_follows_a_language_switch_and_reports_model_and_detection():
    pipe, live = make_live([fused("en", "en", qtext="hello everyone"), fused("hi", "ml"), fused("hi", "ml")])
    seen = []
    for dur in (3.0, 5.0, 5.0):
        sid = live.captions.open(0, dur)
        t = types.SimpleNamespace(kind="final", segment_id=sid, audio=tone(dur), start=0.0, end=dur, forced=False, context_s=0.0,
                                  gate_flags=())
        out = live.run(t)
        seen += [m for m in out.messages if m["type"] == "segment"]
    langs = [(m["source_language"], m["asr_model"]) for m in seen]
    assert langs[0] == ("en", "qwen3_asr") and langs[-1] == ("ml", "indic")
    assert seen[-1]["detection"]["voxlingua"]["code"] == "ml" and seen[-1]["detected_language"] == "ml"


def test_live_unresolved_chunk_keeps_the_locked_language_and_says_so():
    pipe, live = make_live([fused("en", "en", qtext="hello everyone"), fused("hi", "en", vconf=0.2)])
    for i, dur in enumerate((3.0, 3.0)):
        sid = live.captions.open(0, dur)
        out = live.run(types.SimpleNamespace(kind="final", segment_id=sid, audio=tone(dur), start=0.0, end=dur, forced=False,
                                             context_s=0.0, gate_flags=()))
    m = [x for x in out.messages if x["type"] == "segment"][0]
    assert m["source_language"] == "en" and m["uncertain"] and "kept en" in m["reason"]


# ---- FastAPI upload endpoint: segments/results structure the frontend reads ----------------------------------------------------
def test_upload_endpoint_returns_original_and_translated_text_in_the_frontend_structure(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import service.app as appmod
    seq = [fused("hi", "ml"), fused("en", "en", qtext="hello team"), fused("hi", "en", vconf=0.2)]
    p, *_ = pipeline_with(seq, [(0, 2), (3, 5), (6, 8)], translator=FakeTranslator())
    monkeypatch.setattr(appmod, "pipeline", p)
    monkeypatch.setattr(appmod, "TRANSLATION_ENABLED", True)
    buf = io.BytesIO(); sf.write(buf, tone(9), SR, format="WAV"); buf.seek(0)
    r = TestClient(appmod.app).post("/transcribe", files={"file": ("a.wav", buf, "audio/wav")},
                                    data={"source_language": "auto", "target_language": "en"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["segments"] == d["results"] and len(d["segments"]) == 3
    s0, s1, s2 = d["segments"]
    assert s0["original_text"] == "indic text" and s0["translated_text"].startswith("[en]") and s0["asr_model"] == "indic"
    assert s0["detected_language"] == "ml" and s0["language_detection"]["voxlingua"]["code"] == "ml"
    assert s1["original_text"] == "hello team" and s1["translation_status"] == "same_language"
    assert s2["transcription_status"] == "ok" and s2["original_text"] and s2["uncertain"]      # settled from the neighbouring chunk, not blank
    assert d["original_transcript"] and "ml" in d["detected_languages"]


# ---- VoxLingua hosted on the GPU notebook (/identify-language) ----------------------------------------------------------------
def remote_vox(reply, supported=("en", "hi", "ml", "ta", "de", "fr")):
    from src.language import RemoteVoxLinguaDetector
    s = Session(reply)
    return RemoteVoxLinguaDetector("http://k.test", "key", supported=set(supported), session=s), s


def test_remote_voxlingua_full_probabilities_use_closed_set_decision():
    # raw top-1 is only 0.30 (would fail a 0.5 gate); among the routable languages Malayalam is clearly ahead
    probs = {"ml: Malayalam": 0.30, "mr: Marathi": 0.25, "ne: Nepali": 0.2, "hi: Hindi": 0.05, "en: English": 0.04, "ta: Tamil": 0.06,
             "ur: Urdu": 0.05, "yue: Cantonese": 0.05}
    v, s = remote_vox({"success": True, "predicted_language": "ml: Malayalam", "probabilities": probs})
    r = v.detect(tone(3))
    assert r.code == "ml" and r.confidence > 0.55 and abs(sum(r.probs.values()) - 1) < 1e-6
    c = s.calls[0]
    assert c["url"] == "http://k.test/identify-language" and c["headers"] == {"Authorization": "Bearer key"}


def test_remote_voxlingua_old_top_candidates_with_log_probabilities():
    cands = [{"language": "ta: Tamil", "confidence": -0.357}, {"language": "ml: Malayalam", "confidence": -2.3},
             {"language": "hi: Hindi", "confidence": -3.5}]
    v, _ = remote_vox({"predicted_language": "ta: Tamil", "confidence": -0.357, "top_candidates": cands})
    r = v.detect(tone(3))
    assert r.code == "ta" and r.confidence > 0.7


def test_remote_voxlingua_failure_degrades_to_qwen_alone_and_is_reported():
    from src.language import DetectorError
    v, _ = remote_vox(Resp({"detail": "bad key"}, 401))
    with pytest.raises(DetectorError): v.detect(tone(3))
    cfg = load_models_config(); cfg["language"]["detector"] = "compare"
    d = LanguageDetector(cfg, qwen=type("Q", (), {"detect": lambda self, a: LangResult("en", "English", None, "qwen", text="hi", qwen_code="en")})(),
                         voxlingua=v)
    r = d.detect(tone(3))
    assert r.code == "en" and "single detector" in r.note and d.status()["degraded"] == "voxlingua"


def test_voxlingua_backend_is_remote_when_url_and_key_are_set(monkeypatch):
    from src.language import RemoteVoxLinguaDetector, VoxLinguaDetector
    cfg = load_models_config(); cfg["language"]["detector"] = "compare"
    dummy = type("Q", (), {"detect": lambda self, a: None})()
    monkeypatch.delenv("VOXLINGUA_API_URL", raising=False); monkeypatch.delenv("VOXLINGUA_API_KEY", raising=False)
    assert type(LanguageDetector(cfg, qwen=dummy)._vox) is VoxLinguaDetector
    monkeypatch.setenv("VOXLINGUA_API_URL", "http://k.test"); monkeypatch.setenv("VOXLINGUA_API_KEY", "x")
    assert isinstance(LanguageDetector(cfg, qwen=dummy)._vox, RemoteVoxLinguaDetector)


def test_remote_voxlingua_exact_reply_of_the_current_notebook_cell():
    # the notebook cell returns the top 5 only, as rounded log-probabilities, labels like "ml: Malayalam"
    reply = {"success": True, "model": "VoxLingua107", "sample_rate": 16000, "predicted_language": "mr: Marathi", "confidence": -0.9,
             "top_candidates": [{"language": "mr: Marathi", "confidence": -0.9}, {"language": "ml: Malayalam", "confidence": -1.2},
                                {"language": "ne: Nepali", "confidence": -2.5}, {"language": "hi: Hindi", "confidence": -3.0},
                                {"language": "ta: Tamil", "confidence": -3.3}]}
    r = remote_vox(reply)[0].detect(tone(3))
    assert r.code == "ml" and 0.6 < r.confidence < 0.9                 # Marathi is not routable, Malayalam wins among the routable ones
    reply["top_candidates"] = [{"language": "mr: Marathi", "confidence": -0.1}, {"language": "ne: Nepali", "confidence": -3.0},
                               {"language": "bn: Bengali", "confidence": -4.0}, {"language": "ml: Malayalam", "confidence": -5.0}]
    assert remote_vox(reply)[0].detect(tone(3)).code == "mr"           # clearly some other language stays unroutable
