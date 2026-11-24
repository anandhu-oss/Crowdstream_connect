import os
import numpy as np
import pytest

from src.utils.constants import QWEN_LANGUAGES
from src.transcription import (FakeASR, IndicASR, InferenceError, InvalidAudio, ModelLoadError, ModelNotConfigured,
                               ModelSet, QwenASR, UnsupportedLanguage, get_model, text_flags)
from src.language import ConfigError

A = np.zeros(16000 * 3, dtype="float32")
SPEECH = np.full(16000 * 3, 0.1, dtype="float32")


class FakeQwenModel:
    """Same call shape as qwen_asr.Qwen3ASRModel.transcribe."""
    def __init__(self, text=" hello ", lang="English", resp="auto"): self.text, self.lang, self.resp, self.last = text, lang, resp, None
    def transcribe(self, audio, language=None):
        self.last = (audio, language)
        if self.resp != "auto": return self.resp
        return [type("R", (), {"text": self.text, "language": self.lang})()]


def test_qwen_passes_language_name_and_cleans_text():
    m = FakeQwenModel(); r = QwenASR(model=m).transcribe(A, "hi")
    assert m.last[1] == ["Hindi"] and r.text == "hello" and r.raw_language == "English" and r.language == "en" and not r.empty


def test_qwen_auto_language_when_none():
    r = QwenASR(model=FakeQwenModel(lang="Hindi")).transcribe(A, None)
    assert r.language == "hi"


@pytest.mark.parametrize("code", ["ta", "ml", "xx"])
def test_qwen_refuses_unsupported_language_instead_of_guessing(code):
    m = FakeQwenModel()
    with pytest.raises(UnsupportedLanguage): QwenASR(model=m).transcribe(A, code)
    assert m.last is None                                    # model never called


def test_every_listed_qwen_language_is_accepted():
    q = QwenASR(model=FakeQwenModel())
    for code in QWEN_LANGUAGES.values(): assert q.supports(code) and q.transcribe(A, code)


def test_unknown_label_is_kept_not_mapped():
    r = QwenASR(model=FakeQwenModel(lang="Klingon")).transcribe(A, None)
    assert r.language is None and r.raw_language == "Klingon"
    r = QwenASR(model=FakeQwenModel(lang="Klingon")).transcribe(A, "hi")
    assert r.language == "hi" and r.raw_language == "Klingon"          # forced language kept, raw label preserved


@pytest.mark.parametrize("resp", [[], None])
def test_empty_model_response_gives_consistent_empty_result(resp):
    r = QwenASR(model=FakeQwenModel(resp=resp)).transcribe(A, None)
    assert r.text == "" and r.empty and r.language is None and r.raw_language is None


@pytest.mark.parametrize("text", ["", "   ", None, 5])
def test_missing_or_blank_text(text):
    r = QwenASR(model=FakeQwenModel(text=text)).transcribe(A, None)
    assert r.text == "" and r.empty and r.raw_language == "English"


def test_missing_language_label():
    r = QwenASR(model=FakeQwenModel(lang=None)).transcribe(A, None)
    assert r.language is None and r.raw_language is None and r.text == "hello"


@pytest.mark.parametrize("bad", [np.zeros(0, dtype="float32"), np.array([np.nan], dtype="float32"), np.zeros((2, 8), dtype="float32"),
                                 [0.1, 0.2], "audio", None])
def test_invalid_audio_rejected_before_the_model(bad):
    m = FakeQwenModel()
    for model in (QwenASR(model=m), FakeASR(), IndicASR("x", ["ml"])):
        with pytest.raises(InvalidAudio): model.transcribe(bad, None if not isinstance(model, IndicASR) else "ml")
    assert m.last is None


def test_inference_failure_hides_content():
    class Boom:
        def transcribe(self, audio, language): raise RuntimeError("secret transcript text")
    with pytest.raises(InferenceError) as e: QwenASR(model=Boom()).transcribe(A, None)
    assert "secret" not in str(e.value)


def test_model_loading_failures_are_clear(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "qwen_asr", None)                 # package missing
    with pytest.raises(ModelLoadError): QwenASR().load()
    with pytest.raises(ModelLoadError): QwenASR().transcribe(A, None)


def test_cuda_unavailable_falls_back_to_cpu_without_crashing(monkeypatch):
    torch = pytest.importorskip("torch"); import types, sys
    got = {}
    fake = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: got.update(k) or FakeQwenModel()))
    monkeypatch.setitem(sys.modules, "qwen_asr", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    q = QwenASR(); q.load()
    assert q.device == "cpu" and got["device_map"] == "cpu" and got["dtype"] == torch.float32


def test_gpu_without_bf16_uses_fp16_and_capability_errors_are_safe(monkeypatch):
    torch = pytest.importorskip("torch"); import types, sys
    got = {}
    fake = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: got.update(k) or FakeQwenModel()))
    monkeypatch.setitem(sys.modules, "qwen_asr", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    QwenASR().load(); assert got["dtype"] == torch.float16 and got["device_map"] == "cuda:0"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    QwenASR().load(); assert got["dtype"] == torch.float16


def test_unload_releases_the_model():
    q = QwenASR(model=FakeQwenModel()); q.unload(); assert q.model is None


def test_models_are_shared_not_loaded_twice():
    ms = ModelSet(); assert ms.get("qwen3_asr") is ms.get("qwen3_asr")
    ms.get("qwen3_asr").model = FakeQwenModel(); ms.unload_all(); assert ms.get("qwen3_asr").model is None
    with pytest.raises(ConfigError): ms.get("nope")


def test_indic_slot_does_not_claim_languages_it_was_not_given():
    assert IndicASR().languages == set() and not IndicASR().supports("ml") and not IndicASR().supports(None)
    i = IndicASR(None, ["ml", "ta"]); assert i.supports("ml") and not i.supports("en") and not i.available


def test_indic_not_configured_is_a_clear_error():
    with pytest.raises(ModelNotConfigured): IndicASR(None, ["ml"]).transcribe(A, "ml")
    with pytest.raises(UnsupportedLanguage): IndicASR(None, ["ml"]).transcribe(A, "en")
    with pytest.raises(NotImplementedError): IndicASR("some/checkpoint", ["ml"]).transcribe(A, "ml")


def test_fake_and_factory():
    f = FakeASR(); assert f.transcribe(A, "en").text == "[3.0s]"
    assert isinstance(get_model("qwen3_asr"), QwenASR) and isinstance(get_model("indic"), IndicASR)
    assert get_model("indic").languages == {"ta", "ml"}
    with pytest.raises(ConfigError): get_model("nope")


# --- opt-in: real models (RUN_INTEGRATION=1, GPU, data/recordings/meeting_rec.mp3) ---------
RUN = os.environ.get("RUN_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1")
def test_real_qwen_on_recording():
    pytest.importorskip("qwen_asr")
    from src.audio import cut, load_track, speech_segments
    p = "data/recordings/meeting_rec.mp3"
    if not os.path.exists(p): pytest.skip("no recording")
    a = load_track(p); seg = speech_segments(a)[0]
    r = QwenASR().transcribe(cut(a, seg), None)
    assert r.text and r.raw_language


@pytest.mark.integration
@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1")
def test_real_indic_model_once_chosen():
    if not get_model("indic").available: pytest.skip("Indic model not chosen yet")


# --- hallucination guard: flags only, never discards --------------------------------------
def test_normal_speech_is_not_flagged():
    for t in ("hello how are you doing today", "no no no no", "I said no, no, no", "ok ok ok", "നമസ്കാരം എല്ലാവർക്കും"):
        assert text_flags(t, SPEECH) == (), t


def test_text_in_silence_is_flagged():
    assert text_flags("thanks for watching", A) == ("text_in_silence",)
    assert text_flags("", A) == () and text_flags("   ", A) == () and text_flags(None, A) == ()


@pytest.mark.parametrize("t", ["thank you thank you thank you thank you", "so so so so so so so so",
                               "i do not know i do not know i do not know"])
def test_repeated_phrase_is_flagged(t):
    assert "repeated_phrase" in text_flags(t, SPEECH)


def test_repeated_characters_without_spaces():
    assert "repeated_phrase" in text_flags("你好" * 8, SPEECH)
    assert text_flags("今天天气很好我们去公园散步吧", SPEECH) == ()


def test_implausibly_dense_text_is_flagged_but_short_audio_is_not_judged():
    assert "implausibly_dense" in text_flags(" ".join(f"w{i}" for i in range(60)), SPEECH)
    assert text_flags("one two three four five six", SPEECH[:1600]) == ()


def test_guard_works_without_audio():
    assert text_flags("a a a a a a a a", None) == ("repeated_phrase",)


def test_asr_result_carries_flags_and_keeps_the_text():
    m = FakeQwenModel(text="thank you thank you thank you thank you")
    r = QwenASR(model=m).transcribe(SPEECH, None)
    assert r.text == "thank you thank you thank you thank you" and r.flags == ("repeated_phrase",) and not r.empty
    r = QwenASR(model=FakeQwenModel(text="hello there")).transcribe(A, None)
    assert r.text == "hello there" and r.flags == ("text_in_silence",)
    assert QwenASR(model=FakeQwenModel()).transcribe(SPEECH, None).flags == ()
