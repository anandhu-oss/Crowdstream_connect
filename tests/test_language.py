import numpy as np
import pytest

from src.utils.constants import CONFUSABLE, QWEN_LANGUAGES
from src.language import (ConfigError, DetectorError, LangResult, LanguageDetector, QwenDetector, VoxLinguaDetector,
                          load_models_config, parse_confidence, parse_voxlingua, uncertain_reason)
from src.transcription import (AsrResult, FakeASR, LanguageRejected, ModelNotConfigured, ModelSet, Router,
                               UnsupportedLanguage)

A = np.zeros(16000 * 40, dtype="float32")


class Fake:
    def __init__(self, name, code, raw=None, conf=None, suspect=(), text=None, fail=None):
        self.name, self.code, self.raw, self.conf, self.suspect, self.text, self.fail, self.seen = name, code, raw or code, conf, suspect, text, fail, None
    def detect(self, audio):
        if self.fail: raise DetectorError(self.fail)
        self.seen = len(audio)
        return LangResult(self.code, self.raw, self.conf, self.name, text=self.text, suspect=self.suspect)


def cfg(**over):
    c = load_models_config(); c["language"].update(over); return c


def det(mode, q=None, v=None, **over):
    return LanguageDetector(cfg(detector=mode, **over), qwen=q or Fake("qwen", "en", "English"),
                            voxlingua=v or Fake("voxlingua", "ml", "ml: Malayalam", 0.9))


# --- label handling ----------------------------------------------------------------------
def test_qwen_language_table_matches_model_card_size():
    assert len(QWEN_LANGUAGES) == 30 and QWEN_LANGUAGES["Hindi"] == "hi" and QWEN_LANGUAGES["Malay"] == "ms"
    assert "ml" not in QWEN_LANGUAGES.values() and "ta" not in QWEN_LANGUAGES.values()


@pytest.mark.parametrize("label,code", [("ml: Malayalam", "ml"), (" HI : Hindi", "hi"), ("fil: Filipino", "fil"),
                                         ("", None), ("Malayalam", None), (None, None), (7, None), ("1: x", None), ("toolongcode: x", None)])
def test_voxlingua_label_parsing(label, code):
    assert parse_voxlingua(label) == code


@pytest.mark.parametrize("v,out", [(0.9, 0.9), (0.0, 1.0), (-0.1, pytest.approx(0.905, abs=0.01)), (-30.0, pytest.approx(0, abs=1e-6)),
                                   (float("nan"), None), (float("inf"), None), (5.0, None)])
def test_confidence_parsing(v, out):
    assert parse_confidence(v) == out


# --- detector interface and modes --------------------------------------------------------
def test_backend_selection_and_window():
    q, v = Fake("qwen", "en"), Fake("voxlingua", "ml", conf=0.9)
    assert det("qwen", q, v).detect(A).backend == "qwen"
    assert det("voxlingua", q, v).detect(A).backend == "voxlingua"
    assert q.seen == 20 * 16000


@pytest.mark.parametrize("bad", [np.zeros(0, dtype="float32"), np.zeros((2, 5), dtype="float32"),
                                 np.array([np.nan] * 100, dtype="float32"), [0.0] * 10])
def test_empty_or_invalid_audio_rejected(bad):
    for mode in ("qwen", "voxlingua", "compare"):
        with pytest.raises(ValueError): det(mode).detect(bad)


def test_qwen_detector_keeps_transcript_from_the_single_inference():
    class Asr:
        calls = 0
        def transcribe(self, audio, language):
            Asr.calls += 1; assert language is None
            return AsrResult("namaste", "hi", "Hindi")
    r = QwenDetector(Asr()).detect(A[:16000 * 5])
    assert r.code == "hi" and r.text == "namaste" and Asr.calls == 1


def test_qwen_malay_is_kept_but_marked_suspect_not_silently_malayalam():
    class Asr:
        def transcribe(self, audio, language): return AsrResult("x", "ms", "Malay")
    r = QwenDetector(Asr()).detect(A[:16000 * 5])
    assert r.code == "ms" and r.suspect == ("ml",) and "ml" in uncertain_reason(r, 0.5)
    assert CONFUSABLE["ms"] == ("ml",)


def test_qwen_detector_needs_the_shared_model():
    with pytest.raises(ValueError): LanguageDetector(cfg(detector="qwen"))


def test_voxlingua_short_audio_gives_no_language_instead_of_guessing():
    r = VoxLinguaDetector().detect(np.zeros(1000, dtype="float32"))
    assert r.code is None and r.note == "too_short"


def test_voxlingua_load_failure_is_a_detector_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.classifiers", None)
    with pytest.raises(DetectorError): VoxLinguaDetector().detect(np.zeros(16000, dtype="float32"))


# --- config validation -------------------------------------------------------------------
@pytest.mark.parametrize("over", [dict(detector="nope"), dict(primary="x"), dict(window_seconds=0), dict(min_confidence=2),
                                  dict(min_confidence="high")])
def test_invalid_language_config_is_an_understandable_error(over):
    with pytest.raises(ConfigError): LanguageDetector(cfg(**over), qwen=Fake("q", "en"), voxlingua=Fake("v", "ml"))


# --- compare mode ------------------------------------------------------------------------
def test_compare_reports_both_and_agreement():
    # Qwen's known "Malay" confusion for Malayalam is settled when VoxLingua names Malayalam
    r = det("compare", Fake("qwen", "ms", "Malay", suspect=("ml",)), Fake("voxlingua", "ml", conf=0.9)).detect(A)
    assert r.code == "ml" and r.agree is True and not uncertain_reason(r, 0.5) and r.qwen_code == "ms"
    r = det("compare", Fake("qwen", "en"), Fake("voxlingua", "en", conf=0.9)).detect(A)
    assert r.agree is True and r.code == "en" and r.confidence == 0.9


def test_compare_ambiguous_chunks_are_transcribed_flagged_weak_not_dropped():
    # moderate VoxLingua confidence, Qwen disagrees: best guess is used, marked weak
    r = det("compare", Fake("qwen", "hi"), Fake("voxlingua", "en", conf=0.5)).detect(A)
    assert r.code == "en" and r.weak and r.resolved and not uncertain_reason(r, 0.5)
    # only when VoxLingua is below the floor and nothing else supports an answer is the chunk left unresolved
    r = det("compare", Fake("qwen", "hi"), Fake("voxlingua", "en", conf=0.2)).detect(A)
    assert not r.resolved and "disagree" in uncertain_reason(r, 0.5) and set(r.candidates()) == {"hi", "en"}


def test_compare_voxlingua_decides_languages_qwen_cannot_identify():
    # Qwen3-ASR has no Malayalam/Tamil, so it will call that audio Hindi (or Malay): VoxLingua must be able to win
    for qwen_says in ("hi", "en", None):
        r = det("compare", Fake("qwen", qwen_says), Fake("voxlingua", "ml", conf=0.8)).detect(A)
        assert r.code == "ml" and r.resolved and not r.weak and not uncertain_reason(r, 0.5), qwen_says
    r = det("compare", Fake("qwen", "hi"), Fake("voxlingua", "ta", conf=0.4)).detect(A)      # modest but above blind_min: accepted, weak
    assert r.code == "ta" and r.weak and r.resolved
    r = det("compare", Fake("qwen", "hi"), Fake("voxlingua", "ta", conf=0.2)).detect(A)      # nearly no evidence for ta: unresolved
    assert not r.resolved and uncertain_reason(r, 0.5)


def test_compare_confident_detector_may_overrule_only_a_routable_language():
    cfgd = cfg(detector="compare")
    r = LanguageDetector(cfgd, qwen=Fake("qwen", "hi"), voxlingua=Fake("voxlingua", "en", conf=0.95)).detect(A)
    assert r.code == "en" and r.resolved and r.agree is False and not uncertain_reason(r, 0.5)
    r = LanguageDetector(cfgd, qwen=Fake("qwen", "hi"), voxlingua=Fake("voxlingua", "mr", conf=0.95)).detect(A)   # clear Marathi: not routable
    assert not r.resolved and r.code == "mr"
    r = LanguageDetector(cfgd, qwen=Fake("qwen", "hi"), voxlingua=Fake("voxlingua", "mr", conf=0.5)).detect(A)    # weak Marathi: trust Qwen, flagged
    assert r.code == "hi" and r.weak and r.resolved
    r = LanguageDetector(cfgd, qwen=Fake("qwen", "de", conf=0.9), voxlingua=Fake("voxlingua", "en", conf=0.9)).detect(A)   # both confident
    assert not r.resolved
    r = LanguageDetector(cfgd, qwen=Fake("qwen", "de", conf=0.9), voxlingua=Fake("voxlingua", "en", conf=0.4)).detect(A)   # Qwen confident
    assert r.code == "de" and r.resolved


def test_voxlingua_closed_set_posterior_beats_raw_top1():
    from src.language import VoxLinguaDetector
    v = VoxLinguaDetector(supported={"en", "hi", "ml", "ta"})
    v._labels = {0: "en", 1: "hi", 2: "ml", 3: "ta", 4: "mr", 5: "ne", 6: "ur", 7: "yue"}
    # raw top-1 is a tiny 0.22 spread over many labels; among routable languages Malayalam is clearly ahead
    post = v.posterior(np.log(np.array([0.02, 0.05, 0.22, 0.03, 0.2, 0.2, 0.2, 0.08])))
    code, conf, probs, _ = v.classify(post)
    assert code == "hi" or code == "ml"
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    post = v.posterior(np.log(np.array([0.02, 0.03, 0.30, 0.03, 0.2, 0.2, 0.01, 0.21])))
    code, conf, probs, open_top = v.classify(post)
    assert code == "ml" and conf > 0.6 > 0.3           # 0.30 raw would never have passed a 0.5 threshold
    post = v.posterior(np.log(np.array([0.01, 0.01, 0.01, 0.01, 0.9, 0.02, 0.02, 0.02])))   # clearly Marathi: unroutable stays unroutable
    assert v.classify(post)[0] == "mr"


def test_compare_keeps_qwen_transcript_and_reports_per_detector_scores():
    r = det("compare", Fake("qwen", "en", "English", text="hello"), Fake("voxlingua", "en", conf=0.9), primary="voxlingua").detect(A)
    assert r.text == "hello" and r.qwen_code == "en"             # transcript survives even when VoxLingua is primary
    d = r.describe()
    assert d["voxlingua"]["code"] == "en" and d["voxlingua"]["confidence"] == 0.9 and d["agreed"] is True


def test_voxlingua_legacy_codes_are_normalised():
    assert parse_voxlingua("iw: Hebrew") == "he" and parse_voxlingua("tl: Tagalog") == "fil"


def test_compare_with_unknown_on_one_or_both_sides():
    r = det("compare", Fake("qwen", None, "Xyz"), Fake("voxlingua", "en", conf=0.2)).detect(A)
    assert uncertain_reason(r, 0.5)                                  # low VoxLingua confidence, Qwen gave nothing
    r = det("compare", Fake("qwen", None), Fake("voxlingua", None, conf=0.2)).detect(A)
    assert r.code is None and r.agree is None and uncertain_reason(r, 0.5)


def test_compare_survives_one_failing_detector_and_reports_two():
    r = det("compare", Fake("qwen", "en", fail="gpu busy"), Fake("voxlingua", "ml", conf=0.9)).detect(A)
    assert r.code == "ml" and "failed" in r.note
    with pytest.raises(DetectorError): det("compare", Fake("q", "en", fail="a"), Fake("v", "ml", fail="b")).detect(A)


def test_uncertainty_rules():
    assert uncertain_reason(LangResult(None, "x", None, "qwen"), .5)
    assert uncertain_reason(LangResult("hi", "hi", 0.3, "voxlingua"), .5)
    assert uncertain_reason(LangResult("hi", "hi", 0.8, "voxlingua"), .5) is None
    assert uncertain_reason(LangResult("en", "English", None, "qwen"), .5) is None      # missing score is not low confidence


# --- the configured fallback behaviour really happens -----------------------------------
def router(**over):
    c = cfg(**over)
    return Router(c, ModelSet(c, overrides={"qwen3_asr": FakeASR(set(QWEN_LANGUAGES.values())),
                                            "indic": _Indic()}))


class _Indic(FakeASR):
    name = "indic"
    def __init__(self): super().__init__({"ta", "ml"})


def R(code, raw=None, conf=None, **k): return LangResult(code, raw or code, conf, "t", **k)


def test_routing_by_language():
    r = router(); assert r.route(R("en")).model == "qwen3_asr" and r.route(R("ml")).model == "indic" and r.route(R("ta")).language == "ta"


def test_malayalam_is_never_sent_to_qwen_even_via_the_suspect_malay_label():
    r = router()                                               # default: on_uncertain reject
    with pytest.raises(LanguageRejected): r.route(R("ms", "Malay", suspect=("ml",)))
    r = router(on_uncertain="best_guess", best_guess_model="qwen3_asr")
    out = r.route(R("ms", "Malay", suspect=("ml",)))
    assert out.uncertain and out.language is None and "ml" in out.reason     # explicit, flagged, auto-language


def test_no_language_rejected_or_best_guess_flagged():
    with pytest.raises(LanguageRejected): router().route(R(None, "???"))
    out = router(on_uncertain="best_guess", best_guess_model="qwen3_asr").route(R(None, "???"))
    assert out.uncertain and out.model == "qwen3_asr" and out.language is None


def test_low_confidence_known_language():
    with pytest.raises(LanguageRejected): router().route(R("hi", conf=0.2))
    out = router(on_uncertain="best_guess", best_guess_model="qwen3_asr").route(R("hi", conf=0.2))
    assert out.model == "qwen3_asr" and out.language == "hi" and out.uncertain


def test_unsupported_language_policy():
    with pytest.raises(UnsupportedLanguage): router().route(R("bn"))
    out = router(on_unsupported="best_guess", best_guess_model="qwen3_asr").route(R("bn"))
    assert out.uncertain and "unsupported" in out.reason and out.language is None


def test_invalid_policy_config_is_an_understandable_error():
    with pytest.raises(ConfigError): router(on_uncertain="maybe")
    with pytest.raises(ConfigError): router(on_unsupported="best_guess")                      # no best_guess_model named
    with pytest.raises(ConfigError): router(on_uncertain="best_guess", best_guess_model="nope")


def test_routing_table_is_checked_against_model_support():
    c = cfg(); c["asr"]["routing"]["ml"] = "qwen3_asr"          # Qwen cannot do Malayalam
    with pytest.raises(ConfigError): Router(c, ModelSet(c))
    c = cfg(); c["asr"]["routing"]["en"] = "ghost"
    with pytest.raises(ConfigError): Router(c, ModelSet(c))
    c = cfg(); c["asr"]["default_model"] = "ghost"
    with pytest.raises(ConfigError): Router(c, ModelSet(c))


def test_indian_model_not_configured_fails_clearly_at_routing():
    with pytest.raises(ModelNotConfigured): Router(cfg(), ModelSet(cfg())).route(R("ml"))     # real, unconfigured Indic slot
