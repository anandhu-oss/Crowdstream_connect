import sys
import types
import numpy as np
import pytest

from evaluation.translation import chrf, evaluate_translation
from src.translation import (FakeTranslator, MadladTranslator, TranslationError, check_target, check_translation,
                             get_translator, split_sentences)


def test_split_sentences_handles_several_scripts():
    assert split_sentences("Hello there. How are you? Fine!") == ["Hello there.", "How are you?", "Fine!"]
    assert split_sentences("नमस्ते। आप कैसे हैं?") == ["नमस्ते।", "आप कैसे हैं?"]
    assert split_sentences("  one sentence  ") == ["one sentence"] and split_sentences("") == []


@pytest.mark.parametrize("bad", ["", "english", "E", "en-US", None, 5, "EN"])
def test_target_validation(bad):
    with pytest.raises(ValueError): check_target(bad)
    with pytest.raises(ValueError): FakeTranslator().translate("hello", "hi", bad)


def test_translate_sentence_by_sentence_and_same_language_passthrough():
    t = FakeTranslator(); r = t.translate("One. Two.", "hi", "en")
    assert r.text == "[en] One. [en] Two." and len(t.calls) == 2 and r.target == "en" and r.seconds >= 0
    s = t.translate("hello", "en", "en"); assert s.text == "hello" and s.flags == ["same_language"] and len(t.calls) == 2


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_nothing_to_translate_raises(bad):
    with pytest.raises(ValueError): FakeTranslator().translate(bad, "hi", "en")


def test_cache_returns_the_same_result_without_running_the_model_again():
    t = FakeTranslator(cache_size=2)
    t.translate("alpha", "hi", "en"); r = t.translate("alpha", "hi", "en")
    assert r.cached and len(t.calls) == 1
    t.translate("beta", "hi", "en"); t.translate("gamma", "hi", "en")          # alpha is evicted (size 2)
    t.translate("alpha", "hi", "en"); assert len(t.calls) == 4


def test_output_checks_flag_but_do_not_change_text():
    assert check_translation("hello world today", "") == ["empty_translation"]
    assert "unchanged" in check_translation("hello world today", "hello world today")
    assert "length_mismatch" in check_translation("hi", "x" * 100)
    assert check_translation("namaste dost", "hello friend") == []


class FakeTok:
    def __call__(self, text, return_tensors=None): self.last = text; return types.SimpleNamespace(to=lambda d: {"input_ids": text})
    def decode(self, ids, skip_special_tokens=True): return "translated"


class FakeModel:
    def generate(self, **kw): self.kw = kw; return [[1, 2]]


def test_madlad_uses_the_target_prefix_and_passes_settings():
    torch = pytest.importorskip("torch")
    tok, mod = FakeTok(), FakeModel()
    t = MadladTranslator(model=mod, tokenizer=tok, num_beams=2, max_new_tokens=50)
    r = t.translate("namaste dost", "hi", "en")
    assert tok.last == "<2en> namaste dost" and r.text == "translated" and mod.kw["num_beams"] == 2 and mod.kw["max_new_tokens"] == 50


def test_madlad_failures_are_clear_and_hide_text(monkeypatch):
    pytest.importorskip("torch")
    class Boom:
        def generate(self, **k): raise RuntimeError("secret words")
    t = MadladTranslator(model=Boom(), tokenizer=FakeTok())
    with pytest.raises(TranslationError) as e: t.translate("hello there friend", "en", "hi")
    assert "secret" not in str(e.value)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(TranslationError): MadladTranslator().load()


def test_madlad_dtype_choice(monkeypatch):
    torch = pytest.importorskip("torch")
    got = {}
    fake = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTok()),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: got.update(k) or types.SimpleNamespace(to=lambda d: types.SimpleNamespace(eval=lambda: FakeModel()))))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t = MadladTranslator(); t.load(); assert t.device == "cpu" and got["torch_dtype"] == torch.float32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    got.clear()
    class Dev:
        def eval(self): return FakeModel()
    fake.AutoModelForSeq2SeqLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: got.update(k) or types.SimpleNamespace(to=lambda d: Dev()))
    MadladTranslator().load(); assert got["torch_dtype"] == torch.float16
    with pytest.raises(ValueError): MadladTranslator(dtype="int4").load()


def test_factory_reads_config():
    t = get_translator(); assert isinstance(t, MadladTranslator) and t.checkpoint == "google/madlad400-3b-mt" and t.num_beams == 1


# --- evaluation helpers -----------------------------------------------------------------
def test_chrf():
    assert chrf("the cat sat on the mat", "the cat sat on the mat") == 100.0
    assert chrf("completely different", "the cat sat on the mat") < 30 and chrf("", "x") == 0.0
    assert chrf("the cat sat on a mat", "the cat sat on the mat") > chrf("dog", "the cat sat on the mat")


def test_evaluate_translation():
    r = evaluate_translation([("namaste dost kaise", "", None), ("good morning all", "good morning all", "good morning everyone"),
                              ("namaste dost", "hello friend", "hello friend")])
    assert r["items"] == 3 and r["flag_counts"]["empty_translation"] == 1 and r["flag_counts"]["unchanged"] == 1
    assert r["references"] == 2 and r["chrf"] > 50
    with pytest.raises(ValueError): evaluate_translation([])
