import json
import numpy as np
import pytest

from evaluation.language_detection import compare_detectors, evaluate_detector
from evaluation.metrics import score_counts
from evaluation.transcription import align, evaluate, load_reference
from src.language import LangResult
from src.transcription import Transcript, TranscriptSegment


def seg(a, b, text, speaker=None, status="ok", flags=()): return TranscriptSegment(a, b, text, "x", status, speaker, flags=list(flags))


REF = [{"start": "00:00:00.000", "end": "00:00:05.000", "speaker": "Asha", "language": "Malayalam", "text": "namaskaram ellavarkkum"},
       {"start": "00:00:05.000", "end": "00:00:09.000", "speaker": "Asha", "language": "English", "text": "good morning everyone here"},
       {"start": "00:00:09.000", "end": "00:00:12.000", "speaker": "Ben", "language": "English", "text": "thank you"}]


def ref_file(tmp_path, data=REF):
    p = tmp_path / "ref.json"; p.write_text(json.dumps(data), encoding="utf-8"); return p


def test_load_reference_parses_times_and_sorts(tmp_path):
    r = load_reference(ref_file(tmp_path, list(reversed(REF))))
    assert [x["start"] for x in r] == [0.0, 5.0, 9.0] and r[0]["language"] == "Malayalam"


@pytest.mark.parametrize("bad", [[{"start": "x", "end": "00:00:01.000", "text": "a"}], [{"start": "00:00:02.000", "end": "00:00:01.000", "text": "a"}],
                                 [{"start": "00:00:00.000", "end": "00:00:01.000"}]])
def test_load_reference_rejects_bad_items(tmp_path, bad):
    with pytest.raises(ValueError): load_reference(ref_file(tmp_path, bad))


def test_per_language_scores_add_up_errors_not_average_rates(tmp_path):
    ref = load_reference(ref_file(tmp_path))
    t = Transcript("t", "Asha", 12, [seg(0.2, 4.8, "namaskaram ellavarkkum"), seg(5.2, 8.5, "good morning everyone")])
    r = evaluate(t, ref)["languages"]
    assert r["Malayalam"]["wer"] == 0 and r["Malayalam"]["cer"] == 0
    assert r["English"]["words"] == 4 and r["English"]["word_errors"] == 1 and r["English"]["wer"] == 25.0
    assert r["ALL"]["segments"] == 2                                              # only Asha's items: the transcript is Asha's track


def test_speaker_filter_and_override(tmp_path):
    ref = load_reference(ref_file(tmp_path)); t = Transcript("t", None, 12, [seg(9.5, 11, "thank you")])
    assert evaluate(t, ref)["languages"]["ALL"]["segments"] == 3                  # no speaker known: all reference items
    assert evaluate(t, ref, speaker="ben")["languages"]["ALL"]["segments"] == 1 and evaluate(t, ref, speaker="ben")["languages"]["ALL"]["wer"] == 0


def test_midpoint_rule_counts_nothing_twice_and_reports_missing():
    ref = [{"start": 0.0, "end": 5.0, "speaker": None, "language": "English", "text": "one two"},
           {"start": 5.0, "end": 10.0, "speaker": None, "language": "English", "text": "three four"}]
    t = Transcript(None, None, 10, [seg(4.0, 7.0, "one two three four"), seg(20, 21, "stray")])        # midpoint 5.5 -> second item
    rows, unmatched = align(t, ref)
    assert rows[0][1] == "" and rows[1][1] == "one two three four" and unmatched == 1
    r = evaluate(t, ref)["languages"]["English"]; assert r["no_hypothesis"] == 1 and r["word_errors"] == 4


def test_rejected_pieces_are_not_scored_as_text_and_flags_are_counted():
    ref = [{"start": 0.0, "end": 5.0, "speaker": None, "language": "Tamil", "text": "vanakkam nanbare"}]
    t = Transcript(None, None, 5, [seg(0, 4, "", status="rejected")]); r = evaluate(t, ref)["languages"]["Tamil"]
    assert r["no_hypothesis"] == 1 and r["wer"] == 100.0
    t = Transcript(None, None, 5, [seg(0, 4, "vanakkam nanbare", flags=["probable_bleed"])]); assert evaluate(t, ref)["languages"]["Tamil"]["flagged"] == 1


def test_cer_is_primary_for_chinese_labels():
    ref = [{"start": 0.0, "end": 5.0, "speaker": None, "language": "Chinese", "text": "你好世界"}]
    r = evaluate(Transcript(None, None, 5, [seg(0, 4, "你好世间")]), ref)["languages"]["Chinese"]
    assert r["primary_metric"] == "cer" and r["primary"] == 25.0


def test_score_counts():
    assert score_counts("a b c", "a x c") == {"word_errors": 1, "words": 3, "char_errors": 1, "chars": 3}


# --- language detection evaluation ------------------------------------------------------
class D:
    def __init__(self, answers, conf=None): self.a, self.i, self.conf = answers, 0, conf
    def detect(self, audio):
        c = self.a[self.i % len(self.a)]; self.i += 1; return LangResult(c, c, self.conf, "d")


ITEMS = [(np.zeros(16000, dtype="float32"), l) for l in ("ml", "ml", "hi", "en")]


def test_evaluate_detector():
    r = evaluate_detector(D(["ml", "ms", "hi", "en"]), ITEMS)
    assert r["clips"] == 4 and r["accuracy"] == 75.0 and r["recall"] == {"en": 100.0, "hi": 100.0, "ml": 50.0}
    assert r["top_confusions"] == [{"true": "ml", "predicted": "ms", "count": 1}] and r["unknown_rate"] == 0 and r["mean_ms"] >= 0
    r = evaluate_detector(D([None, "ml", "hi", "en"], conf=0.3), ITEMS)
    assert r["unknown_rate"] == 25.0 and r["uncertain_rate"] == 100.0
    with pytest.raises(ValueError): evaluate_detector(D(["en"]), [])


def test_detector_errors_are_counted_not_fatal():
    class Bad:
        def detect(self, a): raise RuntimeError("x")
    assert evaluate_detector(Bad(), ITEMS) == {"clips": 0, "errors": 4}


def test_compare_detectors_reports_agreement():
    r = compare_detectors({"qwen": D(["ms", "ms", "hi", "en"]), "vox": D(["ml", "ml", "hi", "en"], conf=0.9)}, ITEMS)
    assert r["qwen"]["accuracy"] == 50.0 and r["vox"]["accuracy"] == 100.0 and r["agreement_pct"] == 50.0
