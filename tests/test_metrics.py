import numpy as np
import pytest

from evaluation.metrics import (FORMATS, compare_conditions, edit_distance, encode_variants, normalize, score,
                                signal_metrics)


def sine(sec=1, sr=16000, f=300, amp=0.2): return (amp * np.sin(2 * np.pi * f * np.arange(int(sec * sr)) / sr)).astype("float32")


def test_normalize():
    assert normalize("Hello,  World! [laughs] ३४ नमस्ते।") == "hello world 34 नमस्ते"
    assert normalize("") == "" and normalize(None) == ""


def test_edit_distance_and_scores():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    s = score("the cat sat", "the cat sat down"); assert s["wer"] == pytest.approx(33.33, abs=0.01) and s["cer"] > 0
    assert score("a b", "a b")["wer"] == 0
    assert score("", "")["wer"] == 0 and score("", "x")["wer"] is None


def test_cer_is_primary_for_chinese_and_japanese():
    s = score("你好世界", "你好世间", "zh"); assert s["primary_metric"] == "cer" and s["primary"] == 25.0
    assert score("a b", "a c", "en")["primary_metric"] == "wer"


def test_compare_conditions_against_baseline():
    rows = compare_conditions("the quick brown fox", {"flac": "the quick brown fox", "mp3_64k": "the quick brown box"})
    assert rows[0]["baseline"] and rows[0]["delta_vs_baseline"] == 0 and rows[1]["delta_vs_baseline"] == 25.0
    assert compare_conditions("a", {}) == []
    with pytest.raises(ValueError): compare_conditions("a", {"x": "a"}, baseline="y")


def test_signal_metrics_identical_scaled_and_different():
    y = sine(2)
    same = signal_metrics(y, y); assert same["si_sdr_db"] > 60 and same["correlation"] == pytest.approx(1, abs=1e-6)
    assert same["log_spectral_distance_db"] < 0.5 and same["rms_change_db"] == 0
    louder = signal_metrics(y, 3 * y); assert louder["rms_change_db"] == pytest.approx(9.54, abs=0.05) and louder["si_sdr_db"] > 60
    other = signal_metrics(y, sine(2, f=900)); assert other["si_sdr_db"] < 0 and abs(other["correlation"]) < 0.1


def test_signal_metrics_see_the_highpass_and_clipping():
    sr = 16000; t = np.arange(sr * 2) / sr
    rumble = (0.05 * np.sin(2 * np.pi * 30 * t)).astype("float32"); orig = rumble + sine(2, f=500)
    from scipy.signal import butter, sosfiltfilt
    hp = sosfiltfilt(butter(2, 80, btype="highpass", fs=sr, output="sos"), orig).astype("float32")
    m = signal_metrics(orig, hp); assert m["low_band_change_db"] < -10 and m["si_sdr_db"] > 8
    assert signal_metrics(np.clip(sine(1, amp=2), -1, 1), sine(1))["clipped_ratio_original"] > 0.2


def test_signal_metrics_edge_cases():
    assert signal_metrics(sine(), np.zeros(16000, dtype="float32"))["silent"] is True
    assert signal_metrics(sine(2), sine(1))["samples"] == 16000          # shorter length used
    for bad in (np.zeros(0, dtype="float32"), np.array([np.nan, 1.0]), np.zeros((2, 2))):
        with pytest.raises(ValueError): signal_metrics(sine(), bad)


def test_encode_variants_validation_and_missing_ffmpeg(tmp_path, monkeypatch):
    with pytest.raises(ValueError): encode_variants("x.wav", str(tmp_path), ("nope",))
    import subprocess
    def no_ffmpeg(*a, **k): raise FileNotFoundError()
    monkeypatch.setattr(subprocess, "run", no_ffmpeg)
    with pytest.raises(RuntimeError, match="ffmpeg is not installed"): encode_variants("x.wav", str(tmp_path), ("flac",))
    assert {"flac", "mp3_64k", "opus_16k"} <= set(FORMATS)
