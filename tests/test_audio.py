import io, os
import numpy as np
import pytest
import soundfile as sf

from src.audio import (AudioError, Segment, SileroVAD, VADError, cut, drop_flagged, fixed_segments, flag_bleed_cross, flag_segments,
                       load_audio, load_config, load_track, make_segments, preprocess, segment_level_db,
                       speech_level, speech_segments, to_mono)

CFG = load_config()
PEAK_CFG = {**CFG, "audio": {**CFG["audio"], "normalize": "peak"}}
MX = CFG["segments"]["max_seconds"]


def wav_bytes(y, sr):
    b = io.BytesIO(); sf.write(b, y, sr, format="WAV"); return b.getvalue()


def sine(sec, sr=16000, f=300, amp=0.2):
    return (amp * np.sin(2 * np.pi * f * np.arange(int(sec * sr)) / sr)).astype("float32")


# --- loading -----------------------------------------------------------------------------
def test_load_bytes_and_path(tmp_path):
    y, sr = load_audio(wav_bytes(sine(0.1), 16000)); assert sr == 16000 and y.shape[1] == 1
    p = tmp_path / "a.wav"; p.write_bytes(wav_bytes(sine(0.1), 16000))
    assert load_audio(p)[1] == 16000 and load_audio(str(p))[1] == 16000


@pytest.mark.parametrize("bad", [b"", b"not audio", b"RIFF" + b"\x00" * 40, wav_bytes(np.zeros(0, dtype="float32"), 16000), 123, None])
def test_load_rejects_bad_input(bad):
    with pytest.raises(AudioError): load_audio(bad)


def test_load_missing_file_and_directory(tmp_path):
    with pytest.raises(AudioError): load_audio(tmp_path / "nope.wav")
    with pytest.raises(AudioError): load_audio(tmp_path)


def test_load_rejects_invalid_sample_rate():
    with pytest.raises(AudioError): load_audio(wav_bytes(sine(0.1, 4000), 4000))    # below the supported range


def test_load_multichannel():
    y, _ = load_audio(wav_bytes(np.stack([sine(0.1)] * 6, axis=1), 16000)); assert y.shape[1] == 6


# --- preprocessing -----------------------------------------------------------------------
def test_preprocess_resamples_mixes_and_normalises():
    s = sine(1, 44100); y = preprocess(np.stack([s, s], axis=1), 44100, PEAK_CFG)
    assert y.ndim == 1 and abs(len(y) - 16000) <= 2 and y.dtype == np.float32
    assert abs(np.max(np.abs(y)) - CFG["audio"]["normalize_peak"]) < 0.02


def test_opposite_phase_channels_do_not_cancel():
    s = sine(1); mono = to_mono(np.stack([s, -s], axis=1))
    assert np.max(np.abs(mono)) > 0.1                         # plain averaging would give silence
    assert np.max(np.abs(to_mono(np.stack([s, s], axis=1)))) > 0.1


def test_one_silent_channel_keeps_the_other():
    s = sine(1); mono = to_mono(np.stack([s, np.zeros_like(s)], axis=1))
    assert np.max(np.abs(mono)) > 0.05


@pytest.mark.parametrize("bad", [np.zeros(0, dtype="float32"), np.array([np.nan, 0.1], dtype="float32"),
                                 np.array([np.inf, 0.1], dtype="float32"), np.zeros((2, 2, 2), dtype="float32"),
                                 np.array(["a", "b"]), [0.1, 0.2]])
def test_preprocess_rejects_bad_arrays(bad):
    with pytest.raises(AudioError): preprocess(bad, 16000, CFG)


@pytest.mark.parametrize("sr", [0, -1, 100, 1e9, 16000.5, None, True])
def test_preprocess_rejects_bad_sample_rate(sr):
    with pytest.raises(AudioError): preprocess(sine(0.1), sr, CFG)


def test_silence_stays_silent_and_noise_gain_is_capped():
    assert np.max(np.abs(preprocess(np.zeros(16000, dtype="float32"), 16000, CFG))) == 0
    assert np.max(np.abs(preprocess(np.full(16000, 1e-5, dtype="float32"), 16000, CFG))) == 0
    for cfg in (CFG, PEAK_CFG):
        quiet = preprocess(sine(1, amp=0.001), 16000, cfg)
        assert np.max(np.abs(quiet)) <= 0.001 * 10 ** (CFG["audio"]["max_gain_db"] / 20) * 1.3      # gain cap holds (filter start-up adds a little)


def test_very_short_recording():
    y = preprocess(sine(0.01), 16000, CFG); assert 0 < len(y) < 400
    assert len(preprocess(np.array([0.1, 0.2], dtype="float32"), 8000, CFG)) >= 1     # a couple of samples do not crash


def test_load_track_end_to_end():
    assert abs(len(load_track(wav_bytes(sine(1, 8000), 8000), CFG)) - 16000) <= 2


# --- segments ----------------------------------------------------------------------------
def pairs(segs): return [(round(s.start, 2), round(s.end, 2)) for s in segs]


def test_merge_close_keep_far_apart():
    assert pairs(make_segments([Segment(0, 2), Segment(2.3, 4), Segment(10, 12)], CFG)) == [(0, 4), (10, 12)]


def test_overlapping_regions_are_joined():
    assert pairs(make_segments([Segment(0, 5), Segment(3, 8), Segment(7, 9)], CFG)) == [(0, 9)]
    assert pairs(make_segments([Segment(0, 20), Segment(10, 40)], CFG, audio=None))[0][0] == 0   # overlap never lost


def test_unsorted_input():
    assert pairs(make_segments([Segment(10, 12), Segment(0, 2)], CFG)) == [(0, 2), (10, 12)]


@pytest.mark.parametrize("bad", [Segment(float("nan"), 1), Segment(0, float("inf")), Segment(5, 1)])
def test_invalid_times_raise(bad):
    with pytest.raises(AudioError): make_segments([bad], CFG)


def test_out_of_range_clipped_and_zero_length_dropped():
    assert pairs(make_segments([Segment(-3, 2)], CFG)) == [(0, 2)]
    assert pairs(make_segments([Segment(9, 15)], CFG, total_seconds=10)) == [(9, 10)]
    assert make_segments([Segment(20, 30)], CFG, total_seconds=10) == []
    assert make_segments([Segment(3, 3)], CFG) == []


def test_long_region_split_never_exceeds_max_and_covers_everything():
    for audio in (None, np.random.default_rng(0).normal(0, 0.1, 16000 * 80).astype("float32")):
        segs = make_segments([Segment(0, 80)], CFG, audio=audio)
        assert all(s.duration <= MX + 1e-6 for s in segs)
        assert segs[0].start == 0 and abs(segs[-1].end - 80) < 1e-6
        assert all(abs(a.end - b.start) < 1e-9 for a, b in zip(segs, segs[1:]))     # contiguous, nothing lost


def test_split_prefers_a_pause_over_the_middle_of_speech():
    sr = 16000; y = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr * 40) / sr)).astype("float32")
    y[int(21.0 * sr): int(21.4 * sr)] = 0                      # a pause inside the allowed search window (15-25 s)
    segs = make_segments([Segment(0, 40)], CFG, audio=y)
    assert 21.0 <= segs[0].end <= 21.4


def test_merge_stops_at_max_length_and_final_piece_is_kept():
    segs = make_segments([Segment(0, 15), Segment(15.2, 30)], CFG)
    assert len(segs) == 2 and all(s.duration <= MX for s in segs)
    segs = make_segments([Segment(0, 51)], CFG, total_seconds=51)
    assert abs(segs[-1].end - 51) < 1e-6 and len(segs) == 3


def test_tiny_region_dropped_but_short_real_speech_kept():
    assert make_segments([Segment(1, 1.1)], CFG) == []
    assert pairs(make_segments([Segment(1, 1.3)], CFG)) == [(1, 1.3)]


def test_all_segments_inside_recording():
    total = 33.3
    segs = make_segments([Segment(-1, 10), Segment(8, 90), Segment(100, 120)], CFG, total_seconds=total)
    assert all(0 <= s.start < s.end <= total + 1e-9 and s.duration <= MX + 1e-6 for s in segs) and segs[-1].end == total


def test_fixed_segments_cover_the_final_partial_piece():
    segs = fixed_segments(60.0, 25); assert pairs(segs) == [(0, 25), (25, 50), (50, 60)]
    segs = fixed_segments(50.0, 25); assert pairs(segs) == [(0, 25), (25, 50)]
    assert fixed_segments(0, 25) == [] and fixed_segments(float("nan"), 25) == []
    assert abs(fixed_segments(27.3, 25)[-1].end - 27.3) < 1e-9


def test_continuous_speech_without_silence_is_still_cut_to_max():
    segs = make_segments([Segment(0, 300)], CFG, audio=np.full(16000 * 300, 0.1, dtype="float32"))
    assert len(segs) >= 12 and all(s.duration <= MX + 1e-6 for s in segs)


# --- cut ---------------------------------------------------------------------------------
def test_cut_exact_and_clamped():
    a = np.zeros(32000, dtype="float32")
    assert len(cut(a, Segment(0.5, 1.5))) == 16000
    assert len(cut(a, Segment(1.5, 99))) == 8000
    assert len(cut(a, Segment(-5, 0.5))) == 8000


@pytest.mark.parametrize("seg", [Segment(5, 6), Segment(1, 1), Segment(1.0, 0.5), Segment(float("nan"), 1)])
def test_cut_never_returns_empty_or_invalid(seg):
    with pytest.raises(AudioError): cut(np.zeros(32000, dtype="float32"), seg)
    with pytest.raises(AudioError): cut(np.zeros((10, 2), dtype="float32"), Segment(0, 0.0001))


# --- VAD ---------------------------------------------------------------------------------
def test_vad_input_validation_before_any_model():
    v = SileroVAD(CFG)
    for bad in (np.zeros((2, 16000), dtype="float32"), np.zeros(16000, dtype="float64"), [0.0] * 16000,
                np.full(16000, np.nan, dtype="float32"), np.zeros(0, dtype="float32")):
        with pytest.raises(AudioError): v.detect(bad)
    with pytest.raises(AudioError): v.detect(np.zeros(16000, dtype="float32"), sr=8000)
    assert v.detect(np.zeros(100, dtype="float32")) == []     # shorter than one frame: no model needed


def test_vad_model_loading_failure_is_a_clear_error(monkeypatch):
    v = SileroVAD(CFG)
    def boom(): raise VADError("x")
    monkeypatch.setattr(v, "_load", boom)
    with pytest.raises(VADError): v.detect(np.zeros(16000, dtype="float32"))


def test_vad_missing_package_is_a_clear_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "silero_vad", None)       # import raises ImportError
    with pytest.raises(VADError): SileroVAD(CFG).detect(np.zeros(16000, dtype="float32"))


def test_speech_segments_with_a_stub_vad_uses_audio_for_splitting():
    class Stub:
        def detect(self, audio): return [Segment(0, 60)]
    segs = speech_segments(sine(60, amp=0.1), CFG, vad=Stub())
    assert all(s.duration <= MX + 1e-6 for s in segs) and abs(segs[-1].end - 60) < 1e-6


def test_vad_real_model_silence_and_tone():
    pytest.importorskip("silero_vad"); pytest.importorskip("torch")
    v = SileroVAD(CFG)
    assert v.detect(np.zeros(32000, dtype="float32")) == []
    assert v.detect(np.zeros(16000, dtype="float32") + 1e-4) == []


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1")
def test_real_recording_has_speech():
    pytest.importorskip("silero_vad"); p = "data/recordings/meeting_rec.mp3"
    if not os.path.exists(p): pytest.skip("no recording")
    segs = speech_segments(load_track(p, CFG), CFG)
    assert segs and all(s.duration <= MX + 1e-6 for s in segs)


# --- loudness, high-pass, clipping report -------------------------------------------------
def test_rms_normalisation_levels_quiet_and_loud_tracks_alike():
    lo, hi = preprocess(sine(2, amp=0.02), 16000, CFG), preprocess(sine(2, amp=0.4), 16000, CFG)
    assert abs(20 * np.log10(speech_level(lo, 16000) / speech_level(hi, 16000))) < 1.5


def test_one_click_does_not_set_the_gain():
    y = sine(4, amp=0.05); y[8000] = 0.99                      # a single loud click
    out = preprocess(y, 16000, CFG)
    assert speech_level(out, 16000) > 0.06 and np.max(np.abs(out)) <= 0.95   # click clipped, speech level kept


def test_long_silence_does_not_lower_the_speech_level():
    y = np.concatenate([sine(1, amp=0.1), np.zeros(16000 * 20, dtype="float32")])
    assert speech_level(preprocess(y, 16000, CFG), 16000) > 0.05


def test_highpass_removes_rumble_but_keeps_speech_band():
    sr = 16000; t = np.arange(sr * 2) / sr
    rumble = (0.3 * np.sin(2 * np.pi * 30 * t)).astype("float32"); voice = sine(2, f=500, amp=0.3)
    raw = {**CFG, "audio": {**CFG["audio"], "highpass_hz": 0, "normalize": "peak"}}
    hp = {**CFG, "audio": {**CFG["audio"], "highpass_hz": 80, "normalize": "peak"}}
    def band(y, f): return abs(np.fft.rfft(y)[int(f * 2)])
    a, b = preprocess(rumble + voice, sr, raw), preprocess(rumble + voice, sr, hp)
    assert band(b, 30) < 0.15 * band(a, 30) and band(b, 500) > 0.8 * band(a, 500)


def test_report_flags_clipping_silence_and_noise():
    clipped = np.clip(sine(2, amp=3.0), -1, 1)
    _, r = preprocess(clipped, 16000, CFG, return_report=True)
    assert r.clipped_ratio > 0.1 and any("clipping" in w for w in r.warnings)
    _, r = preprocess(np.zeros(16000, dtype="float32"), 16000, CFG, return_report=True)
    assert r.silent and r.warnings == ["silent track"]
    _, r = preprocess(sine(2, amp=0.2), 16000, CFG, return_report=True)
    assert r.clipped_ratio == 0 and r.duration_s == pytest.approx(2, abs=0.01) and not r.silent
    noisy = sine(2, amp=0.05) + np.random.default_rng(1).normal(0, 0.05, 32000).astype("float32")
    _, r = preprocess(noisy, 16000, CFG, return_report=True); assert any("signal-to-noise" in w for w in r.warnings)


# --- flags (nothing is removed unless asked) ---------------------------------------------
def speech_and_quiet(levels, sr=16000):
    """One second of tone per level (linear amplitude), 1 s of silence between; returns audio and the segments."""
    parts, segs, t = [], [], 0.0
    for amp in levels:
        parts += [sine(1, amp=amp), np.zeros(sr, dtype="float32")]; segs.append(Segment(t, t + 1)); t += 2
    return np.concatenate(parts), segs


def test_near_silent_piece_is_flagged_not_removed():
    audio, segs = speech_and_quiet([0.2, 0.0001, 0.2])
    out = flag_segments(segs, audio, CFG)
    assert len(out) == 3 and out[1].flags == ["near_silent"] and out[0].flags == [] and out[2].flags == []


def test_bleed_piece_far_below_own_voice_is_flagged_and_quiet_speaker_is_not():
    audio, segs = speech_and_quiet([0.3, 0.28, 0.3, 0.29, 0.01])       # last one ~30 dB down
    out = flag_segments(segs, audio, CFG)
    assert [x.flags for x in out] == [[], [], [], [], ["probable_bleed"]]
    audio, segs = speech_and_quiet([0.01, 0.012, 0.011, 0.009])           # everyone equally quiet: nothing is bleed
    assert all(x.flags == [] for x in flag_segments(segs, audio, CFG))


def test_too_few_pieces_to_judge_bleed():
    audio, segs = speech_and_quiet([0.3, 0.01])
    assert all(x.flags == [] for x in flag_segments(segs, audio, CFG))


def test_bleed_check_can_be_turned_off_in_config():
    audio, segs = speech_and_quiet([0.3, 0.3, 0.3, 0.01])
    cfg = {**CFG, "gate": {**CFG["gate"], "bleed": False}}
    assert all(x.flags == [] for x in flag_segments(segs, audio, cfg))


def test_drop_flagged_is_opt_in():
    audio, segs = speech_and_quiet([0.3, 0.3, 0.3, 0.01])
    flag_segments(segs, audio, CFG)
    assert len(segs) == 4 and len(drop_flagged(segs)) == 3 and len(drop_flagged(segs, flags=("near_silent",))) == 4


def test_speech_segments_flags_by_default_and_drops_only_when_configured():
    audio, _ = speech_and_quiet([0.3, 0.3, 0.3, 0.01])
    class Stub:
        def detect(self, a): return [Segment(0, 1), Segment(2, 3), Segment(4, 5), Segment(6, 7)]
    cfg = {**CFG, "segments": {**CFG["segments"], "merge_gap_seconds": 0.1}}
    kept = speech_segments(audio, cfg, vad=Stub())
    assert len(kept) == 4 and kept[3].flags == ["probable_bleed"]
    dropped = speech_segments(audio, {**cfg, "gate": {**cfg["gate"], "action": "drop"}}, vad=Stub())
    assert len(dropped) == 3


def test_cross_track_bleed_flags_the_quieter_copy():
    sr = 16000; a = np.concatenate([sine(2, amp=0.3), np.zeros(sr * 2, dtype="float32")])       # A speaks 0-2 s
    b = np.concatenate([sine(2, amp=0.01), sine(2, amp=0.3)])                                    # B hears A faintly, speaks 2-4 s
    sa, sb = [Segment(0, 2)], [Segment(0, 2), Segment(2, 4)]
    out = flag_bleed_cross({"A": (a, sa), "B": (b, sb)}, 12.0)
    assert out["A"][0].flags == [] and out["B"][0].flags == ["probable_bleed"] and out["B"][1].flags == []


def test_cross_track_uses_levels_as_recorded_not_after_normalisation():
    a = sine(2, amp=0.3); b_norm = sine(2, amp=0.01) * 30             # B was boosted 30x (29.5 dB) by per-track normalisation
    naive = flag_bleed_cross({"A": (a, [Segment(0, 2)]), "B": (b_norm, [Segment(0, 2)])}, 12.0)
    assert naive["B"][0].flags == []
    fixed = flag_bleed_cross({"A": (a, [Segment(0, 2)]), "B": (b_norm, [Segment(0, 2)])}, 12.0, gains_db={"B": 29.5})
    assert fixed["B"][0].flags == ["probable_bleed"]


def test_segment_level_helper():
    assert segment_level_db(np.zeros(100, dtype="float32"), 5, 6) == -180.0
    assert segment_level_db(sine(1, amp=1.0), 0, 1) == pytest.approx(-3.0, abs=0.1)


def test_denoise_is_off_by_default_and_explicit_when_on(monkeypatch):
    import sys
    assert CFG["audio"]["denoise"] is False
    y = sine(1, amp=0.2); on = {**CFG, "audio": {**CFG["audio"], "denoise": True}}
    monkeypatch.setitem(sys.modules, "noisereduce", None)
    with pytest.raises(AudioError): preprocess(y, 16000, on)          # asked for, not installed: a clear error, not silence
    assert len(preprocess(y, 16000, CFG)) == 16000


def test_denoise_runs_when_enabled_and_installed(monkeypatch):
    import sys, types
    called = {}
    fake = types.SimpleNamespace(reduce_noise=lambda y, sr, stationary: called.update(n=len(y)) or y)
    monkeypatch.setitem(sys.modules, "noisereduce", fake)
    preprocess(sine(1, amp=0.2), 16000, {**CFG, "audio": {**CFG["audio"], "denoise": True}})
    assert called["n"] == 16000
