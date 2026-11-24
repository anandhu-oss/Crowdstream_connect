#!/usr/bin/env python3
"""Run one ASR model on a recording: VAD -> segments -> transcribe each. Prints text with track times, RTF, GPU memory.
  python scripts/test_asr.py data/recordings/meeting_rec.mp3 --model qwen3_asr [--language en] [--no-vad]
Exit codes: 0 ok, 1 nothing to transcribe or some pieces failed, 2 bad arguments or audio, 3 model could not load."""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio import AudioError, VADError, cut, fixed_segments, load_config, load_track, speech_segments
from src.language import ConfigError
from src.transcription import InferenceError, ModelLoadError, ModelNotConfigured, UnsupportedLanguage, get_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audio"); ap.add_argument("--model", default="qwen3_asr")
    ap.add_argument("--language", help="ISO code to force, e.g. en; default: the model detects it")
    ap.add_argument("--no-vad", action="store_true", help="fixed consecutive pieces instead of VAD, for comparison")
    a = ap.parse_args()
    try:
        cfg = load_config(); model = get_model(a.model)
        if a.language is not None and not model.supports(a.language):
            print(f"{a.model} does not support language {a.language!r}"); return 2
        audio = load_track(a.audio, cfg)
        total = len(audio) / 16000
        segs = fixed_segments(total, cfg["segments"]["max_seconds"]) if a.no_vad else speech_segments(audio, cfg)
    except (AudioError, VADError, ConfigError) as e:
        print(f"error: {e}"); return 2
    speech = sum(s.duration for s in segs)
    print(f"audio {total:.1f}s, {len(segs)} segments, {speech:.1f}s processed")
    if not segs or speech <= 0:
        print("no speech found, nothing to transcribe"); return 1
    try:
        model.load()
        model.transcribe(cut(audio, segs[0]), a.language)               # warm-up, not timed
    except (ModelLoadError, ModelNotConfigured, NotImplementedError) as e:
        print(f"model error: {e}"); return 3
    except (InferenceError, UnsupportedLanguage, AudioError) as e:
        print(f"warm-up failed: {e}"); return 1
    failed, t0 = 0, time.time()
    for s in segs:
        try:
            r = model.transcribe(cut(audio, s), a.language)
            print(f"[{s.start:7.2f} - {s.end:7.2f}] ({r.language or r.raw_language}) {r.text or '(no text)'}")
        except (InferenceError, UnsupportedLanguage, AudioError) as e:
            failed += 1; print(f"[{s.start:7.2f} - {s.end:7.2f}] FAILED: {e}")
    proc = time.time() - t0
    print(f"\ntotal audio {total:.1f}s | speech processed {speech:.1f}s | processing {proc:.1f}s | "
          f"RTF {proc / speech:.3f} (on speech only) | failed pieces {failed} | {model.info()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
