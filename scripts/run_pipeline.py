#!/usr/bin/env python3
"""Transcribe one or more people's tracks: timestamps, language per piece, silence skipped; optional translation and scoring.
  python scripts/run_pipeline.py data/recordings/meeting_rec.mp3
  python scripts/run_pipeline.py alice=a.wav bob=b.wav --translate-to en --out outputs/transcripts
  python scripts/run_pipeline.py data/recordings/meeting_rec.mp3 --reference data/recordings/meeting_rec.json
Tracks are  path  or  speaker=path. Several tracks are also merged into one meeting transcript by start time."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.transcription import evaluate, load_reference
from src.audio import AudioError, VADError
from src.language import ConfigError
from src.translation import get_translator
from src.transcription import ModelLoadError, TranscriptionPipeline, merge_transcripts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tracks", nargs="+")
    ap.add_argument("--language", help="force one language (ISO code) instead of detecting it")
    ap.add_argument("--translate-to", help="ISO code, e.g. en")
    ap.add_argument("--reference", help="reference json (meeting_rec.json) to score against")
    ap.add_argument("--out", default="outputs/transcripts"); ap.add_argument("--formats", nargs="+", default=["json", "txt", "srt"])
    a = ap.parse_args()
    try:
        pipe = TranscriptionPipeline(translator=get_translator() if a.translate_to else None)
        ref = load_reference(a.reference) if a.reference else None
        done = []
        for spec in a.tracks:
            speaker, _, path = spec.rpartition("=") if "=" in spec else (None, "", spec)
            t = pipe.transcribe_track(path, speaker or None, a.language, a.translate_to)
            stem = (speaker or Path(path).stem)
            print(f"\n{stem}: {t.duration_s:.1f}s audio, {t.speech_s:.1f}s speech, {len(t.segments)} pieces, RTF {t.rtf}")
            for w in t.warnings:
                print(f"  warning: {w}")
            print("  files:", ", ".join(t.save(a.out, stem, a.formats)))
            if ref:
                print(json.dumps(evaluate(t, ref), indent=2, ensure_ascii=False))
            done.append(t)
        if len(done) > 1:
            m = merge_transcripts(done); print("\nmerged:", ", ".join(m.save(a.out, "merged", a.formats)))
    except (AudioError, VADError, ConfigError, ModelLoadError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}"); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
