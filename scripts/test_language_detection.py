#!/usr/bin/env python3
"""Compare language detectors on labelled clips.
  python scripts/test_language_detection.py clips.csv --detectors qwen voxlingua --seconds 5 10
clips.csv has two columns, path and language (ISO code), no header, e.g.   data/recordings/ml_01.wav,ml
Each clip is cut to its first N seconds. Use short pieces and code-mixed clips separately: one label is only the dominant language."""
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.language_detection import compare_detectors
from src.audio import AudioError, load_config, load_track
from src.language import ConfigError, DetectorError, LanguageDetector, QwenDetector, VoxLinguaDetector, load_models_config
from src.transcription import ModelLoadError, ModelSet


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv")
    ap.add_argument("--detectors", nargs="+", default=["qwen", "voxlingua"], choices=["qwen", "voxlingua"])
    ap.add_argument("--seconds", nargs="+", type=float, default=[5.0])
    a = ap.parse_args()
    try:
        cfg = load_config(); rows = [r for r in csv.reader(open(a.csv, encoding="utf-8")) if len(r) >= 2]
        if not rows:
            print("the csv has no rows"); return 2
        clips = [(load_track(p.strip(), cfg), lang.strip().lower()) for p, lang in rows]
        models = ModelSet(load_models_config())
        det = {}
        if "qwen" in a.detectors:
            det["qwen"] = QwenDetector(models.get("qwen3_asr"))
        if "voxlingua" in a.detectors:
            det["voxlingua"] = VoxLinguaDetector()
    except (AudioError, ConfigError, FileNotFoundError) as e:
        print(f"error: {e}"); return 2
    for sec in a.seconds:
        items = [(audio[: int(sec * 16000)], lang) for audio, lang in clips]
        try:
            res = compare_detectors(det, items)
        except (DetectorError, ModelLoadError) as e:
            print(f"model error: {e}"); return 3
        print(f"\n=== first {sec:g} s of each clip, {len(items)} clips ===")
        print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
