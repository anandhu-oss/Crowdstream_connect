#!/usr/bin/env python3
"""Feed a recording to the live captioner in small chunks and report the caption delay.
  python scripts/test_live_captions.py data/recordings/meeting_rec.mp3 --translate-to en [--realtime] [--partial 1.5]
--realtime sleeps between chunks so wall-clock delay is meaningful; without it the file is fed as fast as possible and the
stream-time estimate (endpoint wait + ASR + translation) is the number to read."""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.latency import summarize_captions
from src.audio import AudioError, VADError, load_config, load_track
from src.captions import LiveCaptioner
from src.language import ConfigError
from src.translation import get_translator
from src.transcription import ModelLoadError, SegmentTranscriber


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audio")
    ap.add_argument("--translate-to"); ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--partial", type=float, default=0.0, help="seconds between partial captions (0 = off)")
    a = ap.parse_args()
    try:
        cfg = load_config(); cfg["streaming"]["partial_interval_s"] = a.partial
        audio = load_track(a.audio, cfg)
        cap = LiveCaptioner(SegmentTranscriber(), get_translator() if a.translate_to else None, a.translate_to, cfg)
        n = int(cfg["streaming"]["chunk_ms"] * 16000 / 1000); caps = []
        for i in range(0, len(audio), n):
            for c in cap.feed(audio[i:i + n]):
                caps.append(c)
                if c.kind == "final":
                    print(f"[{c.start:7.2f}-{c.end:7.2f}] ({c.language}) {c.text}" + (f"\n    -> {c.translation}" if c.translation else "")
                          + f"   delay~{c.estimated_delay_s:.2f}s" + (f" {c.flags}" if c.flags else ""))
            if a.realtime:
                time.sleep(n / 16000)
        caps += cap.flush()
    except (AudioError, VADError, ConfigError, ModelLoadError, FileNotFoundError) as e:
        print(f"error: {e}"); return 2
    print("\n" + json.dumps(summarize_captions(caps), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
