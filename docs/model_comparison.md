# Model comparison (fill in from real runs)

Verified from model cards: Qwen3-ASR supports 30 languages (list in `src/utils/constants.py`), no Tamil or Malayalam; MADLAD-400-3B is Apache 2.0, uses a `<2xx>` target prefix, weights 11.8 GB in fp32.

| Item | Status |
|---|---|
| Qwen3-ASR 1.7B on `meeting_rec.mp3` per language (WER/CER, RTF, GPU memory) | to run: `scripts/run_pipeline.py --reference` |
| Indian-language model (Indic-Transcribe Flex under test) | not chosen; licence text unread |
| Qwen vs VoxLingua on mixed speech (accuracy, delay) | to run: `scripts/test_language_detection.py` |
| VoxLingua107 licence | to check |
| Translation quality and delay (MADLAD-400-3B) | to run: `scripts/test_live_captions.py --translate-to en` |
| Lossless vs lossy recording | to run: `evaluation/metrics.py` `encode_variants` + `compare_conditions` |
| Raw vs high-pass vs loudness-normalised | to run: same, compare WER/CER |
