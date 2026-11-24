# Architecture: transcription phase

Flow for one person's track (file) or live stream:

    audio -> preprocess -> VAD + segments -> language per piece -> route -> ASR -> (translate) -> transcript / live caption

| Step | Module | Notes |
|---|---|---|
| Load, preprocess | `src/audio/preprocess.py` | mono, 16 kHz, 80 Hz high-pass (configurable), loudness normalisation, clipping and noise report. No denoising unless `audio.denoise: true`. |
| VAD, segments | `src/audio/vad.py` | Silero VAD; pieces of at most 25 s cut at the quietest point; `near_silent` and `probable_bleed` are flags, removal is opt-in. |
| Live input | `src/audio/streaming.py` | Silero VADIterator, 512 samples at a time; utterances end after `endpoint_silence_ms` of silence. |
| Language | `src/language/detector.py` | `qwen`, `voxlingua` or `compare`. Qwen names the language while transcribing, so its transcript is kept and reused. |
| Models, routing | `src/transcription/` | Qwen3-ASR adapter, Indian-model slot (not chosen), router with `on_uncertain` / `on_unsupported` policy. |
| Output | `src/transcription/pipeline.py` | JSON, TXT, SRT, VTT; merging several people's tracks. |
| Translation | `src/translation/translator.py` | MADLAD-400-3B, sentence by sentence; output flagged, never altered. |
| Live captions | `src/captions/live_captions.py` | final caption per utterance, optional partials, delay measured per caption. |

Rules
- Nothing is guessed: unsure or unsupported languages are rejected unless the config names a `best_guess_model`; the result is then marked uncertain.
- Suspicious audio and output are flagged (`near_silent`, `probable_bleed`, `text_in_silence`, `repeated_phrase`, `implausibly_dense`), never silently dropped.
- Logs hold ids, times and counts only (`src/utils/logging.py` refuses content fields).

Not built yet: the FastAPI service (Task 2 design is separate), summaries (Task 5), interviewer voice (Task 6).
