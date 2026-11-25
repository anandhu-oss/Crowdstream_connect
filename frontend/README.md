# Multilingual AI Meeting Assistant (frontend)

Plain HTML/CSS/JS demo frontend for live transcription and translation. No build step, no dependencies.

## Run

```bash
cd frontend
python -m http.server 5500
```

Open http://localhost:5500. Microphone access needs `localhost` or HTTPS.

## Connect to FastAPI

1. Set `Api.config.baseUrl` (and paths if different) at the top of `js/api.js`.
2. Allow the frontend origin in FastAPI:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5500"], allow_methods=["*"], allow_headers=["*"])
```

3. The status pill turns green when `GET /health` returns 200.

## Expected contract (not yet verified against a real backend)

The backend was not available when this was written, so the contract is documented at the top of `js/api.js`. If the real backend differs, change the paths in `Api.config`, the `start` message in `js/websocket.js`, and `Api.normalizeSegment`, which is the only place raw field names are mapped.

- `GET /health`
- `GET /languages` (optional) returns `{"source": [...], "target": [...]}`. Without it, `Api.config.fallbackLanguages` is used.
- `POST /transcribe`: multipart `file`, `source_language` (`auto` or code), `target_language`; returns `{"segments": [...]}`
- `WS /ws/stream?source_language=&target_language=`
  - client sends JSON `start`, then binary audio chunks (MediaRecorder, 250 ms, WebM/Opus where supported; the `mime_type` is sent in `start`), then JSON `stop`
  - server sends `{"type":"status","status":...}`, `{"type":"segment","segment":{...}}`, `{"type":"error","message":...}`
- Segment fields: `segment_id`, `speaker_id`, `start_time`, `end_time`, `source_language`, `original_text`, `target_language`, `translated_text`, `status` (`partial` or `completed`), `translation_error`

Segments sharing a `segment_id` update in place, so partial results, late translations, and reconnect replays do not create duplicates. Completed segments are never overwritten by later partials.

## Behaviour notes

- Statuses come only from backend `status` messages. For file upload, the interface shows "Uploading, awaiting backend response" until the HTTP response arrives, then "Completed".
- If translation fails or is missing, the original text stays and a warning tag appears in the translated panel.
- On WebSocket drop the client reconnects (up to 5 attempts, backoff) with a single socket, and restarts the recorder so the new connection receives a fresh WebM header. Audio during the gap is not buffered.
- Clear Captions only clears the browser view.

## Mock mode

Tick **Mock mode** in the header to test without a backend. A banner and a SIMULATED tag on every caption mark the data as fake. Mock mode lives only in `js/mock.js`; delete that file and its script tag to remove it. It has English translations only; other targets show the translation-error state.

## AI Interview tab (read-aloud only)

Separate section (`#interview`) in `index.html`, `css/interview.css`, `js/interview.js`. It does not use `audio.js`, the WebSocket code or captions. It only reads prepared questions aloud; there is no candidate recording, speech recognition, follow-up, evaluation or scoring.

- **Backend URL**: set `baseUrl` in `INTERVIEW_CONFIG` at the top of `js/interview.js` (or `window.INTERVIEW_CONFIG` before the script), or type it under *TTS backend* in the tab (saved in this browser). Swap the temporary Kaggle URL for the production GPU server there.
- **TTS contract (unverified)**: `POST {baseUrl}/tts` with JSON `{"text","language","voice"}` returns `audio/wav`. Change `ttsPath` / `buildTtsBody()` if the real API differs. Voices come from `INTERVIEW_CONFIG.voices`; languages default to `Api.config.fallbackLanguages`.
- **Questions**: with `questionsPath` empty, a 6-question sample list is used and tagged SAMPLE QUESTIONS. Set `questionsPath` (e.g. `/interview/questions`) to `GET` `["q1", ...]` or `{candidate, role, interview_id, questions: [...]}`.
- **Flow**: one request per question; the next one is requested only after the current audio ends (plus `pauseBetweenMs`). Replay reuses the audio already received and does not change the question number. Leaving the tab pauses playback.

## Files

`index.html`, `css/style.css`, `js/app.js` (UI wiring), `js/audio.js` (mic and file checks), `js/api.js` (HTTP and mapping), `js/websocket.js` (live stream), `js/captions.js` (rendering), `js/mock.js` (simulated data), `css/interview.css` + `js/interview.js` (AI Interview tab).
