# Crowdstream Connect

Multilingual meeting assistant: speech-to-text, language identification, translation and structured meeting summaries for English, Hindi, Malayalam, Tamil and other languages.

## Features

- **Transcription**: multilingual speech recognition with voice activity detection (VAD).
- **Language identification**: routes audio to the right transcription and translation path.
- **Translation**: text translation between supported languages, English by default.
- **Summarization**: structured summaries from final meeting transcripts, including long meetings.

## Models

| Component | Model / technology |
|---|---|
| Speech-to-text | Qwen3-ASR-1.7B |
| Voice activity detection | Silero VAD |
| Translation | MADLAD-400-3B-MT |
| Summarization | Qwen3.5-9B, served through an OpenAI-compatible API |
| Backend / frontend | FastAPI / HTML, CSS, JavaScript |

Integration status can vary by component; the configuration files hold the current settings.

## Architecture

```text
Audio -> VAD -> Language ID -> Qwen3-ASR -> Transcript --+--> Translation
                                                         |
                                                         +--> Summarization service -> Qwen3.5-9B -> Summary -> Frontend
```

Summarization runs as a **separate service** and takes the final transcript, never raw audio. The only place that maps the transcription output to the summarizer's input is `summarization/adapters.py`, so connecting the two later means editing that file.

## Project structure

```text
connect_interview_ai/
├── configs/                  configuration files
├── docs/                     documentation
├── evaluation/               model evaluation scripts
├── frontend/                 web UI
├── scripts/                  utility scripts
├── service/                  main FastAPI backend (transcription), port 8000
├── src/                      core application code
├── summarization/            meeting summarization module
├── tests/                    tests and evaluation cases
├── summarization_app.py      summarization FastAPI app, port 8100
├── requirements.txt
├── requirements-models.txt
├── requirements-summarization.txt
└── .env.example
```

## Setup

Requires Python 3.10+ and Git.

```bash
git clone https://github.com/anandhu-oss/Crowdstream_connect.git
cd Crowdstream_connect

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
pip install -r requirements-models.txt
pip install -r requirements-summarization.txt
```

Create the environment files from the examples:

```bash
copy .env.example .env                                       # main app
copy summarization\.env.example summarization\.env           # summarization
```

Never commit `.env` files, API keys or model weights.

## Running

Run each service in its own terminal from the project root, with the virtual environment active.

| Service | Command | Port |
|---|---|---:|
| Transcription backend | `uvicorn service.app:app --host 127.0.0.1 --port 8000 --reload` | 8000 |
| Summarization backend | `uvicorn summarization_app:app --host 127.0.0.1 --port 8100 --reload` | 8100 |
| Frontend | `cd frontend` then `python -m http.server 5500` | 5500 |

Open `http://localhost:5500`. Interactive API docs for summarization are at `http://127.0.0.1:8100/docs`.

## Summarization

### Model server

Qwen3.5-9B runs on a GPU and is exposed through an ngrok URL.

1. Start the model-server notebook and wait for the "API READY" message.
2. Copy the two printed values into `summarization/.env`:

   ```env
   SUMMARY_LLM_BASE_URL=https://<your-ngrok-url>/v1
   SUMMARY_LLM_API_KEY=<your-api-key>
   SUMMARY_LLM_MODEL=Qwen/Qwen3.5-9B
   SUMMARY_JSON_MODE=off
   SUMMARY_REQUEST_TIMEOUT=900
   SUMMARY_TOTAL_TIMEOUT=3600
   SUMMARY_MAX_CONCURRENCY=1
   ```

3. Restart the summarization backend. The ngrok URL changes on every Kaggle run, and `--reload` does not re-read `.env`.

`GET /summarize/health` checks that the model server is reachable. All settings are listed in `summarization/.env.example`.

### How it works

1. Clean the transcript: remove fillers, stutters and ASR loops, keep corrections and numbers, merge speaker turns.
2. Short meetings: one summary call. Long meetings: split at speaker turns, summarize each chunk, merge the notes, then write the final summary.
3. Audit pass (short meetings): a second call lists anything important the draft missed. Each addition must include a quote that is verified against the transcript.
4. Date safety net: any date or time found in the transcript but missing from the summary is added from the transcript sentence.
5. Return validated JSON.

### API

| Endpoint | Purpose |
|---|---|
| `POST /summarize` | Summarize a transcript |
| `POST /summarize/raw` | Same, taking the transcription module's output as is |
| `GET /summarize/health` | Model server status |

Request body for `/summarize`:

```json
{
  "meeting_id": "demo-1",
  "transcript": "[00:00:05] Rahul: We need to finish the website by December 5.",
  "output_language": "English"
}
```

Instead of `transcript`, send `segments`: a list of `{text, speaker, start, end, language}`.

The response contains `meeting_overview`, `executive_summary`, `key_discussion_points`, `decisions`, `action_items` (task, assigned_to, deadline, status), `key_dates`, `unresolved_issues`, `next_steps`, `processing_time_seconds`, `status`, plus `warnings` and `quality` for diagnostics. Missing information is `null`; owners, deadlines and decisions are never invented.

### Troubleshooting

| Message | Likely cause |
|---|---|
| Cannot reach the summarization service | Summarization backend is not running on port 8100 |
| Model unavailable / HTTP 500 or 503 | Kaggle session stopped, or `.env` still has the old ngrok URL |
| Output cut off (hit max tokens) | Model server is in thinking mode, or its output limit is too low |
| Very slow responses | Expected on a T4; keep the long timeouts above |

## Testing and evaluation

```bash
python -m pytest tests -q
python -m summarization.evaluate tests/eval_cases --tag <model-name>
```

The tests use a mock model and need no GPU. The evaluation script runs the real model over the meetings in `tests/eval_cases/`, compares against expected answers (decisions, action items, dates, unresolved issues, invented dates or owners, time taken) and saves a JSON report in `eval_reports/`, so models can be compared. Add a meeting by putting `name.txt` and `name.expected.json` in that folder; the format is described at the top of `summarization/evaluate.py`.

## Repository

https://github.com/anandhu-oss/Crowdstream_connect
