# summarization (standalone module)

Takes a **final transcript** (never audio), returns a validated structured meeting summary.
Does not import or modify the transcription code. The LLM runs on **your own server**
(OpenAI-compatible endpoint, e.g. vLLM on the RunPod GPU) and is configured by env vars.

## Files

| File | Role |
|---|---|
| `config.py` | env-driven settings (`SUMMARY_*`) |
| `schemas.py` | request / LLM-output / response models (lenient to small-LLM JSON noise) |
| `preprocessing.py` | normalise, parse plain text, merge speaker turns, validate, chunk |
| `prompts.py` | system + chunk / merge / final / repair prompts |
| `llm_engine.py` | OpenAI-compatible client (retries, timeouts, concurrency) + mock engine |
| `pipeline.py` | short: 1 call. long: chunk -> notes -> merge rounds -> final |
| `api.py` | `POST /summarize`, `POST /summarize/raw`, `GET /summarize/health` |
| `dates.py` | regex scan of every date/time in the transcript (checklist + post-check) |
| `evaluate.py` | scores a model against expected answers (recall, hallucinations, time) |
| `adapters.py` | **the only place** that maps transcription output -> `SummarizeRequest` |
| `errors.py`, `cli.py` | typed errors, quick CLI |

## Run it (from `connect_interview_ai/`)

```bash
pip install -r requirements-summarization.txt
cp summarization/.env.example summarization/.env     # set URL + model

# 1) no GPU needed: mock model
python -m pytest tests -q
python -m summarization.cli tests/sample_transcripts/en_standup.txt --mock

# 2) real model: serve it somewhere (example on the GPU box)
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --port 8000
#    then in .env: SUMMARY_LLM_BASE_URL=http://<host>:8000/v1  SUMMARY_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
uvicorn summarization_app:app --port 8100
curl localhost:8100/summarize/health           # confirms the LLM server + model name
curl -X POST localhost:8100/summarize -H 'content-type: application/json' \
     -d '{"meeting_id":"m1","transcript":"[00:00:05] Speaker 1: ..."}'
```

Swapping candidate models for evaluation = change `SUMMARY_LLM_MODEL` / base URL only.
For Qwen3-8B / 1.7B (hybrid thinking) keep `SUMMARY_DISABLE_THINKING=true`.
`SUMMARY_CONTEXT_TOKENS` must equal the server's `--max-model-len`.

## Integrating later (only `adapters.py` changes)

1. Mount in the main app: `app.include_router(summarization.api.router)` in `service/app.py`.
2. When a meeting's final transcript is ready, either
   - in-process: `await summarize(from_transcription_output(final_transcript, meeting_id))`
     (run as a background task, store the result), or
   - over HTTP: `POST /summarize/raw` with `{"meeting_id", "transcription": <final transcript>}`.
3. Adjust the alias tables at the top of `adapters.py` to the real transcript JSON.

## Quality safeguards

* Prompts (`prompts.py`) demand completeness, define what counts as a decision / action item,
  ask for topic-level key points, and include one short worked example.
* `dates.py` scans the transcript for every date/time (English months, numeric dates, weekdays
  incl. Hindi/Malayalam/Tamil names, today/tomorrow/next week, clock times). The list is added to the
  prompt as a checklist; after the summary is generated each mention is verified, and if any is
  missing the model is asked once more with the missing ones listed. Anything still missing is
  returned in `warnings`. Turn off with `SUMMARY_COVERAGE_CHECK=false`.
* Response field `key_dates` lists every date/deadline/milestone with what it refers to and owner.

## Evaluating a model (do this for each candidate)

```bash
# point .env at the model, restart nothing needed for the CLI, then:
python -m summarization.evaluate tests/eval_cases --tag qwen3.5-9b-int4
```

It runs every `tests/eval_cases/<name>.txt|json` that has a `<name>.expected.json`, prints a table
(decision / action-item / key-date / unresolved recall, invented dates, invented owners, rule
violations, seconds, chunks), lists exactly what was missed, and saves a JSON report in
`eval_reports/`. Add your own recorded meetings the same way (see the docstring at the top of
`evaluate.py` for the expected-file format). `--mock` only checks the plumbing.
