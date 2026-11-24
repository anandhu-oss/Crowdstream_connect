import asyncio
import heapq
import itertools
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.audio import AudioError, VADError, decode_container, section, sniff_container
from src.language import load_models_config
from src.transcription import LiveSession, ModelSet, SegmentTranscriber, TranscriptionPipeline
from src.translation.translator import get_translator, translation_enabled
from src.utils.logging import get_logger, log_event

log = get_logger()

models_cfg = load_models_config()
TRANSLATION_ENABLED = translation_enabled(models_cfg)
CONC = models_cfg["concurrency"]
SERVICE = models_cfg.get("service") or {}

models = ModelSet(cfg=models_cfg)
transcriber = SegmentTranscriber(cfg=models_cfg, models=models)
translator = get_translator(models_cfg) if TRANSLATION_ENABLED else None
pipeline = TranscriptionPipeline(models_cfg=models_cfg, transcriber=transcriber, translator=translator,
                                 translate=TRANSLATION_ENABLED)

STREAM = section(pipeline.audio_cfg, "streaming")
SAMPLE_RATE = int(section(pipeline.audio_cfg, "audio")["sample_rate"])
MAX_UPLOAD_BYTES = int(SERVICE.get("max_upload_mb", 200)) * 1024 * 1024
SHED_WAITING = int(CONC.get("shed_waiting", 4))
LANG_CODE = re.compile(r"^[a-z]{2,3}$")
CONTAINER_NAMES = ("webm", "ogg", "opus", "mp4", "m4a", "wav", "mp3", "flac", "container")

cpu_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="audio")
asr_executor = ThreadPoolExecutor(max_workers=int(CONC["asr"]) * 2 + int(CONC["uploads"]) + 2, thread_name_prefix="asr")
mt_executor = ThreadPoolExecutor(max_workers=int(CONC["translation"]) * 2 + 1, thread_name_prefix="mt")


class PriorityGate:
    def __init__(self, slots):
        self.slots = max(1, int(slots))
        self.busy = 0
        self._heap = []
        self._seq = itertools.count()

    @property
    def waiting(self):
        return sum(1 for _, _, f in self._heap if not f.done())

    async def acquire(self, priority=1):
        if self.busy < self.slots and not self._heap:
            self.busy += 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                fut.cancel()
            raise

    def release(self):
        self.busy = max(0, self.busy - 1)
        while self._heap:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                self.busy += 1
                fut.set_result(True)
                break


asr_gate = PriorityGate(CONC["asr"])
mt_gate = PriorityGate(CONC["translation"])
upload_gate = PriorityGate(CONC["uploads"])

RESUME = {}
ACTIVE = {"live": 0}

app = FastAPI(title="Multilingual AI Meeting Assistant", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=SERVICE.get("cors_origins", ["*"]), allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "hi", "name": "Hindi"},
    {"code": "ml", "name": "Malayalam"},
    {"code": "ta", "name": "Tamil"},
    {"code": "de", "name": "German"},
    {"code": "fr", "name": "French"},
    {"code": "it", "name": "Italian"},
    {"code": "es", "name": "Spanish"},
    {"code": "ar", "name": "Arabic"},
    {"code": "ja", "name": "Japanese"},
    {"code": "zh", "name": "Chinese"},
]


def language_id_status():
    try:
        d = pipeline.transcriber._detector
        return d.status() if d is not None else {"mode": models_cfg["language"].get("detector"), "voxlingua_loaded": False}
    except Exception:
        return {}


def asr_coverage():
    try:
        return pipeline.transcriber.router.coverage()
    except Exception:
        return {}


@app.on_event("startup")
async def warm_language_id():
    """Load VoxLingua in the background at startup: its first load is slow and would otherwise stall (and degrade) the first live chunk."""
    if models_cfg["language"].get("detector") not in ("voxlingua", "compare"):
        return
    loop = asyncio.get_running_loop()

    def warm():
        try:
            pipeline.transcriber.detector.warm_up()
            log.info("language ID warmed up")
        except Exception as e:
            log.warning("language ID warm-up failed: %s", e)

    loop.run_in_executor(cpu_executor, warm)


@app.on_event("shutdown")
async def shutdown():
    for ex in (cpu_executor, asr_executor, mt_executor):
        ex.shutdown(wait=False, cancel_futures=True)
    try:
        models.close()
        if translator is not None:
            translator.close()
    except Exception:
        pass


@app.get("/health")
async def health():
    cov = asr_coverage()
    return {
        "status": "healthy",
        "service": "multilingual-meeting-assistant",
        "mode": "windowed_near_realtime_qwen_madlad",
        "asr_model": "Qwen3-ASR-1.7B",
        "indic_asr_model": models_cfg["asr"]["models"]["indic"].get("checkpoint"),
        "indic_mode": models_cfg["asr"]["models"]["indic"].get("mode"),
        "language_id": models_cfg["language"].get("detector"),
        "language_id_status": language_id_status(),
        "indic_asr_ready": bool(cov.get("ml", {}).get("ready")) and bool(cov.get("ta", {}).get("ready")),
        "translation_model": "google/madlad400-3b-mt",
        "translation_enabled": TRANSLATION_ENABLED,
        "live_audio": "continuous PCM windows with VAD endpointing; token streaming is not used",
        "live_interim_interval_s": STREAM["partial_interval_s"],
        "live_sessions": ACTIVE["live"],
        "sample_rate": SAMPLE_RATE,
        "asr_waiting": asr_gate.waiting,
        "translation_waiting": mt_gate.waiting,
    }


@app.get("/health/models")
async def health_models():
    loop = asyncio.get_running_loop()

    def probe():
        out = {}
        for key in models.keys():
            try:
                out[key] = models.get(key).health()
            except Exception as e:
                out[key] = {"available": False, "reason": type(e).__name__}
        return out

    return {"asr": await loop.run_in_executor(cpu_executor, probe),
            "translation": translator.info() if translator is not None else {"enabled": False}}


@app.get("/languages")
async def languages():
    cov = asr_coverage()
    source = [{"code": "auto", "name": "Auto Detect", "asr_ready": True}]
    source += [{**x, "asr_ready": bool(cov.get(x["code"], {}).get("ready"))} for x in LANGUAGES]
    return {"source": source, "target": LANGUAGES, "translation_enabled": TRANSLATION_ENABLED}


def clean_language(value, allow_auto=False, field="language"):
    value = (value or "").strip().lower()
    if allow_auto and value in ("", "auto"):
        return None
    if not LANG_CODE.match(value):
        raise ValueError(f"{field} must be a 2-3 letter language code")
    return value


def upload_segment(index, segment, target, source):
    translated, tstatus = segment.translation, "none"
    flags = segment.flags or []
    not_done = segment.status in ("rejected", "failed", "busy")
    if not TRANSLATION_ENABLED:
        tstatus = "disabled"
    elif segment.translation:
        tstatus = "completed"
    elif target and segment.language == target and segment.text:
        translated, tstatus = segment.text, "same_language"
    elif "translation_failed" in flags:
        tstatus = "failed"
    # The frontend drops segments with empty original_text (blank panel), so a chunk that could not be transcribed
    # is shown with an explicit placeholder instead; original_text stays the recognised text for every other chunk.
    original = segment.text
    if not original and not_done:
        original = f"[not transcribed: {segment.reason or segment.status}]"
    det = segment.detection or {}
    return {
        "segment_id": str(index + 1),
        "speaker_id": segment.speaker,
        "start_time": segment.start,
        "end_time": segment.end,
        "source_language": segment.language or "unknown",
        "detected_language": segment.language or "unknown",
        "asr_model": segment.model,
        "asr_mode": segment.mode,
        "language_detection": {k: det.get(k) for k in ("voxlingua", "qwen", "agreed", "selected", "resolution", "fallback_from")
                               if det.get(k) is not None},
        "original_text": original,
        "target_language": segment.translation_language or (target if tstatus == "same_language" else None),
        "translated_text": translated,
        "status": "completed" if segment.status == "ok" else segment.status,
        "transcription_status": segment.status,
        "translation_status": tstatus,
        "translation_error": "Translation failed" if tstatus == "failed" else None,
        "uncertain": segment.uncertain,
        "reason": segment.reason,
        "flags": flags,
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), source_language: str = Form("auto"), target_language: str = Form("en")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded audio file is too large")
    try:
        language = clean_language(source_language, True, "source_language")
        target = clean_language(target_language, True, "target_language")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    translate_to = target if TRANSLATION_ENABLED else None
    loop = asyncio.get_running_loop()
    try:
        await upload_gate.acquire(1)
        try:
            transcript = await loop.run_in_executor(
                asr_executor, lambda: pipeline.transcribe_track(audio_bytes, language=language, translate_to=translate_to))
        finally:
            upload_gate.release()
    except (AudioError, VADError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.error("transcription failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Audio processing failed. Check backend logs.")
    segments = [upload_segment(i, s, target, language) for i, s in enumerate(transcript.segments)]
    ok = [s for s in segments if s["transcription_status"] == "ok"]
    langs = {}
    for s in ok:
        langs[s["detected_language"]] = langs.get(s["detected_language"], 0.0) + (s["end_time"] - s["start_time"])
    return {"segments": segments, "results": segments, "duration_s": transcript.duration_s,
            "processing_s": transcript.processing_s, "warnings": transcript.warnings,
            "detected_languages": sorted(langs, key=langs.get, reverse=True),
            "asr_models": sorted({s["asr_model"] for s in ok if s["asr_model"]}),
            "original_transcript": " ".join(s["original_text"] for s in ok),
            "translated_transcript": " ".join(s["translated_text"] for s in ok if s["translated_text"]),
            "mode": "remote_qwen_madlad", "translation_enabled": TRANSLATION_ENABLED}


class LiveConnection:
    def __init__(self, ws):
        self.ws = ws
        self.q = ws.query_params
        self.loop = asyncio.get_running_loop()
        self.send_lock = asyncio.Lock()
        self.closed = False
        self.state = "idle"
        self.session = None
        self.session_id = None
        self.pcm = True
        self.audio_q = None
        self.final_q = None
        self.ingest_task = None
        self.final_task = None
        self.stop_task = None
        self.tasks = set()
        self.asr_active = 0
        self.mt_active = 0
        self.interim_mt = False
        self.last_status = ("connected", 0.0)
        self.started = None
        self.warn_times = {}
        self.counted = False

    async def send(self, payload):
        if self.closed:
            return
        async with self.send_lock:
            try:
                await self.ws.send_json(payload)
            except Exception:
                self.closed = True

    async def error(self, message, fatal=False):
        await self.send({"type": "error", "message": message, "fatal": fatal})

    async def warn(self, key, message, every=8.0):
        now = time.monotonic()
        if now - self.warn_times.get(key, -1e9) >= every:
            self.warn_times[key] = now
            await self.send({"type": "warning", "message": message})

    async def status(self, name, message, force=False):
        last, at = self.last_status
        now = time.monotonic()
        if not force and (name == last or now - at < 0.6):
            return
        self.last_status = (name, now)
        await self.send({"type": "status", "status": name, "message": message})

    async def refresh_status(self):
        if self.state != "recording":
            return
        if self.asr_active:
            await self.status("transcribing", "Transcribing speech")
        elif self.mt_active:
            await self.status("translating", "Translating completed phrases")
        else:
            await self.status("receiving", "Listening")

    def spawn(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def run(self):
        await self.send({"type": "status", "status": "connected", "message": "WebSocket connected successfully"})
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    await self.on_text(message["text"])
                elif message.get("bytes") is not None:
                    await self.on_bytes(message["bytes"])
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.error("websocket error: %s", type(exc).__name__)
            await self.error("Live audio connection failed. Check backend logs.", True)
        finally:
            await self.cleanup()

    async def on_text(self, text):
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError
        except ValueError:
            await self.error("Invalid JSON control message")
            return
        command = data.get("type") or data.get("event") or data.get("action") or data.get("command")
        if command == "start":
            await self.on_start(data)
        elif command == "stop":
            if self.state != "recording":
                await self.error("No active recording session.")
                return
            self.state = "draining"
            self.stop_task = asyncio.create_task(self.on_stop())
        elif command == "config":
            if self.session is None:
                await self.error("No active recording session.")
                return
            try:
                src = data.get("source_language")
                tgt = data.get("target_language")
                src = clean_language(src, True, "source_language") or "auto" if src is not None else None
                tgt = (clean_language(tgt, True, "target_language") or "") if tgt is not None else None
            except ValueError as exc:
                await self.error(str(exc))
                return
            self.session.update_languages(src, tgt)
            await self.send({"type": "status", "status": "recording", "message": "Language settings updated"})
        elif command == "ping":
            await self.send({"type": "pong", "time": time.time()})
        else:
            await self.error(f"Unknown command: {command}")

    async def on_start(self, data):
        if self.state in ("recording", "draining"):
            await self.error("A recording session is already active.")
            return
        if ACTIVE["live"] >= int(CONC["max_live_sessions"]):
            await self.error("The server has reached its live session limit. Try again shortly.", True)
            return
        try:
            source = clean_language(data.get("source_language", self.q.get("source_language", "auto")), True, "source_language")
            raw_target = data.get("target_language", self.q.get("target_language", "en"))
            target = None if raw_target in (None, "", "none") else clean_language(raw_target, False, "target_language")
            fmt = str(data.get("audio_format") or "").lower()
            mime = str(data.get("mime_type") or "").lower()
            channels = int(data.get("channels", 1))
            rate = int(data.get("sample_rate", SAMPLE_RATE))
        except (ValueError, TypeError) as exc:
            await self.error(str(exc) or "Invalid start message")
            return
        mode = str(data.get("mode") or self.q.get("mode") or "").lower()
        if fmt.startswith("pcm"):
            self.pcm = True
        elif fmt in CONTAINER_NAMES or mime:
            self.pcm = False
        else:
            self.pcm = mode == "live"
        pcm_format = fmt if fmt in ("pcm_s16le", "pcm_f32le") else "pcm_s16le"
        session_id = str(data.get("session_id") or uuid.uuid4().hex[:8])[:64]
        now = time.monotonic()
        for k in [k for k, (_, exp) in RESUME.items() if exp < now]:
            RESUME.pop(k, None)
        resume = RESUME.pop(session_id, (None, 0))[0] if data.get("reconnect") else None
        speaker = data.get("speaker_id") or data.get("speaker")
        try:
            self.session = await self.loop.run_in_executor(
                cpu_executor,
                lambda: LiveSession(pipeline, source or "auto", target, speaker, session_id,
                                    rate if self.pcm else SAMPLE_RATE, pcm_format, channels, state=resume))
        except (AudioError, VADError, ValueError) as exc:
            await self.error(str(exc))
            return
        except Exception as exc:
            log.error("live session could not start: %s", type(exc).__name__)
            await self.error("The live session could not be started. Check backend logs.")
            return
        self.session_id = session_id
        self.audio_q = asyncio.Queue()
        self.final_q = asyncio.Queue()
        self.ingest_task = asyncio.create_task(self.ingest())
        self.final_task = asyncio.create_task(self.final_worker())
        self.state = "recording"
        self.started = time.monotonic()
        ACTIVE["live"] += 1
        self.counted = True
        log_event(log, "live_start", session=session_id, source=source or "auto", target=target or "none", pcm=self.pcm)
        await self.send({
            "type": "status", "status": "recording", "message": "Microphone connected; receiving live audio",
            "session_id": session_id, "live_mode": "windowed_near_realtime", "token_streaming": False,
            "sample_rate": SAMPLE_RATE, "audio_format": pcm_format if self.pcm else "container",
            "source_language": source or "auto", "target_language": target,
            "translation": "enabled" if self.session.translate_on else "disabled",
            "interim_interval_s": STREAM["partial_interval_s"],
        })
        self.last_status = ("recording", time.monotonic())

    async def on_bytes(self, data):
        if self.state == "draining":
            return
        if self.state != "recording":
            await self.warn("early", "Audio received before the recording session was started.")
            return
        if not data:
            return
        if len(data) > int(STREAM["max_chunk_bytes"]):
            await self.error("Received audio chunk exceeds the size limit.")
            return
        if self.audio_q.qsize() >= int(STREAM["max_inflight_chunks"]):
            await self.warn("behind", "The server is falling behind; some audio was skipped.")
            return
        self.audio_q.put_nowait(bytes(data))

    async def ingest(self):
        try:
            while True:
                item = await self.audio_q.get()
                if item is None:
                    return
                try:
                    if self.pcm:
                        tasks, msgs = await self.loop.run_in_executor(cpu_executor, self.session.feed_bytes, item)
                    else:
                        if sniff_container(item) is None:
                            raise AudioError("Unrecognised audio container received.")
                        samples = await self.loop.run_in_executor(cpu_executor, decode_container, item, SAMPLE_RATE)
                        tasks, msgs = await self.loop.run_in_executor(cpu_executor, self.session.feed_samples, samples)
                except AudioError as exc:
                    await self.warn("decode", str(exc))
                    continue
                except Exception as exc:
                    log.error("audio ingest failed: %s", type(exc).__name__)
                    await self.warn("ingest", "An audio chunk could not be processed.")
                    continue
                await self.emit(msgs, tasks)
        except asyncio.CancelledError:
            raise

    async def emit(self, messages, tasks):
        for m in messages:
            await self.send(m)
        for t in tasks:
            self.dispatch(t)

    def dispatch(self, task):
        if task.kind == "final":
            self.final_q.put_nowait(task)
        elif task.kind == "interim":
            self.spawn(self.run_interim(task))
        elif task.kind == "translate":
            self.spawn(self.run_translate(task))

    async def final_worker(self):
        while True:
            task = await self.final_q.get()
            try:
                if task is None:
                    return
                self.asr_active += 1
                await self.refresh_status()
                await asr_gate.acquire(0)
                try:
                    outcome = await self.loop.run_in_executor(asr_executor, self.session.run, task)
                finally:
                    asr_gate.release()
                    self.asr_active -= 1
                await self.emit(outcome.messages, outcome.followups)
                await self.refresh_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("final segment failed: %s", type(exc).__name__)
                await self.warn("final", "A phrase could not be transcribed.")
            finally:
                self.final_q.task_done()

    async def run_interim(self, task):
        if asr_gate.waiting >= SHED_WAITING or self.state != "recording":
            self.session.abandon(task)
            return
        acquired = False
        try:
            await asr_gate.acquire(2)
            acquired = True
            self.asr_active += 1
            await self.refresh_status()
            outcome = await self.loop.run_in_executor(asr_executor, self.session.run, task)
            await self.emit(outcome.messages, outcome.followups)
        except asyncio.CancelledError:
            self.session.abandon(task)
            raise
        except Exception as exc:
            log.error("interim update failed: %s", type(exc).__name__)
            self.session.abandon(task)
        finally:
            if acquired:
                self.asr_active -= 1
                asr_gate.release()
        await self.refresh_status()

    async def run_translate(self, task):
        if not task.final:
            if self.interim_mt or mt_gate.waiting >= SHED_WAITING or self.state != "recording":
                return
            self.interim_mt = True
        acquired = False
        try:
            await mt_gate.acquire(0 if task.final else 1)
            acquired = True
            self.mt_active += 1
            await self.refresh_status()
            outcome = await self.loop.run_in_executor(mt_executor, self.session.run, task)
            await self.emit(outcome.messages, outcome.followups)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("translation task failed: %s", type(exc).__name__)
        finally:
            if acquired:
                self.mt_active -= 1
                mt_gate.release()
            if not task.final:
                self.interim_mt = False

    async def on_stop(self):
        timeout = float(STREAM["drain_timeout_s"])
        deadline = time.monotonic() + timeout
        await self.send({"type": "status", "status": "processing", "message": "Processing the final audio segment"})
        self.last_status = ("processing", time.monotonic())
        timed_out = False
        try:
            self.audio_q.put_nowait(None)
            await asyncio.wait_for(self.ingest_task, max(1.0, deadline - time.monotonic()))
            tasks, msgs = await self.loop.run_in_executor(cpu_executor, self.session.flush)
            await self.emit(msgs, tasks)
            self.final_q.put_nowait(None)
            await asyncio.wait_for(self.final_task, max(1.0, deadline - time.monotonic()))
            while self.tasks:
                left = deadline - time.monotonic()
                if left <= 0:
                    timed_out = True
                    break
                await asyncio.wait(set(self.tasks), timeout=left)
        except asyncio.TimeoutError:
            timed_out = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("stop failed: %s", type(exc).__name__)
            await self.error("Live transcription could not be finished cleanly.")
        if timed_out:
            await self.warn("drain", "Some final captions did not finish in time.", 0)
        duration = self.session.seconds_received if self.session else 0.0
        finals = self.session.finals if self.session else 0
        await self.send({"type": "status", "status": "completed",
                         "message": f"Live transcription finished ({duration:.1f} seconds of audio, {finals} captions)",
                         "duration_s": round(duration, 2), "segments": finals})
        log_event(log, "live_done", session=self.session_id, audio_s=round(duration, 1), segments=finals)
        await self.end_session()
        self.state = "idle"

    async def end_session(self):
        for t in (self.ingest_task, self.final_task):
            if t is not None and not t.done():
                t.cancel()
        for t in list(self.tasks):
            t.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.session is not None:
            ttl = float(STREAM["session_resume_ttl_s"])
            if self.session_id and ttl > 0:
                RESUME[self.session_id] = (self.session.export_state(), time.monotonic() + ttl)
            session = self.session
            self.session = None
            await self.loop.run_in_executor(cpu_executor, session.close)
        if self.counted:
            ACTIVE["live"] = max(0, ACTIVE["live"] - 1)
            self.counted = False
        self.ingest_task = self.final_task = None
        self.audio_q = self.final_q = None

    async def cleanup(self):
        self.closed = True
        if self.stop_task is not None and not self.stop_task.done():
            self.stop_task.cancel()
        await self.end_session()


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    log.info("live websocket connected")
    conn = LiveConnection(websocket)
    await conn.run()
    log.info("live websocket closed")
