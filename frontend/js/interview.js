/* AI Interview section (v1: read prepared questions aloud with TTS).
 *
 * Independent of the Meeting Assistant code: it does not use audio.js, WebSocket or Captions.
 * Out of scope on purpose: candidate recording, speech recognition, follow-ups, evaluation, scoring.
 *
 * TTS contract (not yet verified against the real server, change buildTtsBody() if it differs):
 *   POST {baseUrl}{ttsPath}   JSON {"text": "...", "language": "English", "speaker": "ryan"}
 *   -> 200 audio/wav (binary). Errors: non-2xx, ideally JSON {"detail"|"error"|"message": "..."}.
 * Optional question source: GET {baseUrl}{questionsPath} -> ["q1", ...] or
 *   {"candidate": "...", "role": "...", "interview_id": "...", "questions": ["q1" | {"text": "..."}, ...]}
 *   Leave questionsPath empty to use the sample list below.
 *
 * Override any setting before this script loads:
 *   <script>window.INTERVIEW_CONFIG = { baseUrl: 'https://gpu.example.com', voices: [...] };</script>
 */
(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const apiCfg = (typeof Api !== 'undefined' && Api.config) ? Api.config : {};

  const cfg = Object.assign({
    baseUrl: apiCfg.baseUrl || 'http://localhost:8000', // temporary Kaggle URL now, production GPU server later
    ttsPath: '/tts',
    questionsPath: '',          // e.g. '/interview/questions'; empty = sample questions
    timeoutMs: 120000,          // first request on a cold GPU can be slow
    pauseBetweenMs: 1200,       // silence between one question ending and the next being requested
    headers: {},                // extra request headers
    languages: [
      { code: 'en', name: 'English' },
      { code: 'de', name: 'German' },
      { code: 'fr', name: 'French' },
      { code: 'es', name: 'Spanish' },
      { code: 'it', name: 'Italian' },
      { code: 'ja', name: 'Japanese' },
      { code: 'zh', name: 'Chinese' },
      { code: 'ko', name: 'Korean' },
      { code: 'ru', name: 'Russian' },
      { code: 'pt', name: 'Portuguese' }
    ],
    voices: [   // values must match GET /voices "speakers" exactly (lowercase)
      { value: 'ryan', label: 'Ryan' },
      { value: 'aiden', label: 'Aiden' },
      { value: 'dylan', label: 'Dylan' },
      { value: 'eric', label: 'Eric' },
      { value: 'vivian', label: 'Vivian' },
      { value: 'serena', label: 'Serena' },
      { value: 'ono_anna', label: 'Ono Anna' },
      { value: 'sohee', label: 'Sohee' },
      { value: 'uncle_fu', label: 'Uncle Fu' }
    ],
    defaultLanguage: 'en',
    defaultVoice: 'ryan'
  }, window.INTERVIEW_CONFIG || {});

  const SAMPLE = {
    candidate: 'Sample candidate',
    role: 'Junior Python Developer',
    id: 'sample-001',
    questions: [
      'Please introduce yourself and tell us a little about your background.',
      'How does Python manage memory for objects?',
      'What does a database index speed up, and what does it cost?',
      'Walk us through a REST API you have built. What did you use and why?',
      'How would you handle validation and errors in a FastAPI endpoint?',
      'Tell us about a bug that was hard to find, and how you fixed it.'
    ]
  };

  const LS_BASE = 'interview.baseUrl';
  const lsGet = k => { try { return localStorage.getItem(k); } catch { return null; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } };

  /* ---------------- state ---------------- */
  const st = {
    questions: [], meta: { candidate: '-', role: '-', id: '-' }, mock: true,
    phase: 'idle',      // idle | running | completed
    sub: '',            // loading | preparing | held | blocked | speaking | gap | error | done
    paused: false, stopped: false, advance: true,
    index: -1, run: 0,
    ctrl: null, timer: null, error: '',
    url: null, urlKey: ''
  };
  const audio = new Audio();
  audio.preload = 'auto';

  /* ---------------- helpers ---------------- */
  const base = () => (lsGet(LS_BASE) || cfg.baseUrl).trim().replace(/\/+$/, '');
  const current = () => st.questions[st.index] || '';
  const total = () => st.questions.length;

  function releaseUrl() {
    if (st.url) { try { URL.revokeObjectURL(st.url); } catch {} }
    st.url = null; st.urlKey = '';
  }
  function cancelNetwork() {
    if (st.ctrl) { st.ctrl.abort(); st.ctrl = null; }
  }
  function clearTimer() {
    if (st.timer) { clearTimeout(st.timer); st.timer = null; }
  }
  function silence() { audio.pause(); }

  function cleanQuestion(q) {
    const t = typeof q === 'string' ? q : (q && (q.text || q.question));
    return typeof t === 'string' ? t.trim() : '';
  }
  function normalizeQuestions(d) {
    const arr = Array.isArray(d) ? d : (d && Array.isArray(d.questions) ? d.questions : []);
    const questions = arr.map(cleanQuestion).filter(Boolean);
    const o = Array.isArray(d) ? {} : (d || {});
    return {
      questions,
      candidate: o.candidate || o.candidate_name || '-',
      role: o.role || '-',
      id: o.interview_id || o.id || '-'
    };
  }

  /* ---------------- questions ---------------- */
  async function loadQuestions() {
    if (!cfg.questionsPath) {
      return { ...SAMPLE, questions: SAMPLE.questions.slice(), mock: true };
    }
    const ctrl = new AbortController();
    st.ctrl = ctrl;
    const timer = setTimeout(() => ctrl.abort(), 15000);
    try {
      const r = await fetch(base() + cfg.questionsPath, { headers: reqHeaders(), signal: ctrl.signal });
      if (!r.ok) throw new Error(`Questions endpoint returned ${r.status}.`);
      const n = normalizeQuestions(await r.json());
      if (!n.questions.length) throw new Error('The interview has no questions.');
      return { ...n, mock: false };
    } finally {
      clearTimeout(timer);
      if (st.ctrl === ctrl) st.ctrl = null;
    }
  }

  /* ---------------- TTS ---------------- */
  function reqHeaders(extra) {
    const h = Object.assign({}, cfg.headers, extra);
    if (/ngrok/i.test(base())) h['ngrok-skip-browser-warning'] = 'true'; // free ngrok shows an HTML interstitial otherwise
    return h;
  }
  function buildTtsBody(text) {
    const languageNames = {
      en: 'English',
      de: 'German',
      fr: 'French',
      es: 'Spanish',
      it: 'Italian',
      ja: 'Japanese',
      zh: 'Chinese',
      ko: 'Korean',
      ru: 'Russian',
      pt: 'Portuguese'
    };

    return {
      text: text,
      language: languageNames[$('ivLang').value],
      speaker: $('ivVoice').value
    };
  }
  async function errorFromResponse(r) {
    let detail = '';
    try {
      const ct = r.headers.get('content-type') || '';
      if (ct.includes('json')) {
        const j = await r.json();
        const d = j.detail ?? j.error ?? j.message;
        detail = typeof d === 'string' ? d : (d ? JSON.stringify(d) : '');
      }
    } catch { /* ignore body */ }
    if (r.status === 404) return `TTS endpoint not found (POST ${cfg.ttsPath}). Check the backend URL.`;
    if (r.status === 422 || r.status === 400) return `The TTS server rejected the request${detail ? ': ' + detail : '.'}`;
    if (r.status >= 500) return `The TTS server failed (${r.status})${detail ? ': ' + detail : '.'}`;
    return `TTS request failed (${r.status})${detail ? ': ' + detail : '.'}`;
  }

  async function fetchTts(text) {
    const ctrl = new AbortController();
    st.ctrl = ctrl;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, cfg.timeoutMs);
    try {
      const r = await fetch(base() + cfg.ttsPath, {
        method: 'POST',
        headers: reqHeaders({ 'Content-Type': 'application/json', 'Accept': 'audio/wav, audio/*' }),
        body: JSON.stringify(buildTtsBody(text)),
        signal: ctrl.signal
      });
      if (!r.ok) throw new Error(await errorFromResponse(r));
      const ct = (r.headers.get('content-type') || '').toLowerCase();
      if (ct.includes('text/html')) {
        throw new Error('The TTS URL returned a web page instead of audio. Check the backend URL (a tunnel warning page can cause this).');
      }
      if (ct.includes('json')) {
        throw new Error('The TTS server returned JSON instead of audio.');
      }
      const blob = await r.blob();
      if (blob.size < 100) throw new Error('The TTS server returned empty audio.');
      return blob.type && blob.type.startsWith('audio/') ? blob : new Blob([blob], { type: 'audio/wav' });
    } catch (e) {
      if (e.name === 'AbortError') {
        if (timedOut) throw new Error(`TTS timed out after ${Math.round(cfg.timeoutMs / 1000)} s.`);
        throw e; // cancelled by Stop or Replay
      }
      if (e instanceof TypeError) {
        throw new Error('Cannot reach the TTS server. Check the backend URL, that the server is running, and that CORS allows this page.');
      }
      throw e;
    } finally {
      clearTimeout(timer);
      if (st.ctrl === ctrl) st.ctrl = null;
    }
  }

  /* ---------------- playback flow ---------------- */
  function bindAudio() {
    audio.onended = () => { if (st.sub === 'speaking') onSpoken(); };
    audio.onerror = () => {
      if (st.sub !== 'speaking') return;
      fail('The audio could not be played. The server response may not be a valid WAV file.');
    };
  }
  bindAudio();

  function fail(msg) {
    clearTimer();
    audio.pause();
    st.sub = 'error';
    st.error = msg;
    render();
  }

  // Request (or reuse) audio for one question and play it. Only one run is ever active: a newer run
  // cancels the older one, and stale async results are dropped via the run id.
  async function speak(index, advance) {
    cancelNetwork();
    clearTimer();
    silence();
    const id = ++st.run;
    st.index = index;
    st.advance = advance;
    st.error = '';
    st.sub = 'preparing';
    render();

    const key = `${index}|${$('ivLang').value}|${$('ivVoice').value}`;
    if (st.url && st.urlKey === key) {
      /* replay: reuse the audio already generated for this question, no new request */
    } else {
      releaseUrl();
      try {
        const blob = await fetchTts(st.questions[index]);
        if (id !== st.run) return;
        st.url = URL.createObjectURL(blob);
        st.urlKey = key;
      } catch (e) {
        if (id !== st.run || e.name === 'AbortError') return;
        fail(e.message || 'TTS request failed.');
        return;
      }
    }
    if (st.paused) { st.sub = 'held'; render(); return; }
    playLoaded(id);
  }

  async function playLoaded(id) {
    audio.src = st.url;
    audio.currentTime = 0;
    st.sub = 'speaking';
    render();
    try {
      await audio.play();
    } catch (e) {
      if (id !== st.run || e.name === 'AbortError') return;
      if (e.name === 'NotAllowedError') { st.sub = 'blocked'; render(); return; } // autoplay policy
      fail('The audio could not be played.');
    }
  }

  function onSpoken() {
    if (!st.advance) { st.sub = 'done'; render(); return; }          // replay after completion
    if (st.index >= total() - 1) { st.phase = 'completed'; st.sub = 'done'; render(); return; }
    st.sub = 'gap';
    render();
    if (!st.paused) scheduleNext(cfg.pauseBetweenMs);
  }

  function scheduleNext(ms) {
    clearTimer();
    const id = st.run;
    st.timer = setTimeout(() => {
      st.timer = null;
      if (id === st.run && st.sub === 'gap' && !st.paused) speak(st.index + 1, true);
    }, ms);
  }

  /* ---------------- controls ---------------- */
  async function start() {
    if (st.phase === 'running' || st.sub === 'loading') return;
    cancelNetwork(); clearTimer(); silence(); releaseUrl();
    const id = ++st.run;
    st.stopped = false; st.paused = false; st.error = '';
    st.sub = 'loading';
    render();
    try {
      const q = await loadQuestions();
      if (id !== st.run) return;
      st.questions = q.questions;
      st.meta = { candidate: q.candidate, role: q.role, id: q.id };
      st.mock = q.mock;
    } catch (e) {
      if (id !== st.run) return;
      st.sub = 'error'; st.phase = 'idle';
      st.error = e.name === 'AbortError' || e instanceof TypeError
        ? 'Could not load the interview questions. Check the backend URL.'
        : e.message;
      render();
      return;
    }
    st.phase = 'running';
    speak(0, true);
  }

  function pause() {
    if (st.phase !== 'running' || st.paused || st.sub === 'error' || st.sub === 'blocked') return;
    st.paused = true;
    if (st.sub === 'speaking') audio.pause();
    if (st.sub === 'gap') clearTimer();
    render();            // a request that is still in flight finishes quietly and then waits (held)
  }

  function resume() {
    if (!st.paused) return;
    st.paused = false;
    if (st.sub === 'speaking') {
      const id = st.run;
      audio.play().catch(e => { if (id === st.run && e.name !== 'AbortError') { st.sub = 'blocked'; render(); } });
    } else if (st.sub === 'held') {
      playLoaded(st.run);
    } else if (st.sub === 'gap') {
      scheduleNext(0);
    }
    render();
  }

  function stop() {
    st.run++;
    cancelNetwork(); clearTimer(); silence(); releaseUrl();
    st.phase = 'idle'; st.sub = ''; st.index = -1;
    st.paused = false; st.error = ''; st.stopped = true;
    render();
  }

  // Replays (or retries) the current question. It never moves to another question by itself;
  // during a running interview the normal flow continues after the replay ends.
  function replay() {
    if (st.index < 0 || st.sub === 'preparing' || st.sub === 'loading') return;
    st.paused = false;
    speak(st.index, st.phase === 'running');
  }

  /* ---------------- rendering ---------------- */
  const STATUS = {
    idle: ['waiting', 'Ready'], stopped: ['waiting', 'Stopped'], loading: ['preparing', 'Loading questions'],
    preparing: ['preparing', 'Preparing audio'], speaking: ['speaking', 'Speaking'], gap: ['speaking', 'Next question coming up'],
    held: ['paused', 'Paused'], blocked: ['paused', 'Press Play to hear the question'],
    paused: ['paused', 'Paused'], error: ['error', 'Error'], completed: ['completed', 'Completed']
  };

  function statusKey() {
    if (st.sub === 'error') return 'error';
    if (st.sub === 'loading') return 'loading';
    if (st.paused) return 'paused';
    if (['preparing', 'speaking', 'gap', 'held', 'blocked'].includes(st.sub)) return st.sub;
    if (st.phase === 'completed') return 'completed';
    return st.stopped ? 'stopped' : 'idle';
  }

  function render() {
    const key = statusKey();
    const [state, label] = STATUS[key];
    $('ivStatus').dataset.state = state;
    $('ivStatusText').textContent = label;

    $('ivCandidate').textContent = st.meta.candidate;
    $('ivRole').textContent = st.meta.role;
    $('ivId').textContent = st.meta.id;
    $('ivSource').hidden = !(st.mock && st.questions.length);

    const n = total();
    const active = st.index >= 0 && (st.phase !== 'idle');
    $('ivCounter').textContent = active ? `Question ${st.index + 1} of ${n}` : (st.sub === 'loading' ? 'Loading…' : (n ? `${n} questions` : 'Not started'));
    const pct = n && active ? Math.round(((st.phase === 'completed' ? n : Math.max(0, st.index)) / n) * 100) : 0;
    $('ivProgressBar').style.width = pct + '%';
    $('ivProgress').setAttribute('aria-valuenow', String(pct));

    const q = $('ivQuestion');
    if (active && current()) {
      q.textContent = current(); q.classList.remove('placeholder');
    } else {
      q.textContent = st.phase === 'idle'
        ? 'Press Start Interview. Each question is read aloud, one at a time.'
        : ''; q.classList.add('placeholder');
    }

    const err = $('ivError');
    err.textContent = st.error; err.hidden = !st.error;

    const running = st.phase === 'running';
    const busy = st.sub === 'loading';
    $('ivBtnStart').disabled = running || busy;
    $('ivBtnPause').disabled = !running || st.paused || st.sub === 'error' || st.sub === 'blocked';
    $('ivBtnResume').disabled = !running || !st.paused;
    $('ivBtnStop').disabled = !(running || busy);

    const rp = $('ivBtnReplay');
    rp.disabled = st.index < 0 || st.sub === 'preparing' || busy || (st.phase === 'idle' && st.sub !== 'error');
    rp.textContent = st.sub === 'error' ? 'Retry question' : (st.sub === 'blocked' || st.sub === 'held' ? 'Play question' : 'Replay question');

    $('ivWave').classList.toggle('on', st.sub === 'speaking' && !st.paused);
    ['ivLang', 'ivVoice', 'ivBase'].forEach(id => { $(id).disabled = running || busy; });

    const hint = $('ivHint');
    const needEn = st.mock && $('ivLang').value !== 'en';
    hint.hidden = !needEn;
    hint.textContent = needEn ? 'The sample questions are in English. Choose English, or connect real questions, for matching speech.' : '';
  }

  /* ---------------- navigation (independent of app.js tab logic) ---------------- */
  const view = $('viewInterview'), tab = $('tabInterview');
  const otherTabs = ['tabTranscribe', 'tabSummary'];

  function openView() {
    $('viewTranscribe').hidden = true;
    $('viewSummary').hidden = true;
    view.hidden = false;
    otherTabs.forEach(id => { const t = $(id); t.classList.remove('active'); t.removeAttribute('aria-current'); });
    tab.classList.add('active'); tab.setAttribute('aria-current', 'page');
    try { history.replaceState(null, '', '#interview'); } catch {}
  }
  function leaveView() {
    if (view.hidden) return;
    view.hidden = true;
    tab.classList.remove('active'); tab.removeAttribute('aria-current');
    pause(); // do not keep talking from a hidden screen
  }

  /* ---------------- init ---------------- */
  function init() {
    if (!cfg.questionsPath) { st.questions = SAMPLE.questions.slice(); st.meta = { candidate: SAMPLE.candidate, role: SAMPLE.role, id: SAMPLE.id }; st.mock = true; }
    cfg.languages.forEach(l => $('ivLang').add(new Option(l.name || l.code, l.code)));
    $('ivLang').value = cfg.defaultLanguage;
    cfg.voices.forEach(v => $('ivVoice').add(new Option(v.label || v.value, v.value)));
    $('ivVoice').value = cfg.defaultVoice;
    $('ivBase').value = base();
    $('ivPathLabel').textContent = cfg.ttsPath;

    $('ivBase').addEventListener('change', e => { lsSet(LS_BASE, e.target.value.trim()); });
    $('ivLang').addEventListener('change', render);
    $('ivBtnStart').onclick = start;
    $('ivBtnPause').onclick = pause;
    $('ivBtnResume').onclick = resume;
    $('ivBtnStop').onclick = stop;
    $('ivBtnReplay').onclick = replay;

    tab.addEventListener('click', openView);
    otherTabs.forEach(id => $(id).addEventListener('click', leaveView));
    window.addEventListener('beforeunload', () => { cancelNetwork(); audio.pause(); releaseUrl(); });

    render();
    if (location.hash === '#interview') openView();
  }

  init();
  window.Interview = { start, pause, resume, stop, replay, config: cfg };
})();