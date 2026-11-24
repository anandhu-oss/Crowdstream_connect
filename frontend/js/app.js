(() => {
  const $ = id => document.getElementById(id);
  const STATUS = {
    waiting: 'Waiting for audio', receiving: 'Receiving audio', detecting: 'Detecting language',
    transcribing: 'Transcribing', translating: 'Translating', completed: 'Completed', error: 'Error',
    uploading: 'Uploading, awaiting backend response', connecting: 'Connecting to backend…',
    recording: 'Recording, streaming audio', processing: 'Processing, waiting for remaining results',
    live: 'Live, streaming audio'
  };
  const CONN = { connected: 'Backend connected', disconnected: 'Backend disconnected', connecting: 'Connecting', mock: 'Mock mode (simulated)' };
  const SOCKET = {
    connecting: ['connecting', 'WebSocket connecting'], open: ['connected', 'WebSocket connected'],
    reconnecting: ['connecting', 'WebSocket reconnecting'], closed: ['disconnected', 'WebSocket closed']
  };
  const state = {
    mock: false, conn: 'connecting', rec: 'idle', cancel: false, uploading: false, file: null, stream: null,
    token: 0, group: 0, sessionError: false,
    live: 'idle', liveCancel: false, liveFailed: false, liveStream: null, liveToken: 0
  };

  function setStatus(key, text) {
    $('status').dataset.state = key;
    $('statusText').textContent = text || STATUS[key] || key;
  }
  function setConn(c) {
    state.conn = c;
    $('conn').dataset.state = c;
    $('connLabel').textContent = CONN[c];
  }
  function setSocket(key) {
    $('session').hidden = !key;
    if (!key) return;
    $('session').dataset.state = SOCKET[key][0];
    $('sessionLabel').textContent = SOCKET[key][1];
  }
  function showError(msg) { $('errorBar').textContent = msg; $('errorBar').hidden = false; }
  function clearError() { $('errorBar').hidden = true; }
  function micError(msg) { $('micError').textContent = msg || ''; $('micError').hidden = !msg; }
  function fileError(msg) { $('fileError').textContent = msg || ''; $('fileError').hidden = !msg; }
  function liveErr(msg) { $('liveError').textContent = msg || ''; $('liveError').hidden = !msg; }
  function setLiveLabel(t) { $('liveStatus').textContent = t; }
  function liveDot(on) { $('liveDot').classList.toggle('on', on); }
  function mmss(t) { return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`; }
  function recUi(on) {
    $('recDot').classList.toggle('on', on);
    $('recStatus').textContent = on ? 'Recording' : 'Not recording';
  }
  function uploadStatus(key, text) { if (state.rec === 'idle' && state.live === 'idle') setStatus(key, text); }

  function updateButtons() {
    const idle = state.rec === 'idle' && state.live === 'idle';
    $('btnStart').disabled = !idle;
    $('btnStop').disabled = !(state.rec === 'connecting' || state.rec === 'recording');
    $('btnStartLive').disabled = !idle;
    $('btnStopLive').disabled = !['connecting', 'live', 'reconnecting'].includes(state.live);
    $('btnUpload').disabled = !state.file || state.uploading;
    $('fileInput').disabled = state.uploading;
    $('srcLang').disabled = $('tgtLang').disabled = !idle;
    $('mockToggle').disabled = !idle || state.uploading;
  }

  function langParams() {
    return { source_language: $('srcLang').value, target_language: $('tgtLang').value };
  }

  function canUseBackend() {
    if (state.mock || state.conn === 'connected') return true;
    showError('Backend unavailable. Check the API base URL and that FastAPI is running, or enable mock mode to test the interface.');
    return false;
  }

  /* ---------- Test recording (unchanged) ---------- */

  function liveHandlersTest(group, token) {
    const alive = () => token === state.token && state.rec !== 'idle';
    return {
      onStatus: s => {
        if (!alive()) return;
        if (state.rec === 'recording' && s === 'completed') return;
        if (s === 'error') state.sessionError = true;
        setStatus(s);
      },
      onSegment: seg => { if (token === state.token) Captions.upsert(seg, group); },
      onError: msg => { if (token !== state.token) return; state.sessionError = true; showError(msg); },
      onState: s => { if (token === state.token) onSocketState(s, group); }
    };
  }

  function onSocketState(s, group) {
    if (s === 'open') setSocket('open');
    else if (s === 'reconnecting') {
      setSocket('reconnecting');
      if (state.rec === 'recording') setStatus('error', 'Live connection lost, reconnecting…');
    } else if (s === 'reconnected') {
      setSocket('open'); clearError();
      if (state.rec === 'recording') setStatus('recording');
    } else if (s === 'closed') endSession(group);
  }

  function endSession(group) {
    const wasRecording = state.rec === 'recording';
    if (!wasRecording && state.rec !== 'processing') return;
    if (wasRecording) {
      Recorder.stop(); recUi(false);
      state.sessionError = true;
      if ($('errorBar').hidden) showError('Live connection closed unexpectedly.');
    }
    Captions.finalize(group);
    state.rec = 'idle'; state.stream = null;
    setSocket(null);
    setStatus(state.sessionError ? 'error' : 'completed');
    updateButtons();
  }

  async function startRecording() {
    if (state.rec !== 'idle' || state.live !== 'idle') return;
    clearError(); micError();
    if (!canUseBackend()) return;
    const token = ++state.token;
    const group = ++state.group;
    state.rec = 'connecting'; state.cancel = false; state.sessionError = false;
    updateButtons(); setStatus('connecting'); setSocket('connecting');
    const cancelled = () => { if (state.cancel) throw new Error('cancelled'); };
    let stream = null;
    try {
      const mime = await Recorder.acquire();
      cancelled();
      const h = liveHandlersTest(group, token);
      stream = state.mock ? Mock.createStream(h) : WS.createStream(h);
      state.stream = stream;
      await stream.open({ ...langParams(), mime_type: mime });
      cancelled();
      Recorder.begin(chunk => stream.send(chunk), t => { $('recTime').textContent = mmss(t); }, onMicFail);
    } catch (e) {
      const wasCancelled = state.cancel;
      state.rec = 'idle'; state.cancel = false; state.stream = null;
      if (stream) stream.close();
      Recorder.release();
      setSocket(null);
      if (wasCancelled) setStatus('waiting');
      else {
        if (e.mic) micError(e.message); else showError(e.message);
        setStatus('error');
      }
      updateButtons();
      return;
    }
    state.rec = 'recording';
    recUi(true); $('recTime').textContent = '00:00';
    setStatus('recording');
    updateButtons();
  }

  function onMicFail(msg) {
    state.sessionError = true;
    micError(msg);
    stopRecording();
  }

  async function stopRecording() {
    if (state.rec === 'connecting') {
      state.cancel = true;
      if (state.stream) state.stream.close();
      return;
    }
    if (state.rec !== 'recording') return;
    const stream = state.stream, token = state.token;
    state.rec = 'processing';
    recUi(false); setStatus('processing'); updateButtons();
    await Recorder.stop();
    if (token === state.token && state.rec === 'processing' && stream) stream.stop();
  }

  /* ---------- Live Meeting (real backend only) ---------- */

  function liveHandlers(group, token) {
    const alive = () => token === state.liveToken;
    return {
      onStatus: s => {
        if (!alive() || state.live === 'idle') return;
        if (s === 'completed' && state.live !== 'processing') return;
        if (s === 'error') state.liveFailed = true;
        if (s === 'waiting' || s === 'receiving') setStatus(state.live === 'processing' ? 'processing' : 'live');
        else setStatus(s);
      },
      onSegment: (seg, epoch) => { if (alive()) Captions.upsert(seg, group + (epoch || 0) / 1000); },
      onRemove: (id, epoch) => { if (alive()) Captions.remove(id, group + (epoch || 0) / 1000); },
      onError: msg => { if (!alive()) return; state.liveFailed = true; showError(msg); },
      onWarn: msg => { if (alive()) liveErr(msg); },
      onState: s => { if (alive()) onLiveSocketState(s, group); }
    };
  }

  function onLiveSocketState(s, group) {
    if (s === 'open') setSocket('open');
    else if (s === 'reconnecting') {
      setSocket('reconnecting');
      if (state.live === 'live') {
        state.live = 'reconnecting';
        setLiveLabel('Connection lost, reconnecting…');
        setStatus('error', 'Live connection lost, reconnecting…');
        updateButtons();
      }
    } else if (s === 'reconnected') {
      setSocket('open'); clearError(); liveErr();
      if (state.live === 'reconnecting') {
        state.live = 'live';
        setLiveLabel('Live, streaming audio'); setStatus('live');
        updateButtons();
      }
    } else if (s === 'closed') endLive(group);
  }

  function endLive(group) {
    const was = state.live;
    if (was !== 'live' && was !== 'reconnecting' && was !== 'processing') return;
    if (was !== 'processing') {
      LiveAudio.stop();
      state.liveFailed = true;
      if ($('errorBar').hidden) showError('Live connection closed unexpectedly.');
    }
    liveDot(false);
    Captions.finalize(group);
    state.live = 'idle'; state.liveStream = null;
    setSocket(null);
    setLiveLabel(state.liveFailed ? 'Live session ended with errors' : 'Live session ended');
    setStatus(state.liveFailed ? 'error' : 'completed');
    updateButtons();
  }

  async function startLive() {
    if (state.live !== 'idle' || state.rec !== 'idle') return;
    clearError(); liveErr();
    if (state.mock) { liveErr('Live Meeting uses the real backend only. Turn off Mock mode to use it.'); return; }
    if (state.conn !== 'connected') {
      showError('Backend unavailable. Check the API base URL and that FastAPI is running.');
      return;
    }
    const token = ++state.liveToken;
    const group = ++state.group;
    state.live = 'connecting'; state.liveCancel = false; state.liveFailed = false;
    updateButtons(); setLiveLabel('Requesting microphone…'); setStatus('connecting'); setSocket('connecting');
    const cancelled = () => { if (state.liveCancel) throw new Error('cancelled'); };
    let stream = null;
    try {
      await LiveAudio.acquire();
      cancelled();
      setLiveLabel('Connecting to backend…');
      stream = WS.createLiveStream(liveHandlers(group, token));
      state.liveStream = stream;
      const sessionId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now());
      await stream.open({ ...langParams(), mode: 'live', ...LiveAudio.format(), session_id: sessionId });
      cancelled();
      LiveAudio.begin(chunk => stream.send(chunk), t => { $('liveTime').textContent = mmss(t); }, onLiveMicFail);
    } catch (e) {
      const wasCancelled = state.liveCancel;
      state.live = 'idle'; state.liveCancel = false; state.liveStream = null;
      if (stream) stream.close();
      LiveAudio.release();
      setSocket(null); liveDot(false); setLiveLabel('Not live');
      if (wasCancelled) setStatus('waiting');
      else {
        if (e.mic) liveErr(e.message); else showError(e.message);
        setStatus('error');
      }
      updateButtons();
      return;
    }
    state.live = 'live';
    liveDot(true); $('liveTime').textContent = '00:00';
    setLiveLabel('Live, streaming audio'); setStatus('live');
    updateButtons();
  }

  function onLiveMicFail(msg) {
    state.liveFailed = true;
    liveErr(msg);
    stopLive();
  }

  async function stopLive() {
    if (state.live === 'connecting') {
      state.liveCancel = true;
      if (state.liveStream) state.liveStream.close();
      return;
    }
    if (state.live !== 'live' && state.live !== 'reconnecting') return;
    const stream = state.liveStream, token = state.liveToken;
    state.live = 'processing';
    liveDot(false); setLiveLabel('Stopping, processing remaining audio…'); setStatus('processing');
    updateButtons();
    await LiveAudio.stop();
    if (token === state.liveToken && state.live === 'processing' && stream) stream.stop();
  }

  /* ---------- Upload (unchanged) ---------- */

  async function upload() {
    if (!state.file || state.uploading) return;
    clearError(); fileError();
    if (!canUseBackend()) return;
    const group = ++state.group;
    state.uploading = true; updateButtons();
    uploadStatus('uploading'); $('fileStatus').textContent = 'Uploading…';
    try {
      if (state.mock) {
        await Mock.uploadAudio(state.file, langParams(), {
          onStatus: s => uploadStatus(s),
          onSegment: seg => Captions.upsert(seg, group)
        });
      } else {
        const segs = await Api.uploadAudio(state.file, langParams());
        segs.forEach(s => Captions.upsert(s, group));
        if (!segs.length) showError('The backend returned no speech segments for this file.');
        uploadStatus('completed');
      }
      Captions.finalize(group);
      $('fileStatus').textContent = 'Processing finished';
    } catch (e) {
      uploadStatus('error'); showError(e.message); $('fileStatus').textContent = 'Upload failed';
    } finally { state.uploading = false; updateButtons(); }
  }

  async function onFile() {
    const file = $('fileInput').files[0];
    state.file = null; fileError(); $('fileStatus').textContent = '';
    $('fileName').textContent = 'No file selected'; $('fileDuration').textContent = '';
    if (!file) return updateButtons();
    $('fileName').textContent = file.name;
    try {
      const d = await Files.inspect(file);
      state.file = file;
      $('fileDuration').textContent = d != null ? mmss(d) : '';
    } catch (e) { fileError(e.message); }
    updateButtons();
  }

  function fillSelect(sel, list, def, withAuto) {
    sel.replaceChildren();
    if (withAuto) sel.add(new Option('Auto Detect', 'auto'));
    list.forEach(l => sel.add(new Option(l.name, l.code)));
    sel.value = def;
  }

  async function loadLanguages() {
    const remote = state.mock ? null : await Api.getLanguages();
    const src = remote?.source?.length ? remote.source : Api.config.fallbackLanguages;
    const tgt = remote?.target?.length ? remote.target : Api.config.fallbackLanguages;
    fillSelect($('srcLang'), src.filter(l => l.code !== 'auto'), 'auto', true);
    fillSelect($('tgtLang'), tgt, tgt.some(l => l.code === 'en') ? 'en' : tgt[0].code, false);
    $('transTitle').textContent = 'Translated transcription';
  }

  async function checkHealth() {
    if (state.mock) return setConn('mock');
    const ok = await Api.health();
    if (state.mock) return;
    setConn(ok ? 'connected' : 'disconnected');
  }

  async function setMock(on) {
    state.mock = on;
    $('mockBanner').hidden = !on;
    clearError(); liveErr();
    setConn(on ? 'mock' : 'connecting');
    await loadLanguages();
    if (!on) checkHealth();
  }

  function init() {
    Captions.init($('origList'), $('transList'));
    $('btnStart').onclick = startRecording;
    $('btnStop').onclick = stopRecording;
    $('btnStartLive').onclick = startLive;
    $('btnStopLive').onclick = stopLive;
    $('btnUpload').onclick = upload;
    $('btnClear').onclick = Captions.clear;
    $('fileInput').onchange = onFile;
    $('mockToggle').onchange = e => setMock(e.target.checked);
    window.addEventListener('beforeunload', () => {
      Recorder.release(); LiveAudio.release();
      if (state.stream) state.stream.close();
      if (state.liveStream) state.liveStream.close();
    });
    setStatus('waiting'); updateButtons();
    loadLanguages();
    checkHealth();
    setInterval(checkHealth, 5000);
  }

  init();
})();

/* ===================== Summarization section =====================
   Independent of the transcription / live code above: it only needs the
   Summarization tab and Api.summarize(). It works with manually entered text and
   never touches audio, WebSockets or the caption state.

   Future integration (no UI change needed): once a session has a final transcript,
     Summary.run({ segments: Captions.getTranscript(), meetingId: 'abc' })
   fills the Summarization tab and calls POST /summarize with `segments`
   (or Summary.run({ transcript: '...' }) for plain text). */
const Summary = (() => {
  const $ = id => document.getElementById(id);
  const UNSPEC = 'Unspecified';
  const MIN_CHARS = 20; // matches the backend minimum
  const STATUS = { open: 'Open', in_progress: 'In progress', done: 'Done', unspecified: UNSPEC };
  const st = { busy: false, ctrl: null, segments: null, result: null, title: '', ticker: null, flashT: null };

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function has(v) { return typeof v === 'string' && v.trim() !== ''; }
  function clean(v) { return has(v) ? v.trim() : null; }
  function unspec(tag = 'p') { return el(tag, 'unspec', UNSPEC); }

  /* ---------- navigation ---------- */
  function show(view) {
    const isSum = view === 'viewSummary';
    $('viewTranscribe').hidden = isSum;
    $('viewSummary').hidden = !isSum;
    const tabs = { tabTranscribe: !isSum, tabSummary: isSum };
    Object.entries(tabs).forEach(([id, on]) => {
      $(id).classList.toggle('active', on);
      if (on) $(id).setAttribute('aria-current', 'page'); else $(id).removeAttribute('aria-current');
    });
    try { history.replaceState(null, '', isSum ? '#summarization' : location.pathname + location.search); } catch {}
  }

  /* ---------- small UI helpers ---------- */
  function showErr(msg) { $('sumError').textContent = msg; $('sumError').hidden = false; }
  function clearErr() { $('sumError').hidden = true; $('sumError').textContent = ''; }
  function updateCount() {
    const n = $('sumTranscript').value.length;
    $('sumCount').textContent = `${n.toLocaleString()} character${n === 1 ? '' : 's'}`;
  }
  function setBusy(on) {
    st.busy = on;
    ['sumTranscript', 'sumTitle', 'sumLang', 'btnSummarize', 'btnSumClear'].forEach(id => { $(id).disabled = on; });
    $('sumLoading').hidden = !on;
    clearInterval(st.ticker);
    if (!on) return;
    const t0 = Date.now();
    const tick = () => {
      const s = Math.floor((Date.now() - t0) / 1000);
      $('sumLoadingText').textContent = s < 20
        ? `Summarizing… ${s}s`
        : `Summarizing… ${s}s (long transcripts are processed in parts and can take a few minutes)`;
    };
    tick();
    st.ticker = setInterval(tick, 1000);
  }

  /* ---------- transcript helpers ---------- */
  function hms(t) {
    const s = Math.max(0, Math.floor(t || 0));
    return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map(x => String(x).padStart(2, '0')).join(':');
  }
  function segmentsToText(segs) {
    return segs.map(s => {
      const ts = s.start != null ? `[${hms(s.start)}] ` : '';
      const sp = s.speaker ? `${s.speaker}: ` : '';
      return ts + sp + s.text;
    }).join('\n');
  }

  /* ---------- rendering ---------- */
  function listOrUnspec(items) {
    if (!Array.isArray(items) || !items.length) return unspec();
    const ul = el('ul');
    items.forEach(i => { const li = el('li', null, i); li.dir = 'auto'; ul.append(li); });
    return ul;
  }
  function textOrUnspec(v) {
    if (!has(v)) return unspec();
    const p = el('p', null, v.trim()); p.dir = 'auto'; return p;
  }
  function block(title, content, full) {
    const b = el('section', 'sum-block' + (full ? ' full' : ''));
    b.append(el('h3', null, title), content);
    return b;
  }
  function cell(v, tag = 'td') {
    const c = el(tag);
    if (has(v)) { c.textContent = v.trim(); c.dir = 'auto'; } else { c.className = 'unspec'; c.textContent = UNSPEC; }
    return c;
  }
  function actionTable(items) {
    if (!items.length) return unspec();
    const wrap = el('div', 'sum-table-wrap'), table = el('table', 'sum-table');
    const head = el('tr');
    ['Task', 'Assigned to', 'Deadline', 'Status'].forEach(h => head.append(el('th', null, h)));
    const thead = el('thead'); thead.append(head); table.append(thead);
    const body = el('tbody');
    items.forEach(a => {
      const tr = el('tr');
      tr.append(cell(a.task), cell(a.assigned_to), cell(a.deadline), cell(STATUS[a.status] || UNSPEC));
      body.append(tr);
    });
    table.append(body); wrap.append(table);
    return wrap;
  }
  function persons(items) {
    if (!items.length) return unspec();
    const counts = new Map();
    items.forEach(a => { const k = clean(a.assigned_to) || UNSPEC; counts.set(k, (counts.get(k) || 0) + 1); });
    const box = el('div', 'chips');
    counts.forEach((n, name) => {
      const c = el('span', 'chip' + (name === UNSPEC ? ' unspec' : ''), `${name} · ${n} task${n === 1 ? '' : 's'}`);
      c.dir = 'auto'; box.append(c);
    });
    return box;
  }
  function deadlines(items) {
    if (!items.length) return unspec();
    const ul = el('ul');
    items.forEach(a => {
      const li = el('li'); li.dir = 'auto';
      li.append(el('span', null, `${a.task}: `));
      li.append(has(a.deadline) ? el('strong', null, a.deadline.trim()) : unspec('span'));
      ul.append(li);
    });
    return ul;
  }

  function render(d) {
    const items = Array.isArray(d.action_items) ? d.action_items : [];
    const root = $('sumResult');
    root.replaceChildren(
      block('Meeting overview', textOrUnspec(d.meeting_overview), true),
      block('Executive summary', textOrUnspec(d.executive_summary), true),
      block('Key discussion points', listOrUnspec(d.key_discussion_points), true),
      block('Decisions made', listOrUnspec(d.decisions), true),
      block('Action items', actionTable(items), true),
      block('Assigned persons', persons(items)),
      block('Deadlines', deadlines(items)),
      block('Unresolved issues', listOrUnspec(d.unresolved_issues)),
      block('Next steps', listOrUnspec(d.next_steps))
    );
    $('sumResultTitle').textContent = st.title || 'Meeting summary';
    const bits = [`Output language: ${d.summary_language || $('sumLang').value}`];
    if (typeof d.processing_time_seconds === 'number') bits.push(`${d.processing_time_seconds.toFixed(1)}s`);
    if (d.model) bits.push(d.model);
    $('sumResultMeta').textContent = bits.join(' · ');
    const w = $('sumWarnings');
    const warnings = Array.isArray(d.warnings) ? d.warnings.filter(has) : [];
    w.textContent = warnings.length ? 'Notes: ' + warnings.join(' ') : '';
    w.hidden = !warnings.length;
    $('sumResultCard').hidden = false;
  }

  /* ---------- plain-text export (used by Copy and Download) ---------- */
  function toText(d, title = st.title) {
    const list = a => (Array.isArray(a) && a.length ? a.map((x, i) => `${i + 1}. ${x}`).join('\n') : UNSPEC);
    const items = Array.isArray(d.action_items) ? d.action_items : [];
    const L = [];
    L.push('MEETING SUMMARY');
    if (title) L.push(`Meeting: ${title}`);
    L.push(`Output language: ${d.summary_language || UNSPEC}`);
    L.push(`Generated: ${new Date().toLocaleString()}`, '', '='.repeat(40), '');
    const sec = (h, body) => L.push(h, body, '');
    sec('MEETING OVERVIEW', has(d.meeting_overview) ? d.meeting_overview.trim() : UNSPEC);
    sec('EXECUTIVE SUMMARY', has(d.executive_summary) ? d.executive_summary.trim() : UNSPEC);
    sec('KEY DISCUSSION POINTS', list(d.key_discussion_points));
    sec('DECISIONS MADE', list(d.decisions));
    sec('ACTION ITEMS', items.length
      ? items.map((a, i) => `${i + 1}. ${a.task}\n   Assigned to: ${clean(a.assigned_to) || UNSPEC}\n   Deadline: ${clean(a.deadline) || UNSPEC}\n   Status: ${STATUS[a.status] || UNSPEC}`).join('\n')
      : UNSPEC);
    const counts = new Map();
    items.forEach(a => { const k = clean(a.assigned_to) || UNSPEC; counts.set(k, (counts.get(k) || 0) + 1); });
    sec('ASSIGNED PERSONS', counts.size ? [...counts].map(([n, c]) => `- ${n} (${c} task${c === 1 ? '' : 's'})`).join('\n') : UNSPEC);
    sec('DEADLINES', items.length ? items.map(a => `- ${a.task}: ${clean(a.deadline) || UNSPEC}`).join('\n') : UNSPEC);
    sec('UNRESOLVED ISSUES', list(d.unresolved_issues));
    sec('NEXT STEPS', list(d.next_steps));
    return L.join('\n').trimEnd() + '\n';
  }

  function flash(btn, text) {
    const orig = btn.dataset.label || btn.textContent;
    btn.dataset.label = orig;
    btn.textContent = text;
    clearTimeout(st.flashT);
    st.flashT = setTimeout(() => { btn.textContent = orig; }, 2000);
  }

  async function copy() {
    if (!st.result) return;
    const text = toText(st.result);
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch {
      const ta = el('textarea'); ta.value = text; ta.style.cssText = 'position:fixed;left:-9999px;top:0';
      document.body.append(ta); ta.select();
      try { ok = document.execCommand('copy'); } catch {}
      ta.remove();
    }
    flash($('btnSumCopy'), ok ? 'Copied!' : 'Copy failed');
  }

  function download() {
    if (!st.result) return;
    const slug = (st.title || 'meeting').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-+|-+$/g, '').slice(0, 50) || 'meeting';
    const date = new Date().toISOString().slice(0, 10);
    const url = URL.createObjectURL(new Blob([toText(st.result)], { type: 'text/plain;charset=utf-8' }));
    const a = el('a'); a.href = url; a.download = `summary-${slug}-${date}.txt`;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ---------- submit ---------- */
  async function submit() {
    if (st.busy) return;
    clearErr();
    const text = $('sumTranscript').value.trim();
    if (!st.segments && !text) return showErr('Please paste or enter a meeting transcript before summarizing.');
    if (!st.segments && text.length < MIN_CHARS) return showErr(`The transcript is too short to summarize. Please enter at least ${MIN_CHARS} characters.`);

    const title = $('sumTitle').value.trim();
    const payload = {
      meeting_id: title || `meeting-${Date.now()}`,
      output_language: $('sumLang').value || 'English'
    };
    if (st.segments) payload.segments = st.segments; else payload.transcript = text;

    st.ctrl = new AbortController();
    setBusy(true);
    try {
      const d = await Api.summarize(payload, { signal: st.ctrl.signal });
      st.result = d; st.title = title;
      render(d);
      $('sumResultCard').scrollIntoView({ block: 'start' });
    } catch (e) {
      if (e.kind !== 'cancelled') showErr(e.message || 'Summarization failed.');
      else showErr('Summarization cancelled.');
    } finally {
      st.ctrl = null; setBusy(false);
    }
  }

  function clearAll() {
    if (st.busy) return;
    $('sumTranscript').value = ''; $('sumTitle').value = '';
    st.segments = null; st.result = null; st.title = '';
    $('sumResultCard').hidden = true; $('sumResult').replaceChildren();
    clearErr(); updateCount();
  }

  /* Programmatic entry point for future integration (see header comment). */
  function run(opts = {}) {
    if (st.busy) return Promise.resolve();
    if (opts.meetingId != null) $('sumTitle').value = String(opts.meetingId);
    if (opts.outputLanguage) $('sumLang').value = opts.outputLanguage;
    st.segments = null;
    if (Array.isArray(opts.segments) && opts.segments.length) {
      st.segments = opts.segments;
      $('sumTranscript').value = segmentsToText(opts.segments);
    } else if (typeof opts.transcript === 'string') {
      $('sumTranscript').value = opts.transcript;
    }
    updateCount(); show('viewSummary');
    return submit();
  }

  function init() {
    const sel = $('sumLang');
    Api.config.summaryLanguages.forEach(n => sel.add(new Option(n, n)));
    sel.value = 'English';
    document.querySelectorAll('.tab').forEach(t => { t.onclick = () => show(t.dataset.view); });
    $('btnSummarize').onclick = submit;
    $('btnSumClear').onclick = clearAll;
    $('btnSumCancel').onclick = () => { if (st.ctrl) st.ctrl.abort(); };
    $('btnSumCopy').onclick = copy;
    $('btnSumDownload').onclick = download;
    $('sumTranscript').addEventListener('input', () => { st.segments = null; updateCount(); });
    updateCount();
    if (location.hash === '#summarization') show('viewSummary');
  }

  init();
  return { run, show, toText };
})();