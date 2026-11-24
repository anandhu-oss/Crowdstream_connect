const Api = (() => {
  const config = {
    // Existing local/Qwen backend
    baseUrl: 'http://localhost:8000',

    healthPath: '/health',
    languagesPath: '/languages',
    uploadPath: '/transcribe',
    wsPath: '/ws/stream',

    // IndicConformer backend running on Kaggle
    indicBaseUrl: 'https://stowaway-large-reason.ngrok-free.dev',
    indicPath: '/transcribe-indic',

    // Set this at runtime (e.g. from a local, uncommitted file) with a NEW key. The old key was committed and must be rotated.
    // Normal use does not need it: the backend calls the Kaggle API itself (INDIC_API_URL / INDIC_API_KEY).
    indicApiKey: '',

    // Independent summarization service (uvicorn summarization_app:app --port 8100).
    // When it is later mounted into service/app.py, point this to the same URL as baseUrl.
    summarizeBaseUrl: 'http://localhost:8100',
    summarizePath: '/summarize',
    summarizeTimeoutMs: 300000,
    summaryLanguages: ['English', 'Hindi', 'Malayalam', 'Tamil', 'German', 'French', 'Italian', 'Spanish', 'Arabic', 'Japanese', 'Chinese'],

    fallbackLanguages: [
      { code: 'en', name: 'English' },
      { code: 'hi', name: 'Hindi' },
      { code: 'ml', name: 'Malayalam' },
      { code: 'ta', name: 'Tamil' },
      { code: 'de', name: 'German' },
      { code: 'fr', name: 'French' },
      { code: 'it', name: 'Italian' }
    ]
  };

  let autoId = 0;
  const names = new Intl.DisplayNames(['en'], { type: 'language' });

  function langName(code) {
    if (!code || code === 'unknown') {
      return 'Unknown language';
    }

    try {
      return names.of(code) || code;
    } catch {
      return code;
    }
  }

  function wsUrl(params) {
    const u = new URL(config.baseUrl);

    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    u.pathname = config.wsPath;

    u.searchParams.set('source_language', params.source_language);
    u.searchParams.set('target_language', params.target_language);

    return u.toString();
  }

  async function health() {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 3000);

    try {
      const r = await fetch(
        config.baseUrl + config.healthPath,
        { signal: ctrl.signal }
      );

      return r.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(t);
    }
  }

  function toLangList(list) {
    return (list || []).map(x =>
      typeof x === 'string'
        ? { code: x, name: langName(x) }
        : { code: x.code, name: x.name || langName(x.code) }
    );
  }

  async function getLanguages() {
    try {
      const r = await fetch(config.baseUrl + config.languagesPath);

      if (!r.ok) return null;

      const d = await r.json();
      const shared = d.languages || null;

      return {
        source: toLangList(d.source || shared),
        target: toLangList(d.target || shared)
      };
    } catch {
      return null;
    }
  }

  function normalizeStatus(s) {
    s = String(s || '').toLowerCase();

    if (s.includes('detect')) return 'detecting';
    if (s.includes('transcri')) return 'transcribing';
    if (s.includes('translat')) return 'translating';
    if (s.includes('receiv') || s.includes('listen')) return 'receiving';
    if (s.includes('complet') || s.includes('done')) return 'completed';
    if (s.includes('error') || s.includes('fail')) return 'error';

    return 'waiting';
  }

  function normalizeSegment(raw) {
    const start = raw.start_time ?? raw.start ?? null;

    const final = raw.status
      ? raw.status !== 'partial'
      : !(raw.is_final === false || raw.partial === true);

    const failed =
      raw.translation_error ||
      raw.translation_status === 'failed';

    return {
      segment_id: String(
        raw.segment_id ??
        raw.id ??
        raw.seq ??
        `auto_${start ?? autoId++}`
      ),

      speaker_id: raw.speaker_id ?? raw.speaker ?? null,
      start_time: start,
      end_time: raw.end_time ?? raw.end ?? null,

      source_language:
        raw.source_language ??
        raw.language ??
        raw.detected_language ??
        'unknown',

      original_text:
        raw.original_text ??
        raw.text ??
        raw.transcript ??
        '',

      target_language: raw.target_language ?? null,
      translated_text: raw.translated_text ?? raw.translation ?? null,

      status: final ? 'completed' : 'partial',

      translation_status: raw.translation_status ?? null,
      transcription_status: raw.transcription_status ?? null,

      uncertain: !!raw.uncertain,

      detected_language: raw.detected_language ?? null,
      asr_model: raw.asr_model ?? raw.model ?? null,

      translation_error: failed
        ? (typeof failed === 'string' ? failed : 'Translation failed')
        : null,

      simulated: !!raw.simulated
    };
  }

  // Existing Qwen audio upload
  async function uploadAudio(file, params) {
    const form = new FormData();

    form.append('file', file);
    form.append('source_language', params.source_language);
    form.append('target_language', params.target_language);

    let r;

    try {
      r = await fetch(
        config.baseUrl + config.uploadPath,
        {
          method: 'POST',
          body: form
        }
      );
    } catch {
      throw new Error('Upload failed: backend unreachable.');
    }

    if (!r.ok) {
      let detail = '';

      try {
        detail = (await r.json()).detail || '';
      } catch {}

      throw new Error(
        `Upload failed (${r.status})${detail ? ': ' + detail : ''}`
      );
    }

    const d = await r.json();

    return (d.segments || d.results || []).map(normalizeSegment);
  }

  // New IndicConformer audio upload
  async function uploadIndicAudio(file, language = 'ml', mode = 'mixed') {
    if (!config.indicBaseUrl ||
        config.indicBaseUrl.includes('PASTE_')) {
      throw new Error('Configure the IndicConformer ngrok URL first.');
    }

    if (!config.indicApiKey ||
        config.indicApiKey.includes('PASTE_')) {
      throw new Error('Set Api.config.indicApiKey first (or use the main upload, which routes through the backend).');
    }

    const form = new FormData();

    form.append('file', file);
    form.append('language', language);
    if (!['native', 'mixed', 'romanized'].includes(mode)) {
      throw new Error(`Unknown Indic mode: ${mode}`);
    }

    form.append('mode', mode);

    let r;

    try {
      r = await fetch(
        config.indicBaseUrl.replace(/\/$/, '') + config.indicPath,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${config.indicApiKey}`
          },
          body: form
        }
      );
    } catch {
      throw new Error(
        'IndicConformer upload failed: backend unreachable. Check the Kaggle session and ngrok URL.'
      );
    }

    let d;

    try {
      d = await r.json();
    } catch {
      throw new Error(
        `IndicConformer returned an invalid response (${r.status}).`
      );
    }

    if (!r.ok) {
      throw new Error(
        `IndicConformer error (${r.status}): ${
          d.detail || 'Unknown error'
        }`
      );
    }

    return {
      success: d.success,
      model: d.model,
      language: d.language,
      mode: d.mode,
      transcription: d.transcription ?? d.text,
      duration_seconds: d.duration_seconds
    };
  }

  // Summarization: POST /summarize on the independent summarization service.
  // payload: { meeting_id, output_language, transcript | segments }
  // Throws Error with .kind = network | timeout | cancelled | http | invalid_response and .status/.code when known.
  function sumError(kind, message, extra) {
    const e = new Error(message);
    e.kind = kind;
    return Object.assign(e, extra || {});
  }

  function detailText(d) {
    if (!d || d.detail == null) return '';
    if (typeof d.detail === 'string') return d.detail;
    if (Array.isArray(d.detail)) {
      return d.detail.map(x => (x && x.msg) ? x.msg : String(x)).join('; ');
    }
    return '';
  }

  async function summarize(payload, opts = {}) {
    const base = config.summarizeBaseUrl.replace(/\/$/, '');
    const timeoutMs = opts.timeoutMs ?? config.summarizeTimeoutMs;
    const ctrl = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, timeoutMs);

    if (opts.signal) {
      if (opts.signal.aborted) ctrl.abort();
      else opts.signal.addEventListener('abort', () => ctrl.abort(), { once: true });
    }

    try {
      let r, d = null;

      try {
        r = await fetch(base + config.summarizePath, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: ctrl.signal
        });
        try { d = await r.json(); } catch { d = null; }
      } catch (e) {
        if (e && e.name === 'AbortError') {
          throw timedOut
            ? sumError('timeout', `The summarization request timed out after ${Math.round(timeoutMs / 1000)} seconds. Try a shorter transcript or try again.`)
            : sumError('cancelled', 'Summarization cancelled.');
        }
        throw sumError('network', `Cannot reach the summarization service at ${base}. Make sure it is running and that it allows requests from this page (CORS).`);
      }

      if (!r.ok) {
        const detail = detailText(d);
        const code = d && d.error_code ? d.error_code : null;
        let msg;

        if (r.status === 404) msg = 'Summarization endpoint not found (POST ' + config.summarizePath + '). Check Api.config.summarizeBaseUrl.';
        else if (r.status === 422) msg = 'The transcript could not be processed' + (detail ? ': ' + detail : '.');
        else if (r.status === 503) msg = 'The summarization model is unavailable right now. Please try again in a moment.' + (detail ? ' (' + detail + ')' : '');
        else if (r.status === 504) msg = 'Summarization took too long on the server. Try a shorter transcript or try again.';
        else if (r.status === 502) msg = 'The summarization model did not return a valid summary. Please try again.' + (detail ? ' (' + detail + ')' : '');
        else msg = `Summarization failed (${r.status}).` + (detail ? ' ' + detail : '');

        throw sumError('http', msg, { status: r.status, code });
      }

      if (!d || typeof d !== 'object' || typeof d.executive_summary !== 'string') {
        throw sumError('invalid_response', 'The summarization service returned an unexpected response.');
      }

      return d;
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    config,
    langName,
    wsUrl,
    health,
    getLanguages,
    normalizeStatus,
    normalizeSegment,
    uploadAudio,
    uploadIndicAudio,
    summarize
  };
})();