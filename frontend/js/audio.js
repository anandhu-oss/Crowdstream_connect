const Recorder = (() => {
  const TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  const CHUNK_MS = 2500;
  let media = null, rec = null, running = false, cycleTimer = null, tickTimer = null;
  let startedAt = 0, onChunk = null, onFail = null, mime = '', waiters = [];

  function pickMime() {
    return window.MediaRecorder ? (TYPES.find(t => MediaRecorder.isTypeSupported(t)) || '') : '';
  }

  function micError(msg) { const e = new Error(msg); e.mic = true; return e; }

  function explain(e) {
    switch (e.name) {
      case 'NotAllowedError':
      case 'SecurityError': return 'Microphone permission was denied. Allow microphone access in the browser and try again.';
      case 'NotFoundError':
      case 'OverconstrainedError': return 'No microphone was found on this device.';
      case 'NotReadableError': return 'The microphone is being used by another application.';
      default: return 'Could not start the microphone: ' + e.message;
    }
  }

  async function acquire() {
    if (media) throw micError('The microphone is already in use by this page.');
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      throw micError('Recording is not supported here. Use a modern browser on https or localhost.');
    }
    mime = pickMime();
    if (!mime) throw micError('This browser supports none of the recording formats the app can send (WebM/Opus, Ogg/Opus, MP4).');
    try { media = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { throw micError(explain(e)); }
    media.getAudioTracks().forEach(t => { t.onended = () => fail('The microphone was disconnected.'); });
    return mime;
  }

  function begin(chunkCb, tickCb, failCb) {
    onChunk = chunkCb; onFail = failCb; running = true;
    startedAt = Date.now();
    tickTimer = setInterval(() => tickCb((Date.now() - startedAt) / 1000), 250);
    startCycle();
  }

  function startCycle() {
    if (!media || !running) return;
    const r = new MediaRecorder(media, { mimeType: mime });
    rec = r;
    r.ondataavailable = e => { if (e.data && e.data.size > 0) onChunk(e.data); };
    r.onerror = () => fail('Recording error. Audio capture stopped.');
    r.onstop = () => {
      if (rec === r) rec = null;
      if (running) startCycle(); else settle();
    };
    r.start();
    cycleTimer = setTimeout(() => { if (r.state === 'recording') r.stop(); }, CHUNK_MS);
  }

  function fail(msg) {
    if (!running) return;
    running = false;
    clearTimeout(cycleTimer);
    if (rec && rec.state === 'recording') { try { rec.stop(); } catch {} }
    if (onFail) onFail(msg);
  }

  function settle() { waiters.splice(0).forEach(f => f()); }

  function release() {
    running = false;
    clearTimeout(cycleTimer); clearInterval(tickTimer);
    if (media) media.getTracks().forEach(t => { t.onended = null; t.stop(); });
    media = rec = null;
  }

  function stop() {
    running = false;
    clearTimeout(cycleTimer); clearInterval(tickTimer);
    return new Promise(resolve => {
      const done = () => { release(); resolve(); };
      if (rec && rec.state === 'recording') { waiters.push(done); rec.stop(); } else done();
    });
  }

  return { acquire, begin, stop, release, pickMime, CHUNK_MS };
})();

/*
 Live meeting capture (separate from Recorder above).
 Audio format sent to the backend: raw 16-bit little-endian PCM, mono, 16 kHz, in 200 ms binary chunks.
 The microphone is read through an AudioWorklet (created from an inline Blob, so no extra file),
 downsampled from the device rate to 16 kHz with a box filter, and sent continuously while live.
*/
const LiveAudio = (() => {
  const RATE = 16000, CHUNK_MS = 200, CHUNK_SAMPLES = RATE * CHUNK_MS / 1000;
  const WORKLET = `
    class LivePcm extends AudioWorkletProcessor {
      constructor() {
        super();
        this.buf = new Float32Array(1024); this.n = 0;
        this.port.onmessage = () => {
          if (this.n) this.port.postMessage(this.buf.slice(0, this.n));
          this.n = 0;
          this.port.postMessage(null);
        };
      }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch) for (let i = 0; i < ch.length; i++) {
          this.buf[this.n++] = ch[i];
          if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice(0)); this.n = 0; }
        }
        return true;
      }
    }
    registerProcessor('live-pcm', LivePcm);`;

  let stream = null, ctx = null, src = null, node = null, sink = null, url = null;
  let active = false, onChunk = null, onFail = null, tickTimer = null, startedAt = 0, flushWaiter = null;
  let out = new Int16Array(CHUNK_SAMPLES), outN = 0, acc = 0, cnt = 0, phase = 0, ratio = 1;

  function micError(msg) { const e = new Error(msg); e.mic = true; return e; }

  function explain(e) {
    switch (e.name) {
      case 'NotAllowedError':
      case 'SecurityError': return 'Microphone permission was denied. Allow microphone access in the browser and try again.';
      case 'NotFoundError':
      case 'OverconstrainedError': return 'No microphone was found on this device.';
      case 'NotReadableError': return 'The microphone is being used by another application.';
      default: return 'Could not start the microphone: ' + e.message;
    }
  }

  function format() {
    return { audio_format: 'pcm_s16le', sample_rate: RATE, channels: 1, chunk_ms: CHUNK_MS };
  }

  async function acquire() {
    if (stream || ctx) throw micError('The microphone is already in use by this page.');
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!navigator.mediaDevices?.getUserMedia || !AC || !window.AudioWorkletNode) {
      throw micError('Live capture is not supported here. Use a current browser on https or localhost.');
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
    } catch (e) { throw micError(explain(e)); }
    try {
      ctx = new AC({ latencyHint: 'interactive' });
      if (!ctx.audioWorklet) throw new Error('AudioWorklet is unavailable.');
      url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      await ctx.audioWorklet.addModule(url);
      ratio = ctx.sampleRate / RATE;
    } catch (e) {
      release();
      throw micError('Could not start audio capture: ' + e.message);
    }
  }

  function emit(v) {
    v = v > 1 ? 1 : v < -1 ? -1 : v;
    out[outN++] = Math.round(v < 0 ? v * 0x8000 : v * 0x7fff);
    if (outN === CHUNK_SAMPLES) { outN = 0; onChunk(out.slice(0).buffer); }
  }

  function onBlock(block) {
    if (block === null) {
      if (flushWaiter) { const f = flushWaiter; flushWaiter = null; f(); }
      return;
    }
    for (let i = 0; i < block.length; i++) {
      acc += block[i]; cnt++; phase++;
      if (phase >= ratio) { emit(acc / cnt); acc = 0; cnt = 0; phase -= ratio; }
    }
  }

  function begin(chunkCb, tickCb, failCb) {
    if (!ctx || !stream) throw micError('Audio capture was not prepared.');
    onChunk = chunkCb; onFail = failCb;
    outN = 0; acc = 0; cnt = 0; phase = 0; active = true;
    src = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, 'live-pcm', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: 'explicit' });
    node.port.onmessage = e => onBlock(e.data);
    sink = ctx.createGain(); sink.gain.value = 0;
    src.connect(node); node.connect(sink); sink.connect(ctx.destination);
    stream.getAudioTracks().forEach(t => { t.onended = () => fail('The microphone was disconnected.'); });
    if (ctx.state === 'suspended') ctx.resume();
    startedAt = Date.now();
    tickTimer = setInterval(() => tickCb((Date.now() - startedAt) / 1000), 250);
  }

  function fail(msg) {
    if (!active) return;
    active = false;
    clearInterval(tickTimer);
    if (onFail) onFail(msg);
  }

  function release() {
    active = false;
    clearInterval(tickTimer);
    try { if (node) node.port.onmessage = null; } catch {}
    [src, node, sink].forEach(n => { try { if (n) n.disconnect(); } catch {} });
    if (stream) stream.getTracks().forEach(t => { t.onended = null; t.stop(); });
    if (ctx) { try { ctx.close(); } catch {} }
    if (url) URL.revokeObjectURL(url);
    stream = ctx = src = node = sink = url = null;
    outN = 0; acc = 0; cnt = 0; phase = 0; flushWaiter = null;
  }

  function stop() {
    active = false;
    clearInterval(tickTimer);
    if (!node) { release(); return Promise.resolve(); }
    return new Promise(resolve => {
      const finish = () => {
        if (cnt > 0) emit(acc / cnt);
        if (outN > 0 && onChunk) onChunk(out.slice(0, outN).buffer);
        release();
        resolve();
      };
      flushWaiter = finish;
      node.port.postMessage('flush');
      setTimeout(() => { if (flushWaiter) { const f = flushWaiter; flushWaiter = null; f(); } }, 300);
    });
  }

  return { acquire, begin, stop, release, format, CHUNK_MS };
})();

const Files = (() => {
  const EXT = /\.(wav|mp3|m4a|flac|ogg|webm|aac)$/i;

  function probeDuration(file) {
    return new Promise(resolve => {
      const url = URL.createObjectURL(file);
      const a = new Audio();
      const done = v => { URL.revokeObjectURL(url); resolve(v); };
      a.preload = 'metadata';
      a.onloadedmetadata = () => done(isFinite(a.duration) ? a.duration : null);
      a.onerror = () => done(null);
      setTimeout(() => done(null), 5000);
      a.src = url;
    });
  }

  async function inspect(file) {
    if (file.size === 0) throw new Error('The selected file is empty.');
    if (!file.type.startsWith('audio/') && !EXT.test(file.name)) {
      throw new Error('Unsupported file type. Use a WAV or MP3 audio file (other formats depend on the backend).');
    }
    return probeDuration(file);
  }

  return { inspect };
})();
