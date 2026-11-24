const WS = (() => {
  const MAX_RETRIES = 5, CONNECT_TIMEOUT = 8000, STOP_TIMEOUT = 15000, MAX_QUEUE = 8;
  const LIVE_MAX_QUEUE = 25;
  const LIVE_MAX_BUFFERED = 256 * 1024;

  function createStream(h) {
    let sock = null, params = null, wantOpen = false, stopping = false, ended = false, pending = null;
    let retries = 0, retryTimer = null, closeTimer = null, connectTimer = null, badJsonShown = false, queue = [];

    function open(p) {
      if (wantOpen || sock) return Promise.reject(new Error('A live session is already open.'));
      params = p; wantOpen = true; stopping = false; retries = 0;
      return new Promise((resolve, reject) => {
        pending = { resolve, reject };
        connectTimer = setTimeout(() => {
          if (!pending) return;
          const p2 = pending; pending = null; wantOpen = false;
          if (sock) { sock.onclose = null; try { sock.close(); } catch {} sock = null; }
          p2.reject(new Error('Timed out connecting to the backend WebSocket.'));
        }, CONNECT_TIMEOUT);
        connect();
      });
    }

    function connect() {
      if (sock && sock.readyState <= 1) return;
      let ws;
      try { ws = new WebSocket(Api.wsUrl(params)); }
      catch { return fail('Invalid WebSocket URL. Check the API base URL.'); }
      ws.binaryType = 'arraybuffer';
      sock = ws;
      ws.onopen = () => {
        clearTimeout(connectTimer);
        const again = retries > 0;
        retries = 0;
        ws.send(JSON.stringify({ type: 'start', ...params }));
        queue.splice(0).forEach(c => ws.send(c));
        h.onState(again ? 'reconnected' : 'open');
        if (pending) { pending.resolve(); pending = null; }
      };
      ws.onmessage = e => handle(e.data);
      ws.onclose = () => {
        if (sock !== ws) return;
        sock = null;
        if (pending) {
          clearTimeout(connectTimer);
          const p2 = pending; pending = null; wantOpen = false;
          return p2.reject(new Error('Could not open live connection to the backend.'));
        }
        if (!wantOpen || stopping) return finish();
        scheduleRetry();
      };
    }

    function fail(msg) {
      if (pending) { const p2 = pending; pending = null; wantOpen = false; clearTimeout(connectTimer); p2.reject(new Error(msg)); }
      else { h.onError(msg); finish(); }
    }

    function scheduleRetry() {
      retries++;
      if (retries > MAX_RETRIES) return fail('Live connection lost and could not be restored.');
      h.onState('reconnecting');
      retryTimer = setTimeout(connect, Math.min(1000 * 2 ** retries, 8000));
    }

    function finish() {
      if (ended) return;
      ended = true; wantOpen = false;
      clearTimeout(retryTimer); clearTimeout(closeTimer); clearTimeout(connectTimer);
      queue = [];
      if (sock) { sock.onclose = null; try { sock.close(1000); } catch {} sock = null; }
      if (pending) { const p2 = pending; pending = null; p2.reject(new Error('Connection cancelled.')); }
      h.onState('closed');
    }

    function handle(data) {
      if (typeof data !== 'string') return;
      let m;
      try { m = JSON.parse(data); } catch {
        if (!badJsonShown) { badJsonShown = true; h.onError('Received an invalid message from the backend.'); }
        return;
      }
      if (!m || typeof m !== 'object') return;
      if (m.type === 'status') {
        const s = Api.normalizeStatus(m.status);
        h.onStatus(s);
        if (stopping && s === 'completed') finish();
      } else if (m.type === 'segment' || m.type === 'caption') {
        h.onSegment(Api.normalizeSegment(m.segment || m));
      } else if (Array.isArray(m.segments)) {
        m.segments.forEach(s => h.onSegment(Api.normalizeSegment(s)));
      } else if (m.type === 'error') {
        h.onStatus('error');
        h.onError(m.message || 'Backend reported an error.');
      }
    }

    function send(chunk) {
      if (!chunk || chunk.size === 0 || stopping || ended) return;
      if (sock && sock.readyState === WebSocket.OPEN) sock.send(chunk);
      else if (wantOpen) { queue.push(chunk); if (queue.length > MAX_QUEUE) queue.shift(); }
    }

    function stop() {
      stopping = true;
      if (sock && sock.readyState === WebSocket.OPEN) {
        sock.send(JSON.stringify({ type: 'stop' }));
        closeTimer = setTimeout(finish, STOP_TIMEOUT);
      } else finish();
    }

    return { open, send, stop, close: finish };
  }

  /*
   Live meeting stream (real backend only). Same endpoint, /ws/stream, with ?mode=live.
   client -> JSON {"type":"start","mode":"live","source_language","target_language","audio_format":"pcm_s16le",
                   "sample_rate":16000,"channels":1,"chunk_ms":200,"session_id":"...","reconnect":<n>}
   client -> binary PCM chunks, continuously, while live
   client -> JSON {"type":"stop"}
   server -> {"type":"segment","segment_id","status":"partial"|"completed","start_time","end_time",
              "source_language","original_text","target_language","translated_text"}
             ("partial"/"final" message types are also accepted); the same segment_id updates the same caption
   server -> {"type":"status","status":...}, {"type":"error","message":...}
   Segments are tagged with a reconnect epoch so ids that restart on a new connection never overwrite earlier captions.
  */
  function createLiveStream(h) {
    let sock = null, params = null, wantOpen = false, stopping = false, ended = false, pending = null;
    let retries = 0, epoch = 0, retryTimer = null, closeTimer = null, connectTimer = null;
    let badJsonShown = false, warned = false, queue = [];

    function liveUrl() {
      const u = new URL(Api.wsUrl(params));
      u.searchParams.set('mode', 'live');
      return u.toString();
    }

    function open(p) {
      if (wantOpen || sock) return Promise.reject(new Error('A live session is already open.'));
      params = p; wantOpen = true; stopping = false; retries = 0; epoch = 0;
      return new Promise((resolve, reject) => {
        pending = { resolve, reject };
        connectTimer = setTimeout(() => {
          if (!pending) return;
          const p2 = pending; pending = null; wantOpen = false;
          if (sock) { sock.onclose = null; try { sock.close(); } catch {} sock = null; }
          p2.reject(new Error('Timed out connecting to the live backend WebSocket.'));
        }, CONNECT_TIMEOUT);
        connect();
      });
    }

    function connect() {
      if (sock && sock.readyState <= 1) return;
      let ws;
      try { ws = new WebSocket(liveUrl()); }
      catch { return fail('Invalid WebSocket URL. Check the API base URL.'); }
      ws.binaryType = 'arraybuffer';
      sock = ws;
      ws.onopen = () => {
        clearTimeout(connectTimer);
        const again = retries > 0;
        if (again) epoch++;
        retries = 0; warned = false;
        ws.send(JSON.stringify({ type: 'start', ...params, reconnect: epoch }));
        queue.splice(0).forEach(c => ws.send(c));
        h.onState(again ? 'reconnected' : 'open');
        if (pending) { pending.resolve(); pending = null; }
      };
      ws.onmessage = e => handle(e.data);
      ws.onclose = () => {
        if (sock !== ws) return;
        sock = null;
        if (pending) {
          clearTimeout(connectTimer);
          const p2 = pending; pending = null; wantOpen = false;
          return p2.reject(new Error('Could not open the live connection to the backend.'));
        }
        if (!wantOpen || stopping) return finish();
        scheduleRetry();
      };
    }

    function fail(msg) {
      if (pending) { const p2 = pending; pending = null; wantOpen = false; clearTimeout(connectTimer); p2.reject(new Error(msg)); }
      else { h.onError(msg); finish(); }
    }

    function scheduleRetry() {
      retries++;
      if (retries > MAX_RETRIES) return fail('Live connection lost and could not be restored.');
      h.onState('reconnecting');
      retryTimer = setTimeout(connect, Math.min(1000 * 2 ** retries, 8000));
    }

    function finish() {
      if (ended) return;
      ended = true; wantOpen = false;
      clearTimeout(retryTimer); clearTimeout(closeTimer); clearTimeout(connectTimer);
      queue = [];
      if (sock) { sock.onclose = null; try { sock.close(1000); } catch {} sock = null; }
      if (pending) { const p2 = pending; pending = null; p2.reject(new Error('Connection cancelled.')); }
      h.onState('closed');
    }

    function handle(data) {
      if (typeof data !== 'string') return;
      let m;
      try { m = JSON.parse(data); } catch {
        if (!badJsonShown) { badJsonShown = true; h.onError('Received an invalid message from the backend.'); }
        return;
      }
      if (!m || typeof m !== 'object') return;
      if (m.type === 'status') {
        const s = Api.normalizeStatus(m.status);
        h.onStatus(s);
        if (stopping && s === 'completed') finish();
      } else if (m.type === 'segment' || m.type === 'caption' || m.type === 'partial' || m.type === 'final') {
        const seg = Api.normalizeSegment(m.segment || m);
        if (m.type === 'partial') seg.status = 'partial';
        if (m.type === 'final') seg.status = 'completed';
        h.onSegment(seg, epoch);
      } else if (m.type === 'remove') {
        if (h.onRemove && m.segment_id != null) h.onRemove(String(m.segment_id), epoch);
      } else if (m.type === 'warning') {
        if (h.onWarn && m.message) h.onWarn(m.message);
      } else if (Array.isArray(m.segments)) {
        m.segments.forEach(s => h.onSegment(Api.normalizeSegment(s), epoch));
      } else if (m.type === 'error') {
        h.onStatus('error');
        h.onError(m.message || 'Backend reported an error.');
      }
    }

    function send(chunk) {
      if (!chunk || chunk.byteLength === 0 || stopping || ended) return;
      if (sock && sock.readyState === WebSocket.OPEN) {
        if (sock.bufferedAmount > LIVE_MAX_BUFFERED) {
          if (!warned && h.onWarn) { warned = true; h.onWarn('Network is slow; dropping audio to stay live.'); }
          return;
        }
        sock.send(chunk);
      } else if (wantOpen) {
        queue.push(chunk);
        if (queue.length > LIVE_MAX_QUEUE) queue.shift();
      }
    }

    function stop() {
      stopping = true;
      if (sock && sock.readyState === WebSocket.OPEN) {
        sock.send(JSON.stringify({ type: 'stop' }));
        closeTimer = setTimeout(finish, STOP_TIMEOUT);
      } else finish();
    }

    return { open, send, stop, close: finish };
  }

  return { createStream, createLiveStream };
})();
