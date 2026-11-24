const Captions = (() => {
  const segs = new Map();
  const rows = new Map();
  let seq = 0, origEl, transEl, tail = null;

  function init(o, t) { origEl = o; transEl = t; render(); }

  function fmt(t) {
    if (t == null) return '';
    const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, '0');
    return `${String(m).padStart(2, '0')}:${s}`;
  }

  function cmp(a, b) {
    return a._group - b._group || (a.start_time ?? 1e9) - (b.start_time ?? 1e9) || a._order - b._order;
  }

  function upsert(seg, group = 0) {
    const key = `${group}:${seg.segment_id}`;
    const prev = segs.get(key);
    if (!prev) {
      if (!seg.original_text) return;
      const s = { ...seg, _key: key, _group: group, _order: seq++ };
      segs.set(key, s);
      if (!tail || cmp(tail, s) <= 0) { appendRows(s); tail = s; } else render();
      return;
    }
    if (prev.status === 'completed' && seg.status === 'partial') return;
    const out = { ...prev };
    for (const [k, v] of Object.entries(seg)) if (v !== null && v !== undefined && v !== '') out[k] = v;
    if (seg.translated_text) { out.translation_error = null; out.translation_stale = false; }
    else if (seg.original_text && seg.original_text !== prev.original_text && prev.translated_text) out.translation_stale = true;
    segs.set(key, out);
    if (tail && tail._key === key) tail = out;
    if (prev.start_time !== out.start_time) render(); else replaceRows(out);
  }

  function finalize(group) {
    segs.forEach(s => {
      if (group !== undefined && Math.floor(s._group) !== Math.floor(group)) return;
      if (s.status === 'completed' && !s.translated_text && !s.translation_error && s.translation_status !== 'disabled' && s.translation_status !== 'none') s.translation_error = 'Translation missing';
    });
    render();
  }

  function remove(id, group = 0) {
    const key = `${group}:${id}`;
    if (!segs.has(key)) return;
    segs.delete(key);
    const r = rows.get(key);
    rows.delete(key);
    if (r) { r.o.remove(); r.t.remove(); }
    if (tail && tail._key === key) tail = [...segs.values()].sort(cmp).pop() || null;
  }

  function clear() {
    segs.clear(); rows.clear(); tail = null;
    origEl.replaceChildren(); transEl.replaceChildren();
  }

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function meta(s, lang, extra) {
    const m = el('div', 'cap-meta');
    if (s.start_time != null) m.append(el('span', 'mono', fmt(s.start_time)));
    if (s.speaker_id) m.append(el('span', 'tag', s.speaker_id));
    m.append(el('span', 'tag lang', Api.langName(lang)));
    if (s.status === 'partial') m.append(el('span', 'tag', 'partial'));
    if (s.uncertain && s.status !== 'partial') m.append(el('span', 'tag warn', 'uncertain'));
    if (s.simulated) m.append(el('span', 'tag sim', 'SIMULATED'));
    if (extra) m.append(extra);
    return m;
  }

  function row(s, kind) {
    const r = el('div', 'cap' + (s.status === 'partial' ? ' partial' : ''));
    r.dataset.id = s._key;
    let text = el('div', 'cap-text');
    text.dir = 'auto';
    if (kind === 'orig') {
      text.textContent = s.original_text;
      r.append(meta(s, s.source_language), text);
    } else {
      let warn = null;
      if (s.translated_text) {
        text.textContent = s.translated_text;
        if (s.translation_stale || s.translation_status === 'interim') { text.className = 'cap-text muted'; warn = el('span', 'tag', 'updating'); }
      } else if (s.translation_error) {
        text.className = 'cap-text muted'; text.textContent = 'Translation unavailable';
        warn = el('span', 'tag warn', '⚠ ' + s.translation_error);
      } else if (s.translation_status === 'disabled') { text.className = 'cap-text muted'; text.textContent = 'Translation off'; }
      else { text.className = 'cap-text muted'; text.textContent = s.status === 'partial' ? '…' : 'Translating…'; }
      r.append(meta(s, s.target_language, warn), text);
    }
    r.onmouseenter = () => document.querySelectorAll(`.cap[data-id="${CSS.escape(s._key)}"]`).forEach(n => n.classList.add('hl'));
    r.onmouseleave = () => document.querySelectorAll('.cap.hl').forEach(n => n.classList.remove('hl'));
    return r;
  }

  function keepBottom(container, change) {
    const stick = container.scrollHeight - container.scrollTop - container.clientHeight < 40;
    change();
    if (stick) container.scrollTop = container.scrollHeight;
  }

  function appendRows(s) {
    const o = row(s, 'orig'), t = row(s, 'trans');
    rows.set(s._key, { o, t });
    keepBottom(origEl, () => origEl.append(o));
    keepBottom(transEl, () => transEl.append(t));
  }

  function replaceRows(s) {
    const r = rows.get(s._key);
    if (!r) return render();
    const o = row(s, 'orig'), t = row(s, 'trans');
    keepBottom(origEl, () => r.o.replaceWith(o));
    keepBottom(transEl, () => r.t.replaceWith(t));
    rows.set(s._key, { o, t });
  }

  function render() {
    const list = [...segs.values()].sort(cmp);
    rows.clear();
    const os = [], ts = [];
    list.forEach(s => {
      const o = row(s, 'orig'), t = row(s, 'trans');
      rows.set(s._key, { o, t }); os.push(o); ts.push(t);
    });
    tail = list.length ? list[list.length - 1] : null;
    keepBottom(origEl, () => origEl.replaceChildren(...os));
    keepBottom(transEl, () => transEl.replaceChildren(...ts));
  }

  // Read-only: final transcript in the shape POST /summarize accepts as `segments`.
  // Not used by the UI yet; reserved for passing the transcript straight to Summary.run().
  function getTranscript(opts = {}) {
    const translated = !!opts.translated;
    return [...segs.values()].filter(s => s.status === 'completed' && s.original_text).sort(cmp).map(s => {
      const useT = translated && s.translated_text;
      return {
        text: useT ? s.translated_text : s.original_text,
        speaker: s.speaker_id || null,
        start: s.start_time,
        end: s.end_time,
        language: useT ? s.target_language : s.source_language
      };
    });
  }

  return { init, upsert, remove, finalize, clear, getTranscript };
})();