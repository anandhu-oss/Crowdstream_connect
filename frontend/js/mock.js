const Mock = (() => {
  const SAMPLES = [
    { src: 'de', speaker: 'speaker_1', text: 'Guten Morgen, wie geht es Ihnen?', en: 'Good morning, how are you?' },
    { src: 'hi', speaker: 'speaker_2', text: 'आज हम इस प्रोजेक्ट पर चर्चा करेंगे।', en: 'Today we will discuss this project.' },
    { src: 'ml', speaker: 'speaker_3', text: 'നമുക്ക് മീറ്റിംഗ് ആരംഭിക്കാം.', en: "Let's start the meeting." },
    { src: 'fr', speaker: 'speaker_1', text: "Je pense que le calendrier est réaliste.", en: 'I think the schedule is realistic.' }
  ];
  const wait = ms => new Promise(r => setTimeout(r, ms));

  function build(i, params, t0, extra) {
    const s = SAMPLES[i % SAMPLES.length];
    const mapped = params.target_language === 'en';
    return {
      segment_id: `mock_${i}`, speaker_id: s.speaker, start_time: t0, end_time: t0 + 3,
      source_language: s.src, original_text: s.text, target_language: params.target_language,
      translated_text: mapped ? s.en : null,
      translation_error: mapped ? null : 'No simulated translation for this language',
      status: 'completed', simulated: true, ...extra
    };
  }

  async function emit(i, params, t0, h) {
    h.onStatus('detecting'); await wait(500);
    h.onStatus('transcribing');
    const full = build(i, params, t0, { translated_text: null, translation_error: null });
    h.onSegment({ ...full, original_text: full.original_text.slice(0, Math.ceil(full.original_text.length / 2)), status: 'partial' });
    await wait(700);
    h.onSegment(full);
    h.onStatus('translating'); await wait(700);
    h.onSegment(build(i, params, t0));
  }

  function createStream(h) {
    let timer = null, n = 0, t0 = Date.now(), started = false;
    return {
      async open(params) {
        await wait(150);
        h.onState('open'); h.onStatus('waiting');
        timer = setInterval(() => {
          if (!started) return;
          emit(n, params, Math.round((Date.now() - t0) / 100) / 10, h); n++;
        }, 4500);
      },
      send() { if (!started) { started = true; h.onStatus('receiving'); } },
      stop() { clearInterval(timer); setTimeout(() => { h.onStatus('completed'); h.onState('closed'); }, 2500); },
      close() { clearInterval(timer); h.onState('closed'); }
    };
  }

  async function uploadAudio(file, params, h) {
    h.onStatus('receiving'); await wait(600);
    for (let i = 0; i < SAMPLES.length; i++) await emit(i, params, i * 4, h);
    h.onStatus('completed');
  }

  return { createStream, uploadAudio };
})();
