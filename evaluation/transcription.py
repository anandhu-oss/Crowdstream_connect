"""Score a transcript against a reference meeting file (like meeting_rec.json) per language.
The reference is a list of {start, end, speaker, language, text} with times as "HH:MM:SS.mmm". A hypothesis segment belongs to the
reference segment that contains its midpoint, so nothing is counted twice. Rates are errors added up over all clips, not averages."""
import json
from collections import defaultdict
from pathlib import Path

from src.utils.timestamps import parse_time

from .metrics import score_counts

CER_LABELS = {"chinese", "japanese", "mandarin", "cantonese"}


def load_reference(path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("segments", data.get("items", []))
    out = []
    for i, d in enumerate(data):
        try:
            s, e = parse_time(d["start"]), parse_time(d["end"])
            out.append({"start": s, "end": e, "speaker": d.get("speaker"), "language": d.get("language") or "unknown", "text": d["text"]})
        except (KeyError, ValueError) as ex:
            raise ValueError(f"reference item {i} is invalid: {ex}") from ex
        if e <= s:
            raise ValueError(f"reference item {i}: end is not after start")
    return sorted(out, key=lambda r: r["start"])


def align(transcript, reference: list[dict], speaker: str | None = None):
    """-> (rows, unmatched) where rows = [(reference item, joined hypothesis text, [hypothesis segments])]."""
    speaker = speaker if speaker is not None else getattr(transcript, "speaker", None)
    refs = [r for r in reference if not speaker or not r["speaker"] or str(r["speaker"]).lower() == str(speaker).lower()]
    hyps = [s for s in transcript.segments if s.status in ("ok", "empty")]
    rows, used = [], set()
    for r in refs:
        mine = [(i, s) for i, s in enumerate(hyps) if r["start"] <= (s.start + s.end) / 2 < r["end"]]
        used.update(i for i, _ in mine)
        rows.append((r, " ".join(s.text for _, s in mine if s.text), [s for _, s in mine]))
    return rows, len(hyps) - len(used)


def evaluate(transcript, reference: list[dict], speaker: str | None = None) -> dict:
    """Per-language and overall WER/CER for a transcript. Also counts reference segments that got no hypothesis at all."""
    rows, unmatched = align(transcript, reference, speaker)
    agg = defaultdict(lambda: {"segments": 0, "word_errors": 0, "words": 0, "char_errors": 0, "chars": 0, "no_hypothesis": 0, "flagged": 0})
    for ref, hyp, segs in rows:
        c = score_counts(ref["text"], hyp)
        for key in (ref["language"], "ALL"):
            a = agg[key]; a["segments"] += 1; a["no_hypothesis"] += (not segs)
            a["flagged"] += any(s.flags for s in segs)
            for k in ("word_errors", "words", "char_errors", "chars"):
                a[k] += c[k]
    out = {}
    for key, a in agg.items():
        rate = lambda e, n: None if not n else round(100 * e / n, 2)
        out[key] = {**a, "wer": rate(a["word_errors"], a["words"]), "cer": rate(a["char_errors"], a["chars"]),
                    "primary_metric": "cer" if key.lower() in CER_LABELS else "wer"}
        out[key]["primary"] = out[key][out[key]["primary_metric"]]
    return {"languages": out, "unmatched_hypothesis_segments": unmatched}
