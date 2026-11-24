"""Score language detectors on labelled clips: accuracy, recall per language, common confusions, unknown and uncertain rates, time.
Run it on short speech pieces and on code-mixed recordings separately: one label per clip is only the dominant language."""
import time
from collections import Counter, defaultdict

from src.language import uncertain_reason


def evaluate_detector(detector, items, min_confidence: float = 0.5) -> dict:
    """items = [(audio float32 array, true ISO code)]. detector = anything with detect(audio) -> LangResult."""
    if not items:
        raise ValueError("no items to evaluate")
    n = correct = unknown = uncertain = 0
    per, conf, times, errors = defaultdict(lambda: [0, 0]), Counter(), [], 0
    for audio, true in items:
        t0 = time.perf_counter()
        try:
            r = detector.detect(audio)
        except (RuntimeError, ValueError):
            errors += 1; continue
        times.append(time.perf_counter() - t0)
        n += 1; per[true][1] += 1
        if r.code is None:
            unknown += 1
        if uncertain_reason(r, min_confidence):
            uncertain += 1
        if r.code == true:
            correct += 1; per[true][0] += 1
        else:
            conf[(true, r.code)] += 1
    if not n:
        return {"clips": 0, "errors": errors}
    times.sort()
    return {"clips": n, "errors": errors, "accuracy": round(100 * correct / n, 2),
            "recall": {k: round(100 * a / b, 1) for k, (a, b) in sorted(per.items())},
            "top_confusions": [{"true": t, "predicted": p, "count": c} for (t, p), c in conf.most_common(8)],
            "unknown_rate": round(100 * unknown / n, 2), "uncertain_rate": round(100 * uncertain / n, 2),
            "mean_ms": round(1000 * sum(times) / len(times), 1), "p95_ms": round(1000 * times[int(0.95 * (len(times) - 1))], 1)}


def compare_detectors(detectors: dict, items, min_confidence: float = 0.5) -> dict:
    """{name: detector} on the same clips -> {name: summary, 'agreement_pct': how often the first two give the same code}."""
    out = {name: evaluate_detector(d, items, min_confidence) for name, d in detectors.items()}
    names = list(detectors)
    if len(names) >= 2:
        same = tot = 0
        for audio, _ in items:
            try:
                a, b = detectors[names[0]].detect(audio).code, detectors[names[1]].detect(audio).code
            except (RuntimeError, ValueError):
                continue
            tot += 1; same += a == b
        out["agreement_pct"] = round(100 * same / tot, 2) if tot else None
    return out
