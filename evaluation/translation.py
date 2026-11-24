"""Checks for translation output. chrF (character n-gram F-score) needs reference translations written by a person; without them
only the automatic flags (empty, unchanged, length mismatch) can be reported, and they are warnings, not quality scores."""
from collections import Counter

from src.translation.translator import check_translation


def _ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF in 0..100: average over character n-gram orders of the F-beta score (recall weighted more)."""
    scores = []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hypothesis, n), _ngrams(reference, n)
        if not h or not r:
            continue
        match = sum((h & r).values())
        p, rec = match / sum(h.values()), match / sum(r.values())
        scores.append(0.0 if p + rec == 0 else (1 + beta ** 2) * p * rec / (beta ** 2 * p + rec))
    return round(100 * sum(scores) / len(scores), 2) if scores else 0.0


def evaluate_translation(items) -> dict:
    """items = [(source text, translated text, reference translation or None)]."""
    if not items:
        raise ValueError("no items to evaluate")
    flags, ratios, chrfs = Counter(), [], []
    for src, hyp, ref in items:
        for f in check_translation(src, hyp):
            flags[f] += 1
        ratios.append(len(hyp) / max(1, len(src)))
        if ref:
            chrfs.append(chrf(hyp, ref))
    n = len(items)
    return {"items": n, "flag_counts": dict(flags), "flagged_pct": round(100 * sum(1 for s, h, _ in items if check_translation(s, h)) / n, 2),
            "mean_length_ratio": round(sum(ratios) / n, 2), "chrf": round(sum(chrfs) / len(chrfs), 2) if chrfs else None,
            "references": len(chrfs)}
