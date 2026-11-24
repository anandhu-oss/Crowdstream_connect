"""Scoring and comparison helpers.
- WER / CER with the same text normalisation for reference and hypothesis.
- signal_metrics: how much a processed recording differs from the original (high-pass, normalisation, denoising, lossy coding).
- compare_conditions: WER/CER per condition (lossless vs lossy, raw vs processed) against a baseline.
- encode_variants: makes lossless and lossy copies of one recording with ffmpeg, for the recorder-format decision.
A signal metric says how different two files are, not which one a speech model reads better: decide on WER/CER."""
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

CER_PRIMARY = {"zh", "ja", "yue"}          # scripts written without spaces


# --- text ---------------------------------------------------------------------------------
def normalize(text: str) -> str:
    """NFC, lower case, [tags] and punctuation (including the danda) removed, native digits to ASCII, spaces collapsed."""
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Nd":
            out.append(str(unicodedata.decimal(ch)))
        elif cat[0] in "PS":
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def error_rate(ref: list, hyp: list) -> dict:
    errors = edit_distance(ref, hyp)
    return {"errors": errors, "ref_len": len(ref), "rate": (errors / len(ref)) if ref else (0.0 if not hyp else None)}


def score(reference: str, hypothesis: str, lang: str | None = None) -> dict:
    """WER and CER in percent (None when the reference is empty but the hypothesis is not), plus the primary metric."""
    r, h = normalize(reference), normalize(hypothesis)
    w = error_rate(r.split(), h.split())
    c = error_rate(list(r.replace(" ", "")), list(h.replace(" ", "")))
    pct = lambda x: None if x["rate"] is None else round(100 * x["rate"], 2)
    out = {"wer": pct(w), "cer": pct(c), "ref_words": w["ref_len"], "ref_chars": c["ref_len"]}
    out["primary"] = out["cer"] if lang in CER_PRIMARY else out["wer"]
    out["primary_metric"] = "cer" if lang in CER_PRIMARY else "wer"
    return out


def score_counts(reference: str, hypothesis: str) -> dict:
    """Raw error counts, so many clips can be added up into one rate (errors / reference length), not averaged."""
    r, h = normalize(reference), normalize(hypothesis)
    w = error_rate(r.split(), h.split()); c = error_rate(list(r.replace(" ", "")), list(h.replace(" ", "")))
    return {"word_errors": w["errors"], "words": w["ref_len"], "char_errors": c["errors"], "chars": c["ref_len"]}


def compare_conditions(reference: str, hypotheses: dict, lang: str | None = None, baseline: str | None = None) -> list[dict]:
    """hypotheses = {condition name: transcript}. Rows with WER, CER and the change against the baseline (default: first)."""
    if not hypotheses:
        return []
    baseline = baseline or next(iter(hypotheses))
    if baseline not in hypotheses:
        raise ValueError(f"baseline {baseline!r} is not one of the conditions")
    base = score(reference, hypotheses[baseline], lang)["primary"]
    rows = []
    for name, hyp in hypotheses.items():
        s = score(reference, hyp, lang)
        delta = None if s["primary"] is None or base is None else round(s["primary"] - base, 2)
        rows.append({"condition": name, **s, "delta_vs_baseline": delta, "baseline": name == baseline})
    return rows


# --- signal -------------------------------------------------------------------------------
def _db(x: float) -> float:
    return 20 * np.log10(max(float(x), 1e-9))


def _rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, dtype="float64") ** 2)))


def signal_metrics(original: np.ndarray, processed: np.ndarray, sr: int = 16000, clip_level: float = 0.999) -> dict:
    """Both signals mono, same sample rate, same timeline (the shorter length is used).
    si_sdr_db: how closely the processed signal matches the original, ignoring loudness (higher = closer; 30+ is near identical).
    correlation: waveform similarity, -1..1.
    log_spectral_distance_db: average spectral shape difference after matching loudness (0 = same spectrum).
    rms_change_db / low_band_change_db: loudness change overall and below 80 Hz.
    clipped_ratio_*: share of samples at full scale."""
    o, p = np.asarray(original, dtype="float64"), np.asarray(processed, dtype="float64")
    if o.ndim != 1 or p.ndim != 1 or not o.size or not p.size:
        raise ValueError("both signals must be non-empty 1-D arrays")
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(p))):
        raise ValueError("signals contain NaN or infinite values")
    n = min(len(o), len(p)); o, p = o[:n], p[:n]
    ro, rp = _rms(o), _rms(p)
    if ro < 1e-9 or rp < 1e-9:
        return {"samples": n, "silent": True, "rms_change_db": None, "si_sdr_db": None, "correlation": None,
                "log_spectral_distance_db": None, "low_band_change_db": None,
                "clipped_ratio_original": float(np.mean(np.abs(o) >= clip_level)),
                "clipped_ratio_processed": float(np.mean(np.abs(p) >= clip_level))}
    alpha = float(np.dot(p, o) / np.dot(o, o))
    target = alpha * o
    noise = p - target
    si_sdr = 10 * np.log10(max(np.dot(target, target), 1e-18) / max(np.dot(noise, noise), 1e-18))
    corr = float(np.corrcoef(o, p)[0, 1]) if np.std(o) > 0 and np.std(p) > 0 else 0.0
    from scipy.signal import stft
    nper = min(512, n)
    _, _, So = stft(o / ro, fs=sr, nperseg=nper); _, _, Sp = stft(p / rp, fs=sr, nperseg=nper)
    lo, lp = 10 * np.log10(np.abs(So) ** 2 + 1e-10), 10 * np.log10(np.abs(Sp) ** 2 + 1e-10)
    lsd = float(np.mean(np.sqrt(np.mean((lo - lp) ** 2, axis=0))))
    f = np.fft.rfftfreq(n, 1 / sr); low = f < 80
    low_o, low_p = float(np.sum(np.abs(np.fft.rfft(o))[low] ** 2)), float(np.sum(np.abs(np.fft.rfft(p))[low] ** 2))
    return {"samples": n, "silent": False, "rms_change_db": round(_db(rp) - _db(ro), 2), "si_sdr_db": round(float(si_sdr), 2),
            "correlation": round(corr, 4), "log_spectral_distance_db": round(lsd, 2),
            "low_band_change_db": round(10 * np.log10((low_p + 1e-12) / (low_o + 1e-12)), 2),
            "clipped_ratio_original": float(np.mean(np.abs(o) >= clip_level)),
            "clipped_ratio_processed": float(np.mean(np.abs(p) >= clip_level))}


# --- recorder format comparison ----------------------------------------------------------
FORMATS = {                                  # name -> (file extension, ffmpeg arguments)
    "flac": ("flac", []),
    "wav16": ("wav", ["-c:a", "pcm_s16le"]),
    "mp3_128k": ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
    "mp3_64k": ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    "opus_32k": ("opus", ["-c:a", "libopus", "-b:a", "32k"]),
    "opus_16k": ("opus", ["-c:a", "libopus", "-b:a", "16k"]),
}


def encode_variants(source: str, out_dir: str, formats=("flac", "mp3_128k", "mp3_64k", "opus_32k", "opus_16k")) -> dict:
    """Re-encode one recording into lossless and lossy copies with ffmpeg (must be installed). Returns {name: path}.
    Transcribe each copy, then pass the transcripts to compare_conditions with the lossless one as baseline."""
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown formats {unknown}; choose from {sorted(FORMATS)}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in formats:
        ext, args = FORMATS[name]
        dst = out / f"{Path(source).stem}_{name}.{ext}"
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-ac", "1", *args, str(dst)],
                           check=True, capture_output=True)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg is not installed") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed for {name}: {e.stderr.decode(errors='ignore')[:200]}") from e
        paths[name] = str(dst)
    return paths
