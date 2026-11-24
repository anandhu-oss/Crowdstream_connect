"""Evaluation harness: run the pipeline over test meetings and score the results.

    python -m summarization.evaluate tests/eval_cases --tag qwen3.5-9b
    python -m summarization.evaluate tests/eval_cases --mock          # plumbing check, no LLM

A test case is a transcript (``name.txt`` or ``name.json``) next to ``name.expected.json``:

    {
      "decisions":    [["december 2", "dec 2"]],            # each entry = alternatives; any one must appear
      "actions": [                                          # all given fields must match ONE action item
        {"owner": ["rahul"], "task": ["payment"], "deadline": ["november 28", "nov 28"]}
      ],
      "key_dates":    [["december 5", "dec 5"], ["today"]], # must appear in the key_dates list
      "unresolved":   [["payment"]],                        # must appear in unresolved_issues
      "expect_no_decisions": false,                         # meeting has NO confirmed decision
      "expect_no_action_items": false,
      "expect_no_key_dates": false,
      "must_not_contain": ["december 25"]                   # strings that would be hallucinations
    }

Matching is case-insensitive substring matching, so write alternatives for each way the
model might phrase it (the expected answers are in English; use output language English).

Besides recall, two hallucination checks run automatically on every case:
  * invented dates  - date/time mentions in the summary that the transcript never contained
  * invented owners - action-item owners whose names/labels never appear in the transcript

A report (JSON, including every raw response) is saved so different models can be compared.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .dates import mention_keys

GENERIC_OWNERS = {"everyone", "everybody", "team", "all", "we", "both"}


# --------------------------------------------------------------------------- scoring


def _n(s: Any) -> str:
    return " ".join(str(s or "").lower().split())


def _hit(text: Any, alternatives: list[str]) -> bool:
    t = _n(text)
    return any(_n(a) in t for a in alternatives)


def _recall(flags: list[bool], labels: list[str]) -> dict:
    total = len(flags)
    return {
        "found": sum(flags),
        "total": total,
        "recall": (sum(flags) / total) if total else None,
        "missing": [labels[i] for i, ok in enumerate(flags) if not ok],
    }


def flatten(resp: dict) -> str:
    """All summary text in one string (used for 'appears anywhere' and date checks)."""
    parts: list[str] = [resp.get("meeting_overview", ""), resp.get("executive_summary", "")]
    for k in ("key_discussion_points", "decisions", "unresolved_issues", "next_steps"):
        parts += [str(x) for x in resp.get(k, [])]
    for a in resp.get("action_items", []):
        parts += [str(a.get("task") or ""), str(a.get("assigned_to") or ""), str(a.get("deadline") or "")]
    for d in resp.get("key_dates", []):
        parts += [str(d.get("date") or ""), str(d.get("what") or ""), str(d.get("owner") or "")]
    return "\n".join(parts)


def _supported_date_key(key: str, transcript_keys: set[str], transcript_days: set[str]) -> bool:
    if key in transcript_keys:
        return True
    if re.fullmatch(r"\d\d-\d\d", key):          # "November 28" is supported by a bare "the 28th"
        return f"d{key[3:]}" in transcript_keys
    if key.startswith("d"):
        return key[1:] in transcript_days
    return False


def score_case(resp: dict, expected: dict, transcript_text: str) -> dict:
    """Score one response dict against one expected-answers dict. Pure function (no LLM)."""
    decisions = [str(d) for d in resp.get("decisions", [])]
    actions = resp.get("action_items", [])
    key_dates = resp.get("key_dates", [])
    unresolved = [str(u) for u in resp.get("unresolved_issues", [])]
    everything = flatten(resp)

    exp_dec = expected.get("decisions", [])
    dec = _recall([any(_hit(d, alts) for d in decisions) for alts in exp_dec],
                  [" | ".join(a) for a in exp_dec])

    exp_act = expected.get("actions", [])

    def match_action(item: dict, exp: dict, fields: tuple[str, ...]) -> bool:
        mapping = {"owner": item.get("assigned_to"), "task": item.get("task"), "deadline": item.get("deadline")}
        return all(_hit(mapping[f], exp[f]) for f in fields if f in exp)

    labels = [json.dumps(e, ensure_ascii=False) for e in exp_act]
    act_full = _recall([any(match_action(i, e, ("owner", "task", "deadline")) for i in actions) for e in exp_act], labels)
    act_task = _recall([any(match_action(i, e, ("task",)) for i in actions) for e in exp_act], labels)

    exp_kd = expected.get("key_dates", [])
    kd_text = [f"{d.get('date', '')} {d.get('what', '')}" for d in key_dates]
    kd = _recall([any(_hit(t, alts) for t in kd_text) for alts in exp_kd], [" | ".join(a) for a in exp_kd])
    anywhere = _recall([_hit(everything, alts) for alts in exp_kd], [" | ".join(a) for a in exp_kd])

    exp_unres = expected.get("unresolved", [])
    unres = _recall([any(_hit(u, alts) for u in unresolved) for alts in exp_unres],
                    [" | ".join(a) for a in exp_unres])

    violations: list[str] = []
    if expected.get("expect_no_decisions") and decisions:
        violations.append(f"expected no decisions, got {len(decisions)}: {decisions[:3]}")
    if expected.get("expect_no_action_items") and actions:
        violations.append(f"expected no action items, got {len(actions)}")
    if expected.get("expect_no_key_dates") and key_dates:
        violations.append(f"expected no key dates, got {len(key_dates)}")
    for bad in expected.get("must_not_contain", []):
        if _n(bad) in _n(everything):
            violations.append(f"forbidden text present: {bad!r}")

    tr_keys = mention_keys(transcript_text)
    tr_days = {k[3:] for k in tr_keys if re.fullmatch(r"\d\d-\d\d", k)} | {k[1:] for k in tr_keys if k.startswith("d")}
    invented_dates = sorted(k for k in mention_keys(everything) if not _supported_date_key(k, tr_keys, tr_days))

    tl = _n(transcript_text)
    invented_owners: list[str] = []
    for item in actions:
        owner = item.get("assigned_to")
        if not owner:
            continue
        for part in re.split(r",|/|&|\band\b|\bwith\b", str(owner), flags=re.IGNORECASE):
            part = _n(part)
            if part and part not in GENERIC_OWNERS and part not in tl:
                invented_owners.append(part)

    return {
        "decisions": dec,
        "actions": act_full,
        "actions_task_only": act_task,
        "key_dates": kd,
        "dates_anywhere": anywhere,
        "unresolved": unres,
        "violations": violations,
        "invented_dates": invented_dates,
        "invented_owners": sorted(set(invented_owners)),
    }


RECALL_METRICS = ("decisions", "actions", "actions_task_only", "key_dates", "dates_anywhere", "unresolved")


def aggregate(cases: list[dict]) -> dict:
    out: dict[str, Any] = {"cases": len(cases), "cases_ok": sum(1 for c in cases if c["ok"])}
    scored = [c for c in cases if c["ok"]]
    for m in RECALL_METRICS:
        vals = [c["score"][m]["recall"] for c in scored if c["score"][m]["recall"] is not None]
        out[f"{m}_recall"] = round(sum(vals) / len(vals), 3) if vals else None
    out["violations"] = sum(len(c["score"]["violations"]) for c in scored)
    out["invented_dates"] = sum(len(c["score"]["invented_dates"]) for c in scored)
    out["invented_owners"] = sum(len(c["score"]["invented_owners"]) for c in scored)
    times = [c["seconds"] for c in scored]
    out["avg_seconds"] = round(sum(times) / len(times), 1) if times else None
    return out


# --------------------------------------------------------------------------- running


def find_cases(directory: Path, only: Optional[str] = None) -> list[tuple[Path, Path]]:
    cases = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in (".txt", ".json") or p.name.endswith(".expected.json"):
            continue
        if only and only.lower() not in p.name.lower():
            continue
        exp = p.with_name(p.stem + ".expected.json")
        if exp.exists():
            cases.append((p, exp))
        else:
            print(f"  (skipping {p.name}: no {exp.name})")
    return cases


async def run_cases(cases: list[tuple[Path, Path]], lang: str) -> list[dict]:
    from .adapters import request_from_file
    from .config import get_settings
    from .errors import SummarizationError
    from .pipeline import summarize
    from .preprocessing import prepare

    results = []
    for path, exp_path in cases:
        expected = json.loads(exp_path.read_text(encoding="utf-8"))
        rec: dict[str, Any] = {"name": path.stem, "ok": False, "error": None, "seconds": 0.0,
                               "chunks": 0, "score": None, "response": None}
        print(f"- {path.name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            request = request_from_file(path, output_language=lang)
            transcript_text = prepare(request, get_settings()).text()
            resp = await summarize(request)
            data = json.loads(resp.model_dump_json())
            rec.update(ok=True, response=data, chunks=data["chunks_processed"],
                       score=score_case(data, expected, transcript_text))
        except SummarizationError as exc:
            rec["error"] = f"{exc.code}: {exc}"
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(rec)
    return results


def _fmt(m: dict) -> str:
    return "  -  " if m["recall"] is None else f"{m['found']}/{m['total']}"


def print_report(results: list[dict], agg: dict) -> None:
    head = f"{'case':28} {'dec':>5} {'act':>5} {'task':>5} {'dates':>6} {'any':>5} {'unres':>6} {'badDt':>5} {'badOw':>5} {'viol':>4} {'sec':>6} {'chk':>4}"
    print("\n" + head + "\n" + "-" * len(head))
    for c in results:
        if not c["ok"]:
            print(f"{c['name'][:28]:28} FAILED  {c['error']}")
            continue
        s = c["score"]
        print(f"{c['name'][:28]:28} {_fmt(s['decisions']):>5} {_fmt(s['actions']):>5} {_fmt(s['actions_task_only']):>5} "
              f"{_fmt(s['key_dates']):>6} {_fmt(s['dates_anywhere']):>5} {_fmt(s['unresolved']):>6} "
              f"{len(s['invented_dates']):>5} {len(s['invented_owners']):>5} {len(s['violations']):>4} "
              f"{c['seconds']:>6} {c['chunks']:>4}")
    print("\nMISSING / PROBLEMS")
    any_issue = False
    for c in results:
        if not c["ok"]:
            continue
        s = c["score"]
        for m in ("decisions", "actions", "actions_task_only", "key_dates", "unresolved"):
            for item in s[m]["missing"]:
                any_issue = True
                print(f"  [{c['name']}] missing {m}: {item}")
        for v in s["violations"]:
            any_issue = True
            print(f"  [{c['name']}] {v}")
        if s["invented_dates"]:
            any_issue = True
            print(f"  [{c['name']}] invented dates: {s['invented_dates']}")
        if s["invented_owners"]:
            any_issue = True
            print(f"  [{c['name']}] invented owners: {s['invented_owners']}")
    if not any_issue:
        print("  none")
    print("\nAVERAGES (macro, over cases that have that kind of expectation)")
    for k, v in agg.items():
        print(f"  {k}: {v}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="summarization.evaluate")
    p.add_argument("cases_dir", nargs="?", default="tests/eval_cases")
    p.add_argument("--tag", default=None, help="name for this run / model, used in the report file name")
    p.add_argument("--out", default=None, help="report path (default eval_reports/<tag>_<time>.json)")
    p.add_argument("--lang", default="English")
    p.add_argument("--only", default=None, help="run only cases whose file name contains this text")
    p.add_argument("--mock", action="store_true", help="use the mock model (checks plumbing only)")
    args = p.parse_args(argv)

    if args.mock:
        os.environ["SUMMARY_LLM_BACKEND"] = "mock"
    from .config import reload_settings

    settings = reload_settings()
    directory = Path(args.cases_dir)
    if not directory.is_dir():
        print(f"No such directory: {directory}")
        return 2
    cases = find_cases(directory, args.only)
    if not cases:
        print("No test cases found.")
        return 2

    print(f"Model: {'mock' if settings.llm_backend == 'mock' else settings.llm_model} | {len(cases)} case(s)")
    results = asyncio.run(run_cases(cases, args.lang))
    agg = aggregate(results)
    print_report(results, agg)

    tag = args.tag or ("mock" if settings.llm_backend == "mock" else re.sub(r"[^\w.-]+", "_", settings.llm_model))
    out = Path(args.out) if args.out else Path("eval_reports") / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"tag": tag, "model": settings.llm_model, "backend": settings.llm_backend,
         "created": datetime.now().isoformat(timespec="seconds"), "aggregate": agg, "cases": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
