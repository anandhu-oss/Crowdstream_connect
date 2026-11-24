"""Quick command-line runner:

    python -m summarization.cli path/to/transcript.txt|.json [--lang Hindi] [--mock]

.json files go through adapters.py (same path as /summarize/raw); anything else is
treated as a plain-text transcript.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="summarization.cli")
    p.add_argument("file")
    p.add_argument("--meeting-id", default=None)
    p.add_argument("--lang", default="English")
    p.add_argument("--mock", action="store_true", help="use the built-in mock model (no GPU/server)")
    args = p.parse_args(argv)

    if args.mock:
        os.environ["SUMMARY_LLM_BACKEND"] = "mock"

    from .adapters import request_from_file
    from .config import reload_settings
    from .errors import SummarizationError
    from .pipeline import summarize

    reload_settings()
    try:
        request = request_from_file(args.file, args.meeting_id, args.lang)
        result = asyncio.run(summarize(request))
    except SummarizationError as exc:
        print(json.dumps({"status": "failed", "error_code": exc.code, "detail": str(exc)}, indent=2))
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
