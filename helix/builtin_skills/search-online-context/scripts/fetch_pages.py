#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _search_common import err_payload, fetch_urls, format_fetched_context, ok_payload


def run(
    *,
    urls: list[str],
    context_chars: int,
    max_total_context_chars: int,
    timeout: int,
) -> dict[str, object]:
    normalized_urls: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        url = str(raw_url).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        normalized_urls.append(url)

    if not normalized_urls:
        return err_payload(
            "fetch",
            fetched_context="fetch_error: no urls provided",
        )

    rows = fetch_urls(
        urls=normalized_urls,
        context_chars=context_chars,
        max_total_context_chars=max_total_context_chars,
        timeout=timeout,
    )
    readable_rows = [row for row in rows if str(row.get("status", "")).strip().lower() != "error"]
    summary = (
        "fetch_ok: "
        f"requested={len(normalized_urls)}; fetched={len(rows)}; "
        f"readable={len(readable_rows)}; max_total_context_chars={max_total_context_chars}"
    )
    details = format_fetched_context(rows)
    if not readable_rows:
        return err_payload(
            "fetch",
            fetched_context=summary if not details else f"{summary}\n\n{details}",
        )
    return ok_payload(
        "fetch",
        fetched_context=summary if not details else f"{summary}\n\n{details}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and clean text from selected URLs.")
    parser.add_argument("--url", action="append", default=[], help="URL to fetch; repeat for multiple URLs")
    parser.add_argument("--context-chars", type=int, default=2500, help="Max chars kept per fetched page")
    parser.add_argument(
        "--max-total-context-chars",
        type=int,
        default=15000,
        help="Global max chars across all fetched contexts",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(
            urls=[str(item) for item in args.url],
            context_chars=max(200, int(args.context_chars)),
            max_total_context_chars=max(1000, int(args.max_total_context_chars)),
            timeout=max(5, int(args.timeout)),
        )
        print(json.dumps(result, ensure_ascii=True))
        return 0 if str(result.get("status", "")).strip().lower() == "ok" else 1
    except Exception as exc:
        out = err_payload(
            "fetch",
            fetched_context=f"fetch_error: unexpected exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
