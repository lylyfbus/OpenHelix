#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _search_common import err_payload, format_search_results, ok_payload, search_searxng


def run(
    *,
    query: str,
    limit: int,
    timeout: int,
    searxng_base_url: str,
    language: str,
    categories: str,
    safesearch: int,
) -> dict[str, object]:
    try:
        results = search_searxng(
            base_url=searxng_base_url,
            query=query,
            limit=limit,
            timeout=timeout,
            language=language,
            categories=categories,
            safesearch=safesearch,
        )
    except Exception as exc:
        return err_payload(
            "search",
            query=query,
            search_results=f"search_error: searxng: {exc}",
        )

    if not results:
        return err_payload(
            "search",
            query=query,
            search_results="search_error: searxng: no results found",
        )

    summary = (
        "search_ok: "
        f"query={query!r}; search_results={len(results)}; backend=searxng"
    )
    details = format_search_results(results)
    return ok_payload(
        "search",
        query=query,
        search_results=summary if not details else f"{summary}\n\n{details}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search SearXNG and return ranked results only.")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--limit", type=int, default=8, help="Max search results to keep")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument(
        "--searxng-base-url",
        default=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8888"),
        help="SearXNG base URL",
    )
    parser.add_argument("--language", default="en-US", help="SearXNG language code")
    parser.add_argument("--categories", default="general", help="SearXNG categories")
    parser.add_argument("--safesearch", type=int, default=1, help="SearXNG safesearch level (0-2)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = str(args.query).strip()
    try:
        result = run(
            query=query,
            limit=max(1, int(args.limit)),
            timeout=max(5, int(args.timeout)),
            searxng_base_url=str(args.searxng_base_url).strip(),
            language=str(args.language).strip() or "en-US",
            categories=str(args.categories).strip() or "general",
            safesearch=max(0, min(2, int(args.safesearch))),
        )
        print(json.dumps(result, ensure_ascii=True))
        return 0 if str(result.get("status", "")).strip().lower() == "ok" else 1
    except Exception as exc:
        out = err_payload(
            "search",
            query=query,
            search_results=f"search_error: unexpected exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
