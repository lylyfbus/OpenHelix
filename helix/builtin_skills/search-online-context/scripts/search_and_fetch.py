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

from _search_common import err_payload, fetch_urls, format_fetched_context, ok_payload, search_searxng


def run(
    *,
    query: str,
    limit: int,
    fetch_count: int,
    context_chars: int,
    max_total_context_chars: int,
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
            "search-and-fetch",
            query=query,
            fetched_context=f"search_error: searxng: {exc}",
        )

    if not results:
        return err_payload(
            "search-and-fetch",
            query=query,
            fetched_context="search_error: searxng: no results found",
        )

    fetch_urls_list = [
        str(item.get("url", "")).strip()
        for item in results[: max(0, fetch_count)]
        if str(item.get("url", "")).strip()
    ]
    title_by_url = {
        str(item.get("url", "")).strip(): str(item.get("title", "")).strip()
        for item in results[: max(0, fetch_count)]
        if str(item.get("url", "")).strip()
    }
    fetched_rows = fetch_urls(
        urls=fetch_urls_list,
        context_chars=context_chars,
        max_total_context_chars=max_total_context_chars,
        timeout=timeout,
    )
    for row in fetched_rows:
        url = str(row.get("url", "")).strip()
        if url in title_by_url and title_by_url[url]:
            row["title"] = title_by_url[url]

    summary = (
        "search_fetch_ok: "
        f"query={query!r}; search_results={len(results)}; fetched={len(fetched_rows)}; "
        f"backend=searxng; max_total_context_chars={max_total_context_chars}"
    )
    details = format_fetched_context(fetched_rows)
    return ok_payload(
        "search-and-fetch",
        query=query,
        fetched_context=summary if not details else f"{summary}\n\n{details}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search via SearXNG and fetch context from top results."
    )
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--limit", type=int, default=8, help="Max search results to keep")
    parser.add_argument("--fetch", type=int, default=4, help="How many top links to fetch for context")
    parser.add_argument("--context-chars", type=int, default=2500, help="Max chars kept per fetched page")
    parser.add_argument(
        "--max-total-context-chars",
        type=int,
        default=15000,
        help="Global max chars across all fetched contexts",
    )
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
            fetch_count=max(0, int(args.fetch)),
            context_chars=max(200, int(args.context_chars)),
            max_total_context_chars=max(1000, int(args.max_total_context_chars)),
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
            "search-and-fetch",
            query=query,
            fetched_context=f"search_error: unexpected exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
