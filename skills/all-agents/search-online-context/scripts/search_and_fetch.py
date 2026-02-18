#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from html import unescape
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _http_get_text(url: str, timeout: int) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (AgenticSystemSkill/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read(1_500_000)
    return data.decode(charset, errors="replace")


def _http_get_json(url: str, timeout: int) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (AgenticSystemSkill/1.0)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        body = resp.read(1_500_000).decode(charset, errors="replace")
    parsed = json.loads(body)
    return parsed if isinstance(parsed, dict) else {}


def _clean_text(text: str) -> str:
    out = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    out = re.sub(r"(?s)<!--.*?-->", " ", out)
    out = re.sub(r"(?is)<[^>]+>", " ", out)
    out = unescape(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _clean_inline_html(text: str) -> str:
    out = re.sub(r"(?is)<[^>]+>", " ", text)
    out = unescape(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def search_searxng(
    *,
    base_url: str,
    query: str,
    limit: int,
    timeout: int,
    language: str,
    categories: str,
    safesearch: int,
) -> list[dict[str, Any]]:
    base = base_url.rstrip("/")
    params = {
        "q": query,
        "format": "json",
        "language": language,
        "categories": categories,
        "safesearch": str(safesearch),
    }
    url = f"{base}/search?{urlencode(params)}"
    payload = _http_get_json(url, timeout=timeout)
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        href = str(item.get("url", "")).strip()
        if not href or not href.lower().startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        title = str(item.get("title", "")).strip() or href
        snippet = _clean_inline_html(str(item.get("content", "")))
        engines = item.get("engines", [])
        if not isinstance(engines, list):
            engines = []
        results.append(
            {
                "rank": len(results) + 1,
                "title": title,
                "url": href,
                "snippet": snippet,
                "engines": [str(v) for v in engines if str(v).strip()],
            }
        )
        if len(results) >= limit:
            break
    return results


def fetch_context(url: str, max_chars: int, timeout: int) -> tuple[str, str]:
    html = _http_get_text(url, timeout=timeout)
    text = _clean_text(html)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "...", ""
    return text, ""


def run(
    *,
    query: str,
    limit: int,
    fetch_count: int,
    context_chars: int,
    timeout: int,
    searxng_base_url: str,
    language: str,
    categories: str,
    safesearch: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "search_engine": "searxng",
        "searxng_base_url": searxng_base_url,
        "search_results": [],
        "fetched_context": [],
        "errors": [],
    }

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
        payload["search_results"] = results
    except Exception as exc:
        payload["errors"].append(f"search_error: {exc}")
        return payload

    if not payload["search_results"]:
        payload["errors"].append("search_error: no results found")
        return payload

    for item in payload["search_results"][: max(0, fetch_count)]:
        rank = int(item.get("rank", 0))
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        row: dict[str, Any] = {
            "rank": rank,
            "title": title,
            "url": url,
            "status": "ok",
            "context": "",
            "error": "",
        }
        try:
            context, error = fetch_context(url=url, max_chars=context_chars, timeout=timeout)
            row["context"] = context
            row["error"] = error
            if error:
                row["status"] = "partial"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        payload["fetched_context"].append(row)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search via SearXNG and fetch context from top results.")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--limit", type=int, default=6, help="Max search results to keep")
    parser.add_argument("--fetch", type=int, default=3, help="How many top links to fetch for context")
    parser.add_argument("--context-chars", type=int, default=1800, help="Max chars kept per fetched page")
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
    result = run(
        query=str(args.query).strip(),
        limit=max(1, int(args.limit)),
        fetch_count=max(0, int(args.fetch)),
        context_chars=max(200, int(args.context_chars)),
        timeout=max(5, int(args.timeout)),
        searxng_base_url=str(args.searxng_base_url).strip(),
        language=str(args.language).strip() or "en-US",
        categories=str(args.categories).strip() or "general",
        safesearch=max(0, min(2, int(args.safesearch))),
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
