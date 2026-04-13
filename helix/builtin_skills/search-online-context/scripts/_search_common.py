from __future__ import annotations

import json
import re
from html import unescape
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "search-online-context"


def ok_payload(phase: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "executed_skill": _EXECUTED_SKILL,
        "phase": str(phase),
        "status": "ok",
    }
    payload.update(fields)
    return payload


def err_payload(phase: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "executed_skill": _EXECUTED_SKILL,
        "phase": str(phase),
        "status": "error",
    }
    payload.update(fields)
    return payload


def http_get_text(url: str, timeout: int) -> str:
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


def http_get_json(url: str, timeout: int) -> dict[str, Any]:
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


def clean_text(text: str) -> str:
    out = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    out = re.sub(r"(?s)<!--.*?-->", " ", out)
    out = re.sub(r"(?is)<[^>]+>", " ", out)
    out = unescape(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def clean_inline_html(text: str) -> str:
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
    payload = http_get_json(url, timeout=timeout)
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        href = str(item.get("url", "")).strip()
        if not href.lower().startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        title = str(item.get("title", "")).strip() or href
        snippet = clean_inline_html(str(item.get("content", "")))
        engines = item.get("engines", [])
        if not isinstance(engines, list):
            engines = []
        out.append(
            {
                "rank": len(out) + 1,
                "title": title,
                "url": href,
                "snippet": snippet,
                "engines": [str(v) for v in engines if str(v).strip()],
            }
        )
        if len(out) >= limit:
            break
    return out


def fetch_page_context(url: str, max_chars: int, timeout: int) -> tuple[str, str]:
    html = http_get_text(url, timeout=timeout)
    text = clean_text(html)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "...", ""
    return text, ""


def fetch_urls(
    *,
    urls: list[str],
    context_chars: int,
    max_total_context_chars: int,
    timeout: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    remaining_context_chars = max(0, int(max_total_context_chars))
    for raw_url in urls:
        if remaining_context_chars <= 0:
            break
        url = str(raw_url).strip()
        if not url:
            continue
        row: dict[str, Any] = {
            "title": url,
            "url": url,
            "status": "ok",
            "context": "",
            "error": "",
        }
        try:
            context, error = fetch_page_context(
                url=url,
                max_chars=context_chars,
                timeout=timeout,
            )
            if context and len(context) > remaining_context_chars:
                context = context[:remaining_context_chars].rstrip() + "..."
                error = (f"{error} | " if error else "") + "truncated by max_total_context_chars"
            if context:
                remaining_context_chars = max(0, remaining_context_chars - len(context))
            row["context"] = context
            row["error"] = error
            if error:
                row["status"] = "partial"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)
    return rows


def format_search_results(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    blocks: list[str] = []
    for row in rows:
        rank = int(row.get("rank", 0))
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        engines = row.get("engines", [])
        if not isinstance(engines, list):
            engines = []
        blocks.append(
            "\n".join(
                [
                    f"rank: {rank}",
                    f"title: {title}",
                    f"url: {url}",
                    f"snippet: {snippet if snippet else '(empty)'}",
                    f"engines: {', '.join(str(v) for v in engines) if engines else '(none)'}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def format_fetched_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    blocks: list[str] = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        status = str(row.get("status", "")).strip()
        context = str(row.get("context", "")).strip()
        error = str(row.get("error", "")).strip()
        blocks.append(
            "\n".join(
                [
                    f"# {title}".strip(),
                    f"url: {url}",
                    f"status: {status}",
                    "context:",
                    context if context else "(empty)",
                    f"error: {error}" if error else "error: (none)",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)

