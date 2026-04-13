#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "analyze-image"
_MODEL_ID = "glm-ocr"
_DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _err(*, image_source: str, analysis: str, error_code: str, message: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "image_source": image_source,
        "analysis": analysis,
        "model_used": _MODEL_ID,
        "error_code": error_code,
        "message": message,
    }


def _ok(*, image_source: str, analysis: str, message: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "image_source": image_source,
        "analysis": analysis,
        "model_used": _MODEL_ID,
        "error_code": "",
        "message": message,
    }


def _ollama_base_url() -> str:
    return str(os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)).strip().rstrip("/")


def _resolve_relative_path(path_text: str) -> str:
    raw = str(path_text or "").strip()
    if not raw:
        raise ValueError("workspace-relative path is required")
    path = Path(raw).expanduser()
    cwd = Path.cwd().resolve()
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(cwd))
        except ValueError as exc:
            raise ValueError("path must stay inside the workspace") from exc
    if ".." in path.parts:
        raise ValueError("path traversal is not allowed")
    return str(path)


def _download_to_workspace(image_url: str, timeout: int) -> str:
    req = Request(
        image_url,
        headers={
            "User-Agent": "Mozilla/5.0 (HelixOllamaImageSkill/1.0)",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read(10_000_000)
    if not raw:
        raise ValueError("downloaded image is empty")
    suffix = Path(urlparse(image_url).path).suffix or ".bin"
    rel_dir = Path(".runtime") / "ollama-image-analysis" / "downloads"
    rel_dir.mkdir(parents=True, exist_ok=True)
    rel_path = rel_dir / f"download_{_utc_now_compact()}{suffix}"
    rel_path.write_bytes(raw)
    return str(rel_path)


def _load_image_base64(relative_image_path: str) -> str:
    resolved = (Path.cwd().resolve() / relative_image_path).resolve(strict=True)
    workspace_root = Path.cwd().resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("path must stay inside the workspace") from exc
    return base64.b64encode(resolved.read_bytes()).decode("ascii")


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any] | None, str]:
    req = Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body.strip() else None
            return int(getattr(resp, "status", 200)), parsed if isinstance(parsed, dict) else None, body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body.strip() else None
        except json.JSONDecodeError:
            parsed = None
        return int(exc.code), parsed if isinstance(parsed, dict) else None, body
    except URLError as exc:
        raise RuntimeError(f"ollama request failed: {exc}") from exc


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    query = str(args.query or "").strip()
    image_url = str(args.image_url or "").strip()
    image_path = str(args.image_path or "").strip()
    image_source = image_url or image_path
    if not query:
        return _err(
            image_source=image_source,
            analysis="",
            error_code="image_query_missing",
            message="image query context is required; provide --query",
        ), 1

    timeout = max(5, int(args.timeout))
    try:
        if image_path:
            relative_image_path = _resolve_relative_path(image_path)
        else:
            relative_image_path = _download_to_workspace(image_url, timeout)
        image_b64 = _load_image_base64(relative_image_path)
    except (ValueError, OSError, URLError, HTTPError) as exc:
        return _err(
            image_source=image_source,
            analysis="",
            error_code="image_input_error",
            message=str(exc),
        ), 1

    payload: dict[str, Any] = {
        "model": _MODEL_ID,
        "prompt": query,
        "images": [image_b64],
        "stream": False,
    }
    keep_alive = str(os.getenv("OLLAMA_KEEP_ALIVE", "")).strip()
    if keep_alive:
        payload["keep_alive"] = keep_alive

    base_url = _ollama_base_url()
    try:
        status_code, parsed, body = _post_json(
            f"{base_url}/api/generate",
            payload,
            timeout,
        )
    except RuntimeError as exc:
        return _err(
            image_source=image_source,
            analysis="",
            error_code="ollama_unavailable",
            message=f"{exc}. Start Ollama with 'ollama serve' and install the model with 'ollama pull {_MODEL_ID}'.",
        ), 1

    parsed = parsed or {}
    if status_code != 200:
        message = str(parsed.get("error", "")).strip() or body.strip() or "image analysis failed"
        return _err(
            image_source=image_source,
            analysis="",
            error_code="ollama_request_failed",
            message=message,
        ), 1

    response_text = str(parsed.get("response", "")).strip()
    if not response_text:
        return _err(
            image_source=image_source,
            analysis="",
            error_code="image_analysis_failed",
            message=str(parsed.get("error", "")).strip() or "ollama returned an empty response",
        ), 1

    return _ok(
        image_source=image_source,
        analysis=response_text,
        message="image analysis complete",
    ), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an image with the built-in Ollama GLM-OCR backend.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image-path", default="")
    source_group.add_argument("--image-url", default="")
    parser.add_argument("--query", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out, code = run(args)
        print(json.dumps(out, ensure_ascii=True))
        return int(code)
    except Exception as exc:
        out = _err(
            image_source=str(args.image_url or args.image_path or ""),
            analysis="",
            error_code="image_unexpected_exception",
            message=f"unexpected runtime exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
