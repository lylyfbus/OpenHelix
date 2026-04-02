#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "generate-image-from-pytorch"
_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _err(*, prompt: str, output_path: str, error_code: str, message: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "prompt": prompt,
        "output_path": output_path,
        "model_used": _MODEL_ID,
        "error_code": error_code,
        "message": message,
    }


def _ok(*, prompt: str, output_path: str, message: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "prompt": prompt,
        "output_path": output_path,
        "model_used": _MODEL_ID,
        "error_code": "",
        "message": message,
    }


def _local_service_config() -> tuple[str, str]:
    base_url = str(os.getenv("HELIX_LOCAL_MODEL_SERVICE_URL", "")).strip().rstrip("/")
    token = str(os.getenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "")).strip()
    return base_url, token


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


def _choose_output_path(output_path: str, output_dir: str) -> str:
    output_path_raw = str(output_path or "").strip()
    if output_path_raw:
        return _resolve_relative_path(output_path_raw)

    out_dir = str(output_dir or "").strip() or "generated_images"
    rel_dir = Path(_resolve_relative_path(out_dir))
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", _MODEL_ID).strip("-") or "model"
    return str(rel_dir / f"image_{safe_model}_{_utc_now_compact()}.png")


def _post_json(url: str, payload: dict[str, Any], token: str, timeout: int) -> tuple[int, dict[str, Any] | None, str]:
    req = Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
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
        raise RuntimeError(f"local model service request failed: {exc}") from exc


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    prompt = str(args.prompt or "").strip()
    if not prompt:
        return _err(
            prompt="",
            output_path="",
            error_code="image_prompt_missing",
            message="image prompt is required; provide --prompt",
        ), 1

    base_url, token = _local_service_config()
    if not base_url or not token:
        return _err(
            prompt=prompt,
            output_path="",
            error_code="local_model_service_unavailable",
            message="HELIX_LOCAL_MODEL_SERVICE_URL/TOKEN are not configured",
        ), 1

    try:
        output_path = _choose_output_path(str(args.output_path or ""), str(args.output_dir or ""))
    except ValueError as exc:
        return _err(
            prompt=prompt,
            output_path="",
            error_code="image_output_path_invalid",
            message=str(exc),
        ), 1

    payload = {
        "model_id": _MODEL_ID,
        "prompt": prompt,
        "size": str(args.size or "1024x1024"),
        "output_path": output_path,
        "workspace_root": str(Path.cwd().resolve()),
    }
    timeout = max(5, int(args.timeout))
    try:
        status_code, parsed, body = _post_json(
            f"{base_url}/v1/image/generate",
            payload,
            token,
            timeout,
        )
    except RuntimeError as exc:
        return _err(
            prompt=prompt,
            output_path="",
            error_code="local_model_service_unavailable",
            message=str(exc),
        ), 1

    parsed = parsed or {}
    if status_code != 200 or parsed.get("status") != "ok":
        return _err(
            prompt=prompt,
            output_path=str(parsed.get("output_path", "")).strip(),
            error_code=str(parsed.get("error_code", "")).strip() or "image_generation_failed",
            message=str(parsed.get("message", "")).strip() or body.strip() or "image generation failed",
        ), 1

    return _ok(
        prompt=prompt,
        output_path=str(parsed.get("output_path", output_path)).strip(),
        message=str(parsed.get("message", "")).strip() or "image generation complete",
    ), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with the built-in local PyTorch backend.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out, code = run(args)
        print(json.dumps(out, ensure_ascii=True))
        return int(code)
    except Exception as exc:
        out = _err(
            prompt=str(args.prompt or ""),
            output_path="",
            error_code="image_unexpected_exception",
            message=f"unexpected runtime exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
