#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "generate-video"
_PHASE = "prepare"
_SKILL_ROOT = Path(__file__).resolve().parent.parent
_MODEL_SPEC_PATH = _SKILL_ROOT / "model_spec.json"


def _load_model_spec() -> dict[str, Any]:
    return json.loads(_MODEL_SPEC_PATH.read_text(encoding="utf-8"))


def _model_used() -> str:
    spec = _load_model_spec()
    source = spec.get("source") if isinstance(spec, dict) else {}
    if isinstance(source, dict):
        token = str(source.get("repo_id", "")).strip()
        if token:
            return token
    return str(spec.get("id", "")).strip() or "model"


def _err(*, error_code: str, message: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "phase": _PHASE,
        "status": "error",
        "model_used": _model_used(),
        "error_code": error_code,
        "message": message,
    }


def _ok(*, message: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "phase": _PHASE,
        "status": "ok",
        "model_used": _model_used(),
        "error_code": "",
        "message": message,
    }


def _local_service_config() -> tuple[str, str]:
    base_url = str(os.getenv("HELIX_LOCAL_MODEL_SERVICE_URL", "")).strip().rstrip("/")
    token = str(os.getenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "")).strip()
    return base_url, token


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
    base_url, token = _local_service_config()
    if not base_url or not token:
        return _err(
            error_code="local_model_service_unavailable",
            message="HELIX_LOCAL_MODEL_SERVICE_URL/TOKEN are not configured",
        ), 1

    timeout = max(5, int(args.timeout))
    payload = {
        "skill_name": _EXECUTED_SKILL,
        "model_spec": _load_model_spec(),
        "request_timeout_seconds": timeout,
    }
    try:
        status_code, parsed, body = _post_json(
            f"{base_url}/models/prepare",
            payload,
            token,
            timeout,
        )
    except RuntimeError as exc:
        return _err(
            error_code="local_model_service_unavailable",
            message=str(exc),
        ), 1

    parsed = parsed or {}
    if status_code != 200 or parsed.get("status") != "ok":
        return _err(
            error_code=str(parsed.get("error_code", "")).strip() or "video_prepare_failed",
            message=str(parsed.get("message", "")).strip() or body.strip() or "video prepare failed",
        ), 1

    return _ok(
        message=str(parsed.get("message", "")).strip() or "video model prepared",
    ), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the built-in local PyTorch LTX-Video model.")
    parser.add_argument("--timeout", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out, code = run(args)
        print(json.dumps(out, ensure_ascii=True))
        return int(code)
    except Exception as exc:
        out = _err(
            error_code="video_prepare_unexpected_exception",
            message=f"unexpected runtime exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
