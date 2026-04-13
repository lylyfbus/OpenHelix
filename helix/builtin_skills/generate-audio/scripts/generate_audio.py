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

_EXECUTED_SKILL = "generate-audio"
_PHASE = "generate"
_TASK_TYPE = "text_to_audio"
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


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _err(*, text: str, output_path: str, error_code: str, message: str, sample_rate: int = 0) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "phase": _PHASE,
        "status": "error",
        "text": text,
        "output_path": output_path,
        "sample_rate": sample_rate,
        "model_used": _model_used(),
        "error_code": error_code,
        "message": message,
    }


def _ok(*, text: str, output_path: str, sample_rate: int, message: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "phase": _PHASE,
        "status": "ok",
        "text": text,
        "output_path": output_path,
        "sample_rate": sample_rate,
        "model_used": _model_used(),
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

    out_dir = str(output_dir or "").strip() or "generated_audio"
    rel_dir = Path(_resolve_relative_path(out_dir))
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", _model_used()).strip("-") or "model"
    return str(rel_dir / f"audio_{safe_model}_{_utc_now_compact()}.wav")


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


def _parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    text = str(args.text or "").strip()
    if not text:
        return _err(
            text="",
            output_path="",
            error_code="audio_text_missing",
            message="audio text is required; provide --text",
        ), 1

    base_url, token = _local_service_config()
    if not base_url or not token:
        return _err(
            text=text,
            output_path="",
            error_code="local_model_service_unavailable",
            message="HELIX_LOCAL_MODEL_SERVICE_URL/TOKEN are not configured",
        ), 1

    try:
        output_path = _choose_output_path(str(args.output_path or ""), str(args.output_dir or ""))
    except ValueError as exc:
        return _err(
            text=text,
            output_path="",
            error_code="audio_output_path_invalid",
            message=str(exc),
        ), 1

    try:
        top_k = int(args.top_k)
        top_p = float(args.top_p)
        temperature = float(args.temperature)
        repetition_penalty = float(args.repetition_penalty)
        max_new_tokens = int(args.max_new_tokens)
        seed = int(args.seed)
        do_sample = _parse_bool_arg(args.do_sample)
        non_streaming_mode = _parse_bool_arg(args.non_streaming_mode)
    except ValueError as exc:
        return _err(
            text=text,
            output_path=output_path,
            error_code="audio_parameter_invalid",
            message=str(exc),
        ), 1

    if top_k < 0:
        return _err(
            text=text,
            output_path=output_path,
            error_code="audio_parameter_invalid",
            message="top_k must be >= 0",
        ), 1
    if not (0.0 < top_p <= 1.0):
        return _err(
            text=text,
            output_path=output_path,
            error_code="audio_parameter_invalid",
            message="top_p must be in the range (0, 1]",
        ), 1
    if temperature <= 0.0:
        return _err(
            text=text,
            output_path=output_path,
            error_code="audio_parameter_invalid",
            message="temperature must be > 0",
        ), 1
    if repetition_penalty <= 0.0:
        return _err(
            text=text,
            output_path=output_path,
            error_code="audio_parameter_invalid",
            message="repetition_penalty must be > 0",
        ), 1
    if max_new_tokens < 16:
        return _err(
            text=text,
            output_path=output_path,
            error_code="audio_parameter_invalid",
            message="max_new_tokens must be >= 16",
        ), 1
    if seed < 0:
        return _err(
            text=text,
            output_path=output_path,
            error_code="audio_parameter_invalid",
            message="seed must be >= 0",
        ), 1

    timeout = max(5, int(args.timeout))
    payload = {
        "skill_name": _EXECUTED_SKILL,
        "task_type": _TASK_TYPE,
        "model_spec": _load_model_spec(),
        "request_timeout_seconds": timeout,
        "workspace_root": str(Path.cwd().resolve()),
        "inputs": {
            "text": text,
            "language": str(args.language or "Auto"),
            "speaker": str(args.speaker or "Vivian"),
            "instruct": str(args.instruct or ""),
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "non_streaming_mode": non_streaming_mode,
            "seed": seed,
            "output_path": output_path,
        },
    }
    try:
        status_code, parsed, body = _post_json(
            f"{base_url}/infer",
            payload,
            token,
            timeout,
        )
    except RuntimeError as exc:
        return _err(
            text=text,
            output_path="",
            error_code="local_model_service_unavailable",
            message=str(exc),
        ), 1

    parsed = parsed or {}
    outputs = parsed.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    resolved_output_path = str(outputs.get("output_path", output_path)).strip()
    sample_rate = int(outputs.get("sample_rate") or 0)
    if status_code != 200 or parsed.get("status") != "ok":
        return _err(
            text=text,
            output_path=resolved_output_path if status_code != 200 else "",
            error_code=str(parsed.get("error_code", "")).strip() or "audio_generation_failed",
            message=str(parsed.get("message", "")).strip() or body.strip() or "audio generation failed",
            sample_rate=sample_rate,
        ), 1

    return _ok(
        text=text,
        output_path=resolved_output_path,
        sample_rate=sample_rate,
        message=str(parsed.get("message", "")).strip() or "audio generation complete",
    ), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with the built-in local PyTorch Qwen3-TTS backend.")
    parser.add_argument("--text", default="")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--do-sample", default="true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--non-streaming-mode", default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--output-dir", default="generated_audio")
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
            text=str(args.text or ""),
            output_path="",
            error_code="audio_unexpected_exception",
            message=f"unexpected runtime exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
