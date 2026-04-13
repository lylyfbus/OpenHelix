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

_EXECUTED_SKILL = "generate-video"
_PHASE = "generate"
_TASK_TEXT_TO_VIDEO = "text_to_video"
_TASK_TEXT_IMAGE_TO_VIDEO = "text_image_to_video"
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


def _err(
    *,
    task_type: str,
    prompt: str,
    image_path: str,
    output_path: str,
    fps: int,
    num_frames: int,
    error_code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "phase": _PHASE,
        "status": "error",
        "task_type": task_type,
        "prompt": prompt,
        "image_path": image_path,
        "output_path": output_path,
        "fps": fps,
        "num_frames": num_frames,
        "model_used": _model_used(),
        "error_code": error_code,
        "message": message,
    }


def _ok(
    *,
    task_type: str,
    prompt: str,
    image_path: str,
    output_path: str,
    fps: int,
    num_frames: int,
    message: str,
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "phase": _PHASE,
        "status": "ok",
        "task_type": task_type,
        "prompt": prompt,
        "image_path": image_path,
        "output_path": output_path,
        "fps": fps,
        "num_frames": num_frames,
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

    out_dir = str(output_dir or "").strip() or "generated_videos"
    rel_dir = Path(_resolve_relative_path(out_dir))
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", _model_used()).strip("-") or "model"
    return str(rel_dir / f"video_{safe_model}_{_utc_now_compact()}.mp4")


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
            task_type="",
            prompt="",
            image_path="",
            output_path="",
            fps=0,
            num_frames=0,
            error_code="video_prompt_missing",
            message="video prompt is required; provide --prompt",
        ), 1

    base_url, token = _local_service_config()
    if not base_url or not token:
        return _err(
            task_type="",
            prompt=prompt,
            image_path="",
            output_path="",
            fps=0,
            num_frames=0,
            error_code="local_model_service_unavailable",
            message="HELIX_LOCAL_MODEL_SERVICE_URL/TOKEN are not configured",
        ), 1

    try:
        output_path = _choose_output_path(str(args.output_path or ""), str(args.output_dir or ""))
        image_path = _resolve_relative_path(str(args.image_path or "").strip()) if str(args.image_path or "").strip() else ""
    except ValueError as exc:
        return _err(
            task_type="",
            prompt=prompt,
            image_path="",
            output_path="",
            fps=0,
            num_frames=0,
            error_code="video_path_invalid",
            message=str(exc),
        ), 1

    task_type = _TASK_TEXT_IMAGE_TO_VIDEO if image_path else _TASK_TEXT_TO_VIDEO
    payload_inputs: dict[str, Any] = {
        "prompt": prompt,
        "size": str(args.size or "704x512"),
        "output_path": output_path,
        "fps": int(args.fps),
        "num_frames": int(args.num_frames),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "guidance_rescale": float(args.guidance_rescale),
        "decode_timestep": float(args.decode_timestep),
        "decode_noise_scale": float(args.decode_noise_scale),
        "max_sequence_length": int(args.max_sequence_length),
        "seed": int(args.seed),
    }
    if str(args.negative_prompt or "").strip():
        payload_inputs["negative_prompt"] = str(args.negative_prompt).strip()
    if image_path:
        payload_inputs["image_path"] = image_path

    timeout = max(5, int(args.timeout))
    payload = {
        "skill_name": _EXECUTED_SKILL,
        "task_type": task_type,
        "model_spec": _load_model_spec(),
        "request_timeout_seconds": timeout,
        "workspace_root": str(Path.cwd().resolve()),
        "inputs": payload_inputs,
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
            task_type=task_type,
            prompt=prompt,
            image_path=image_path,
            output_path="",
            fps=0,
            num_frames=0,
            error_code="local_model_service_unavailable",
            message=str(exc),
        ), 1

    parsed = parsed or {}
    outputs = parsed.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    resolved_output_path = str(outputs.get("output_path", output_path)).strip()
    fps = int(outputs.get("fps") or args.fps or 0)
    num_frames = int(outputs.get("num_frames") or args.num_frames or 0)
    if status_code != 200 or parsed.get("status") != "ok":
        return _err(
            task_type=task_type,
            prompt=prompt,
            image_path=image_path,
            output_path=resolved_output_path if status_code != 200 else "",
            fps=fps,
            num_frames=num_frames,
            error_code=str(parsed.get("error_code", "")).strip() or "video_generation_failed",
            message=str(parsed.get("message", "")).strip() or body.strip() or "video generation failed",
        ), 1

    return _ok(
        task_type=task_type,
        prompt=prompt,
        image_path=image_path,
        output_path=resolved_output_path,
        fps=fps,
        num_frames=num_frames,
        message=str(parsed.get("message", "")).strip() or "video generation complete",
    ), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with the built-in local PyTorch LTX-Video backend.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--size", default="704x512")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--num-frames", type=int, default=161)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--decode-timestep", type=float, default=0.03)
    parser.add_argument("--decode-noise-scale", type=float, default=0.025)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--output-dir", default="generated_videos")
    parser.add_argument("--timeout", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out, code = run(args)
        print(json.dumps(out, ensure_ascii=True))
        return int(code)
    except Exception as exc:
        image_path = str(getattr(args, "image_path", "") or "")
        task_type = _TASK_TEXT_IMAGE_TO_VIDEO if image_path else _TASK_TEXT_TO_VIDEO
        out = _err(
            task_type=task_type,
            prompt=str(args.prompt or ""),
            image_path=image_path,
            output_path="",
            fps=int(getattr(args, "fps", 0) or 0),
            num_frames=int(getattr(args, "num_frames", 0) or 0),
            error_code="video_unexpected_exception",
            message=f"unexpected runtime exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
