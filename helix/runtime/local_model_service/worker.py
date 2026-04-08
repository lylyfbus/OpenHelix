"""Worker process entrypoint for local model service."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

from .adapter_discovery import discover_and_register, discover_and_register_builtins
from .protocol import (
    _FAKE_BACKEND_NAME,
    _TASK_TEXT_TO_AUDIO,
    _TASK_TEXT_TO_IMAGE,
    _TASK_TEXT_TO_VIDEO,
    _TASK_TEXT_IMAGE_TO_VIDEO,
    _error_response,
    _ok_response,
    _parse_int,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)
from .registry import _build_backend


# --------------------------------------------------------------------------- #
# Fake backends for testing (only instantiated when backend_mode == "fake")
# --------------------------------------------------------------------------- #

_FAKE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9pob7XUAAAAASUVORK5CYII="
)
_FAKE_WAV_BYTES = base64.b64decode(
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
)
_FAKE_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


class _FakeBackend:
    """Minimal test backend that produces static placeholder outputs."""

    def __init__(self, *, task_type: str, backend: str, model_id: str) -> None:
        self.task_type = task_type
        self.backend = backend
        self.model_id = model_id

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        if inputs.get("prepare_only") in (True, 1, "true", "1", "yes"):
            return _ok_response(
                task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                outputs={"prepared": True},
                message=f"prepared placeholder model state for {self.model_id}",
            )
        workspace_root = _resolve_service_workspace_root(payload)

        if self.task_type == _TASK_TEXT_TO_IMAGE:
            prompt = str(inputs.get("prompt", "")).strip()
            if not prompt:
                return _error_response(
                    task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                    error_code="image_prompt_missing", message="prompt is required",
                )
            output = _resolve_workspace_path(workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False)
            output.write_bytes(_FAKE_PNG_BYTES)
            rel = str(output.relative_to(workspace_root))
            return _ok_response(
                task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                outputs={"output_path": rel}, message=f"generated placeholder image at {rel}",
            )

        if self.task_type in {_TASK_TEXT_TO_VIDEO, _TASK_TEXT_IMAGE_TO_VIDEO}:
            prompt = str(inputs.get("prompt", "")).strip()
            if not prompt:
                return _error_response(
                    task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                    error_code="video_prompt_missing", message="prompt is required",
                )
            if self.task_type == _TASK_TEXT_IMAGE_TO_VIDEO and not str(inputs.get("image_path", "")).strip():
                return _error_response(
                    task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                    error_code="video_image_missing", message="image_path is required for text_image_to_video",
                )
            output = _resolve_workspace_path(workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False)
            output.write_bytes(_FAKE_MP4_BYTES)
            rel = str(output.relative_to(workspace_root))
            fps = _parse_int(inputs.get("fps"), default=8, minimum=1)
            frames = _parse_int(inputs.get("num_frames"), default=16, minimum=1)
            return _ok_response(
                task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                outputs={"output_path": rel, "fps": fps, "num_frames": frames},
                message=f"generated placeholder video at {rel}",
            )

        if self.task_type == _TASK_TEXT_TO_AUDIO:
            text = str(inputs.get("text", "")).strip()
            if not text:
                return _error_response(
                    task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                    error_code="audio_text_missing", message="text is required",
                )
            output = _resolve_workspace_path(workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False)
            output.write_bytes(_FAKE_WAV_BYTES)
            rel = str(output.relative_to(workspace_root))
            return _ok_response(
                task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                outputs={"output_path": rel, "sample_rate": 24000},
                message=f"generated placeholder audio at {rel}",
            )

        return _error_response(
            task_type=self.task_type, backend=self.backend, model_id=self.model_id,
            error_code="unsupported_task_type", message=f"no fake backend for task_type: {self.task_type}",
        )


# --------------------------------------------------------------------------- #
# Worker main
# --------------------------------------------------------------------------- #


def _worker_main(args) -> int:
    # Register adapters from built-in and workspace skills
    discover_and_register_builtins()
    skills_root = str(getattr(args, "skills_root", "") or "").strip()
    if skills_root:
        discover_and_register(Path(skills_root))

    cache_root = Path(args.cache_root).expanduser().resolve()
    python_bin = Path(sys.executable).resolve()
    raw_model_spec = str(getattr(args, "model_spec_json", "") or "").strip()
    model_spec = json.loads(raw_model_spec) if raw_model_spec else None
    model_root = str(getattr(args, "model_root", "") or "").strip()

    if str(args.backend_mode) == _FAKE_BACKEND_NAME:
        display_id = str(args.model_id)
        if model_spec is not None:
            from .model_specs import model_spec_display_id
            display_id = model_spec_display_id(model_spec)
        backend = _FakeBackend(
            task_type=str(args.task_type),
            backend=str(args.backend),
            model_id=display_id,
        )
    else:
        backend = _build_backend(
            task_type=str(args.task_type),
            backend=str(args.backend),
            cache_root=cache_root,
            model_id=str(args.model_id),
            python_bin=python_bin,
            model_spec=model_spec,
            model_root=Path(model_root).expanduser().resolve() if model_root else None,
        )

    print(
        json.dumps(
            {
                "status": "ready",
                "task_type": str(args.task_type),
                "backend": str(args.backend),
                "model_id": str(args.model_id),
                "pid": os.getpid(),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(
                json.dumps(
                        _error_response(
                            task_type=str(args.task_type),
                            backend=str(args.backend),
                            model_id=str(args.model_id),
                            error_code="invalid_json",
                        message="worker request must be a JSON object",
                    ),
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        if not isinstance(payload, dict):
            print(
                json.dumps(
                        _error_response(
                            task_type=str(args.task_type),
                            backend=str(args.backend),
                            model_id=str(args.model_id),
                            error_code="invalid_json",
                        message="worker request must be a JSON object",
                    ),
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        try:
            response = backend.handle(payload)
        except Exception as exc:
            response = _error_response(
                task_type=str(args.task_type),
                backend=str(args.backend),
                model_id=str(args.model_id),
                error_code="worker_runtime_error",
                message=str(exc),
            )
        print(json.dumps(response, ensure_ascii=True), flush=True)
    return 0
