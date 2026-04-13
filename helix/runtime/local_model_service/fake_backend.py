"""Fake backend for testing — produces static placeholder outputs.

Only used when the worker is started with ``--backend-mode fake``.
"""

from __future__ import annotations

import base64
from typing import Any

from .helpers import (
    _error_response,
    _ok_response,
    _parse_int,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)
from .constants import (
    TASK_TEXT_IMAGE_TO_VIDEO,
    TASK_TEXT_TO_AUDIO,
    TASK_TEXT_TO_IMAGE,
    TASK_TEXT_TO_VIDEO,
)

_FAKE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9pob7XUAAAAASUVORK5CYII="
)
_FAKE_WAV_BYTES = base64.b64decode(
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
)
_FAKE_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


class FakeBackend:
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

        if self.task_type == TASK_TEXT_TO_IMAGE:
            return self._handle_image(inputs, workspace_root)
        if self.task_type in {TASK_TEXT_TO_VIDEO, TASK_TEXT_IMAGE_TO_VIDEO}:
            return self._handle_video(inputs, workspace_root)
        if self.task_type == TASK_TEXT_TO_AUDIO:
            return self._handle_audio(inputs, workspace_root)
        return _error_response(
            task_type=self.task_type, backend=self.backend, model_id=self.model_id,
            error_code="unsupported_task_type", message=f"no fake backend for task_type: {self.task_type}",
        )

    def _handle_image(self, inputs: dict, workspace_root) -> dict[str, Any]:
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

    def _handle_video(self, inputs: dict, workspace_root) -> dict[str, Any]:
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return _error_response(
                task_type=self.task_type, backend=self.backend, model_id=self.model_id,
                error_code="video_prompt_missing", message="prompt is required",
            )
        if self.task_type == TASK_TEXT_IMAGE_TO_VIDEO and not str(inputs.get("image_path", "")).strip():
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

    def _handle_audio(self, inputs: dict, workspace_root) -> dict[str, Any]:
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
