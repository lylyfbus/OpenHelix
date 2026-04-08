"""Host adapter for the generate-image skill (MLX Z-Image)."""

from __future__ import annotations

import contextlib
import sys
import urllib.request
from pathlib import Path
from typing import Any

from helix.runtime.local_model_service.registry import _BaseBackend
from helix.runtime.local_model_service.paths import _ensure_worker_dependencies
from helix.runtime.local_model_service.protocol import (
    _HTTP_TIMEOUT_SECONDS,
    _parse_int,
    _parse_size,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

FAMILY = "mlx.z_image"
BACKEND = "mlx"
TASK_TYPES = ["text_to_image"]

_MLX_GENERATION_DEPENDENCIES = (
    "accelerate",
    "diffusers>=0.35.0",
    "hf_transfer",
    "huggingface_hub",
    "mlx>=0.20.0",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
    "tqdm",
)
_MLX_Z_IMAGE_COMMIT = "b508c3555cd49b5fb5afd3434053a55d1710c129"
_MLX_Z_IMAGE_FILES = (
    "lora_utils.py",
    "mlx_pipeline.py",
    "mlx_text_encoder.py",
    "mlx_z_image.py",
)


def _download_public_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=max(_HTTP_TIMEOUT_SECONDS, 60)) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _ensure_mlx_runner_sources(cache_root: Path) -> Path:
    runner_root = cache_root / "sources" / f"mlx_z_image-{_MLX_Z_IMAGE_COMMIT}"
    runner_root.mkdir(parents=True, exist_ok=True)
    for filename in _MLX_Z_IMAGE_FILES:
        target = runner_root / filename
        if target.exists():
            continue
        url = (
            "https://raw.githubusercontent.com/uqer1244/MLX_z-image/"
            f"{_MLX_Z_IMAGE_COMMIT}/{filename}"
        )
        _download_public_file(url, target)
    return runner_root


class _SpecMLXZImageBackend(_BaseBackend):
    def __init__(
        self,
        *,
        task_type: str,
        backend: str,
        model_id: str,
        model_spec: dict[str, Any],
        model_root: Path,
        cache_root: Path,
        python_bin: Path,
    ) -> None:
        super().__init__(
            task_type=task_type,
            backend=backend,
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
            model_spec=model_spec,
            model_root=model_root,
        )
        self.pipeline = None

    def _load(self) -> None:
        assert self.cache_root is not None
        assert self.python_bin is not None
        assert self.model_root is not None
        try:
            import mlx  # noqa: F401
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _MLX_GENERATION_DEPENDENCIES)
            import mlx  # noqa: F401

        source_root = _ensure_mlx_runner_sources(self.cache_root)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from mlx_pipeline import ZImagePipeline

        repo_id = str(self.model_spec["source"]["repo_id"])
        with contextlib.redirect_stdout(sys.stderr):
            self.pipeline = ZImagePipeline(
                model_path=str(self.model_root),
                text_encoder_path=str(self.model_root / "text_encoder"),
                repo_id=repo_id,
            )

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            self._load()

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="image_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        width, height = _parse_size(str(inputs.get("size", "")).strip())
        output_path = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
        )
        try:
            self._ensure_loaded()
            assert self.pipeline is not None
            with contextlib.redirect_stdout(sys.stderr):
                image = self.pipeline.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=_parse_int(inputs.get("num_inference_steps"), default=9, minimum=1),
                    seed=_parse_int(inputs.get("seed"), default=42, minimum=0),
                )
            image.save(output_path)
        except Exception as exc:
            return self._error(error_code="generation_runtime_error", message=str(exc))
        rel = str(output_path.relative_to(workspace_root))
        return self._ok(outputs={"output_path": rel}, message=f"generated image at {rel}")


def create_adapter(*, task_type, backend, model_id, cache_root, python_bin, model_spec, model_root):
    return _SpecMLXZImageBackend(
        task_type=task_type,
        backend=backend,
        model_id=model_id,
        model_spec=model_spec,
        model_root=model_root,
        cache_root=cache_root,
        python_bin=python_bin,
    )
