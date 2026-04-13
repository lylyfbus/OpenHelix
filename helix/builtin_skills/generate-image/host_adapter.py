"""Host adapter for the generate-image skill (MLX Z-Image)."""

from __future__ import annotations

import contextlib
import sys
from typing import Any

from helix.runtime.local_model_service.adapters import _BaseBackend
from helix.runtime.local_model_service.helpers import _ensure_worker_dependencies
from helix.runtime.local_model_service.constants import sources_path
from helix.runtime.local_model_service.helpers import (
    _parse_int,
    _parse_size,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

_MLX_DEPENDENCIES = (
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


class _MLXZImageBackend(_BaseBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipeline = None

    def _load(self) -> None:
        assert self.python_bin is not None
        assert self.model_root is not None
        assert self.model_spec is not None
        try:
            import mlx  # noqa: F401
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _MLX_DEPENDENCIES)
            import mlx  # noqa: F401

        # Locate pre-downloaded source files
        sources = self.model_spec.get("sources", {})
        commit = str(sources.get("commit", "")).strip()
        skill_name = str(sources.get("skill_name", "")).strip()
        if not commit or not skill_name:
            raise RuntimeError("model_spec.sources must define skill_name and commit")
        source_root = sources_path(skill_name, commit)
        if not source_root.exists():
            raise RuntimeError(
                f"MLX runner sources not found at {source_root}. "
                "Run: helix model download --spec <path-to-model_spec.json>"
            )
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

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="image_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        width, height = _parse_size(str(inputs.get("size", "")).strip())
        output_path = _resolve_workspace_path(
            workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False,
        )
        try:
            if self.pipeline is None:
                self._load()
            assert self.pipeline is not None
            with contextlib.redirect_stdout(sys.stderr):
                image = self.pipeline.generate(
                    prompt=prompt, width=width, height=height,
                    steps=_parse_int(inputs.get("num_inference_steps"), default=9, minimum=1),
                    seed=_parse_int(inputs.get("seed"), default=42, minimum=0),
                )
            image.save(output_path)
        except Exception as exc:
            return self._error(error_code="generation_runtime_error", message=str(exc))
        rel = str(output_path.relative_to(workspace_root))
        return self._ok(outputs={"output_path": rel}, message=f"generated image at {rel}")


def create_adapter(**kwargs):
    return _MLXZImageBackend(**kwargs)
