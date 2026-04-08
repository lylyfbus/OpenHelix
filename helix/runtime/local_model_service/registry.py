"""Backend registry for local model service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from .model_specs import model_spec_display_id
from .protocol import (
    _SUPPORTED_BACKENDS,
    _SUPPORTED_TASK_TYPES,
    _error_response,
    _ok_response,
    _supported_backend_task,
)


class _WorkerBackend(Protocol):
    def handle(self, payload: dict) -> dict:
        ...


class _BaseBackend:
    """Shared base for all adapter backends (real and fake)."""

    def __init__(
        self,
        *,
        task_type: str,
        backend: str,
        model_id: str,
        cache_root: Path | None = None,
        python_bin: Path | None = None,
        model_spec: dict[str, Any] | None = None,
        model_root: Path | None = None,
    ) -> None:
        self.task_type = task_type
        self.backend = backend
        self.model_id = model_id
        self.cache_root = cache_root
        self.python_bin = python_bin
        self.model_spec = model_spec
        self.model_root = model_root

    def _ok(self, *, outputs: dict[str, Any] | None, message: str) -> dict[str, Any]:
        return _ok_response(
            task_type=self.task_type,
            backend=self.backend,
            model_id=self.model_id,
            outputs=outputs,
            message=message,
        )

    def _error(
        self,
        *,
        error_code: str,
        message: str,
        outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _error_response(
            task_type=self.task_type,
            backend=self.backend,
            model_id=self.model_id,
            error_code=error_code,
            message=message,
            outputs=outputs,
        )


# --------------------------------------------------------------------------- #
# Dynamic adapter registry
# --------------------------------------------------------------------------- #

# Maps family name → factory function.  Populated at startup by
# adapter_discovery or by direct register_adapter() calls.
_ADAPTER_REGISTRY: dict[str, Callable[..., _WorkerBackend]] = {}


def register_adapter(family: str, factory: Callable[..., _WorkerBackend]) -> None:
    """Register an adapter factory for a model family.

    Args:
        family: The model family name (e.g. "mlx.z_image").
        factory: Callable accepting keyword args (task_type, backend,
                 model_id, model_spec, model_root, cache_root, python_bin)
                 and returning a _WorkerBackend.
    """
    _ADAPTER_REGISTRY[family] = factory


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #


def _build_backend(
    *,
    task_type: str,
    backend: str,
    cache_root,
    model_id: str,
    python_bin,
    model_spec: dict | None = None,
    model_root=None,
) -> _WorkerBackend:
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task_type}")
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if not _supported_backend_task(task_type, backend):
        raise ValueError(f"unsupported backend/task combination: {backend}/{task_type}")

    if model_spec is None:
        raise ValueError("model_spec is required")
    display_model_id = model_spec_display_id(model_spec)
    family = str(model_spec.get("family", "")).strip()
    factory = _ADAPTER_REGISTRY.get(family)
    if factory is not None:
        return factory(
            task_type=task_type,
            backend=backend,
            model_id=display_model_id,
            model_spec=model_spec,
            model_root=model_root,
            cache_root=cache_root,
            python_bin=python_bin,
        )
    raise ValueError(f"unsupported model family: {family}")
