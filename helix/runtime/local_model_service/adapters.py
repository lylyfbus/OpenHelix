"""Adapter registry and backend base class for local model service."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from .helpers import _error_response, _ok_response


class _BaseBackend:
    """Base class for all adapter backends."""

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
            task_type=self.task_type, backend=self.backend,
            model_id=self.model_id, outputs=outputs, message=message,
        )

    def _error(self, *, error_code: str, message: str, outputs: dict[str, Any] | None = None) -> dict[str, Any]:
        return _error_response(
            task_type=self.task_type, backend=self.backend,
            model_id=self.model_id, error_code=error_code,
            message=message, outputs=outputs,
        )


class AdapterRegistry:
    """Registry of skill adapters, keyed by skill name.

    Each skill with a ``host_adapter.py`` exports a ``create_adapter(**kwargs)``
    factory. The registry discovers these and maps skill name → factory.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., _BaseBackend]] = {}

    def discover(self, skills_root: Path) -> list[str]:
        """Scan skills for host adapters and register them.

        Returns a list of registered skill names.
        """
        registered: list[str] = []
        skills_root = Path(skills_root).expanduser().resolve()
        if not skills_root.is_dir():
            return registered
        for adapter_path in sorted(skills_root.rglob("host_adapter.py")):
            try:
                module = self._load_module(adapter_path)
            except Exception:
                continue
            factory = getattr(module, "create_adapter", None)
            if factory is None:
                continue
            skill_name = adapter_path.parent.name
            self._registry[skill_name] = factory
            registered.append(skill_name)
        return registered

    def build_backend(
        self, *, skill_name: str, task_type: str, backend: str,
        model_id: str, cache_root: Path, python_bin: Path,
        model_spec: dict[str, Any] | None = None, model_root: Path | None = None,
    ) -> _BaseBackend:
        """Create a backend instance for the given skill."""
        factory = self._registry.get(skill_name)
        if factory is None:
            raise ValueError(f"no adapter registered for skill: {skill_name}")
        return factory(
            task_type=task_type, backend=backend, model_id=model_id,
            model_spec=model_spec, model_root=model_root,
            cache_root=cache_root, python_bin=python_bin,
        )

    @staticmethod
    def _load_module(path: Path) -> Any:
        module_name = f"_helix_adapter_{path.parent.name}_{id(path)}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
