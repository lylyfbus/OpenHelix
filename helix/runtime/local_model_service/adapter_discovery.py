"""Discover and register skill-provided model adapters at startup.

Scans skill directories for ``host_adapter.py`` files, imports their
``create_adapter`` factory, and registers them in the dynamic adapter
registry.  This allows new generative capabilities to be added by
installing a skill rather than editing the helix package.

Each ``host_adapter.py`` must export:
    FAMILY    — str, the model family name (e.g. "mlx.z_image")
    BACKEND   — str, the backend name (e.g. "pytorch", "mlx")
    TASK_TYPES — list[str], the supported task types
    create_adapter(**kwargs) — factory returning a _WorkerBackend

A single host_adapter.py may register additional families by defining
``FAMILY_<suffix>``, ``BACKEND_<suffix>``, ``TASK_TYPES_<suffix>``,
and ``create_<suffix>_adapter`` for each extra family.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def discover_and_register(skills_root: Path) -> list[str]:
    """Scan skills for host adapters and register them.

    Returns a list of registered family names (for logging).
    """
    from .model_specs import register_family
    from .protocol import register_backend, register_backend_task, register_task_type
    from .registry import register_adapter

    def _register(family: str, backend: str, task_types: list[str], factory: Any) -> None:
        register_family(family, backend=backend, task_types=set(task_types))
        register_backend(backend)
        for tt in task_types:
            register_task_type(tt)
            register_backend_task(tt, backend)
        register_adapter(family, factory)

    registered: list[str] = []
    skills_root = Path(skills_root).expanduser().resolve()
    if not skills_root.is_dir():
        return registered

    for adapter_path in sorted(skills_root.rglob("host_adapter.py")):
        try:
            module = _load_module(adapter_path)
        except Exception:
            continue

        # Primary family
        family = getattr(module, "FAMILY", None)
        backend = getattr(module, "BACKEND", None)
        task_types = getattr(module, "TASK_TYPES", None)
        factory = getattr(module, "create_adapter", None)

        if all((family, backend, task_types, factory)):
            _register(family, backend, task_types, factory)
            registered.append(family)

        # Additional families (FAMILY_<suffix>, create_<suffix>_adapter)
        for attr in dir(module):
            if not attr.startswith("FAMILY_"):
                continue
            suffix = attr[len("FAMILY_"):].lower()
            extra_family = getattr(module, attr, None)
            extra_backend = getattr(module, f"BACKEND_{suffix.upper()}", None)
            extra_task_types = getattr(module, f"TASK_TYPES_{suffix.upper()}", None)
            extra_factory = getattr(module, f"create_{suffix}_adapter", None)
            if all((extra_family, extra_backend, extra_task_types, extra_factory)):
                _register(extra_family, extra_backend, extra_task_types, extra_factory)
                registered.append(extra_family)

    return registered


def discover_and_register_builtins() -> list[str]:
    """Register adapters from built-in skills shipped with helix."""
    builtin_skills = Path(__file__).resolve().parent.parent.parent / "builtin_skills"
    return discover_and_register(builtin_skills)


def _load_module(path: Path) -> Any:
    """Import a Python file as a module without adding it to sys.modules permanently."""
    module_name = f"_helix_adapter_{path.parent.name}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
