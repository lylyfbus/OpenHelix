"""Path and dependency helpers for local model service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

from .protocol import _LOCAL_MODEL_SERVICE_NAME


def helix_home() -> Path:
    override = str(os.environ.get("HELIX_HOME", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".helix").resolve()


def _runtime_root() -> Path:
    return helix_home() / "runtime"


def _service_runtime_dir(service_name: str) -> Path:
    return _runtime_root() / "services" / service_name


def _service_cache_dir(service_name: str) -> Path:
    return helix_home() / "cache" / service_name


def _active_runtime_dir() -> Path:
    return _runtime_root() / "active-runtimes"


def _runtime_marker_path(pid: int | None = None) -> Path:
    token = int(pid or os.getpid())
    return _active_runtime_dir() / f"{token}.json"


def _prune_stale_runtime_markers() -> None:
    markers = _active_runtime_dir()
    if not markers.exists():
        return
    for marker in markers.glob("*.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker.unlink(missing_ok=True)
            continue
        try:
            pid = int(payload.get("pid"))
        except (TypeError, ValueError):
            marker.unlink(missing_ok=True)
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            marker.unlink(missing_ok=True)
        except PermissionError:
            continue


def register_active_runtime(*, workspace: Path, session_id: str) -> Path:
    markers = _active_runtime_dir()
    markers.mkdir(parents=True, exist_ok=True)
    _prune_stale_runtime_markers()
    marker = _runtime_marker_path()
    payload = {
        "pid": os.getpid(),
        "workspace": str(Path(workspace).expanduser().resolve()),
        "session_id": str(session_id or "session").strip() or "session",
        "started_at": time.time(),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def unregister_active_runtime(marker_path: Path | None) -> None:
    if marker_path is not None:
        marker_path.unlink(missing_ok=True)


def has_active_runtimes() -> bool:
    _prune_stale_runtime_markers()
    markers = _active_runtime_dir()
    if not markers.exists():
        return False
    return any(markers.glob("*.json"))


def default_cache_root(workspace: Path | None = None) -> Path:
    return _service_cache_dir(_LOCAL_MODEL_SERVICE_NAME).resolve()


def default_runtime_root() -> Path:
    return _service_runtime_dir(_LOCAL_MODEL_SERVICE_NAME).resolve()


def _backend_cache_root(cache_root: Path, backend: str) -> Path:
    return (cache_root / backend).resolve()


def _worker_python(cache_root: Path) -> Path:
    venv_root = cache_root / "venv"
    python_bin = venv_root / "bin" / "python"
    if python_bin.exists():
        return python_bin
    venv_root.parent.mkdir(parents=True, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=False)
    builder.create(str(venv_root))
    return python_bin


def _ensure_worker_dependencies(python_bin: Path, dependencies: tuple[str, ...]) -> None:
    if not dependencies:
        return
    completed = subprocess.run(
        [str(python_bin), "-m", "pip", "install", *dependencies],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or "failed installing worker dependencies")


def _safe_model_dir_name(model_id: str) -> str:
    return str(model_id or "model").strip().replace("/", "--")
