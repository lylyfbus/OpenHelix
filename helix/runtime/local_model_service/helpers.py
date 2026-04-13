"""Public protocol constants and request/response helpers for local model service."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import venv
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


from .constants import HTTP_TIMEOUT_SECONDS, WORKER_REQUEST_TIMEOUT_SECONDS


def local_model_service_supported() -> bool:
    """True when running on Apple Silicon or a Linux host with an NVIDIA GPU."""
    import platform
    import sys as _sys

    if _sys.platform == "darwin" and platform.machine().lower() == "arm64":
        return True
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _kill_process_tree(pid: int, *, grace_seconds: float = 5.0) -> None:
    if pid <= 0:
        return
    deadline = time.time() + max(0.1, grace_seconds)
    # Try group signal first; fall back to individual process signal when
    # the caller lacks permission (e.g. different session on macOS).
    try:
        os.killpg(pid, signal.SIGTERM)
    except PermissionError:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            break
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except PermissionError:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _http_json_request(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> tuple[int, str, dict[str, Any] | None]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else _json_dumps(payload)
    req = urllib.request.Request(url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body.strip() else None
            return int(getattr(resp, "status", 200)), body, parsed if isinstance(parsed, dict) else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = None
        try:
            candidate = json.loads(body)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            parsed = candidate
        return int(exc.code), body, parsed
    except urllib.error.URLError:
        return 0, "", None


def _resolve_workspace_path(
    workspace_root: Path,
    path_text: str,
    *,
    expect_exists: bool,
) -> Path:
    candidate = str(path_text or "").strip().replace("\\", "/")
    if not candidate:
        raise ValueError("workspace path is required")
    path_obj = Path(candidate)
    if path_obj.is_absolute():
        raise ValueError("absolute paths are not allowed")
    if ".." in path_obj.parts:
        raise ValueError("path traversal is not allowed")
    resolved = (workspace_root / path_obj).resolve(strict=False)
    workspace_resolved = workspace_root.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    if expect_exists:
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"workspace file not found: {candidate}")
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_service_workspace_root(payload: dict[str, Any]) -> Path:
    raw = str(payload.get("workspace_root", "")).strip()
    if not raw:
        raise ValueError("workspace_root is required")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        raise ValueError("workspace_root must be absolute")
    resolved = root.resolve(strict=False)
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("workspace_root must exist and be a directory")
    return resolved


def _parse_size(size_text: str) -> tuple[int, int]:
    token = str(size_text or "").strip().lower()
    if "x" not in token:
        raise ValueError(f"invalid size: {size_text}")
    left, right = token.split("x", 1)
    width = int(left)
    height = int(right)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid size: {size_text}")
    return width, height


def _parse_int(value: Any, *, default: int, minimum: int = 1) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    return parsed if parsed >= minimum else minimum


def _parse_float(value: Any, *, default: float, minimum: float = 0.0) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    return parsed if parsed >= minimum else minimum


def _request_timeout_seconds(payload: dict[str, Any]) -> int:
    raw = payload.get("request_timeout_seconds")
    if raw in (None, ""):
        return WORKER_REQUEST_TIMEOUT_SECONDS
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("request_timeout_seconds must be an integer") from exc
    if parsed < 1:
        raise ValueError("request_timeout_seconds must be >= 1")
    return parsed


def _ok_response(
    *,
    task_type: str,
    backend: str,
    model_id: str,
    outputs: dict[str, Any] | None,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "task_type": task_type,
        "backend": backend,
        "model_id": model_id,
        "outputs": outputs or {},
        "error_code": "",
        "message": message,
    }


def _error_response(
    *,
    task_type: str,
    backend: str,
    model_id: str,
    error_code: str,
    message: str,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "task_type": task_type,
        "backend": backend,
        "model_id": model_id,
        "outputs": outputs or {},
        "error_code": error_code,
        "message": message,
    }


def _request_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be a JSON object")
    return inputs


# ── Venv / dependency helpers ─────────────────────────────────────────── #


def _worker_python(venv_root: Path) -> Path:
    """Return the Python binary inside the venv, creating it if needed."""
    python_bin = venv_root / "bin" / "python"
    if python_bin.exists():
        return python_bin
    venv_root.mkdir(parents=True, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=False)
    builder.create(str(venv_root))
    return python_bin


def _ensure_worker_dependencies(python_bin: Path, dependencies: tuple[str, ...]) -> None:
    """pip-install dependencies into the venv that python_bin belongs to."""
    if not dependencies:
        return
    completed = subprocess.run(
        [str(python_bin), "-m", "pip", "install", *dependencies],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or "failed installing worker dependencies")


