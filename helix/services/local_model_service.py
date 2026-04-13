"""Local model service management.

Usage:
    helix start local-model-service    Start the coordinator process
    helix stop local-model-service     Stop the coordinator process
"""

from __future__ import annotations

import contextlib
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from helix.runtime.local_model_service.constants import (
    COORDINATOR_HEALTH_PATH,
    DEFAULT_BACKEND_MODE,
    DEFAULT_IDLE_SECONDS,
    SERVICE_ROOT,
    STARTUP_TIMEOUT_SECONDS,
)
from helix.runtime.local_model_service.helpers import (
    _find_free_port,
    _http_json_request,
    _kill_process_tree,
)

_STATE_PATH = SERVICE_ROOT / "state.json"


def start(workspace: Path) -> dict[str, Any]:
    """Start the coordinator process and write state.json.

    If an existing coordinator is healthy, reuse it.
    Returns the service state dict.
    """
    existing = discover()
    if existing is not None:
        return existing

    SERVICE_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = Path(workspace).expanduser().resolve()
    port = _find_free_port()
    token = f"tok_{secrets.token_urlsafe(24)}"
    backend_mode = DEFAULT_BACKEND_MODE

    process = subprocess.Popen(
        [
            sys.executable,
            "-m", "helix.runtime.local_model_service",
            "coordinator",
            "--service-root", str(SERVICE_ROOT),
            "--host", "127.0.0.1",
            "--port", str(port),
            f"--token={token}",
            "--idle-seconds", str(DEFAULT_IDLE_SECONDS),
            "--backend-mode", backend_mode,
            "--skills-root", str(workspace / "skills"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    # Wait for health
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    url = f"http://127.0.0.1:{port}{COORDINATOR_HEALTH_PATH}"
    while time.time() < deadline:
        status, _, parsed = _http_json_request(method="GET", url=url, timeout=2)
        if status == 200 and isinstance(parsed, dict) and parsed.get("status") == "ok":
            break
        if process.poll() is not None:
            stderr = ""
            if process.stderr is not None:
                try:
                    stderr = process.stderr.read().strip()
                except Exception:
                    pass
            raise RuntimeError(stderr or "local model service exited during startup")
        time.sleep(0.2)
    else:
        _kill_process_tree(process.pid or 0)
        raise RuntimeError("local model service health check timed out")

    state = {
        "pid": int(process.pid or 0),
        "port": port,
        "token": token,
        "started_at": time.time(),
        "backend_mode": backend_mode,
    }
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def stop() -> None:
    """Stop the coordinator process and remove state.json."""
    pid = 0
    if _STATE_PATH.exists():
        try:
            payload = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pid = 0
    _STATE_PATH.unlink(missing_ok=True)
    if pid > 0:
        _kill_process_tree(pid)


def discover() -> dict[str, Any] | None:
    """Check if the coordinator is running. Returns state dict or None."""
    if not _STATE_PATH.exists():
        return None
    try:
        state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _STATE_PATH.unlink(missing_ok=True)
        return None
    try:
        port = int(state.get("port") or 0)
    except (TypeError, ValueError):
        _STATE_PATH.unlink(missing_ok=True)
        return None
    token = str(state.get("token") or "").strip()
    if port <= 0 or not token:
        _STATE_PATH.unlink(missing_ok=True)
        return None
    # Health check
    status, _, parsed = _http_json_request(
        method="GET",
        url=f"http://127.0.0.1:{port}{COORDINATOR_HEALTH_PATH}",
        timeout=2,
    )
    if status == 200 and isinstance(parsed, dict) and parsed.get("status") == "ok":
        return state
    # Stale — clean up
    pid = 0
    try:
        pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        pass
    if pid > 0:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            _kill_process_tree(pid)
    _STATE_PATH.unlink(missing_ok=True)
    return None
