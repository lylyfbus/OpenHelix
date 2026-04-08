"""Lifecycle manager for the shared local model service."""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from .coordinator import _coordinator_main  # noqa: F401  # package visibility
from .paths import (
    default_cache_root,
    default_runtime_root,
    has_active_runtimes,
    register_active_runtime,
    unregister_active_runtime,
)
from .protocol import (
    _COORDINATOR_HEALTH_PATH,
    _DEFAULT_BACKEND_MODE,
    _DEFAULT_IDLE_SECONDS,
    _STARTUP_TIMEOUT_SECONDS,
    _find_free_port,
    _http_json_request,
    _kill_process_tree,
)


class LocalModelServiceManager:
    """Own the shared local coordinator process."""

    def __init__(
        self,
        workspace: Path,
        *,
        session_id: str,
        cache_root: Path | None = None,
        idle_seconds: int = _DEFAULT_IDLE_SECONDS,
        backend_mode: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.session_id = str(session_id or "session").strip() or "session"
        self.runtime_dir = default_runtime_root()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / "service.json"
        self.log_path = self.runtime_dir / "service.log"
        self.cache_root = Path(cache_root or default_cache_root(self.workspace)).expanduser().resolve()
        self.idle_seconds = max(1, int(idle_seconds))
        self.backend_mode = str(backend_mode or _DEFAULT_BACKEND_MODE).strip().lower() or "real"
        self._process: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._token: str | None = None
        self._runtime_marker_path: Path | None = None

    def start(self) -> None:
        if self._runtime_marker_path is None:
            self._runtime_marker_path = register_active_runtime(
                workspace=self.workspace,
                session_id=self.session_id,
            )
        if self._process is not None and self._process.poll() is None and self._port and self._token:
            return
        if self._adopt_existing_service():
            return
        self.cache_root.mkdir(parents=True, exist_ok=True)
        port = _find_free_port()
        token = f"tok_{secrets.token_urlsafe(24)}"
        with self.log_path.open("a", encoding="utf-8") as log_handle:
            cmd = [
                sys.executable,
                "-m",
                "helix.runtime.local_model_service",
                "coordinator",
                "--cache-root",
                str(self.cache_root),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                f"--token={token}",
                "--idle-seconds",
                str(self.idle_seconds),
                "--backend-mode",
                self.backend_mode,
                "--runtime-dir",
                str(self.runtime_dir),
                "--skills-root",
                str(self.workspace / "skills"),
            ]
            process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=log_handle,
                text=True,
                start_new_session=True,
            )
        self._process = process
        self._port = port
        self._token = token
        try:
            self._wait_for_health()
        except Exception:
            _kill_process_tree(process.pid or 0)
            self._process = None
            self._port = None
            self._token = None
            unregister_active_runtime(self._runtime_marker_path)
            self._runtime_marker_path = None
            raise
        self.state_path.write_text(
            json.dumps(
                {
                    "pid": int(process.pid or 0),
                    "port": port,
                    "token": token,
                    "started_at": time.time(),
                    "backend_mode": self.backend_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def stop(self) -> None:
        unregister_active_runtime(self._runtime_marker_path)
        self._runtime_marker_path = None
        if has_active_runtimes():
            self._process = None
            self._port = None
            self._token = None
            return
        pid = 0
        if self._process is not None:
            pid = int(self._process.pid or 0)
        elif self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            try:
                pid = int(payload.get("pid") or 0)
            except (TypeError, ValueError):
                pid = 0
        self.state_path.unlink(missing_ok=True)
        if pid > 0:
            _kill_process_tree(pid)
        self._process = None
        self._port = None
        self._token = None

    def tool_environment(self) -> dict[str, str]:
        if self._port is None or self._token is None:
            return {}
        return {
            "HELIX_LOCAL_MODEL_SERVICE_URL": f"http://host.docker.internal:{self._port}",
            "HELIX_LOCAL_MODEL_SERVICE_TOKEN": self._token,
        }

    def status_fields(self) -> dict[str, str]:
        if self._port is None:
            return {}
        return {
            "local_model_service": f"http://127.0.0.1:{self._port}",
            "local_model_service_backend": self.backend_mode,
        }

    def _adopt_existing_service(self) -> bool:
        if not self.state_path.exists():
            return False
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.state_path.unlink(missing_ok=True)
            return False
        try:
            pid = int(payload.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        try:
            port = int(payload.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        token = str(payload.get("token") or "").strip()
        backend_mode = str(payload.get("backend_mode") or "").strip().lower()
        if pid <= 0 or port <= 0 or not token:
            self.state_path.unlink(missing_ok=True)
            return False
        if backend_mode and backend_mode != self.backend_mode:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, 0)
                _kill_process_tree(pid)
            self.state_path.unlink(missing_ok=True)
            return False
        status, _, parsed = _http_json_request(
            method="GET",
            url=f"http://127.0.0.1:{port}{_COORDINATOR_HEALTH_PATH}",
            timeout=2,
        )
        if status == 200 and isinstance(parsed, dict) and parsed.get("status") == "ok":
            self._process = None
            self._port = port
            self._token = token
            return True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, 0)
            _kill_process_tree(pid)
        self.state_path.unlink(missing_ok=True)
        return False

    def _wait_for_health(self) -> None:
        assert self._port is not None
        deadline = time.time() + _STARTUP_TIMEOUT_SECONDS
        url = f"http://127.0.0.1:{self._port}{_COORDINATOR_HEALTH_PATH}"
        while time.time() < deadline:
            status, _, parsed = _http_json_request(method="GET", url=url, timeout=2)
            if status == 200 and isinstance(parsed, dict) and parsed.get("status") == "ok":
                return
            if self._process is not None and self._process.poll() is not None:
                detail = self.log_path.read_text(encoding="utf-8", errors="replace").strip()
                raise RuntimeError(detail or "local model service exited during startup")
            time.sleep(0.2)
        detail = self.log_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(detail or "local model service health check timed out")
