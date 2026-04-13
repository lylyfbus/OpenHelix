"""Coordinator process for local model service."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .helpers import _worker_python
from .model_spec import model_spec_signature, normalize_model_spec
from .constants import COORDINATOR_HEALTH_PATH, MODELS_SUBDIR, STARTUP_TIMEOUT_SECONDS, VENVS_SUBDIR
from .helpers import _error_response, _json_dumps, _request_timeout_seconds, _request_inputs


@dataclass
class _WorkerState:
    backend: str
    model_id: str
    model_signature: str
    process: subprocess.Popen[str]
    stdout_queue: "queue.Queue[str]"
    stderr_lines: list[str]
    stdin_lock: threading.Lock
    started_at: float
    task_type: str = ""

    @property
    def pid(self) -> int:
        return int(self.process.pid or 0)

    def captured_stderr(self) -> str:
        return "\n".join(self.stderr_lines).strip()


class _CoordinatorController:
    def __init__(
        self,
        *,
        service_root: Path,
        token: str,
        idle_seconds: int,
        backend_mode: str,
        skills_root: str = "",
    ) -> None:
        self.service_root = service_root
        self.token = token
        self.skills_root = str(skills_root or "").strip()
        self.idle_seconds = max(1, int(idle_seconds))
        self.backend_mode = backend_mode
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_state: _WorkerState | None = None
        self._last_used_at = 0.0
        self._eviction_thread = threading.Thread(target=self._eviction_loop, daemon=True)
        self._eviction_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._stop_worker()

    def authorize(self, header_value: str) -> bool:
        return header_value.strip() == f"Bearer {self.token}"

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            w = self._worker_state
            return {
                "status": "ok",
                "backend_mode": self.backend_mode,
                "worker_active": w is not None and w.process.poll() is None,
                "worker_task_type": w.task_type if w else "",
                "worker_backend": w.backend if w else "",
                "worker_model_id": w.model_id if w else "",
                "worker_pid": w.pid if w else 0,
            }

    # ----- Request handling ------------------------------------------------- #

    def handle_prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Load a model into the worker (pre-warm for inference)."""
        normalized, model_root = self._resolve_request(payload)
        skill_name = self._require_field(payload, "skill_name")
        task_type = str(payload.get("task_type", "")).strip()
        self._ensure_worker(normalized, model_root, skill_name=skill_name, task_type=task_type)
        repo_id = normalized["source"]["repo_id"]
        return {
            "status": "ok",
            "backend": normalized["backend"],
            "model": repo_id,
            "error_code": "",
            "message": f"model {repo_id} loaded and ready",
        }

    def handle_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run inference on the model."""
        normalized, model_root = self._resolve_request(payload)
        skill_name = self._require_field(payload, "skill_name")
        task_type = self._require_field(payload, "task_type")
        _request_inputs(payload)
        payload = dict(payload)
        payload["backend"] = normalized["backend"]
        payload["model_spec"] = normalized
        payload["request_timeout_seconds"] = _request_timeout_seconds(payload)
        worker = self._ensure_worker(normalized, model_root, skill_name=skill_name, task_type=task_type)
        return self._request_worker(worker, payload)

    def _resolve_request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        """Normalize model_spec, verify downloaded, return (normalized, model_root)."""
        raw_spec = payload.get("model_spec")
        if not isinstance(raw_spec, dict):
            raise ValueError("model_spec is required")
        normalized = normalize_model_spec(raw_spec)
        repo_id = normalized["source"]["repo_id"]
        model_root = self.service_root / MODELS_SUBDIR / repo_id.replace("/", "--")
        if not model_root.exists():
            raise RuntimeError(f"model {repo_id} has not been downloaded — run: helix model download")
        return normalized, model_root

    @staticmethod
    def _require_field(payload: dict[str, Any], field: str) -> str:
        value = str(payload.get(field, "")).strip()
        if not value:
            raise ValueError(f"{field} is required")
        return value

    def _ensure_worker(self, normalized: dict[str, Any], model_root: Path, *, skill_name: str, task_type: str = "") -> _WorkerState:
        """Ensure the correct worker is running, start if needed."""
        backend = normalized["backend"]
        repo_id = normalized["source"]["repo_id"]
        signature = model_spec_signature(normalized)
        with self._lock:
            worker = self._worker_state
            if (
                worker is not None
                and worker.process.poll() is None
                and worker.backend == backend
                and worker.model_signature == signature
            ):
                self._last_used_at = time.time()
                return worker
            self._stop_worker()
            self._worker_state = self._start_worker(
                skill_name=skill_name, task_type=task_type, backend=backend, model_id=repo_id,
                model_signature=signature, model_spec=normalized, model_root=model_root,
            )
            self._last_used_at = time.time()
            return self._worker_state

    # ----- Worker lifecycle ------------------------------------------------- #

    def _start_worker(
        self, *, skill_name: str, task_type: str, backend: str, model_id: str,
        model_signature: str, model_spec: dict[str, Any], model_root: Path,
    ) -> _WorkerState:
        python_bin = _worker_python(self.service_root / VENVS_SUBDIR / backend)
        cmd = [
            str(python_bin), "-m", "helix.runtime.local_model_service", "worker",
            "--skill-name", skill_name,
            "--task-type", task_type,
            "--backend", backend,
            "--model-id", model_id,
            "--service-root", str(self.service_root),
            "--backend-mode", self.backend_mode,
            "--model-spec-json", json.dumps(model_spec, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            "--model-root", str(model_root),
        ]
        if self.skills_root:
            cmd.extend(["--skills-root", self.skills_root])
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        hub_root = str(self.service_root / MODELS_SUBDIR)
        env.setdefault("HF_HOME", hub_root)
        env.setdefault("TRANSFORMERS_CACHE", hub_root)
        env.setdefault("HF_HUB_CACHE", hub_root)
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        stdout_queue: queue.Queue[str] = queue.Queue()
        stderr_lines: list[str] = []

        def pump_stdout():
            assert process.stdout is not None
            for line in process.stdout:
                stdout_queue.put(line.rstrip("\n"))

        def pump_stderr():
            assert process.stderr is not None
            for line in process.stderr:
                stderr_lines.append(line.rstrip("\n"))

        threading.Thread(target=pump_stdout, daemon=True).start()
        threading.Thread(target=pump_stderr, daemon=True).start()

        # Wait for ready signal
        deadline = time.time() + STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if process.poll() is not None:
                detail = "\n".join(stderr_lines).strip()
                raise RuntimeError(detail or "worker failed before becoming ready")
            try:
                raw = stdout_queue.get(timeout=0.2)
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("status") == "ready":
                    break
            except (queue.Empty, json.JSONDecodeError):
                continue
        else:
            self._terminate(process)
            detail = "\n".join(stderr_lines).strip()
            raise RuntimeError(detail or "worker startup timed out")

        return _WorkerState(
            backend=backend, model_id=model_id,
            model_signature=model_signature, process=process,
            stdout_queue=stdout_queue, stderr_lines=stderr_lines,
            stdin_lock=threading.Lock(), started_at=time.time(),
            task_type=task_type,
        )

    def _request_worker(self, worker: _WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
        if worker.process.poll() is not None:
            raise RuntimeError("worker exited before request")
        request_text = json.dumps(payload, ensure_ascii=True)
        deadline = time.monotonic() + _request_timeout_seconds(payload)
        stray_lines: list[str] = []
        with worker.stdin_lock:
            assert worker.process.stdin is not None
            worker.process.stdin.write(request_text + "\n")
            worker.process.stdin.flush()
            while True:
                if worker.process.poll() is not None and worker.stdout_queue.empty():
                    detail = worker.captured_stderr()
                    if stray_lines:
                        detail = f"{detail}\nlast worker stdout: {stray_lines[-1]}".strip()
                    raise RuntimeError(detail or "worker exited before response")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f" last worker stdout: {stray_lines[-1]}" if stray_lines else ""
                    raise RuntimeError(f"worker response timed out.{detail}".strip())
                try:
                    raw = worker.stdout_queue.get(timeout=min(remaining, 0.2))
                except queue.Empty:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    stray_lines.append(raw)
                    continue
                if isinstance(parsed, dict):
                    return parsed
                stray_lines.append(raw)

    def _stop_worker(self) -> None:
        worker = self._worker_state
        self._worker_state = None
        if worker is not None:
            self._terminate(worker.process)

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _eviction_loop(self) -> None:
        while not self._stop_event.wait(1.0):
            with self._lock:
                if self._worker_state is None:
                    continue
                if (time.time() - self._last_used_at) < self.idle_seconds:
                    continue
                self._stop_worker()


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


class _CoordinatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], controller: _CoordinatorController) -> None:
        super().__init__(server_address, handler_cls)
        self.controller = controller


class _CoordinatorHandler(BaseHTTPRequestHandler):
    server: _CoordinatorHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:
        if self.path == COORDINATOR_HEALTH_PATH:
            self._send_json(HTTPStatus.OK, self.server.controller.health_payload())
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "not found")

    def do_POST(self) -> None:
        ctrl = self.server.controller
        if not ctrl.authorize(str(self.headers.get("Authorization", ""))):
            self._send_error(HTTPStatus.UNAUTHORIZED, "unauthorized", "missing or invalid token")
            return
        payload = self._read_json_body()
        if payload is None:
            return  # error already sent
        handlers = {
            "/infer": ctrl.handle_infer,
            "/models/prepare": ctrl.handle_prepare,
        }
        handler = handlers.get(self.path)
        if handler is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "not found")
            return
        try:
            out = handler(payload)
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc), payload)
            return
        except RuntimeError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, "model_error", str(exc), payload)
            return
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "service_runtime_error", str(exc), payload)
            return
        status = HTTPStatus.OK if out.get("status") == "ok" else HTTPStatus.BAD_REQUEST
        self._send_json(status, out)

    def _read_json_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(max(0, length)).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_json", "request body must be JSON")
            return None
        if not isinstance(payload, dict):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_json", "request body must be a JSON object")
            return None
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = _json_dumps(payload)
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(
        self, status: HTTPStatus, error_code: str, message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        task_type, backend, model_id = "", "", ""
        if payload is not None:
            task_type, backend, model_id = self._describe_request(payload)
        self._send_json(status, _error_response(
            task_type=task_type, backend=backend, model_id=model_id,
            error_code=error_code, message=message,
        ))

    @staticmethod
    def _describe_request(payload: dict[str, Any]) -> tuple[str, str, str]:
        task_type = str(payload.get("task_type", "")).strip()
        backend = str(payload.get("backend", "")).strip().lower()
        model_id = ""
        raw_spec = payload.get("model_spec")
        if isinstance(raw_spec, dict):
            with contextlib.suppress(Exception):
                normalized = normalize_model_spec(raw_spec)
                backend = normalized["backend"]
                model_id = normalized["source"]["repo_id"]
        return task_type, backend, model_id


def _coordinator_main(args) -> int:
    skills_root = str(getattr(args, "skills_root", "") or "").strip()
    service_root = Path(args.service_root).expanduser().resolve()
    controller = _CoordinatorController(
        service_root=service_root,
        token=str(args.token),
        idle_seconds=int(args.idle_seconds),
        backend_mode=str(args.backend_mode),
        skills_root=skills_root,
    )
    server = _CoordinatorHTTPServer((str(args.host), int(args.port)), _CoordinatorHandler, controller)

    def _shutdown(*_unused):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        controller.close()
        server.server_close()
    return 0
