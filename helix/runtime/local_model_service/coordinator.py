"""Coordinator process for local model service."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .adapter_discovery import discover_and_register, discover_and_register_builtins
from .paths import _backend_cache_root, _worker_python
from .model_specs import (
    model_spec_display_id,
    model_spec_signature,
    normalize_model_spec,
)
from .preparer import ModelPreparationError, ensure_model_prepared, prepare_model_spec
from .protocol import (
    _COORDINATOR_HEALTH_PATH,
    _DEFAULT_BACKEND_MODE,
    _STARTUP_TIMEOUT_SECONDS,
    _error_response,
    _http_json_request,
    _json_dumps,
    _request_timeout_seconds,
    _request_inputs,
)


@dataclass
class _WorkerState:
    task_type: str
    backend: str
    model_id: str
    process: subprocess.Popen[str]
    stdout_queue: "queue.Queue[str]"
    stdin_lock: threading.Lock
    started_at: float
    log_handle: Any
    log_path: Path
    model_signature: str = ""

    @property
    def pid(self) -> int:
        return int(self.process.pid or 0)


class _CoordinatorController:
    def __init__(
        self,
        *,
        cache_root: Path,
        token: str,
        idle_seconds: int,
        backend_mode: str,
        runtime_dir: Path,
        skills_root: str = "",
    ) -> None:
        self.cache_root = cache_root
        self.token = token
        self.skills_root = str(skills_root or "").strip()
        self.idle_seconds = max(1, int(idle_seconds))
        self.backend_mode = backend_mode
        self.runtime_dir = runtime_dir
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_state: _WorkerState | None = None
        self._last_used_at = 0.0
        self._eviction_thread = threading.Thread(target=self._eviction_loop, daemon=True)
        self._eviction_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._stop_worker_locked()

    def authorize(self, header_value: str) -> bool:
        expected = f"Bearer {self.token}"
        return header_value.strip() == expected

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            worker = self._worker_state
            return {
                "status": "ok",
                "backend_mode": self.backend_mode,
                "worker_active": worker is not None and worker.process.poll() is None,
                "worker_task_type": worker.task_type if worker else "",
                "worker_backend": worker.backend if worker else "",
                "worker_kind": worker.task_type if worker else "",
                "worker_model_id": worker.model_id if worker else "",
                "worker_pid": worker.pid if worker else 0,
            }

    def handle_request(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload.get("model_spec"), dict):
            raise ValueError("model_spec is required")
        return self._handle_spec_request(payload=payload)

    def handle_prepare_request(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        raw_spec = payload.get("model_spec")
        timeout_seconds = _request_timeout_seconds(payload)
        normalized, model_root = prepare_model_spec(
            cache_root=self.cache_root,
            model_spec=raw_spec,
            backend_mode=self.backend_mode,
            timeout_seconds=timeout_seconds,
        )
        display_id = model_spec_display_id(normalized)
        return {
            "status": "ok",
            "task_type": normalized["task_type"],
            "backend": normalized["backend"],
            "model_id": display_id,
            "outputs": {
                "prepared": True,
                "model_root": str(model_root),
            },
            "error_code": "",
            "message": f"prepared model {display_id}",
        }

    def _handle_spec_request(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        raw_backend = str(payload.get("backend", "")).strip().lower()
        raw_task_type = str(payload.get("task_type", "")).strip()
        normalized = normalize_model_spec(
            payload.get("model_spec"),
            override_task_type=raw_task_type or None,
        )
        task_type = normalized["task_type"]
        backend = normalized["backend"]
        if raw_backend and raw_backend != backend:
            raise ValueError("payload backend does not match model_spec.backend")
        _request_inputs(payload)
        payload = dict(payload)
        payload["task_type"] = task_type
        payload["backend"] = backend
        payload["model_spec"] = normalized
        payload["request_timeout_seconds"] = _request_timeout_seconds(payload)
        normalized, model_root = ensure_model_prepared(
            cache_root=self.cache_root,
            model_spec=normalized,
            backend_mode=self.backend_mode,
        )
        model_id = model_spec_display_id(normalized)
        model_signature = model_spec_signature(normalized, override_task_type=task_type)
        with self._lock:
            worker = self._ensure_worker_locked(
                task_type=task_type,
                backend=backend,
                model_id=model_id,
                model_signature=model_signature,
                model_spec=normalized,
                model_root=model_root,
            )
            self._last_used_at = time.time()
        return self._request_worker(worker, payload)

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
                    detail = worker.log_path.read_text(encoding="utf-8", errors="replace").strip()
                    if stray_lines:
                        detail = f"{detail}\nlast worker stdout: {stray_lines[-1]}".strip()
                    raise RuntimeError(detail or "worker exited before response")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = ""
                    if stray_lines:
                        detail = f" last worker stdout: {stray_lines[-1]}"
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

    def _ensure_worker_locked(
        self,
        *,
        task_type: str,
        backend: str,
        model_id: str,
        model_signature: str | None = None,
        model_spec: dict[str, Any] | None = None,
        model_root: Path | None = None,
    ) -> _WorkerState:
        effective_signature = model_signature or model_id
        worker = self._worker_state
        if (
            worker is not None
            and worker.process.poll() is None
            and worker.task_type == task_type
            and worker.backend == backend
            and worker.model_signature == effective_signature
        ):
            return worker
        self._stop_worker_locked()
        self._worker_state = self._start_worker_locked(
            task_type=task_type,
            backend=backend,
            model_id=model_id,
            model_signature=effective_signature,
            model_spec=model_spec,
            model_root=model_root,
        )
        return self._worker_state

    def _start_worker_locked(
        self,
        *,
        task_type: str,
        backend: str,
        model_id: str,
        model_signature: str | None = None,
        model_spec: dict[str, Any] | None = None,
        model_root: Path | None = None,
    ) -> _WorkerState:
        effective_signature = model_signature or model_id
        backend_cache_root = _backend_cache_root(self.cache_root, backend)
        python_bin = _worker_python(backend_cache_root)
        log_path = self.runtime_dir / f"worker-{task_type}-{backend}-{int(time.time())}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        cmd = [
            str(python_bin),
            "-m",
            "helix.runtime.local_model_service",
            "worker",
            "--task-type",
            task_type,
            "--backend",
            backend,
            "--model-id",
            model_id,
            "--cache-root",
            str(backend_cache_root),
            "--backend-mode",
            self.backend_mode,
        ]
        if model_spec is not None:
            cmd.extend(
                [
                    "--model-spec-json",
                    json.dumps(model_spec, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                ]
            )
        if model_root is not None:
            cmd.extend(["--model-root", str(model_root)])
        if self.skills_root:
            cmd.extend(["--skills-root", self.skills_root])
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HOME", str(backend_cache_root / "models"))
        env.setdefault("TRANSFORMERS_CACHE", str(backend_cache_root / "models"))
        env.setdefault("HF_HUB_CACHE", str(backend_cache_root / "models"))
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            bufsize=1,
            env=env,
        )
        stdout_queue: "queue.Queue[str]" = queue.Queue()

        def _pump_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stdout_queue.put(line.rstrip("\n"))

        threading.Thread(target=_pump_stdout, daemon=True).start()
        deadline = time.time() + _STARTUP_TIMEOUT_SECONDS
        ready_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            if process.poll() is not None:
                detail = log_path.read_text(encoding="utf-8", errors="replace").strip()
                raise RuntimeError(detail or "worker failed before becoming ready")
            try:
                raw = stdout_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("status") == "ready":
                ready_payload = parsed
                break
        if ready_payload is None:
            self._terminate_process(process)
            detail = log_path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(detail or "worker startup timed out")
        return _WorkerState(
            task_type=task_type,
            backend=backend,
            model_id=model_id,
            model_signature=effective_signature,
            process=process,
            stdout_queue=stdout_queue,
            stdin_lock=threading.Lock(),
            started_at=time.time(),
            log_handle=log_handle,
            log_path=log_path,
        )

    def _stop_worker_locked(self) -> None:
        worker = self._worker_state
        self._worker_state = None
        if worker is None:
            return
        self._terminate_process(worker.process)
        with contextlib.suppress(Exception):
            worker.log_handle.close()

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
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
                self._stop_worker_locked()


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
        if self.path != _COORDINATOR_HEALTH_PATH:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                _error_response(
                    task_type="",
                    backend="",
                    model_id="",
                    error_code="not_found",
                    message="not found",
                ),
            )
            return
        self._send_json(HTTPStatus.OK, self.server.controller.health_payload())

    def do_POST(self) -> None:
        auth = str(self.headers.get("Authorization", "")).strip()
        if not self.server.controller.authorize(auth):
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                _error_response(
                    task_type="",
                    backend="",
                    model_id="",
                    error_code="unauthorized",
                    message="missing or invalid token",
                ),
            )
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(max(0, length)).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                _error_response(
                    task_type="",
                    backend="",
                    model_id="",
                    error_code="invalid_json",
                    message="request body must be JSON",
                ),
            )
            return
        if not isinstance(payload, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                _error_response(
                    task_type="",
                    backend="",
                    model_id="",
                    error_code="invalid_json",
                    message="request body must be a JSON object",
                ),
            )
            return
        try:
            if self.path == "/infer":
                out = self.server.controller.handle_request(payload=payload)
            elif self.path == "/models/prepare":
                out = self.server.controller.handle_prepare_request(payload=payload)
            else:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    _error_response(
                        task_type="",
                        backend="",
                        model_id="",
                        error_code="not_found",
                        message="not found",
                    ),
                )
                return
        except ValueError as exc:
            task_type, backend, model_id = self._describe_request(payload)
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                _error_response(
                    task_type=task_type,
                    backend=backend,
                    model_id=model_id,
                    error_code="invalid_request",
                    message=str(exc),
                ),
            )
            return
        except ModelPreparationError as exc:
            task_type, backend, model_id = self._describe_request(payload)
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                _error_response(
                    task_type=task_type,
                    backend=backend,
                    model_id=model_id,
                    error_code=exc.error_code,
                    message=exc.message,
                ),
            )
            return
        except Exception as exc:
            task_type, backend, model_id = self._describe_request(payload)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                _error_response(
                    task_type=task_type,
                    backend=backend,
                    model_id=model_id,
                    error_code="service_runtime_error",
                    message=str(exc),
                ),
            )
            return
        status_code = HTTPStatus.OK if out.get("status") == "ok" else HTTPStatus.BAD_REQUEST
        self._send_json(status_code, out)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = _json_dumps(payload)
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _describe_request(payload: dict[str, Any]) -> tuple[str, str, str]:
        task_type = str(payload.get("task_type", "")).strip()
        backend = str(payload.get("backend", "")).strip().lower()
        model_id = str(payload.get("model_id", "")).strip()
        raw_spec = payload.get("model_spec")
        if not isinstance(raw_spec, dict):
            return task_type, backend, model_id
        with contextlib.suppress(Exception):
            normalized = normalize_model_spec(raw_spec, override_task_type=task_type or None)
            task_type = normalized["task_type"]
            backend = normalized["backend"]
            model_id = model_spec_display_id(normalized)
        return task_type, backend, model_id


def _coordinator_main(args) -> int:
    # Register adapters from built-in and workspace skills
    discover_and_register_builtins()
    skills_root = str(getattr(args, "skills_root", "") or "").strip()
    if skills_root:
        discover_and_register(Path(skills_root))

    cache_root = Path(args.cache_root).expanduser().resolve()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    controller = _CoordinatorController(
        cache_root=cache_root,
        token=str(args.token),
        idle_seconds=int(args.idle_seconds),
        backend_mode=str(args.backend_mode),
        runtime_dir=runtime_dir,
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
