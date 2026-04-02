"""Host-native local model service for Apple Silicon image skills."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import platform
import queue
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import venv
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol


_DEFAULT_IDLE_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_IDLE_SECONDS", "300"))
_HTTP_TIMEOUT_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_HTTP_TIMEOUT", "30"))
_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_STARTUP_TIMEOUT", "20"))
_WORKER_REQUEST_TIMEOUT_SECONDS = int(
    os.environ.get("HELIX_LOCAL_MODEL_SERVICE_WORKER_TIMEOUT", "900")
)
_COORDINATOR_HEALTH_PATH = "/health"
_FAKE_BACKEND_NAME = "fake"
_REAL_BACKEND_NAME = "real"
_DEFAULT_BACKEND_MODE = os.environ.get(
    "HELIX_LOCAL_MODEL_SERVICE_BACKEND",
    _REAL_BACKEND_NAME,
).strip().lower() or _REAL_BACKEND_NAME
_FAKE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9pob7XUAAAAASUVORK5CYII="
)
_DEFAULT_ANALYSIS_MODEL_ID = "zai-org/GLM-OCR"
_DEFAULT_GENERATION_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
_GENERATION_DEPENDENCIES = (
    "accelerate",
    "diffusers>=0.35.0",
    "huggingface_hub",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)
_ANALYSIS_DEPENDENCIES = (
    "accelerate",
    "pillow",
    "torch",
    "transformers",
)


def local_model_service_supported() -> bool:
    """Return whether the host runtime should use the local model service."""
    return sys.platform == "darwin" and platform.machine().lower() == "arm64"


def default_cache_root() -> Path:
    return (Path.home() / "Library" / "Caches" / "helix" / "local-model-service").resolve()


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
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            break
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def _http_json_request(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = _HTTP_TIMEOUT_SECONDS,
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


def _normalize_relative_path(path_text: str) -> str:
    candidate = str(path_text or "").strip().replace("\\", "/")
    return candidate


def _resolve_workspace_path(
    workspace_root: Path,
    path_text: str,
    *,
    expect_exists: bool,
) -> Path:
    candidate = _normalize_relative_path(path_text)
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


@dataclass
class _WorkerState:
    kind: str
    model_id: str
    process: subprocess.Popen[str]
    stdout_queue: "queue.Queue[str]"
    stdin_lock: threading.Lock
    started_at: float
    log_handle: Any
    log_path: Path

    @property
    def pid(self) -> int:
        return int(self.process.pid or 0)


class _WorkerBackend(Protocol):
    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class _FakeImageGenerationBackend:
    def __init__(self, workspace_root: Path, model_id: str) -> None:
        self.workspace_root = workspace_root
        self.model_id = model_id

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return {
                "status": "error",
                "output_path": "",
                "model_id": self.model_id,
                "error_code": "image_prompt_missing",
                "message": "prompt is required",
            }
        resolved = _resolve_workspace_path(
            self.workspace_root,
            str(payload.get("output_path", "")).strip(),
            expect_exists=False,
        )
        resolved.write_bytes(_FAKE_PNG_BYTES)
        output_path = str(resolved.relative_to(self.workspace_root))
        return {
            "status": "ok",
            "output_path": output_path,
            "model_id": self.model_id,
            "error_code": "",
            "message": f"generated placeholder image at {output_path}",
        }


class _FakeImageAnalysisBackend:
    def __init__(self, workspace_root: Path, model_id: str) -> None:
        self.workspace_root = workspace_root
        self.model_id = model_id

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query", "")).strip()
        if not query:
            return {
                "status": "error",
                "analysis": "",
                "model_id": self.model_id,
                "error_code": "image_query_missing",
                "message": "query is required",
            }
        resolved = _resolve_workspace_path(
            self.workspace_root,
            str(payload.get("image_path", "")).strip(),
            expect_exists=True,
        )
        image_path = str(resolved.relative_to(self.workspace_root))
        return {
            "status": "ok",
            "analysis": f"fake-analysis: {query} ({image_path})",
            "model_id": self.model_id,
            "error_code": "",
            "message": "analysis complete",
        }


class _RealImageAnalysisBackend:
    def __init__(self, workspace_root: Path, model_id: str, cache_root: Path, python_bin: Path) -> None:
        self.workspace_root = workspace_root
        self.model_id = model_id
        self.cache_root = cache_root
        self.python_bin = python_bin
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            from PIL import Image  # noqa: F401
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _ANALYSIS_DEPENDENCIES)
            from PIL import Image  # noqa: F401
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "mps" else torch.float32
        hub_cache = self.cache_root / "models"
        hub_cache.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(hub_cache),
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=dtype,
            cache_dir=str(hub_cache),
        )
        self.model.to(device)

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        resolved = _resolve_workspace_path(
            self.workspace_root,
            str(payload.get("image_path", "")).strip(),
            expect_exists=True,
        )
        query = str(payload.get("query", "")).strip()
        if not query:
            return {
                "status": "error",
                "analysis": "",
                "model_id": self.model_id,
                "error_code": "image_query_missing",
                "message": "query is required",
            }
        try:
            from PIL import Image
            image = Image.open(resolved)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(resolved)},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            assert self.processor is not None
            assert self.model is not None
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            inputs.pop("token_type_ids", None)
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )
            return {
                "status": "ok",
                "analysis": str(output_text).strip(),
                "model_id": self.model_id,
                "error_code": "",
                "message": "analysis complete",
            }
        except Exception as exc:
            return {
                "status": "error",
                "analysis": "",
                "model_id": self.model_id,
                "error_code": "analysis_runtime_error",
                "message": str(exc),
            }


class _RealImageGenerationBackend:
    def __init__(self, workspace_root: Path, model_id: str, cache_root: Path, python_bin: Path) -> None:
        self.workspace_root = workspace_root
        self.model_id = model_id
        self.cache_root = cache_root
        self.python_bin = python_bin
        self.pipeline: Any | None = None
        self.device: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from diffusers import ZImagePipeline
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _GENERATION_DEPENDENCIES)
            import torch
            from diffusers import ZImagePipeline

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "mps" else torch.float32
        hub_cache = self.cache_root / "models"
        hub_cache.mkdir(parents=True, exist_ok=True)
        self.pipeline = ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=str(hub_cache),
            low_cpu_mem_usage=False,
        )
        self.pipeline.to(device)

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return {
                "status": "error",
                "output_path": "",
                "model_id": self.model_id,
                "error_code": "image_prompt_missing",
                "message": "prompt is required",
            }
        try:
            width, height = _parse_size(str(payload.get("size", "")).strip())
            output_path = _resolve_workspace_path(
                self.workspace_root,
                str(payload.get("output_path", "")).strip(),
                expect_exists=False,
            )
            assert self.pipeline is not None
            result = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
            )
            image = result.images[0]
            image.save(output_path)
            rel = str(output_path.relative_to(self.workspace_root))
            return {
                "status": "ok",
                "output_path": rel,
                "model_id": self.model_id,
                "error_code": "",
                "message": f"generated image at {rel}",
            }
        except Exception as exc:
            return {
                "status": "error",
                "output_path": "",
                "model_id": self.model_id,
                "error_code": "generation_runtime_error",
                "message": str(exc),
            }


def _build_backend(
    *,
    kind: str,
    workspace_root: Path,
    cache_root: Path,
    model_id: str,
    backend_mode: str,
    python_bin: Path,
) -> _WorkerBackend:
    if backend_mode == _FAKE_BACKEND_NAME:
        if kind == "generate":
            return _FakeImageGenerationBackend(workspace_root, model_id)
        return _FakeImageAnalysisBackend(workspace_root, model_id)
    if kind == "generate":
        return _RealImageGenerationBackend(workspace_root, model_id, cache_root, python_bin)
    return _RealImageAnalysisBackend(workspace_root, model_id, cache_root, python_bin)


class _CoordinatorController:
    def __init__(
        self,
        *,
        workspace_root: Path,
        cache_root: Path,
        token: str,
        idle_seconds: int,
        backend_mode: str,
        runtime_dir: Path,
    ) -> None:
        self.workspace_root = workspace_root
        self.cache_root = cache_root
        self.token = token
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
                "worker_kind": worker.kind if worker else "",
                "worker_model_id": worker.model_id if worker else "",
                "worker_pid": worker.pid if worker else 0,
            }

    def handle_request(self, *, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            return {
                "status": "error",
                "model_id": "",
                "error_code": "model_id_missing",
                "message": "model_id is required",
            }
        with self._lock:
            worker = self._ensure_worker_locked(kind=kind, model_id=model_id)
            self._last_used_at = time.time()
        return self._request_worker(worker, payload)

    def _request_worker(self, worker: _WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
        if worker.process.poll() is not None:
            raise RuntimeError("worker exited before request")
        request_text = json.dumps(payload, ensure_ascii=True)
        with worker.stdin_lock:
            assert worker.process.stdin is not None
            worker.process.stdin.write(request_text + "\n")
            worker.process.stdin.flush()
            try:
                raw = worker.stdout_queue.get(timeout=_WORKER_REQUEST_TIMEOUT_SECONDS)
            except queue.Empty as exc:
                raise RuntimeError("worker response timed out") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid worker response: {raw}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("worker response must be a JSON object")
        return parsed

    def _ensure_worker_locked(self, *, kind: str, model_id: str) -> _WorkerState:
        worker = self._worker_state
        if (
            worker is not None
            and worker.process.poll() is None
            and worker.kind == kind
            and worker.model_id == model_id
        ):
            return worker
        self._stop_worker_locked()
        self._worker_state = self._start_worker_locked(kind=kind, model_id=model_id)
        return self._worker_state

    def _start_worker_locked(self, *, kind: str, model_id: str) -> _WorkerState:
        python_bin = _worker_python(self.cache_root)
        log_path = self.runtime_dir / f"worker-{kind}-{int(time.time())}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        cmd = [
            str(python_bin),
            "-m",
            "helix.core.local_model_service",
            "worker",
            "--kind",
            kind,
            "--model-id",
            model_id,
            "--workspace",
            str(self.workspace_root),
            "--cache-root",
            str(self.cache_root),
            "--backend-mode",
            self.backend_mode,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HOME", str(self.cache_root / "models"))
        env.setdefault("TRANSFORMERS_CACHE", str(self.cache_root / "models"))
        env.setdefault("HF_HUB_CACHE", str(self.cache_root / "models"))
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
            kind=kind,
            model_id=model_id,
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
                {"status": "error", "error_code": "not_found", "message": "not found"},
            )
            return
        self._send_json(HTTPStatus.OK, self.server.controller.health_payload())

    def do_POST(self) -> None:
        auth = str(self.headers.get("Authorization", "")).strip()
        if not self.server.controller.authorize(auth):
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"status": "error", "error_code": "unauthorized", "message": "missing or invalid token"},
            )
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(max(0, length)).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error_code": "invalid_json", "message": "request body must be JSON"},
            )
            return
        if not isinstance(payload, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error_code": "invalid_json", "message": "request body must be a JSON object"},
            )
            return
        try:
            if self.path == "/v1/image/generate":
                out = self.server.controller.handle_request(kind="generate", payload=payload)
            elif self.path == "/v1/image/analyze":
                out = self.server.controller.handle_request(kind="analyze", payload=payload)
            else:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"status": "error", "error_code": "not_found", "message": "not found"},
                )
                return
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error_code": "invalid_request", "message": str(exc)},
            )
            return
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "error", "error_code": "service_runtime_error", "message": str(exc)},
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


class LocalModelServiceManager:
    """Own the session-scoped local coordinator process."""

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
        self.session_token = secrets.token_hex(8)
        self.runtime_dir = self.workspace / ".runtime" / "local-model-service"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / f"{self.session_id}.json"
        self.log_path = self.runtime_dir / f"{self.session_id}.log"
        self.cache_root = Path(cache_root or default_cache_root()).expanduser().resolve()
        self.idle_seconds = max(1, int(idle_seconds))
        self.backend_mode = str(backend_mode or _DEFAULT_BACKEND_MODE).strip().lower() or _REAL_BACKEND_NAME
        self._process: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._token: str | None = None

    def start(self) -> None:
        self._cleanup_stale()
        if self._process is not None and self._process.poll() is None and self._port and self._token:
            return
        self.cache_root.mkdir(parents=True, exist_ok=True)
        port = _find_free_port()
        token = secrets.token_urlsafe(24)
        with self.log_path.open("a", encoding="utf-8") as log_handle:
            cmd = [
                sys.executable,
                "-m",
                "helix.core.local_model_service",
                "coordinator",
                "--workspace",
                str(self.workspace),
                "--cache-root",
                str(self.cache_root),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--token",
                token,
                "--idle-seconds",
                str(self.idle_seconds),
                "--backend-mode",
                self.backend_mode,
                "--runtime-dir",
                str(self.runtime_dir),
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
            raise
        self.state_path.write_text(
            json.dumps(
                {
                    "pid": int(process.pid or 0),
                    "session_id": self.session_id,
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

    def _cleanup_stale(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.state_path.unlink(missing_ok=True)
            return
        pid = int(payload.get("pid") or 0)
        port = int(payload.get("port") or 0)
        token = str(payload.get("token") or "").strip()
        if pid <= 0 or port <= 0 or not token:
            self.state_path.unlink(missing_ok=True)
            return
        status, _, parsed = _http_json_request(
            method="GET",
            url=f"http://127.0.0.1:{port}{_COORDINATOR_HEALTH_PATH}",
            timeout=2,
        )
        if status == 200 and isinstance(parsed, dict) and parsed.get("status") == "ok":
            _kill_process_tree(pid)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 0)
                _kill_process_tree(pid)
        self.state_path.unlink(missing_ok=True)

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


def _coordinator_main(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    controller = _CoordinatorController(
        workspace_root=workspace_root,
        cache_root=cache_root,
        token=str(args.token),
        idle_seconds=int(args.idle_seconds),
        backend_mode=str(args.backend_mode),
        runtime_dir=runtime_dir,
    )
    server = _CoordinatorHTTPServer((str(args.host), int(args.port)), _CoordinatorHandler, controller)

    def _shutdown(*_unused: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        controller.close()
        server.server_close()
    return 0


def _worker_main(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    python_bin = Path(sys.executable).resolve()
    backend = _build_backend(
        kind=str(args.kind),
        workspace_root=workspace_root,
        cache_root=cache_root,
        model_id=str(args.model_id),
        backend_mode=str(args.backend_mode),
        python_bin=python_bin,
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "kind": str(args.kind),
                "model_id": str(args.model_id),
                "pid": os.getpid(),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_code": "invalid_json",
                        "message": "worker request must be a JSON object",
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        if not isinstance(payload, dict):
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_code": "invalid_json",
                        "message": "worker request must be a JSON object",
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        try:
            response = backend.handle(payload)
        except Exception as exc:
            response = {
                "status": "error",
                "error_code": "worker_runtime_error",
                "message": str(exc),
            }
        print(json.dumps(response, ensure_ascii=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helix local model service")
    subparsers = parser.add_subparsers(dest="role", required=True)

    coordinator = subparsers.add_parser("coordinator")
    coordinator.add_argument("--workspace", required=True)
    coordinator.add_argument("--cache-root", required=True)
    coordinator.add_argument("--runtime-dir", required=True)
    coordinator.add_argument("--host", required=True)
    coordinator.add_argument("--port", required=True)
    coordinator.add_argument("--token", required=True)
    coordinator.add_argument("--idle-seconds", default=str(_DEFAULT_IDLE_SECONDS))
    coordinator.add_argument("--backend-mode", default=_DEFAULT_BACKEND_MODE)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--workspace", required=True)
    worker.add_argument("--cache-root", required=True)
    worker.add_argument("--kind", required=True, choices=["generate", "analyze"])
    worker.add_argument("--model-id", required=True)
    worker.add_argument("--backend-mode", default=_DEFAULT_BACKEND_MODE)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.role == "coordinator":
        return _coordinator_main(args)
    if args.role == "worker":
        return _worker_main(args)
    raise ValueError(f"unsupported role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
