"""Local model service tests."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.local_model_service import (
    LocalModelServiceManager,
    _CoordinatorController,
    _RealImageGenerationBackend,
    _http_json_request,
    _kill_process_tree,
    default_cache_root,
    default_runtime_root,
)


def _configure_helix_home(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = workspace / ".test-helix-home"
    monkeypatch.setenv("HELIX_HOME", str(home))
    return home


def _start_manager(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str = "svc-01",
    idle_seconds: int = 300,
) -> LocalModelServiceManager:
    _configure_helix_home(workspace, monkeypatch)
    manager = LocalModelServiceManager(
        workspace,
        session_id=session_id,
        backend_mode="fake",
        idle_seconds=idle_seconds,
    )
    try:
        manager.start()
    except PermissionError as exc:
        pytest.skip(f"local socket bind is not permitted in this environment: {exc}")
    return manager


def test_local_model_service_start_and_stop(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            status_fields = manager.status_fields()
            status, _, parsed = _http_json_request(
                method="GET",
                url=f"{status_fields['local_model_service']}/health",
            )
            assert status == 200
            assert parsed["status"] == "ok"
            assert parsed["worker_active"] is False
            print("  Local model service start OK")
        finally:
            manager.stop()

        status, _, parsed = _http_json_request(
            method="GET",
            url=f"{status_fields['local_model_service']}/health",
            timeout=2,
        )
        assert status == 0
        assert parsed is None
        print("  Local model service stop OK")


def test_local_model_service_default_cache_root_is_global(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        helix_home = _configure_helix_home(workspace, monkeypatch)
        expected_cache = helix_home / "cache" / "local-model-service"
        expected_runtime = helix_home / "runtime" / "services" / "local-model-service"
        assert default_cache_root(workspace) == expected_cache.resolve()
        assert default_runtime_root() == expected_runtime.resolve()

        manager = LocalModelServiceManager(
            workspace,
            session_id="svc-default-cache",
            backend_mode="fake",
        )
        try:
            manager.start()
        except PermissionError as exc:
            pytest.skip(f"local socket bind is not permitted in this environment: {exc}")
        try:
            assert manager.cache_root == expected_cache.resolve()
            assert manager.cache_root.exists()
            assert manager.runtime_dir == expected_runtime.resolve()
            assert manager.state_path == expected_runtime.resolve() / "service.json"
            print("  Local model service global cache/runtime root OK")
        finally:
            manager.stop()


def test_local_model_service_auth_and_path_validation(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                payload={
                    "task_type": "image_generation",
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "workspace_root": str(workspace.resolve()),
                    "inputs": {
                        "prompt": "hello",
                        "size": "1024x1024",
                        "output_path": "generated_images/test.png",
                    },
                },
            )
            assert status == 401
            assert parsed["error_code"] == "unauthorized"

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "image_generation",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "inputs": {
                        "prompt": "hello",
                        "size": "1024x1024",
                        "output_path": "../escape.png",
                    },
                },
            )
            assert status == 400
            assert parsed["error_code"] == "invalid_request"
            print("  Local model service auth/path validation OK")
        finally:
            manager.stop()


def test_local_model_service_worker_switch_and_idle_eviction(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch, idle_seconds=1)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            token = env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"]

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=token,
                payload={
                    "task_type": "image_generation",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "inputs": {
                        "prompt": "a bright square",
                        "size": "1024x1024",
                        "output_path": "generated_images/first.png",
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            first_pid = int(health["worker_pid"])
            assert health["worker_task_type"] == "image_generation"
            assert health["worker_model_id"] == "Tongyi-MAI/Z-Image-Turbo"

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=token,
                payload={
                    "task_type": "image_generation",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo-q4",
                    "inputs": {
                        "prompt": "a darker square",
                        "size": "1024x1024",
                        "output_path": "generated_images/second.png",
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            assert health["worker_task_type"] == "image_generation"
            assert health["worker_model_id"] == "Tongyi-MAI/Z-Image-Turbo-q4"
            assert int(health["worker_pid"]) != first_pid

            deadline = time.time() + 5
            while time.time() < deadline:
                _, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
                if health and health.get("worker_active") is False:
                    break
                time.sleep(0.2)
            assert health["worker_active"] is False
            print("  Local model service worker switch/eviction OK")
        finally:
            manager.stop()


def test_local_model_service_shared_across_workspaces(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace_one = root / "workspace-one"
        workspace_two = root / "workspace-two"
        workspace_one.mkdir()
        workspace_two.mkdir()
        _configure_helix_home(root, monkeypatch)
        manager_one = LocalModelServiceManager(workspace_one, session_id="svc-one", backend_mode="fake", idle_seconds=1)
        manager_two = LocalModelServiceManager(workspace_two, session_id="svc-two", backend_mode="fake", idle_seconds=1)
        try:
            try:
                manager_one.start()
                manager_two.start()
            except PermissionError as exc:
                pytest.skip(f"local socket bind is not permitted in this environment: {exc}")
            env_one = manager_one.tool_environment()
            env_two = manager_two.tool_environment()
            assert env_one == env_two
            base_url = env_one["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(method="GET", url=f"{base_url}/health")
            assert status == 200
            assert parsed["status"] == "ok"
            manager_one.stop()
            status, _, parsed = _http_json_request(method="GET", url=f"{base_url}/health")
            assert status == 200
            assert parsed["status"] == "ok"
            print("  Local model service sharing OK")
        finally:
            manager_two.stop()


def test_local_model_service_recovers_stale_coordinator_state(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch, session_id="svc-stale")
        state = json.loads(manager.state_path.read_text(encoding="utf-8"))
        stale_pid = int(state["pid"])
        _kill_process_tree(stale_pid)

        replacement = LocalModelServiceManager(
            workspace,
            session_id="svc-stale",
            backend_mode="fake",
            idle_seconds=1,
        )
        try:
            replacement.start()
            fresh_state = json.loads(replacement.state_path.read_text(encoding="utf-8"))
            assert int(fresh_state["pid"]) != stale_pid
            print("  Local model service stale coordinator recovery OK")
        finally:
            replacement.stop()


def test_real_image_generation_backend_loads_lazily(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    def fake_load(self) -> None:
        calls["count"] += 1
        self.device = "cpu"

        class FakePipeline:
            def __call__(self, *, prompt: str, width: int, height: int):
                class FakeImage:
                    def save(self, path: Path) -> None:
                        Path(path).write_bytes(b"png")

                class FakeResult:
                    images = [FakeImage()]

                assert prompt == "demo prompt"
                assert width == 512
                assert height == 512
                return FakeResult()

        self.pipeline = FakePipeline()

    monkeypatch.setattr(_RealImageGenerationBackend, "_load", fake_load)

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealImageGenerationBackend(
            "Tongyi-MAI/Z-Image-Turbo",
            workspace / ".cache",
            workspace / "python",
        )
        assert calls["count"] == 0

        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prompt": "demo prompt",
                    "size": "512x512",
                    "output_path": "generated/demo.png",
                },
            }
        )

        assert calls["count"] == 1
        assert result["status"] == "ok"
        assert result["output_path"] == "generated/demo.png"


def test_worker_env_enables_hf_xet_high_performance(monkeypatch: pytest.MonkeyPatch):
    captured_env: dict[str, str] = {}

    class FakeProcess:
        def __init__(self, env: dict[str, str]) -> None:
            self.pid = 43210
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"status":"ready","task_type":"image_generation","model_id":"Tongyi-MAI/Z-Image-Turbo","pid":43210}\n')
            self._env = env

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(*args, **kwargs):
        nonlocal captured_env
        captured_env = dict(kwargs["env"])
        return FakeProcess(captured_env)

    monkeypatch.setattr("helix.core.local_model_service._worker_python", lambda cache_root: Path("/tmp/fake-python"))
    monkeypatch.setattr("helix.core.local_model_service.subprocess.Popen", fake_popen)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        controller = _CoordinatorController(
            cache_root=root / "cache",
            token="token",
            idle_seconds=300,
            backend_mode="fake",
            runtime_dir=root / "runtime",
        )
        try:
            worker = controller._start_worker_locked(
                task_type="image_generation",
                model_id="Tongyi-MAI/Z-Image-Turbo",
            )
            assert worker.pid == 43210
            assert captured_env["HF_XET_HIGH_PERFORMANCE"] == "1"
        finally:
            controller.close()
