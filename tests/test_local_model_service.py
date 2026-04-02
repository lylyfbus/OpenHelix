"""Local model service tests."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.local_model_service import (
    LocalModelServiceManager,
    _http_json_request,
    _kill_process_tree,
)


def _start_manager(workspace: Path, *, session_id: str = "svc-01", idle_seconds: int = 300) -> LocalModelServiceManager:
    manager = LocalModelServiceManager(
        workspace,
        session_id=session_id,
        cache_root=workspace / ".runtime" / "test-local-model-cache",
        backend_mode="fake",
        idle_seconds=idle_seconds,
    )
    try:
        manager.start()
    except PermissionError as exc:
        pytest.skip(f"local socket bind is not permitted in this environment: {exc}")
    return manager


def test_local_model_service_start_and_stop():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace)
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


def test_local_model_service_auth_and_path_validation():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        image_path = workspace / "sample.png"
        image_path.write_bytes(b"png")
        manager = _start_manager(workspace)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/v1/image/generate",
                payload={
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "prompt": "hello",
                    "size": "1024x1024",
                    "output_path": "generated_images/test.png",
                },
            )
            assert status == 401
            assert parsed["error_code"] == "unauthorized"

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/v1/image/analyze",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "model_id": "zai-org/GLM-OCR",
                    "image_path": "../escape.png",
                    "query": "extract text",
                },
            )
            assert status == 400
            assert parsed["error_code"] == "invalid_request"
            print("  Local model service auth/path validation OK")
        finally:
            manager.stop()


def test_local_model_service_worker_switch_and_idle_eviction():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        image_path = workspace / "sample.png"
        image_path.write_bytes(b"png")
        manager = _start_manager(workspace, idle_seconds=1)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            token = env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"]

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/v1/image/analyze",
                token=token,
                payload={
                    "model_id": "zai-org/GLM-OCR",
                    "image_path": "sample.png",
                    "query": "extract visible text",
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            first_pid = int(health["worker_pid"])
            assert health["worker_kind"] == "analyze"
            assert health["worker_model_id"] == "zai-org/GLM-OCR"

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/v1/image/generate",
                token=token,
                payload={
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "prompt": "a bright square",
                    "size": "1024x1024",
                    "output_path": "generated_images/test.png",
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            assert health["worker_kind"] == "generate"
            assert health["worker_model_id"] == "Tongyi-MAI/Z-Image-Turbo"
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


def test_local_model_service_recovers_stale_coordinator_state():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, session_id="svc-stale")
        state = json.loads(manager.state_path.read_text(encoding="utf-8"))
        stale_pid = int(state["pid"])
        _kill_process_tree(stale_pid)

        replacement = LocalModelServiceManager(
            workspace,
            session_id="svc-stale",
            cache_root=workspace / ".runtime" / "test-local-model-cache",
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
