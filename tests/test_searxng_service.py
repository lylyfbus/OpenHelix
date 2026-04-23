"""SearXNG service management tests (pip/venv backend)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.services import searxng as searxng_mod


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _service_paths(root: Path) -> dict[str, Path]:
    service_dir = root / "searxng"
    return {
        "_SERVICE_DIR": service_dir,
        "_STATE_PATH": service_dir / "state.json",
        "_SOURCE_DIR": service_dir / "source",
        "_VENV_DIR": service_dir / "venv",
        "_CONFIG_DIR": service_dir / "config",
        "_DATA_DIR": service_dir / "data",
    }


def _with_tmp_service():
    """Context manager: redirect module-level paths to a tempdir."""
    tmp = tempfile.TemporaryDirectory()
    paths = _service_paths(Path(tmp.name))
    paths["_SERVICE_DIR"].mkdir(parents=True, exist_ok=True)
    patcher = patch.multiple("helix.services.searxng", **paths)
    patcher.start()

    class _Ctx:
        def __enter__(self):
            return paths
        def __exit__(self, *args):
            patcher.stop()
            tmp.cleanup()
    return _Ctx()


# --------------------------------------------------------------------------- #
# Unit tests for _pid_alive
# --------------------------------------------------------------------------- #


def test_pid_alive_returns_true_for_current_process():
    assert searxng_mod._pid_alive(os.getpid()) is True
    print("  _pid_alive(current) -> True OK")


def test_pid_alive_returns_false_for_nonexistent_pid():
    # PID 0 and massive PIDs that shouldn't exist
    assert searxng_mod._pid_alive(999_999_999) is False
    print("  _pid_alive(nonexistent) -> False OK")


# --------------------------------------------------------------------------- #
# discover()
# --------------------------------------------------------------------------- #


def test_discover_returns_none_when_no_state_file():
    with _with_tmp_service() as _:
        assert searxng_mod.discover() is None
        print("  discover() no-state -> None OK")


def test_discover_unlinks_state_with_dead_pid():
    with _with_tmp_service() as paths:
        paths["_STATE_PATH"].write_text(
            json.dumps({"pid": 999_999_999, "port": 8888, "base_url": "x", "started_at": 0}),
            encoding="utf-8",
        )
        result = searxng_mod.discover()
        assert result is None
        assert not paths["_STATE_PATH"].exists()
        print("  discover() dead PID -> None + file unlinked OK")


def test_discover_unlinks_legacy_docker_state():
    """Docker-era state files had no `pid` key; should be treated as stale."""
    with _with_tmp_service() as paths:
        paths["_STATE_PATH"].write_text(
            json.dumps({
                "container_name": "helix-searxng",
                "network_name": "helix-sandbox-net",
                "base_url": "http://helix-searxng:8080",
                "started_at": 0,
            }),
            encoding="utf-8",
        )
        result = searxng_mod.discover()
        assert result is None
        assert not paths["_STATE_PATH"].exists()
        print("  discover() legacy Docker state -> None + file unlinked OK")


def test_discover_returns_state_when_pid_alive():
    with _with_tmp_service() as paths:
        paths["_STATE_PATH"].write_text(
            json.dumps({"pid": os.getpid(), "port": 8888, "base_url": "http://127.0.0.1:8888", "started_at": 0}),
            encoding="utf-8",
        )
        result = searxng_mod.discover()
        assert result is not None
        assert result["pid"] == os.getpid()
        assert result["base_url"] == "http://127.0.0.1:8888"
        print("  discover() alive PID -> state OK")


# --------------------------------------------------------------------------- #
# stop()
# --------------------------------------------------------------------------- #


def test_stop_is_noop_when_no_state():
    with _with_tmp_service() as paths:
        searxng_mod.stop()  # should not raise
        assert not paths["_STATE_PATH"].exists()
        print("  stop() no-state -> no error OK")


def test_stop_removes_state_file_for_dead_pid():
    """stop() should not crash if the tracked PID is already gone."""
    with _with_tmp_service() as paths:
        paths["_STATE_PATH"].write_text(
            json.dumps({"pid": 999_999_999, "port": 8888, "base_url": "x", "started_at": 0}),
            encoding="utf-8",
        )
        searxng_mod.stop()  # should not raise
        assert not paths["_STATE_PATH"].exists()
        print("  stop() dead PID -> state unlinked OK")


# --------------------------------------------------------------------------- #
# Bootstrap idempotency
# --------------------------------------------------------------------------- #


def test_ensure_source_skips_when_git_dir_exists():
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class _Done:
            returncode = 0
        return _Done()

    with _with_tmp_service() as paths:
        (paths["_SOURCE_DIR"] / ".git").mkdir(parents=True, exist_ok=True)
        with patch("helix.services.searxng.subprocess.run", side_effect=fake_run):
            searxng_mod._ensure_source(paths["_SOURCE_DIR"])
        assert calls == [], f"expected no git calls, got {calls}"
        print("  _ensure_source skips when .git present OK")


def test_ensure_venv_skips_when_python_exists():
    created = {"flag": False}

    class FakeBuilder:
        def __init__(self, *args, **kwargs):
            pass
        def create(self, root):
            created["flag"] = True

    with _with_tmp_service() as paths:
        (paths["_VENV_DIR"] / "bin").mkdir(parents=True, exist_ok=True)
        (paths["_VENV_DIR"] / "bin" / "python").touch()
        with patch("helix.services.searxng.venv.EnvBuilder", FakeBuilder):
            python_bin = searxng_mod._ensure_venv(paths["_VENV_DIR"])
        assert created["flag"] is False
        assert python_bin == paths["_VENV_DIR"] / "bin" / "python"
        print("  _ensure_venv skips when python exists OK")


# --------------------------------------------------------------------------- #
# start() full flow
# --------------------------------------------------------------------------- #


def test_start_reuses_existing_state_when_running():
    with _with_tmp_service() as paths:
        existing = {"pid": os.getpid(), "port": 8888, "base_url": "http://127.0.0.1:8888", "started_at": 0}
        paths["_STATE_PATH"].write_text(json.dumps(existing), encoding="utf-8")
        result = searxng_mod.start()
        assert result["pid"] == os.getpid()
        print("  start() when already running -> reused OK")


def test_start_full_flow_writes_state_with_pid():
    """Mock every outbound call; assert state.json gets written correctly."""
    with _with_tmp_service() as paths:
        with patch("helix.services.searxng._ensure_source") as mock_src, \
             patch("helix.services.searxng._ensure_venv", return_value=paths["_VENV_DIR"] / "bin" / "python") as mock_venv, \
             patch("helix.services.searxng._ensure_deps") as mock_deps, \
             patch("helix.services.searxng._write_settings") as mock_cfg, \
             patch("helix.services.searxng._spawn", return_value=424242) as mock_spawn, \
             patch("helix.services.searxng._wait_ready") as mock_ready:
            state = searxng_mod.start()

        assert state["pid"] == 424242
        assert state["port"] == 8888
        assert state["base_url"] == "http://127.0.0.1:8888"
        assert "started_at" in state

        persisted = json.loads(paths["_STATE_PATH"].read_text(encoding="utf-8"))
        assert persisted["pid"] == 424242
        assert persisted["base_url"] == "http://127.0.0.1:8888"

        mock_src.assert_called_once()
        mock_venv.assert_called_once()
        mock_deps.assert_called_once()
        mock_cfg.assert_called_once()
        mock_spawn.assert_called_once()
        mock_ready.assert_called_once()
        print("  start() full flow -> state.json with pid OK")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    print("=== searxng service ===")
    test_pid_alive_returns_true_for_current_process()
    test_pid_alive_returns_false_for_nonexistent_pid()
    test_discover_returns_none_when_no_state_file()
    test_discover_unlinks_state_with_dead_pid()
    test_discover_unlinks_legacy_docker_state()
    test_discover_returns_state_when_pid_alive()
    test_stop_is_noop_when_no_state()
    test_stop_removes_state_file_for_dead_pid()
    test_ensure_source_skips_when_git_dir_exists()
    test_ensure_venv_skips_when_python_exists()
    test_start_reuses_existing_state_when_running()
    test_start_full_flow_writes_state_with_pid()
    print("\n✅ All searxng service tests passed!")
