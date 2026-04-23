"""SearXNG search service management (pip/venv backend).

Usage:
    helix start searxng    Bootstrap if needed, then launch SearXNG as a
                           host-level Python subprocess.
    helix stop searxng     Signal and remove the managed subprocess.

The service is managed entirely on the host — no Docker required. On first
``start``, SearXNG is cloned from GitHub into ``~/.helix/services/searxng/
source`` and installed editable into ``~/.helix/services/searxng/venv``
(this takes a few minutes). Subsequent starts are instant.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from helix.constants import SERVICES_ROOT


# ----- Configuration -------------------------------------------------------- #

_SEARXNG_REPO = "https://github.com/searxng/searxng.git"
_PORT = 8888
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_READY_TIMEOUT = 60
_READY_POLL = 1.0
_SHUTDOWN_POLL = 0.5
_SHUTDOWN_GRACE_SECONDS = 10
_SUPPORT_DEPS = (
    "pip",
    "setuptools",
    "wheel",
    "pyyaml",
    "msgspec",
    "typing-extensions",
    "pybind11",
)


# ----- Disk layout ---------------------------------------------------------- #

_SERVICE_DIR = SERVICES_ROOT / "searxng"
_STATE_PATH = _SERVICE_DIR / "state.json"
_SOURCE_DIR = _SERVICE_DIR / "source"
_VENV_DIR = _SERVICE_DIR / "venv"
_CONFIG_DIR = _SERVICE_DIR / "config"
_DATA_DIR = _SERVICE_DIR / "data"


# ----- Public API ----------------------------------------------------------- #


def start() -> dict[str, Any]:
    """Bootstrap (if needed) and launch SearXNG; write state.json.

    If SearXNG is already running (state file valid, PID alive), return
    the existing state dict without relaunching.
    """
    existing = discover()
    if existing is not None:
        return existing

    _SERVICE_DIR.mkdir(parents=True, exist_ok=True)

    if not (_SOURCE_DIR / ".git").exists():
        print("Bootstrapping SearXNG source (first run; may take a few minutes)...")
    _ensure_source(_SOURCE_DIR)
    python_bin = _ensure_venv(_VENV_DIR)
    _ensure_deps(python_bin, _SOURCE_DIR)
    _write_settings(_CONFIG_DIR)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    pid = _spawn(python_bin, _CONFIG_DIR / "settings.yml", _DATA_DIR)
    try:
        _wait_ready(_PORT)
    except Exception:
        # Readiness probe failed; try to kill the child we just spawned so
        # we don't leave a half-dead process behind.
        _kill_pid(pid)
        raise

    state: dict[str, Any] = {
        "pid": int(pid),
        "port": _PORT,
        "base_url": _BASE_URL,
        "started_at": time.time(),
    }
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def stop() -> None:
    """Signal the managed SearXNG subprocess and remove state.json."""
    if not _STATE_PATH.exists():
        return
    try:
        state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _STATE_PATH.unlink(missing_ok=True)
        return

    pid = state.get("pid")
    if isinstance(pid, int) and pid > 0:
        _kill_pid(pid)
    _STATE_PATH.unlink(missing_ok=True)


def discover() -> dict[str, Any] | None:
    """Return the state dict iff SearXNG is running; else clean up stale state."""
    if not _STATE_PATH.exists():
        return None
    try:
        state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _STATE_PATH.unlink(missing_ok=True)
        return None

    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0 or not _pid_alive(pid):
        _STATE_PATH.unlink(missing_ok=True)
        return None
    return state


# ----- Bootstrap helpers ---------------------------------------------------- #


def _ensure_source(source_root: Path) -> None:
    """Clone the SearXNG source tree if not already present."""
    if (source_root / ".git").exists():
        return
    source_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", _SEARXNG_REPO, str(source_root)],
        check=True,
    )


def _ensure_venv(venv_root: Path) -> Path:
    """Create a Python venv for SearXNG if not present; return its python binary."""
    python_bin = venv_root / "bin" / "python"
    if python_bin.exists():
        return python_bin
    venv_root.mkdir(parents=True, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(str(venv_root))
    return python_bin


def _ensure_deps(python_bin: Path, source_root: Path) -> None:
    """Install SearXNG + supporting deps into the venv. Idempotent (pip skips satisfied)."""
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", "--upgrade", *_SUPPORT_DEPS],
        check=True,
    )
    subprocess.run(
        [
            str(python_bin), "-m", "pip", "install",
            "--use-pep517", "--no-build-isolation",
            "-e", str(source_root),
        ],
        check=True,
    )


def _write_settings(config_dir: Path) -> None:
    """Write settings.yml. Hardcoded to port 8888, localhost bind, limiter off."""
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = "\n".join([
        "use_default_settings: true",
        "",
        "server:",
        "  bind_address: 127.0.0.1",
        f"  port: {_PORT}",
        '  secret_key: "helix-searxng-local"',
        "  limiter: false",
        "",
        "search:",
        "  safe_search: 0",
        "  formats:",
        "    - html",
        "    - json",
        "",
    ])
    (config_dir / "settings.yml").write_text(settings, encoding="utf-8")


# ----- Runtime helpers ------------------------------------------------------ #


def _spawn(python_bin: Path, settings_path: Path, data_dir: Path) -> int:
    """Launch ``python -m searx.webapp`` as a detached subprocess; return pid."""
    data_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = open(data_dir / "searxng-stdout.log", "w", encoding="utf-8")
    stderr_log = open(data_dir / "searxng-stderr.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["SEARXNG_SETTINGS_PATH"] = str(settings_path)
    try:
        process = subprocess.Popen(
            [str(python_bin), "-m", "searx.webapp"],
            env=env,
            stdout=stdout_log,
            stderr=stderr_log,
            start_new_session=True,
        )
    finally:
        stdout_log.close()
        stderr_log.close()
    return int(process.pid)


def _wait_ready(port: int, timeout: int = _READY_TIMEOUT) -> None:
    """Poll ``/search?q=test&format=json`` until it returns 200 or timeout."""
    deadline = time.time() + max(1, timeout)
    url = f"http://127.0.0.1:{port}/search?q=test&format=json"
    last_error = "SearXNG did not become ready"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as resp:
                if 200 <= getattr(resp, "status", 0) < 300:
                    return
        except (URLError, OSError, ConnectionError) as exc:
            last_error = str(exc)
        except Exception as exc:  # defensive — unusual URL error
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(_READY_POLL)
    raise RuntimeError(f"SearXNG did not become ready: {last_error}")


def _kill_pid(pid: int) -> None:
    """SIGTERM the process, wait briefly, then SIGKILL if still alive."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    deadline = time.time() + _SHUTDOWN_GRACE_SECONDS
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(_SHUTDOWN_POLL)

    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _pid_alive(pid: int) -> bool:
    """Return True iff a process with ``pid`` exists (signal 0 probe)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but not owned by us; treat as alive.
        return True
    return True
