"""SearXNG search service management.

Usage:
    helix start searxng    Start the SearXNG container
    helix stop searxng     Stop the SearXNG container
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from helix.constants import SERVICES_ROOT

_SEARXNG_IMAGE = "docker.io/searxng/searxng:latest"
_NETWORK_NAME = "helix-sandbox-net"
_CONTAINER_NAME = "helix-searxng"
_BASE_URL = f"http://{_CONTAINER_NAME}:8080"
_BUILD_TIMEOUT = 1800
_READY_TIMEOUT = 30
_READY_POLL = 1.0
_SERVICE_DIR = SERVICES_ROOT / "searxng"
_STATE_PATH = _SERVICE_DIR / "state.json"
_CONFIG_DIR = _SERVICE_DIR / "config"
_DATA_DIR = _SERVICE_DIR / "data"


def start() -> dict[str, str]:
    """Start the SearXNG container and write state.json.

    Returns the service state dict.
    """
    _ensure_network()
    _ensure_image()
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_settings()

    # Check if already running
    inspect = _run_docker(
        ["inspect", "-f", "{{.State.Running}}", _CONTAINER_NAME],
        check=False,
    )
    if inspect.returncode == 0 and inspect.stdout.strip() == "true":
        _wait_ready()
        return _write_state()
    if inspect.returncode == 0:
        _run_docker(["rm", "-f", _CONTAINER_NAME], check=False)

    _run_docker(
        [
            "run", "-d",
            "--name", _CONTAINER_NAME,
            "--restart", "unless-stopped",
            "--network", _NETWORK_NAME,
            "-v", f"{_CONFIG_DIR}:/etc/searxng",
            "-v", f"{_DATA_DIR}:/var/cache/searxng",
            _SEARXNG_IMAGE,
        ],
        timeout=_BUILD_TIMEOUT,
    )
    _wait_ready()
    return _write_state()


def stop() -> None:
    """Stop the SearXNG container and remove state.json."""
    _run_docker(["rm", "-f", _CONTAINER_NAME], check=False, timeout=30)
    _STATE_PATH.unlink(missing_ok=True)


def discover() -> dict[str, Any] | None:
    """Check if SearXNG is running. Returns state dict or None."""
    if not _STATE_PATH.exists():
        return None
    try:
        state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Verify container is actually running
    inspect = _run_docker(
        ["inspect", "-f", "{{.State.Running}}", state.get("container_name", "")],
        check=False,
    )
    if inspect.returncode == 0 and inspect.stdout.strip() == "true":
        return state
    # Stale state
    _STATE_PATH.unlink(missing_ok=True)
    return None


# ----- Internal --------------------------------------------------------- #


def _write_state() -> dict[str, Any]:
    state = {
        "container_name": _CONTAINER_NAME,
        "network_name": _NETWORK_NAME,
        "base_url": _BASE_URL,
        "started_at": time.time(),
    }
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _run_docker(
    args: list[str], *, check: bool = True, timeout: int = _BUILD_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["docker", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or f"docker {' '.join(args)} failed")
    return completed


def _ensure_network() -> None:
    inspect = _run_docker(["network", "inspect", _NETWORK_NAME], check=False)
    if inspect.returncode == 0:
        return
    created = _run_docker(["network", "create", _NETWORK_NAME], check=False, timeout=30)
    if created.returncode == 0:
        return
    detail = (created.stderr or created.stdout or "").strip().lower()
    if "fully subnetted" in detail:
        _cleanup_unused_networks()
        created = _run_docker(["network", "create", _NETWORK_NAME], check=False, timeout=30)
        if created.returncode == 0:
            return
    detail = (created.stderr or created.stdout or "").strip()
    raise RuntimeError(detail or f"docker network create {_NETWORK_NAME} failed")


def _cleanup_unused_networks() -> None:
    listed = _run_docker(["network", "ls", "--format", "{{.Name}}"], check=False, timeout=30)
    if listed.returncode != 0:
        return
    for raw_name in listed.stdout.splitlines():
        name = raw_name.strip()
        if not name.startswith(_NETWORK_NAME) or name == _NETWORK_NAME:
            continue
        inspect = _run_docker(["network", "inspect", name], check=False, timeout=30)
        if inspect.returncode != 0:
            continue
        try:
            payload = json.loads(inspect.stdout)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list) or not payload:
            continue
        network = payload[0] if isinstance(payload[0], dict) else {}
        containers = network.get("Containers") or {}
        if isinstance(containers, dict) and containers:
            continue
        _run_docker(["network", "rm", name], check=False, timeout=30)


def _ensure_image() -> None:
    inspect = _run_docker(["image", "inspect", _SEARXNG_IMAGE], check=False)
    if inspect.returncode == 0:
        return
    _run_docker(["pull", _SEARXNG_IMAGE], timeout=_BUILD_TIMEOUT)


def _write_settings() -> None:
    settings = "\n".join([
        "use_default_settings: true",
        "",
        "server:",
        '  secret_key: "helix-docker-sandbox"',
        "  limiter: false",
        "",
        "search:",
        "  safe_search: 0",
        "  formats:",
        "    - html",
        "    - json",
        "",
    ])
    (_CONFIG_DIR / "settings.yml").write_text(settings, encoding="utf-8")


def _wait_ready() -> None:
    deadline = time.time() + max(1, _READY_TIMEOUT)
    last_error = "searxng readiness probe did not return success"
    while time.time() < deadline:
        completed = _run_docker(
            ["exec", _CONTAINER_NAME,
             "python3", "-c",
             "from urllib.request import urlopen; urlopen('http://localhost:8080/search?q=test&format=json', timeout=5).read(64); print('ready')"],
            check=False, timeout=15,
        )
        if completed.returncode == 0:
            return
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            last_error = detail
        time.sleep(max(0.1, _READY_POLL))
    raise RuntimeError(f"SearXNG did not become ready: {last_error}")
