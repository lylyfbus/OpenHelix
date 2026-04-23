"""Shared test helpers."""

from __future__ import annotations

import os
from pathlib import Path

from helix.core.state import Turn
from helix.runtime.sandbox import HostSandboxExecutor


def sandbox_executor(payload: dict, workspace: Path) -> Turn:
    """Run a single exec payload in a throwaway host-shell sandbox.

    Creates a HostSandboxExecutor, runs one payload, and shuts down.
    For tests and direct Environment usage only.
    """
    requested_searxng = os.environ.get("SEARXNG_BASE_URL", "").strip()
    local_service_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("HELIX_LOCAL_MODEL_SERVICE_") and str(value).strip()
    }
    executor = HostSandboxExecutor(
        workspace,
        searxng_base_url=requested_searxng or "https://example.com",
        local_model_service_env=local_service_env,
    )
    try:
        return executor(payload, workspace)
    finally:
        executor.shutdown()
