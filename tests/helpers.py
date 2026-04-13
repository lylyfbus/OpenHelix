"""Shared test helpers."""

from __future__ import annotations

import os
from pathlib import Path

from helix.core.state import Turn
from helix.runtime.sandbox import DockerSandboxExecutor


def sandbox_executor(payload: dict, workspace: Path) -> Turn:
    """Run a single exec payload in a throwaway Docker sandbox.

    Creates a DockerSandboxExecutor, runs one payload, and shuts down.
    For tests and direct Environment usage only.
    """
    requested_searxng = os.environ.get("SEARXNG_BASE_URL", "").strip()
    executor = DockerSandboxExecutor(
        workspace,
        searxng_base_url=requested_searxng or "https://example.com",
    )
    local_service_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("HELIX_LOCAL_MODEL_SERVICE_") and str(value).strip()
    }
    if local_service_env:
        executor.attach_local_model_service(local_service_env)
    try:
        return executor(payload, workspace)
    finally:
        executor.shutdown()
