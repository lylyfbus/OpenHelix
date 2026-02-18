from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def execute(
    *,
    code_type: str,
    script_path: str | None = None,
    script: str | None = None,
    workspace: str | Path,
    timeout_seconds: int = 60,
) -> dict[str, str]:
    cwd = Path(workspace).expanduser().resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    normalized_code_type = str(code_type).strip().lower()
    path_value = str(script_path or "").strip()
    script_value = str(script or "").strip()

    has_path = bool(path_value)
    has_script = bool(script_value)
    if has_path == has_script:
        raise ValueError("Exactly one of script_path or script must be provided")

    if normalized_code_type == "python":
        if has_path:
            result = subprocess.run(
                [sys.executable, path_value],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        else:
            result = subprocess.run(
                [sys.executable, "-c", script_value],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
    elif normalized_code_type == "bash":
        if has_path:
            result = subprocess.run(
                ["bash", path_value],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        else:
            result = subprocess.run(
                script_value,
                cwd=str(cwd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
    else:
        raise ValueError(f"Unsupported code_type: {code_type}")

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
