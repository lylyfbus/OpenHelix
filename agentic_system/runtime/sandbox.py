"""Sandbox executor — subprocess isolation for exec actions.

Provides `sandbox_executor`, the pluggable executor for Environment.
Handles bash/python subprocess launching, stdout/stderr capture,
timeout enforcement, and graceful termination.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agentic_system.core.state import Turn


# --------------------------------------------------------------------------- #
# ExecJob handle
# --------------------------------------------------------------------------- #


@dataclass
class ExecJob:
    """Runtime handle for one launched exec process."""

    job_name: str
    process: subprocess.Popen[Any]
    cwd: Path
    stdout_path: Path
    stderr_path: Path
    started_at: float


# --------------------------------------------------------------------------- #
# Input normalization & command building
# --------------------------------------------------------------------------- #


def _normalize_exec_input(
    action_input: dict[str, object],
) -> tuple[str, bool, str, str, list[str]]:
    """Validate and normalize exec action_input into command-building primitives."""
    if not isinstance(action_input, dict):
        raise ValueError("exec action requires dict action_input")

    code_type = str(action_input.get("code_type", "bash")).strip().lower()
    script_path = str(action_input.get("script_path", "")).strip()
    script = str(action_input.get("script", "")).strip()

    # Normalize script_args
    raw_args = action_input.get("script_args", [])
    if isinstance(raw_args, (list, tuple)):
        script_args = [str(a) for a in raw_args if str(a).strip()]
    elif isinstance(raw_args, str) and raw_args.strip():
        try:
            script_args = [a for a in shlex.split(raw_args.strip()) if a.strip()]
        except ValueError:
            script_args = [raw_args.strip()]
    else:
        script_args = []

    has_path = bool(script_path)
    has_script = bool(script)
    if has_path == has_script:
        raise ValueError("Exactly one of script_path or script must be provided")
    if has_script and script_args:
        raise ValueError("script_args is only supported with script_path")

    return code_type, has_path, script_path, script, script_args


def _build_command(
    code_type: str,
    has_path: bool,
    path_value: str,
    script_value: str,
    args_value: list[str],
) -> list[str]:
    """Build subprocess argv for python/bash execution."""
    if code_type == "python":
        if has_path:
            return [sys.executable, path_value, *args_value]
        return [sys.executable, "-c", script_value]
    if code_type == "bash":
        if has_path:
            return ["bash", path_value, *args_value]
        # Use login shell (-l) so PATH modifications from .bash_profile are available
        return ["bash", "-lc", script_value]
    raise ValueError(f"Unsupported code_type: {code_type}")


def _build_exec_environment(workspace_root: Path) -> dict[str, str]:
    """Create child env with runtime-local tmp directories inside workspace."""
    env = dict(os.environ)
    runtime_tmp = workspace_root / ".runtime" / "tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(runtime_tmp)
    env["TEMP"] = str(runtime_tmp)
    env["TMP"] = str(runtime_tmp)
    return env


# --------------------------------------------------------------------------- #
# Job lifecycle
# --------------------------------------------------------------------------- #


def _start_job(
    action_input: dict[str, object],
    workspace: Path,
    job_name: str = "unnamed",
) -> ExecJob:
    """Launch an exec job, capturing stdout/stderr to per-job log files."""
    workspace_root = Path(workspace).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    code_type, has_path, path_value, script_value, args_value = (
        _normalize_exec_input(action_input)
    )
    command = _build_command(code_type, has_path, path_value, script_value, args_value)

    runtime_logs = workspace_root / ".runtime" / "logs"
    runtime_logs.mkdir(parents=True, exist_ok=True)
    stdout_fd, stdout_name = tempfile.mkstemp(
        prefix=f"{job_name}_stdout_", suffix=".log", dir=str(runtime_logs)
    )
    stderr_fd, stderr_name = tempfile.mkstemp(
        prefix=f"{job_name}_stderr_", suffix=".log", dir=str(runtime_logs)
    )

    env = _build_exec_environment(workspace_root)
    stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
    stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(workspace_root),
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
            env=env,
        )
    finally:
        stdout_file.close()
        stderr_file.close()

    return ExecJob(
        job_name=job_name,
        process=process,
        cwd=workspace_root,
        stdout_path=Path(stdout_name),
        stderr_path=Path(stderr_name),
        started_at=time.time(),
    )


def _terminate_job(
    job: ExecJob,
    *,
    sigint_wait: float = 1.5,
    sigterm_wait: float = 1.5,
) -> None:
    """Terminate a running job with escalating signals: INT → TERM → KILL."""
    if job.process.poll() is not None:
        return

    def _send(sig: int) -> None:
        if job.process.poll() is not None:
            return
        try:
            os.killpg(job.process.pid, sig)
        except ProcessLookupError:
            pass

    try:
        _send(signal.SIGINT)
        job.process.wait(timeout=max(0.1, sigint_wait))
    except subprocess.TimeoutExpired:
        try:
            _send(signal.SIGTERM)
            job.process.wait(timeout=max(0.1, sigterm_wait))
        except subprocess.TimeoutExpired:
            _send(signal.SIGKILL)
            job.process.wait(timeout=1.0)
    except ProcessLookupError:
        pass


def _collect_result(
    job: ExecJob,
    extra_stderr: str = "",
) -> dict[str, Any]:
    """Collect completed job stdout/stderr and clean up log files."""
    if job.process.poll() is None:
        job.process.wait()

    stdout = ""
    stderr = ""
    if job.stdout_path.exists():
        stdout = job.stdout_path.read_text(encoding="utf-8", errors="replace")
        job.stdout_path.unlink(missing_ok=True)
    if job.stderr_path.exists():
        stderr = job.stderr_path.read_text(encoding="utf-8", errors="replace")
        job.stderr_path.unlink(missing_ok=True)

    if extra_stderr:
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += extra_stderr.strip() + "\n"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": int(job.process.returncode or 0),
    }


def _scalar_text(value: Any) -> str:
    """Render a scalar value in a concise, readable form."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _indent_block(text: str, prefix: str) -> str:
    """Indent each line in a text block with the given prefix."""
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _format_structured_value(value: Any, indent: int = 0) -> str:
    """Format JSON-like data into a readable YAML-like block."""
    prefix = "  " * indent

    if isinstance(value, dict):
        if not value:
            return f"{prefix}{{}}"
        lines: list[str] = []
        for key, item in value.items():
            key_text = f"{prefix}{key}:"
            if isinstance(item, str):
                if "\n" in item:
                    lines.append(f"{key_text} |")
                    lines.append(_indent_block(item, prefix + "  "))
                else:
                    lines.append(f"{key_text} {item}")
            elif isinstance(item, (dict, list)):
                lines.append(key_text)
                lines.append(_format_structured_value(item, indent + 1))
            else:
                lines.append(f"{key_text} {_scalar_text(item)}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = []
        for item in value:
            item_prefix = f"{prefix}-"
            if isinstance(item, str):
                if "\n" in item:
                    lines.append(f"{item_prefix} |")
                    lines.append(_indent_block(item, prefix + "  "))
                else:
                    lines.append(f"{item_prefix} {item}")
            elif isinstance(item, (dict, list)):
                lines.append(item_prefix)
                lines.append(_format_structured_value(item, indent + 1))
            else:
                lines.append(f"{item_prefix} {_scalar_text(item)}")
        return "\n".join(lines)

    if isinstance(value, str):
        if "\n" in value:
            return "\n".join([
                f"{prefix}|",
                _indent_block(value, prefix + "  "),
            ])
        return f"{prefix}{value}"

    return f"{prefix}{_scalar_text(value)}"


def _format_output_block(name: str, text: str) -> str:
    """Wrap stdout/stderr in readable tags, prettifying JSON when possible."""
    cleaned = text.rstrip()
    if not cleaned:
        return ""

    rendered = cleaned
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        rendered = _format_structured_value(parsed)

    return f"\n\n<{name}>\n{rendered}\n</{name}>"


# --------------------------------------------------------------------------- #
# Public API — the Environment executor
# --------------------------------------------------------------------------- #


def sandbox_executor(payload: dict, workspace: Path) -> Turn:
    """Execute an exec action in the sandbox.

    This is the pluggable executor function that matches the
    ``SandboxExecutor`` signature: ``(payload, workspace) -> Turn``.
    """
    timeout_str = os.environ.get("AGENTIC_SANDBOX_TIMEOUT", "600")
    try:
        default_timeout = int(timeout_str)
    except ValueError:
        default_timeout = 600

    timeout = payload.get("timeout_seconds", default_timeout)
    job_name = str(payload.get("job_name", "unnamed_job")).strip() or "unnamed_job"

    try:
        job = _start_job(
            action_input=payload,
            workspace=workspace,
            job_name=job_name,
        )
    except Exception as e:
        return Turn(
            role="runtime",
            content=f"Job '{job_name}' failed to start: {e}",
        )

    try:
        job.process.wait(timeout=timeout)
        result = _collect_result(job)
    except subprocess.TimeoutExpired:
        _terminate_job(job)
        result = _collect_result(
            job, extra_stderr=f"\nruntime> exec terminated after {timeout}s timeout"
        )
    except KeyboardInterrupt:
        print("\nruntime> Execution interrupted by user (`Ctrl+C`).")
        _terminate_job(job)
        result = _collect_result(
            job, extra_stderr="\nruntime> exec terminated by user (KeyboardInterrupt)"
        )

    # Format result for LLM consumption
    stdout = result["stdout"]
    stderr = result["stderr"]
    rc = result["return_code"]

    status = "succeeded" if rc == 0 else "failed"
    content = f"Job '{job_name}' {status}. (Exit code: {rc})"
    if stdout:
        content += _format_output_block("stdout", stdout)
    if stderr:
        content += _format_output_block("stderr", stderr)

    return Turn(
        role="runtime",
        content=content,
    )
