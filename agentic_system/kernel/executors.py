from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecJob:
    job_id: str
    process: subprocess.Popen[Any]
    cwd: Path
    stdout_path: Path
    stderr_path: Path
    started_at: float


def _normalize_exec_input(action_input: dict[str, object]) -> tuple[str, bool, str, str, list[str]]:
    if not isinstance(action_input, dict):
        raise ValueError("exec action requires object action_input")

    code_type = str(action_input.get("code_type", "bash")).strip().lower()
    script_path = str(action_input.get("script_path", "")).strip()
    script = str(action_input.get("script", "")).strip()
    raw_script_args = action_input.get("script_args", [])
    if isinstance(raw_script_args, (list, tuple)):
        script_args = [str(arg) for arg in raw_script_args if str(arg).strip()]
    elif isinstance(raw_script_args, str):
        raw_args_text = raw_script_args.strip()
        if raw_args_text:
            try:
                script_args = [arg for arg in shlex.split(raw_args_text) if arg.strip()]
            except ValueError:
                script_args = [raw_args_text]
        else:
            script_args = []
    else:
        script_args = []

    normalized_code_type = str(code_type).strip().lower()
    path_value = str(script_path or "").strip()
    script_value = str(script or "").strip()
    args_value = [str(arg) for arg in (script_args or []) if str(arg).strip()]

    has_path = bool(path_value)
    has_script = bool(script_value)
    if has_path == has_script:
        raise ValueError("Exactly one of script_path or script must be provided")
    if has_script and args_value:
        raise ValueError("script_args is only supported when script_path is provided")
    return normalized_code_type, has_path, path_value, script_value, args_value


def _build_exec_command(
    *,
    normalized_code_type: str,
    has_path: bool,
    path_value: str,
    script_value: str,
    args_value: list[str],
) -> list[str]:
    if normalized_code_type == "python":
        if has_path:
            return [sys.executable, path_value, *args_value]
        return [sys.executable, "-c", script_value]
    if normalized_code_type == "bash":
        if has_path:
            return ["bash", path_value, *args_value]
        return ["bash", "-lc", script_value]
    raise ValueError(f"Unsupported code_type: {normalized_code_type}")


def start_exec_job(
    *,
    action_input: dict[str, object],
    workspace: str | Path,
    job_id: str,
) -> ExecJob:
    cwd = Path(workspace).expanduser().resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    normalized_code_type, has_path, path_value, script_value, args_value = _normalize_exec_input(action_input)
    command = _build_exec_command(
        normalized_code_type=normalized_code_type,
        has_path=has_path,
        path_value=path_value,
        script_value=script_value,
        args_value=args_value,
    )

    stdout_fd, stdout_name = tempfile.mkstemp(prefix=f"{job_id}_stdout_", suffix=".log")
    stderr_fd, stderr_name = tempfile.mkstemp(prefix=f"{job_id}_stderr_", suffix=".log")
    stdout_path = Path(stdout_name)
    stderr_path = Path(stderr_name)

    stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
    stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
    finally:
        stdout_file.close()
        stderr_file.close()

    return ExecJob(
        job_id=job_id,
        process=process,
        cwd=cwd,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=time.time(),
    )


def terminate_exec_job(
    job: ExecJob,
    *,
    reason: str,
    sigint_wait_seconds: float = 1.5,
    sigterm_wait_seconds: float = 1.5,
) -> dict[str, Any]:
    if job.process.poll() is not None:
        return {"reason": reason, "signals": []}

    signals_sent: list[str] = []

    def _send(group_signal: int, name: str) -> None:
        if job.process.poll() is not None:
            return
        os.killpg(job.process.pid, group_signal)
        signals_sent.append(name)

    try:
        _send(signal.SIGINT, "SIGINT")
        job.process.wait(timeout=max(0.1, float(sigint_wait_seconds)))
    except subprocess.TimeoutExpired:
        try:
            _send(signal.SIGTERM, "SIGTERM")
            job.process.wait(timeout=max(0.1, float(sigterm_wait_seconds)))
        except subprocess.TimeoutExpired:
            _send(signal.SIGKILL, "SIGKILL")
            job.process.wait(timeout=1.0)
    except ProcessLookupError:
        pass

    return {
        "reason": reason,
        "signals": signals_sent,
    }


def collect_exec_job_result(
    job: ExecJob,
    *,
    stderr_append: str = "",
) -> dict[str, Any]:
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

    note = str(stderr_append or "").strip()
    if note:
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += note + "\n"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": int(job.process.returncode or 0),
    }


def execute(
    *,
    action_input: dict[str, object],
    workspace: str | Path,
    timeout_seconds: int | None = None,
) -> dict[str, str]:
    job = start_exec_job(
        action_input=action_input,
        workspace=workspace,
        job_id="exec_job_sync",
    )
    try:
        if timeout_seconds is not None:
            timeout_value = max(1, int(timeout_seconds))
            job.process.wait(timeout=timeout_value)
        else:
            job.process.wait()
        result = collect_exec_job_result(job)
        return {
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
        }
    except subprocess.TimeoutExpired:
        terminate_exec_job(job, reason="timeout")
        result = collect_exec_job_result(
            job,
            stderr_append="[runtime] exec terminated due to timeout",
        )
        return {
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
        }
