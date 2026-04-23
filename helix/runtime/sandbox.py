"""Host-shell-backed sandbox executor for exec actions.

Runs bash and python scripts directly on the host machine, inheriting the
user's shell environment. Safety is provided by the ApprovalPolicy hook on
the Environment (see ``helix.runtime.approval``), not by process or
filesystem isolation — the approval prompt is the line of defense against
destructive or outside-workspace operations.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from helix.core.state import Turn


_DEFAULT_EXEC_TIMEOUT = 300


# --------------------------------------------------------------------------- #
# Host sandbox executor
# --------------------------------------------------------------------------- #


class HostSandboxExecutor:
    """Callable host-shell executor for exec actions.

    Executes bash and python scripts directly on the host with ``cwd`` set
    to the workspace. All safety guarantees come from the approval layer;
    this executor deliberately trusts what it is given.
    """

    backend_name = "host"

    def __init__(
        self,
        workspace: Path,
        *,
        session_id: str | None = None,
        searxng_base_url: str = "",
        local_model_service_env: dict[str, str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.session_id = str(session_id or "session").strip() or "session"
        self.approval_profile = "host-shell-v1"
        self._searxng_base_url = searxng_base_url
        self._local_model_service_env = dict(local_model_service_env or {})

    # ----- Public interface ------------------------------------------------- #

    def status_fields(self) -> dict[str, str]:
        fields = {
            "sandbox_backend": self.backend_name,
            "sandbox_profile": self.approval_profile,
            "host_python": sys.executable,
        }
        if self._searxng_base_url:
            fields["searxng"] = self._searxng_base_url
        if self._local_model_service_env.get("HELIX_LOCAL_MODEL_SERVICE_URL"):
            fields["local_model_service"] = self._local_model_service_env["HELIX_LOCAL_MODEL_SERVICE_URL"]
        return fields

    def tool_environment(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self._searxng_base_url:
            env["SEARXNG_BASE_URL"] = self._searxng_base_url
        env.update(self._local_model_service_env)
        return env

    def prepare_runtime(self) -> None:
        """No-op: the host shell is always ready."""
        return None

    def shutdown(self) -> None:
        """No-op: host processes are cleaned up per-exec."""
        return None

    def __call__(self, payload: dict, workspace: Path) -> Turn:
        workspace_root = Path(workspace).expanduser().resolve()
        timeout_seconds = self._parse_timeout(payload)
        job_name = str(payload.get("job_name", "unnamed_job")).strip() or "unnamed_job"

        try:
            code_type, has_path, path_value, script_value, args_value = self._normalize_exec_input(payload)
            command = self._build_command(code_type, has_path, path_value, script_value, args_value)
        except Exception as exc:
            return Turn(role="runtime", content=f"Job '{job_name}' failed to start: {exc}")

        runtime_logs = workspace_root / ".runtime" / "logs"
        runtime_logs.mkdir(parents=True, exist_ok=True)
        stdout_fd, stdout_name = tempfile.mkstemp(prefix=f"{job_name}_stdout_", suffix=".log", dir=str(runtime_logs))
        stderr_fd, stderr_name = tempfile.mkstemp(prefix=f"{job_name}_stderr_", suffix=".log", dir=str(runtime_logs))
        stdout_path = Path(stdout_name)
        stderr_path = Path(stderr_name)

        env = self._build_environment()

        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command, cwd=str(workspace_root), env=env,
                stdout=stdout_file, stderr=stderr_file,
                start_new_session=True,
            )
        finally:
            stdout_file.close()
            stderr_file.close()

        result = self._wait_for_process(process, stdout_path, stderr_path, timeout_seconds)
        return self._build_result_turn(job_name, result)

    # ----- Input / command construction ------------------------------------ #

    @staticmethod
    def _normalize_exec_input(action_input: dict[str, object]) -> tuple[str, bool, str, str, list[str]]:
        if not isinstance(action_input, dict):
            raise ValueError("exec action requires dict action_input")

        code_type = str(action_input.get("code_type", "bash")).strip().lower()
        script_path = str(action_input.get("script_path", "")).strip()
        script = str(action_input.get("script", "")).strip()

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

    @staticmethod
    def _build_command(
        code_type: str, has_path: bool, path_value: str, script_value: str, args_value: list[str],
    ) -> list[str]:
        if code_type == "python":
            interpreter = sys.executable or "python"
            return [interpreter, path_value, *args_value] if has_path else [interpreter, "-c", script_value]
        if code_type == "bash":
            return ["bash", path_value, *args_value] if has_path else ["bash", "-c", script_value]
        raise ValueError(f"Unsupported code_type: {code_type}")

    @staticmethod
    def _parse_timeout(payload: dict) -> int:
        try:
            return int(payload.get("timeout_seconds", _DEFAULT_EXEC_TIMEOUT))
        except (TypeError, ValueError):
            return _DEFAULT_EXEC_TIMEOUT

    def _build_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._searxng_base_url:
            env["SEARXNG_BASE_URL"] = self._searxng_base_url
        env.update(self._local_model_service_env)
        return env

    # ----- Process lifecycle ----------------------------------------------- #

    def _wait_for_process(
        self, process: subprocess.Popen[Any],
        stdout_path: Path, stderr_path: Path, timeout_seconds: int,
    ) -> dict[str, Any]:
        try:
            process.wait(timeout=timeout_seconds)
            return self._collect_result(process, stdout_path, stderr_path)
        except subprocess.TimeoutExpired:
            self._kill_process(process)
            return self._collect_result(
                process, stdout_path, stderr_path,
                extra_stderr=f"\nruntime> exec terminated after {timeout_seconds}s timeout",
            )
        except KeyboardInterrupt:
            self._kill_process(process)
            return self._collect_result(
                process, stdout_path, stderr_path,
                extra_stderr="\nruntime> exec terminated by user (KeyboardInterrupt)",
            )

    @staticmethod
    def _kill_process(process: subprocess.Popen[Any]) -> None:
        # start_new_session=True made the child a process-group leader, so
        # killing the group reliably tears down any subprocesses it spawned.
        try:
            os.killpg(os.getpgid(process.pid), 15)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    @staticmethod
    def _collect_result(
        process: subprocess.Popen[Any],
        stdout_path: Path, stderr_path: Path,
        extra_stderr: str = "",
    ) -> dict[str, Any]:
        if process.poll() is None:
            process.wait()
        stdout = ""
        stderr = ""
        if stdout_path.exists():
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            stdout_path.unlink(missing_ok=True)
        if stderr_path.exists():
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            stderr_path.unlink(missing_ok=True)
        if extra_stderr:
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += extra_stderr.strip() + "\n"
        return {"stdout": stdout, "stderr": stderr, "return_code": int(process.returncode or 0)}

    @staticmethod
    def _build_result_turn(job_name: str, result: dict[str, Any]) -> Turn:
        rc = result["return_code"]
        status = "succeeded" if rc == 0 else "failed"
        content = f"Job '{job_name}' {status}. (Exit code: {rc})"
        stdout = result["stdout"].rstrip()
        stderr = result["stderr"].rstrip()
        if stdout:
            content += f"\n\n<stdout>\n{_format_output(stdout)}\n</stdout>"
        if stderr:
            content += f"\n\n<stderr>\n{_format_output(stderr)}\n</stderr>"
        return Turn(role="runtime", content=content)


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #


def _format_output(text: str) -> str:
    """Format stdout/stderr, prettifying JSON when possible."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return _format_structured(parsed)


def _format_structured(value: Any, indent: int = 0) -> str:
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
                    lines.append("\n".join(f"{prefix}  {ln}" for ln in item.splitlines()))
                else:
                    lines.append(f"{key_text} {item}")
            elif isinstance(item, (dict, list)):
                lines.append(key_text)
                lines.append(_format_structured(item, indent + 1))
            else:
                lines.append(f"{key_text} {_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = []
        for item in value:
            if isinstance(item, str):
                if "\n" in item:
                    lines.append(f"{prefix}- |")
                    lines.append("\n".join(f"{prefix}  {ln}" for ln in item.splitlines()))
                else:
                    lines.append(f"{prefix}- {item}")
            elif isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_format_structured(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, str):
        if "\n" in value:
            return f"{prefix}|\n" + "\n".join(f"{prefix}  {ln}" for ln in value.splitlines())
        return f"{prefix}{value}"
    return f"{prefix}{_scalar(value)}"


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
