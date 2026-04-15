"""Docker-backed sandbox executor for exec actions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from helix.core.state import Turn


_DOCKER_BUILD_ROOT = Path(__file__).resolve().parent.parent / "runtime" / "docker"
_SANDBOX_DOCKERFILE = _DOCKER_BUILD_ROOT / "exec-sandbox.Dockerfile"
_DOCKER_BUILD_TIMEOUT = 1800
_DEFAULT_EXEC_TIMEOUT = 300
_DEFAULT_MEMORY = "2g"
_DEFAULT_CPUS = "2.0"
_DEFAULT_PIDS = "256"
_DEFAULT_NETWORK = "helix-sandbox-net"
_FRAMEWORK_ENV_PREFIXES = ("HELIX_", "SEARXNG_")
_USER_ENV_PREFIX = "SANDBOX_"


def _run_docker(
    args: list[str], *, check: bool = True, timeout: int = _DOCKER_BUILD_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a docker CLI command."""
    completed = subprocess.run(
        ["docker", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or f"docker {' '.join(args)} failed")
    return completed


def _dockerize_loopback_url(url: str) -> str:
    """Translate host-loopback URLs into a Docker-reachable hostname."""
    candidate = str(url).strip()
    if not candidate:
        return candidate
    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname not in {"127.0.0.1", "0.0.0.0", "localhost", "::1"}:
        return candidate
    netloc = "host.docker.internal"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


# --------------------------------------------------------------------------- #
# Docker sandbox executor
# --------------------------------------------------------------------------- #


class DockerSandboxExecutor:
    """Callable Docker-backed executor for exec actions.

    Receives service URLs (SearXNG, local model service) from outside.
    Only responsible for: sandbox image and running exec containers.
    """

    backend_name = "docker"

    def __init__(
        self,
        workspace: Path,
        *,
        session_id: str | None = None,
        network_name: str = _DEFAULT_NETWORK,
        searxng_base_url: str = "",
        local_model_service_env: dict[str, str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.session_id = str(session_id or "session").strip() or "session"
        self.image_tag = f"helix-sandbox:{self._hash_directory(_DOCKER_BUILD_ROOT)}"
        self.network_name = network_name
        self.cache_dir = self.workspace / ".runtime" / "docker" / "cache"
        self.approval_profile = f"docker-online-rw-workspace-v1:{self.image_tag}"
        self._runtime_prepared = False
        self._searxng_base_url = searxng_base_url
        self._local_model_service_env = dict(local_model_service_env or {})

    # ----- Public interface ------------------------------------------------- #

    def status_fields(self) -> dict[str, str]:
        fields = {
            "sandbox_backend": self.backend_name,
            "sandbox_profile": self.approval_profile,
            "docker_image": self.image_tag,
            "docker_network": self.network_name,
            "docker_searxng": self._searxng_base_url,
            "docker_cache_root": str(self.cache_dir),
        }
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
        """Build sandbox image, create the shared network, and prepare cache."""
        if self._runtime_prepared:
            return
        self._ensure_image()
        self._ensure_network()
        self._ensure_cache_dir()
        self._ensure_passwd_files()
        self._ensure_ssh_sources()
        self._ensure_gitconfig()
        self._runtime_prepared = True

    def shutdown(self) -> None:
        self._runtime_prepared = False

    def __call__(self, payload: dict, workspace: Path) -> Turn:
        workspace_root = Path(workspace).expanduser().resolve()
        timeout_seconds = self._parse_timeout(payload)
        memory = self._parse_memory(payload)
        cpus = self._parse_cpus(payload)
        job_name = str(payload.get("job_name", "unnamed_job")).strip() or "unnamed_job"

        try:
            self.prepare_runtime()
            code_type, has_path, path_value, script_value, args_value = self._normalize_exec_input(payload)
            container_command = self._build_container_command(
                code_type, has_path, path_value, script_value, args_value,
            )
        except Exception as exc:
            return Turn(role="runtime", content=f"Job '{job_name}' failed to start: {exc}")

        runtime_logs = workspace_root / ".runtime" / "logs"
        runtime_logs.mkdir(parents=True, exist_ok=True)
        stdout_fd, stdout_name = tempfile.mkstemp(prefix=f"{job_name}_stdout_", suffix=".log", dir=str(runtime_logs))
        stderr_fd, stderr_name = tempfile.mkstemp(prefix=f"{job_name}_stderr_", suffix=".log", dir=str(runtime_logs))
        stdout_path = Path(stdout_name)
        stderr_path = Path(stderr_name)

        container_name = f"helix-exec-{self._workspace_slug()}-{int(time.time() * 1000)}"
        env = self._build_container_environment(workspace_root)
        docker_args = self._build_docker_run_args(
            container_name, workspace_root, env, container_command,
            memory=memory, cpus=cpus,
        )

        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                docker_args, cwd=str(workspace_root),
                stdout=stdout_file, stderr=stderr_file,
                start_new_session=True,
            )
        finally:
            stdout_file.close()
            stderr_file.close()

        result = self._wait_for_process(
            process, container_name, stdout_path, stderr_path, timeout_seconds,
        )
        return self._build_result_turn(job_name, result)

    # ----- Docker helpers --------------------------------------------------- #

    def _ensure_image(self) -> None:
        inspect = _run_docker(["image", "inspect", self.image_tag], check=False)
        if inspect.returncode == 0:
            return
        _run_docker(
            ["build", "-t", self.image_tag, "-f", str(_SANDBOX_DOCKERFILE), str(_DOCKER_BUILD_ROOT)],
            timeout=_DOCKER_BUILD_TIMEOUT,
        )

    def _ensure_network(self) -> None:
        inspect = _run_docker(["network", "inspect", self.network_name], check=False)
        if inspect.returncode == 0:
            return
        _run_docker(["network", "create", self.network_name], timeout=30)

    def _ensure_passwd_files(self) -> None:
        # OpenSSH refuses to run if getpwuid(geteuid()) returns NULL, which
        # happens in the container when --user is a UID the base image doesn't
        # know about. We generate a minimal /etc/passwd and /etc/group that map
        # the host UID/GID to a usable entry with $HOME pointing at the
        # cache-backed home directory, and bind-mount them read-only over the
        # base files.
        uid = os.getuid()
        gid = os.getgid()
        passwd_content = (
            "root:x:0:0:root:/root:/bin/bash\n"
            f"helix:x:{uid}:{gid}:helix:/helix-cache/home:/bin/bash\n"
        )
        group_content = (
            "root:x:0:\n"
            f"helix:x:{gid}:\n"
        )
        passwd_file = self.cache_dir / "passwd"
        group_file = self.cache_dir / "group"
        if not passwd_file.exists() or passwd_file.read_text(encoding="utf-8") != passwd_content:
            passwd_file.write_text(passwd_content, encoding="utf-8")
        if not group_file.exists() or group_file.read_text(encoding="utf-8") != group_content:
            group_file.write_text(group_content, encoding="utf-8")

    def _ensure_ssh_sources(self) -> None:
        # Copy the host's ~/.ssh into the cache-backed home directory so the
        # agent can speak SSH to GitHub. We can't bind-mount the host dir
        # directly because (a) nested bind mounts inside the /helix-cache mount
        # don't play well with Docker Desktop on macOS, and (b) the host's
        # ssh_config may contain macOS-only options like `UseKeychain` that
        # Linux OpenSSH rejects as fatal parse errors. The copy strips those.
        host_ssh = Path.home() / ".ssh"
        if not host_ssh.is_dir():
            return
        target = self.cache_dir / "home" / ".ssh"
        target.mkdir(parents=True, exist_ok=True)
        target.chmod(0o700)
        for item in host_ssh.iterdir():
            if not item.is_file():
                continue
            dest = target / item.name
            if item.name == "config":
                text = item.read_text(encoding="utf-8", errors="replace")
                filtered = [
                    line for line in text.splitlines()
                    if not re.match(r"\s*UseKeychain\b", line, re.IGNORECASE)
                ]
                dest.write_text("\n".join(filtered) + "\n", encoding="utf-8")
            else:
                dest.write_bytes(item.read_bytes())
            dest.chmod(item.stat().st_mode & 0o777)

    def _ensure_gitconfig(self) -> None:
        # Copy the host's ~/.gitconfig into the cache-backed home so commits
        # carry the user's identity and any configured credential helpers or
        # aliases are available inside the sandbox.
        host_gitconfig = Path.home() / ".gitconfig"
        if not host_gitconfig.is_file():
            return
        target = self.cache_dir / "home" / ".gitconfig"
        target.write_bytes(host_gitconfig.read_bytes())
        target.chmod(host_gitconfig.stat().st_mode & 0o777)

    def _ensure_cache_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("home", "pip", "npm", "npm-global", "venv"):
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    # ----- Exec input/output ------------------------------------------------ #

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
    def _parse_timeout(payload: dict) -> int:
        try:
            return int(payload.get("timeout_seconds", _DEFAULT_EXEC_TIMEOUT))
        except (TypeError, ValueError):
            return _DEFAULT_EXEC_TIMEOUT

    @staticmethod
    def _parse_memory(payload: dict) -> str:
        raw = str(payload.get("memory", "")).strip()
        return raw if raw else _DEFAULT_MEMORY

    @staticmethod
    def _parse_cpus(payload: dict) -> str:
        raw = str(payload.get("cpus", "")).strip()
        return raw if raw else _DEFAULT_CPUS

    @staticmethod
    def _build_container_command(
        code_type: str, has_path: bool, path_value: str, script_value: str, args_value: list[str],
    ) -> list[str]:
        if code_type == "python":
            return ["python", path_value, *args_value] if has_path else ["python", "-c", script_value]
        if code_type == "bash":
            return ["bash", path_value, *args_value] if has_path else ["bash", "-c", script_value]
        raise ValueError(f"Unsupported code_type: {code_type}")

    def _build_container_environment(self, workspace_root: Path) -> dict[str, str]:
        tmpdir = workspace_root / ".runtime" / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        env = {
            "HELIX_CACHE_ROOT": "/helix-cache",
            "HOME": "/helix-cache/home",
            "PIP_CACHE_DIR": "/helix-cache/pip",
            "NPM_CONFIG_CACHE": "/helix-cache/npm",
            "NPM_CONFIG_PREFIX": "/helix-cache/npm-global",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "TMPDIR": str(tmpdir),
            "TEMP": str(tmpdir),
            "TMP": str(tmpdir),
            "CHROME_BIN": "/usr/bin/chromium",
            "CHROMEDRIVER": "/usr/bin/chromedriver",
        }
        env.update(self._local_model_service_env)
        # Forward framework-managed env vars (HELIX_*, SEARXNG_*)
        for key, value in os.environ.items():
            if not value or not key.startswith(_FRAMEWORK_ENV_PREFIXES):
                continue
            if key.endswith("_BASE_URL") or key.endswith("_URL"):
                env[key] = _dockerize_loopback_url(value)
            else:
                env[key] = value
        # Forward user-defined pass-through vars (SANDBOX_X → X in container)
        for key, value in os.environ.items():
            if not value or not key.startswith(_USER_ENV_PREFIX):
                continue
            container_key = key[len(_USER_ENV_PREFIX):]
            if container_key:
                env[container_key] = value
        # SearXNG URL set last — overrides any forwarded env value
        if self._searxng_base_url:
            env["SEARXNG_BASE_URL"] = self._searxng_base_url
        return env

    def _build_docker_run_args(
        self, container_name: str, workspace_root: Path,
        env: dict[str, str], command: list[str],
        *, memory: str = _DEFAULT_MEMORY, cpus: str = _DEFAULT_CPUS,
    ) -> list[str]:
        uid_gid = f"{os.getuid()}:{os.getgid()}"
        args = [
            "docker", "run",
            "--name", container_name,
            "--rm", "--init",
            "--network", self.network_name,
            "--read-only",
            "--tmpfs", "/tmp:exec,mode=1777",
            "--tmpfs", "/run:mode=755",
            "--shm-size", "512m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", memory,
            "--cpus", cpus,
            "--pids-limit", _DEFAULT_PIDS,
            "--user", uid_gid,
            "--workdir", str(workspace_root),
            "--mount", f"type=bind,src={workspace_root},dst={workspace_root}",
            "--mount", f"type=bind,src={self.cache_dir},dst=/helix-cache",
            "--mount", f"type=bind,src={self.cache_dir / 'passwd'},dst=/etc/passwd,readonly",
            "--mount", f"type=bind,src={self.cache_dir / 'group'},dst=/etc/group,readonly",
        ]
        # Note: the host's ~/.ssh and ~/.gitconfig are copied into
        # {cache_dir}/home/ during prepare_runtime (_ensure_ssh_sources,
        # _ensure_gitconfig). Those files are visible inside the container at
        # /helix-cache/home/.ssh and /helix-cache/home/.gitconfig via the
        # existing /helix-cache bind mount — no additional mounts needed, and
        # the copy strips macOS-only options like `UseKeychain` that Linux
        # OpenSSH rejects.
        if sys.platform.startswith("linux"):
            args.extend(["--add-host", "host.docker.internal:host-gateway"])
        for key, value in sorted(env.items()):
            args.extend(["-e", f"{key}={value}"])
        args.append(self.image_tag)
        args.extend(command)
        return args

    def _wait_for_process(
        self, process: subprocess.Popen[Any], container_name: str,
        stdout_path: Path, stderr_path: Path, timeout_seconds: int,
    ) -> dict[str, Any]:
        try:
            process.wait(timeout=timeout_seconds)
            return self._collect_result(process, stdout_path, stderr_path)
        except subprocess.TimeoutExpired:
            self._kill_container(process, container_name)
            return self._collect_result(
                process, stdout_path, stderr_path,
                extra_stderr=f"\nruntime> exec terminated after {timeout_seconds}s timeout",
            )
        except KeyboardInterrupt:
            self._kill_container(process, container_name)
            return self._collect_result(
                process, stdout_path, stderr_path,
                extra_stderr="\nruntime> exec terminated by user (KeyboardInterrupt)",
            )

    @staticmethod
    def _kill_container(process: subprocess.Popen[Any], container_name: str) -> None:
        _run_docker(["rm", "-f", container_name], check=False, timeout=30)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

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

    # ----- Static utilities ------------------------------------------------- #

    @staticmethod
    def _hash_directory(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()[:12]

    def _workspace_slug(self) -> str:
        return hashlib.sha256(str(self.workspace).encode("utf-8")).hexdigest()[:10]


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
