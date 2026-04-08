"""Docker-backed sandbox executor for exec actions.

Provides the shared exec input/output helpers plus the runtime-managed
Docker sandbox used by the application.
"""

from __future__ import annotations

import hashlib
import json
import os
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
_SEARXNG_IMAGE = "docker.io/searxng/searxng:latest"
_DOCKER_INFO_TIMEOUT = 5
_DOCKER_BUILD_TIMEOUT = int(os.environ.get("AGENTIC_DOCKER_BUILD_TIMEOUT", "1800"))
_DOCKER_TIMEOUT = int(os.environ.get("AGENTIC_DOCKER_SANDBOX_TIMEOUT", "1800"))
_DOCKER_MEMORY = os.environ.get("AGENTIC_DOCKER_SANDBOX_MEMORY", "2g")
_DOCKER_CPUS = os.environ.get("AGENTIC_DOCKER_SANDBOX_CPUS", "2.0")
_DOCKER_PIDS = os.environ.get("AGENTIC_DOCKER_SANDBOX_PIDS", "256")
_SEARXNG_READY_TIMEOUT = int(os.environ.get("AGENTIC_DOCKER_SEARXNG_READY_TIMEOUT", "30"))
_SEARXNG_READY_POLL = float(os.environ.get("AGENTIC_DOCKER_SEARXNG_READY_POLL", "1.0"))
_GLOBAL_NETWORK_NAME = "helix-sandbox-net"
_GLOBAL_SEARXNG_NAME = "helix-searxng"
_NETWORK_NAME_PREFIX = "helix-sandbox-net"
_PASS_ENV_PREFIXES = (
    "HELIX_LOCAL_MODEL_SERVICE_",
    "SEARXNG_",
    "OLLAMA_",
    "DEEPSEEK_",
    "LMSTUDIO_",
    "LM_API_",
    "ZAI_",
    "OPENAI_COMPAT_",
    "OPENAI_API_KEY",
)


def _helix_home() -> Path:
    override = str(os.environ.get("HELIX_HOME", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".helix").resolve()


def _runtime_root() -> Path:
    return _helix_home() / "runtime"


def _service_runtime_dir(service_name: str) -> Path:
    return _runtime_root() / "services" / service_name


def _active_runtime_dir() -> Path:
    return _runtime_root() / "active-runtimes"


def _runtime_marker_path(pid: int | None = None) -> Path:
    token = int(pid or os.getpid())
    return _active_runtime_dir() / f"{token}.json"


def _prune_stale_runtime_markers() -> None:
    markers = _active_runtime_dir()
    if not markers.exists():
        return
    for marker in markers.glob("*.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker.unlink(missing_ok=True)
            continue
        try:
            pid = int(payload.get("pid"))
        except (TypeError, ValueError):
            marker.unlink(missing_ok=True)
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            marker.unlink(missing_ok=True)
        except PermissionError:
            continue


def _register_active_runtime(*, workspace: Path, session_id: str) -> Path:
    markers = _active_runtime_dir()
    markers.mkdir(parents=True, exist_ok=True)
    _prune_stale_runtime_markers()
    marker = _runtime_marker_path()
    payload = {
        "pid": os.getpid(),
        "workspace": str(Path(workspace).expanduser().resolve()),
        "session_id": str(session_id or "session").strip() or "session",
        "started_at": time.time(),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def _unregister_active_runtime(marker_path: Path | None) -> None:
    if marker_path is not None:
        marker_path.unlink(missing_ok=True)


def _has_active_runtimes() -> bool:
    _prune_stale_runtime_markers()
    markers = _active_runtime_dir()
    if not markers.exists():
        return False
    return any(markers.glob("*.json"))


def docker_is_available() -> tuple[bool, str]:
    """Return whether Docker is usable from this runtime."""
    try:
        completed = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_DOCKER_INFO_TIMEOUT,
            check=False,
        )
    except FileNotFoundError:
        return False, "docker CLI not found"
    except subprocess.TimeoutExpired:
        return False, "docker info timed out"

    if completed.returncode == 0:
        return True, ""
    detail = (completed.stderr or "").strip() or "docker info failed"
    return False, detail


def _hash_directory(root: Path) -> str:
    """Hash all files under a directory to derive a content-addressed image tag."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _workspace_slug(workspace: Path) -> str:
    return hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:10]


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


def _write_searxng_settings(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
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
    (config_dir / "settings.yml").write_text(settings, encoding="utf-8")


def _normalize_exec_input(
    action_input: dict[str, object],
) -> tuple[str, bool, str, str, list[str]]:
    """Validate and normalize exec action_input into command-building primitives."""
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


def _collect_logged_result(
    *,
    process: subprocess.Popen[Any],
    stdout_path: Path,
    stderr_path: Path,
    extra_stderr: str = "",
) -> dict[str, Any]:
    """Collect stdout/stderr from temp log files and remove them."""
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

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": int(process.returncode or 0),
    }


class DockerSandboxExecutor:
    """Callable Docker-backed executor matching ``SandboxExecutor``."""

    backend_name = "docker"

    def __init__(
        self,
        workspace: Path,
        *,
        session_id: str | None = None,
        searxng_base_url: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.slug = _workspace_slug(self.workspace)
        self.session_id = str(session_id or "session").strip() or "session"
        self.image_tag = f"helix-sandbox:{_hash_directory(_DOCKER_BUILD_ROOT)}"
        self.network_name = _GLOBAL_NETWORK_NAME
        self.cache_dir = self.workspace / ".runtime" / "docker" / "cache"
        self.searxng_name = _GLOBAL_SEARXNG_NAME
        self.searxng_runtime_dir = _service_runtime_dir("searxng")
        self.searxng_config_dir = self.searxng_runtime_dir / "config"
        self.searxng_data_dir = self.searxng_runtime_dir / "data"
        requested = str(searxng_base_url or "").strip()
        self._managed_searxng = not requested
        self._effective_searxng_base_url = (
            f"http://{self.searxng_name}:8080"
            if self._managed_searxng
            else _dockerize_loopback_url(requested)
        )
        self.approval_profile = f"docker-online-rw-workspace-v1:{self.image_tag}"
        self._session_registered = False
        self._runtime_prepared = False
        self._local_model_service_env: dict[str, str] = {}
        self._runtime_marker_path: Path | None = None

    def status_fields(self) -> dict[str, str]:
        fields = {
            "sandbox_backend": self.backend_name,
            "sandbox_profile": self.approval_profile,
            "docker_image": self.image_tag,
            "docker_network": self.network_name,
            "docker_searxng": self._effective_searxng_base_url,
            "docker_cache_root": str(self.cache_dir),
        }
        if self._local_model_service_env.get("HELIX_LOCAL_MODEL_SERVICE_URL"):
            fields["local_model_service"] = self._local_model_service_env["HELIX_LOCAL_MODEL_SERVICE_URL"]
        return fields

    def tool_environment(self) -> dict[str, str]:
        env = {"SEARXNG_BASE_URL": self._effective_searxng_base_url}
        env.update(self._local_model_service_env)
        return env

    def attach_local_model_service(self, env: dict[str, str]) -> None:
        cleaned: dict[str, str] = {}
        for key, value in env.items():
            token = str(value or "").strip()
            if token:
                cleaned[str(key)] = token
        self._local_model_service_env = cleaned

    def _run_docker(
        self,
        args: list[str],
        *,
        check: bool = True,
        timeout: int = _DOCKER_BUILD_TIMEOUT,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["docker", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"docker {' '.join(args)} failed")
        return completed

    def _register_active_session(self) -> None:
        if self._session_registered:
            return
        self._runtime_marker_path = _register_active_runtime(
            workspace=self.workspace,
            session_id=self.session_id,
        )
        self._session_registered = True

    def _unregister_active_session(self) -> None:
        _unregister_active_runtime(self._runtime_marker_path)
        self._runtime_marker_path = None
        self._session_registered = False

    def _has_active_sessions(self) -> bool:
        return _has_active_runtimes()

    def _ensure_image(self) -> None:
        inspect = self._run_docker(["image", "inspect", self.image_tag], check=False)
        if inspect.returncode == 0:
            return

        self._run_docker(
            [
                "build",
                "-t",
                self.image_tag,
                "-f",
                str(_SANDBOX_DOCKERFILE),
                str(_DOCKER_BUILD_ROOT),
            ],
            timeout=_DOCKER_BUILD_TIMEOUT,
        )

    def _ensure_searxng_image(self) -> None:
        inspect = self._run_docker(["image", "inspect", _SEARXNG_IMAGE], check=False)
        if inspect.returncode == 0:
            return
        self._run_docker(["pull", _SEARXNG_IMAGE], timeout=_DOCKER_BUILD_TIMEOUT)

    def _cleanup_unused_networks(self) -> None:
        listed = self._run_docker(
            ["network", "ls", "--format", "{{.Name}}"],
            check=False,
            timeout=30,
        )
        if listed.returncode != 0:
            return
        for raw_name in listed.stdout.splitlines():
            name = raw_name.strip()
            if not name.startswith(_NETWORK_NAME_PREFIX):
                continue
            if name == self.network_name:
                continue
            inspect = self._run_docker(["network", "inspect", name], check=False, timeout=30)
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
            self._run_docker(["network", "rm", name], check=False, timeout=30)

    def _ensure_network(self) -> None:
        inspect = self._run_docker(["network", "inspect", self.network_name], check=False)
        if inspect.returncode == 0:
            return
        created = self._run_docker(["network", "create", self.network_name], check=False, timeout=30)
        if created.returncode == 0:
            return
        detail = (created.stderr or created.stdout or "").strip().lower()
        if "fully subnetted" in detail:
            self._cleanup_unused_networks()
            created = self._run_docker(["network", "create", self.network_name], check=False, timeout=30)
            if created.returncode == 0:
                return
        detail = (created.stderr or created.stdout or "").strip()
        raise RuntimeError(detail or f"docker network create {self.network_name} failed")

    def _ensure_cache_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for relative in (
            "home",
            "pip",
            "npm",
            "npm-global",
            "venv",
        ):
            (self.cache_dir / relative).mkdir(parents=True, exist_ok=True)

    def _ensure_searxng_service(self) -> None:
        if not self._managed_searxng:
            return

        self.searxng_config_dir.mkdir(parents=True, exist_ok=True)
        self.searxng_data_dir.mkdir(parents=True, exist_ok=True)
        _write_searxng_settings(self.searxng_config_dir)

        inspect = self._run_docker(
            ["inspect", "-f", "{{.State.Running}}", self.searxng_name],
            check=False,
        )
        if inspect.returncode == 0 and inspect.stdout.strip() == "true":
            self._wait_for_searxng_ready()
            return
        if inspect.returncode == 0:
            self._run_docker(["rm", "-f", self.searxng_name], check=False)

        self._run_docker(
            [
                "run",
                "-d",
                "--name",
                self.searxng_name,
                "--restart",
                "unless-stopped",
                "--network",
                self.network_name,
                "-v",
                f"{self.searxng_config_dir}:/etc/searxng",
                "-v",
                f"{self.searxng_data_dir}:/var/cache/searxng",
                _SEARXNG_IMAGE,
            ],
            timeout=_DOCKER_BUILD_TIMEOUT,
        )
        self._wait_for_searxng_ready()

    def _wait_for_searxng_ready(self) -> None:
        deadline = time.time() + max(1, _SEARXNG_READY_TIMEOUT)
        probe = (
            "from urllib.request import urlopen\n"
            f"urlopen('{self._effective_searxng_base_url.rstrip('/')}/search?q=test&format=json', timeout=5).read(64)\n"
            "print('ready')\n"
        )
        last_error = "searxng readiness probe did not return success"
        while time.time() < deadline:
            completed = self._run_docker(
                [
                    "run",
                    "--rm",
                    "--network",
                    self.network_name,
                    self.image_tag,
                    "python",
                    "-c",
                    probe,
                ],
                check=False,
                timeout=15,
            )
            if completed.returncode == 0:
                return
            detail = (completed.stderr or completed.stdout or "").strip()
            if detail:
                last_error = detail
            time.sleep(max(0.1, _SEARXNG_READY_POLL))
        raise RuntimeError(f"SearXNG did not become ready: {last_error}")

    def prepare_runtime(self) -> None:
        if self._runtime_prepared:
            return
        self._ensure_image()
        self._ensure_network()
        self._ensure_cache_dir()
        self._register_active_session()
        try:
            if self._managed_searxng:
                self._ensure_searxng_image()
                self._ensure_searxng_service()
            self._runtime_prepared = True
        except Exception:
            self._unregister_active_session()
            raise

    def shutdown(self) -> None:
        self._unregister_active_session()
        self._runtime_prepared = False
        if self._has_active_sessions():
            return
        if self._managed_searxng:
            self._run_docker(["rm", "-f", self.searxng_name], check=False, timeout=30)
        self._run_docker(["network", "rm", self.network_name], check=False, timeout=30)

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
            "SEARXNG_BASE_URL": self._effective_searxng_base_url,
        }
        env.update(self._local_model_service_env)
        for key, value in os.environ.items():
            if not value:
                continue
            if not key.startswith(_PASS_ENV_PREFIXES):
                continue
            if key.endswith("_BASE_URL") or key == "SEARXNG_BASE_URL":
                env[key] = _dockerize_loopback_url(value)
            else:
                env[key] = value
        env["SEARXNG_BASE_URL"] = self._effective_searxng_base_url
        return env

    @staticmethod
    def _build_container_command(
        code_type: str,
        has_path: bool,
        path_value: str,
        script_value: str,
        args_value: list[str],
    ) -> list[str]:
        if code_type == "python":
            if has_path:
                return ["python", path_value, *args_value]
            return ["python", "-c", script_value]
        if code_type == "bash":
            if has_path:
                return ["bash", path_value, *args_value]
            return ["bash", "-c", script_value]
        raise ValueError(f"Unsupported code_type: {code_type}")

    def _remove_container(self, name: str) -> None:
        self._run_docker(["rm", "-f", name], check=False, timeout=30)

    def __call__(self, payload: dict, workspace: Path) -> Turn:
        workspace_root = Path(workspace).expanduser().resolve()
        timeout = payload.get("timeout_seconds", _DOCKER_TIMEOUT)
        try:
            timeout_seconds = int(timeout)
        except (TypeError, ValueError):
            timeout_seconds = _DOCKER_TIMEOUT
        job_name = str(payload.get("job_name", "unnamed_job")).strip() or "unnamed_job"

        try:
            self.prepare_runtime()
            code_type, has_path, path_value, script_value, args_value = _normalize_exec_input(payload)
            container_command = self._build_container_command(
                code_type,
                has_path,
                path_value,
                script_value,
                args_value,
            )
        except Exception as exc:
            return Turn(
                role="runtime",
                content=f"Job '{job_name}' failed to start: {exc}",
            )

        runtime_logs = workspace_root / ".runtime" / "logs"
        runtime_logs.mkdir(parents=True, exist_ok=True)
        stdout_fd, stdout_name = tempfile.mkstemp(
            prefix=f"{job_name}_stdout_",
            suffix=".log",
            dir=str(runtime_logs),
        )
        stderr_fd, stderr_name = tempfile.mkstemp(
            prefix=f"{job_name}_stderr_",
            suffix=".log",
            dir=str(runtime_logs),
        )
        stdout_path = Path(stdout_name)
        stderr_path = Path(stderr_name)

        container_name = f"helix-exec-{self.slug}-{int(time.time() * 1000)}"
        env = self._build_container_environment(workspace_root)
        uid_gid = f"{os.getuid()}:{os.getgid()}"
        docker_args = [
            "docker",
            "run",
            "--name",
            container_name,
            "--rm",
            "--init",
            "--network",
            self.network_name,
            "--read-only",
            "--tmpfs",
            "/tmp:exec,mode=1777",
            "--tmpfs",
            "/run:mode=755",
            "--shm-size",
            "512m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            _DOCKER_MEMORY,
            "--cpus",
            _DOCKER_CPUS,
            "--pids-limit",
            _DOCKER_PIDS,
            "--user",
            uid_gid,
            "--workdir",
            str(workspace_root),
            "--mount",
            f"type=bind,src={workspace_root},dst={workspace_root}",
            "--mount",
            f"type=bind,src={self.cache_dir},dst=/helix-cache",
        ]
        if sys.platform.startswith("linux"):
            docker_args.extend(["--add-host", "host.docker.internal:host-gateway"])
        for key, value in sorted(env.items()):
            docker_args.extend(["-e", f"{key}={value}"])
        docker_args.append(self.image_tag)
        docker_args.extend(container_command)

        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                docker_args,
                cwd=str(workspace_root),
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        finally:
            stdout_file.close()
            stderr_file.close()

        try:
            process.wait(timeout=timeout_seconds)
            result = _collect_logged_result(
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except subprocess.TimeoutExpired:
            self._remove_container(container_name)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            result = _collect_logged_result(
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra_stderr=f"\nruntime> exec terminated after {timeout_seconds}s timeout",
            )
        except KeyboardInterrupt:
            self._remove_container(container_name)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            result = _collect_logged_result(
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra_stderr="\nruntime> exec terminated by user (KeyboardInterrupt)",
            )

        stdout = result["stdout"]
        stderr = result["stderr"]
        rc = result["return_code"]

        status = "succeeded" if rc == 0 else "failed"
        content = f"Job '{job_name}' {status}. (Exit code: {rc})"
        if stdout:
            content += _format_output_block("stdout", stdout)
        if stderr:
            content += _format_output_block("stderr", stderr)
        return Turn(role="runtime", content=content)


def sandbox_executor(payload: dict, workspace: Path) -> Turn:
    """Run a single exec payload in the Docker sandbox.

    This helper is mainly for tests and direct `Environment(...)` usage.
    The session-managed runtime should use `DockerSandboxExecutor` directly.
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
