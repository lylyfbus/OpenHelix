"""Docker sandbox integration tests."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.sandbox import DockerSandboxExecutor, docker_is_available
from helix.core.local_model_service import LocalModelServiceManager


def _docker_ready() -> bool:
    available, reason = docker_is_available()
    if not available:
        print(f"  Docker unavailable, skipping docker sandbox tests: {reason}")
        return False
    return True


def _set_helix_home(workspace: Path) -> tuple[str | None, Path]:
    previous = os.environ.get("HELIX_HOME")
    home = workspace / ".test-helix-home"
    os.environ["HELIX_HOME"] = str(home)
    return previous, home


def _restore_helix_home(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("HELIX_HOME", None)
    else:
        os.environ["HELIX_HOME"] = previous


def test_docker_sandbox_uses_global_service_paths():
    with tempfile.TemporaryDirectory() as td:
        workspace_one = (Path(td) / "workspace-one").resolve()
        workspace_two = (Path(td) / "workspace-two").resolve()
        workspace_one.mkdir(parents=True, exist_ok=True)
        workspace_two.mkdir(parents=True, exist_ok=True)
        previous, helix_home = _set_helix_home(workspace_one)
        try:
            executor_one = DockerSandboxExecutor(workspace_one, searxng_base_url="https://example.com")
            executor_two = DockerSandboxExecutor(workspace_two, searxng_base_url="https://example.com")
            assert executor_one.network_name == executor_two.network_name == "helix-sandbox-net"
            assert executor_one.cache_dir == workspace_one / ".runtime" / "docker" / "cache"
            assert executor_two.cache_dir == workspace_two / ".runtime" / "docker" / "cache"
            assert executor_one.searxng_config_dir == helix_home / "runtime" / "services" / "searxng" / "config"
            assert executor_two.searxng_data_dir == helix_home / "runtime" / "services" / "searxng" / "data"
            print("  Docker sandbox global service paths OK")
        finally:
            _restore_helix_home(previous)


def test_docker_sandbox_bash_execution():
    if not _docker_ready():
        return

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        previous, _ = _set_helix_home(workspace)
        executor = DockerSandboxExecutor(workspace, searxng_base_url="https://example.com")
        try:
            turn = executor(
                {
                    "job_name": "docker-bash",
                    "code_type": "bash",
                    "script": "echo hello-from-docker",
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert "hello-from-docker" in turn.content
            print("  Docker sandbox bash execution OK")
        finally:
            executor.shutdown()
            _restore_helix_home(previous)


def test_docker_sandbox_writes_host_workspace():
    if not _docker_ready():
        return

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        previous, _ = _set_helix_home(workspace)
        executor = DockerSandboxExecutor(workspace, searxng_base_url="https://example.com")
        try:
            turn = executor(
                {
                    "job_name": "docker-write",
                    "code_type": "bash",
                    "script": "printf 'from-docker' > docker-output.txt",
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert (workspace / "docker-output.txt").read_text(encoding="utf-8") == "from-docker"
            print("  Docker sandbox host workspace write OK")
        finally:
            executor.shutdown()
            _restore_helix_home(previous)


def test_docker_sandbox_persists_python_installs_in_cache():
    if not _docker_ready():
        return

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        previous, _ = _set_helix_home(workspace)
        package_dir = workspace / "pkgdemo"
        module_dir = package_dir / "demo_pkg"
        module_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "setup.py").write_text(
            "from setuptools import setup\n"
            "setup(name='demo-pkg', version='0.1.0', packages=['demo_pkg'])\n",
            encoding="utf-8",
        )
        (module_dir / "__init__.py").write_text("VALUE = 'persisted'\n", encoding="utf-8")

        executor = DockerSandboxExecutor(workspace, searxng_base_url="https://example.com")
        try:
            assert executor.cache_dir == workspace / ".runtime" / "docker" / "cache"
            install_turn = executor(
                {
                    "job_name": "docker-pip-install",
                    "code_type": "bash",
                    "script": "python -m pip install ./pkgdemo && python -c \"import demo_pkg; print(demo_pkg.VALUE)\"",
                },
                workspace,
            )
            assert "persisted" in install_turn.content

            reuse_turn = executor(
                {
                    "job_name": "docker-pip-reuse",
                    "code_type": "python",
                    "script": "import demo_pkg; print(demo_pkg.VALUE)",
                },
                workspace,
            )
            assert "persisted" in reuse_turn.content
            assert (workspace / ".runtime" / "docker" / "cache" / "venv").exists()
            print("  Docker sandbox Python package cache persistence OK")
        finally:
            executor.shutdown()
            _restore_helix_home(previous)


def test_docker_sandbox_browser_tooling_available():
    if not _docker_ready():
        return

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        previous, _ = _set_helix_home(workspace)
        executor = DockerSandboxExecutor(workspace, searxng_base_url="https://example.com")
        try:
            turn = executor(
                {
                    "job_name": "docker-browser-tooling",
                    "code_type": "python",
                    "script": (
                        "import shutil, selenium\n"
                        "print('selenium_import=1')\n"
                        "print('chromium_path=' + str(shutil.which('chromium') or ''))\n"
                        "print('chromedriver_path=' + str(shutil.which('chromedriver') or ''))\n"
                    ),
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert "selenium_import=1" in turn.content
            assert "chromium_path=/usr/bin/chromium" in turn.content
            assert "chromedriver_path=/usr/bin/chromedriver" in turn.content
            print("  Docker sandbox browser tooling OK")
        finally:
            executor.shutdown()
            _restore_helix_home(previous)


def test_docker_sandbox_can_use_git_metadata():
    if not _docker_ready():
        return

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        previous, _ = _set_helix_home(workspace)
        import subprocess

        subprocess.run(["git", "init"], cwd=workspace, check=True, stdout=subprocess.DEVNULL)
        executor = DockerSandboxExecutor(workspace, searxng_base_url="https://example.com")
        try:
            turn = executor(
                {
                    "job_name": "docker-git",
                    "code_type": "bash",
                    "script": (
                        "git config user.email 'sandbox@example.com' && "
                        "git config user.name 'Sandbox' && "
                        "printf 'x' > tracked.txt && "
                        "git add tracked.txt && "
                        "git status --short"
                    ),
                },
                workspace,
            )
            assert "A  tracked.txt" in turn.content
            print("  Docker sandbox git metadata access OK")
        finally:
            executor.shutdown()
            _restore_helix_home(previous)


def test_docker_sandbox_managed_searxng_returns_json():
    if not _docker_ready():
        return

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        previous, _ = _set_helix_home(workspace)
        executor = DockerSandboxExecutor(workspace)
        try:
            turn = executor(
                {
                    "job_name": "docker-searxng",
                    "code_type": "python",
                    "script": (
                        "import json, os\n"
                        "from urllib.request import urlopen\n"
                        "base = os.environ['SEARXNG_BASE_URL'].rstrip('/')\n"
                        "with urlopen(base + '/search?q=test&format=json', timeout=20) as resp:\n"
                        "    payload = json.loads(resp.read().decode('utf-8', errors='replace'))\n"
                        "print(json.dumps({'status': 'ok', 'keys': sorted(payload.keys())[:5]}))\n"
                    ),
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert "status: ok" in turn.content
            print("  Docker sandbox managed SearXNG JSON OK")
        finally:
            executor.shutdown()
            _restore_helix_home(previous)


def test_docker_sandbox_image_skill_can_reach_local_model_service():
    if not _docker_ready() or sys.platform != "darwin":
        return

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        previous, _ = _set_helix_home(workspace)
        skills_root = workspace / "skills" / "all-agents"
        skills_root.mkdir(parents=True, exist_ok=True)
        source_skill = (
            Path(__file__).resolve().parent.parent
            / "helix"
            / "builtin_skills"
            / "all-agents"
            / "generate-image-from-pytorch"
        )
        shutil.copytree(source_skill, skills_root / "generate-image-from-pytorch")
        manager = LocalModelServiceManager(
            workspace,
            session_id="docker-image-skill",
            backend_mode="fake",
        )
        try:
            manager.start()
        except PermissionError:
            return
        executor = DockerSandboxExecutor(workspace, searxng_base_url="https://example.com")
        executor.attach_local_model_service(manager.tool_environment())
        try:
            turn = executor(
                {
                    "job_name": "docker-image-skill",
                    "code_type": "python",
                    "script_path": "skills/all-agents/generate-image-from-pytorch/scripts/generate_image_from_pytorch.py",
                    "script_args": [
                        "--prompt", "A minimal test image",
                        "--output-dir", "generated_images",
                    ],
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert "generated_images/" in turn.content
            generated = list((workspace / "generated_images").glob("*.png"))
            assert generated, "expected generated image file"
            print("  Docker image skill local-model-service reachability OK")
        finally:
            executor.shutdown()
            manager.stop()
            _restore_helix_home(previous)


if __name__ == "__main__":
    test_docker_sandbox_bash_execution()
    test_docker_sandbox_writes_host_workspace()
    test_docker_sandbox_persists_python_installs_in_cache()
    test_docker_sandbox_browser_tooling_available()
    test_docker_sandbox_can_use_git_metadata()
    test_docker_sandbox_managed_searxng_returns_json()
