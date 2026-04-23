"""Host-shell sandbox integration tests.

These tests exercise HostSandboxExecutor end-to-end by running real
scripts on the host. No isolation is expected — safety comes from the
ApprovalPolicy layer (see test_runtime.py for those tests).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.runtime.sandbox import HostSandboxExecutor


def test_host_sandbox_bash_execution():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        executor = HostSandboxExecutor(workspace)
        try:
            turn = executor(
                {
                    "job_name": "host-bash",
                    "code_type": "bash",
                    "script": "echo hello-from-host",
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert "hello-from-host" in turn.content
            print("  Host sandbox bash execution OK")
        finally:
            executor.shutdown()


def test_host_sandbox_python_execution():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        executor = HostSandboxExecutor(workspace)
        try:
            turn = executor(
                {
                    "job_name": "host-python",
                    "code_type": "python",
                    "script": "print('hello-from-python')",
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert "hello-from-python" in turn.content
            print("  Host sandbox python execution OK")
        finally:
            executor.shutdown()


def test_host_sandbox_writes_workspace_file():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        executor = HostSandboxExecutor(workspace)
        try:
            turn = executor(
                {
                    "job_name": "host-write",
                    "code_type": "bash",
                    "script": "printf 'from-host' > host-output.txt",
                },
                workspace,
            )
            assert "succeeded" in turn.content
            assert (workspace / "host-output.txt").read_text(encoding="utf-8") == "from-host"
            print("  Host sandbox workspace write OK")
        finally:
            executor.shutdown()


def test_host_sandbox_failure_exit_code():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        executor = HostSandboxExecutor(workspace)
        try:
            turn = executor(
                {
                    "job_name": "host-fail",
                    "code_type": "bash",
                    "script": "exit 7",
                },
                workspace,
            )
            assert "failed" in turn.content
            assert "Exit code: 7" in turn.content
            print("  Host sandbox failure exit code OK")
        finally:
            executor.shutdown()


def test_host_sandbox_timeout_kills_runaway():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        executor = HostSandboxExecutor(workspace)
        try:
            turn = executor(
                {
                    "job_name": "host-timeout",
                    "code_type": "bash",
                    "script": "sleep 30",
                    "timeout_seconds": 1,
                },
                workspace,
            )
            assert "terminated after 1s timeout" in turn.content
            print("  Host sandbox timeout OK")
        finally:
            executor.shutdown()


def test_host_sandbox_env_forwarding():
    """tool_environment values should reach the running script."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        executor = HostSandboxExecutor(
            workspace,
            searxng_base_url="http://searxng.local:8080",
            local_model_service_env={
                "HELIX_LOCAL_MODEL_SERVICE_URL": "http://127.0.0.1:9999",
                "HELIX_LOCAL_MODEL_SERVICE_TOKEN": "tok_test",
            },
        )
        try:
            turn = executor(
                {
                    "job_name": "host-env",
                    "code_type": "bash",
                    "script": 'echo "$SEARXNG_BASE_URL|$HELIX_LOCAL_MODEL_SERVICE_URL|$HELIX_LOCAL_MODEL_SERVICE_TOKEN"',
                },
                workspace,
            )
            assert "http://searxng.local:8080|http://127.0.0.1:9999|tok_test" in turn.content
            print("  Host sandbox env forwarding OK")
        finally:
            executor.shutdown()


if __name__ == "__main__":
    print("=== Host sandbox ===")
    test_host_sandbox_bash_execution()
    test_host_sandbox_python_execution()
    test_host_sandbox_writes_workspace_file()
    test_host_sandbox_failure_exit_code()
    test_host_sandbox_timeout_kills_runaway()
    test_host_sandbox_env_forwarding()
    print("\n✅ All host sandbox tests passed!")
