"""Phase 2 verification tests for runtime sandbox and approval policies."""

import builtins
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.action import Action
from helix.core.environment import Environment
from helix.core.sandbox import docker_is_available, sandbox_executor
from helix.core.state import Turn
from helix.runtime.approval import ApprovalPolicy
from helix.runtime.display import TURN_SEPARATOR


def _docker_ready() -> bool:
    available, reason = docker_is_available()
    if not available:
        print(f"  Docker unavailable, skipping runtime sandbox test: {reason}")
        return False
    return True


def test_sandbox_bash_execution():
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            "script": "echo 'hello from bash'",
            "job_name": "test_bash"
        }
        turn = sandbox_executor(payload, workspace)
        assert turn.role == "runtime"
        assert "Job 'test_bash' succeeded." in turn.content
        assert "hello from bash" in turn.content
        assert "Exit code: 0" in turn.content
        print("  Sandbox bash execution OK")


def test_sandbox_python_execution():
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "python",
            "script": "print('hello from python')",
            "job_name": "test_python"
        }
        turn = sandbox_executor(payload, workspace)
        assert "Job 'test_python' succeeded." in turn.content
        assert "hello from python" in turn.content
        assert "Exit code: 0" in turn.content
        print("  Sandbox python execution OK")


def test_sandbox_failure_exit_code():
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            "script": "exit 42",
        }
        turn = sandbox_executor(payload, workspace)
        assert "Job 'unnamed_job' failed." in turn.content
        assert "failed" in turn.content
        assert "Exit code: 42" in turn.content
        print("  Sandbox failure exit code OK")


def test_sandbox_invalid_input():
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            # Missing both script and script_path
        }
        turn = sandbox_executor(payload, workspace)
        assert "Job 'unnamed_job' failed to start" in turn.content
        print("  Sandbox invalid input OK")


def test_sandbox_formats_json_stdout_readably():
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "python",
            "job_name": "json-output",
            "script": (
                "import json\n"
                "print(json.dumps({"
                "\"status\": \"ok\", "
                "\"details\": \"line1\\nline2\", "
                "\"items\": [\"a\", \"b\"]"
                "}))\n"
            ),
        }
        turn = sandbox_executor(payload, workspace)
        assert "Job 'json-output' succeeded." in turn.content
        assert "<stdout>" in turn.content
        assert "status: ok" in turn.content
        assert "details: |" in turn.content
        assert "line1" in turn.content
        assert "line2" in turn.content
        assert '"details"' not in turn.content
        print("  Sandbox JSON stdout formatting OK")


def test_sandbox_wraps_stderr_readably():
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            "job_name": "stderr-output",
            "script": "echo 'problem line' 1>&2; exit 7",
        }
        turn = sandbox_executor(payload, workspace)
        assert "Job 'stderr-output' failed." in turn.content
        assert "<stderr>" in turn.content
        assert "problem line" in turn.content
        print("  Sandbox stderr formatting OK")


def test_approval_policy_auto_mode():
    policy = ApprovalPolicy(mode="auto")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})
    assert policy(env, action) is True
    print("  Approval policy auto mode OK")


def test_approval_policy_controlled_allow_once():
    policy = ApprovalPolicy(mode="controlled", prompt=lambda _prompt: "y")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})
    assert policy(env, action) is True
    print("  Approval allow once OK")


def test_approval_policy_controlled_allow_session():
    policy = ApprovalPolicy(mode="controlled", prompt=lambda _prompt: "s")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})
    # First time prompts 's'
    assert policy(env, action) is True

    # Second time shouldn't prompt
    with patch("builtins.input", side_effect=Exception("Should not prompt")):
        assert policy(env, action) is True

    print("  Approval allow session OK")


def test_approval_policy_pattern_mode():
    policy = ApprovalPolicy(mode="controlled", prompt=lambda _prompt: "p")
    env = Environment(workspace=Path("."))

    action1 = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo 'hello world'"})
    # Approve with pattern mode
    assert policy(env, action1) is True

    # Similar script with different quoted content should also be approved
    action2 = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo 'goodbye world'"})
    with patch("builtins.input", side_effect=Exception("Should not prompt")):
        assert policy(env, action2) is True

    print("  Approval pattern mode OK")


def test_approval_policy_pattern_mode_rejects_script_path():
    prompts: list[str] = []

    def fake_prompt(_prompt: str) -> str:
        prompts.append("p")
        return "p"

    policy = ApprovalPolicy(mode="controlled", prompt=fake_prompt)
    env = Environment(workspace=Path("."))

    action1 = Action(
        response="",
        type="exec",
        payload={
            "code_type": "python",
            "script_path": "skills/a.py",
            "script_args": ["--value", "123"],
        },
    )
    result1 = policy(env, action1)
    assert isinstance(result1, Turn)
    assert "denied by requester" in result1.content.lower()
    assert not policy.approved_patterns

    action2 = Action(
        response="",
        type="exec",
        payload={
            "code_type": "python",
            "script_path": "skills/b.py",
            "script_args": ["--value", "456"],
        },
    )
    result2 = policy(env, action2)
    assert isinstance(result2, Turn)
    assert "denied by requester" in result2.content.lower()
    assert len(prompts) == 2

    print("  Approval pattern mode rejects script_path OK")


def test_approval_policy_controlled_deny():
    policy = ApprovalPolicy(mode="controlled", prompt=lambda _prompt: "n")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "rm -rf /"})
    result = policy(env, action)
    assert isinstance(result, Turn)
    assert "denied by requester" in result.content.lower()
    print("  Approval deny OK")


def test_approval_policy_uses_injected_prompt_instead_of_raw_input():
    policy = ApprovalPolicy(mode="controlled", prompt=lambda _prompt: "y")
    env = Environment(workspace=Path("."))
    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})

    original_input = builtins.input
    try:
        builtins.input = lambda _prompt="": (_ for _ in ()).throw(Exception("raw input should not be used"))
        assert policy(env, action) is True
    finally:
        builtins.input = original_input
    print("  Approval uses injected prompt OK")


def test_approval_policy_keyboard_interrupt_cancels():
    policy = ApprovalPolicy(
        mode="controlled",
        prompt=lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    env = Environment(workspace=Path("."))
    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})
    result = policy(env, action)
    assert isinstance(result, Turn)
    assert "cancelled during approval prompt" in result.content.lower()
    print("  Approval keyboard interrupt cancels OK")


def test_approval_prompt_prints_separator_before_input():
    captured = StringIO()

    def fake_prompt(prompt_text: str) -> str:
        print(prompt_text, end="")
        return "y"

    policy = ApprovalPolicy(mode="controlled", prompt=fake_prompt)
    env = Environment(workspace=Path("."))
    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})

    with patch("sys.stdout", captured):
        assert policy(env, action) is True

    output = captured.getvalue()
    assert f"{TURN_SEPARATOR}\nruntime> Action requires approval:" in output
    assert TURN_SEPARATOR in output
    assert f"{TURN_SEPARATOR}\n> " in output
    print("  Approval separator before input OK")


def test_approval_policy_non_exec_passthrough():
    policy = ApprovalPolicy(mode="controlled")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="think", payload={})
    assert policy(env, action) is True
    print("  Approval non-exec passthrough OK")


def test_environment_integration():
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        # Use our real executor and an auto policy for hands-free test
        env = Environment(workspace=workspace, executor=sandbox_executor)
        policy = ApprovalPolicy(mode="auto")
        env.on_before_execute(policy)

        action = Action(response="Running script", type="exec", payload={
            "code_type": "bash",
            "script": "echo 'integrated bash'"
        })

        turn = env.execute(action)
        assert "Job 'unnamed_job' succeeded." in turn.content
        assert "integrated bash" in turn.content
        print("  Environment integration OK")


if __name__ == "__main__":
    print("=== Sandbox Execution ===")
    test_sandbox_bash_execution()
    test_sandbox_python_execution()
    test_sandbox_failure_exit_code()
    test_sandbox_invalid_input()
    test_sandbox_formats_json_stdout_readably()
    test_sandbox_wraps_stderr_readably()

    print("\n=== Approval Policy ===")
    test_approval_policy_auto_mode()
    test_approval_policy_controlled_allow_once()
    test_approval_policy_controlled_allow_session()
    test_approval_policy_pattern_mode()
    test_approval_policy_pattern_mode_rejects_script_path()
    test_approval_policy_controlled_deny()
    test_approval_policy_uses_injected_prompt_instead_of_raw_input()
    test_approval_policy_keyboard_interrupt_cancels()
    test_approval_prompt_prints_separator_before_input()
    test_approval_policy_non_exec_passthrough()

    print("\n=== Environment Integration ===")
    test_environment_integration()

    print("\n✅ All Runtime/Phase 2 tests passed!")
