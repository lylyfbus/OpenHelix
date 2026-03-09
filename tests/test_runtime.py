"""Phase 2 verification tests for runtime sandbox and approval policies."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_system.core.action import Action
from agentic_system.core.environment import Environment
from agentic_system.runtime.sandbox import sandbox_executor
from agentic_system.runtime.approval import ApprovalPolicy


def test_sandbox_bash_execution():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            "script": "echo 'hello from bash'",
            "job_name": "test_bash"
        }
        turn = sandbox_executor(payload, workspace)
        assert turn.role == "runtime"
        assert "hello from bash" in turn.content
        assert "Exit code: 0" in turn.content
        print("  Sandbox bash execution OK")


def test_sandbox_python_execution():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "python",
            "script": "print('hello from python')",
            "job_name": "test_python"
        }
        turn = sandbox_executor(payload, workspace)
        assert "hello from python" in turn.content
        assert "Exit code: 0" in turn.content
        print("  Sandbox python execution OK")


def test_sandbox_failure_exit_code():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            "script": "exit 42",
        }
        turn = sandbox_executor(payload, workspace)
        assert "failed" in turn.content
        assert "Exit code: 42" in turn.content
        print("  Sandbox failure exit code OK")


def test_sandbox_invalid_input():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            # Missing both script and script_path
        }
        turn = sandbox_executor(payload, workspace)
        assert "Execution failed" in turn.content
        print("  Sandbox invalid input OK")


def test_approval_policy_auto_mode():
    policy = ApprovalPolicy(mode="auto")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})
    assert policy(env, action) is True
    print("  Approval policy auto mode OK")


@patch("builtins.input", side_effect=["y"])
def test_approval_policy_controlled_allow_once(mock_input):
    policy = ApprovalPolicy(mode="controlled")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})
    assert policy(env, action) is True
    print("  Approval allow once OK")


@patch("builtins.input", side_effect=["s"])
def test_approval_policy_controlled_allow_session(mock_input):
    policy = ApprovalPolicy(mode="controlled")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo x"})
    # First time prompts 's'
    assert policy(env, action) is True

    # Second time shouldn't prompt
    with patch("builtins.input") as mock_input2:
        mock_input2.side_effect = Exception("Should not prompt")
        assert policy(env, action) is True

    print("  Approval allow session OK")


@patch("builtins.input", side_effect=["p"])
def test_approval_policy_pattern_mode(mock_input):
    policy = ApprovalPolicy(mode="controlled")
    env = Environment(workspace=Path("."))

    action1 = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo 'hello world'"})
    # Approve with pattern mode
    assert policy(env, action1) is True

    # Similar script with different quoted content should also be approved
    action2 = Action(response="", type="exec", payload={"code_type": "bash", "script": "echo 'goodbye world'"})
    with patch("builtins.input") as mock_input2:
        mock_input2.side_effect = Exception("Should not prompt")
        assert policy(env, action2) is True

    print("  Approval pattern mode OK")


@patch("builtins.input", side_effect=["n"])
def test_approval_policy_controlled_deny(mock_input):
    policy = ApprovalPolicy(mode="controlled")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="exec", payload={"code_type": "bash", "script": "rm -rf /"})
    assert policy(env, action) is False
    print("  Approval deny OK")


def test_approval_policy_non_exec_passthrough():
    policy = ApprovalPolicy(mode="controlled")
    env = Environment(workspace=Path("."))

    action = Action(response="", type="think", payload={})
    assert policy(env, action) is True
    print("  Approval non-exec passthrough OK")


def test_environment_integration():
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
        assert "integrated bash" in turn.content
        print("  Environment integration OK")


if __name__ == "__main__":
    print("=== Sandbox Execution ===")
    test_sandbox_bash_execution()
    test_sandbox_python_execution()
    test_sandbox_failure_exit_code()
    test_sandbox_invalid_input()

    print("\n=== Approval Policy ===")
    test_approval_policy_auto_mode()
    test_approval_policy_controlled_allow_once()
    test_approval_policy_controlled_allow_session()
    test_approval_policy_pattern_mode()
    test_approval_policy_controlled_deny()
    test_approval_policy_non_exec_passthrough()

    print("\n=== Environment Integration ===")
    test_environment_integration()

    print("\n✅ All Runtime/Phase 2 tests passed!")
