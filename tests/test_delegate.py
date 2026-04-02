"""Sub-agent delegation integration tests.

Tests the delegate action flow: core-agent emits delegate → Environment
spawns sub-agent with isolated workspace → sub-agent runs → result
flows back into parent history.
"""

import sys
import tempfile
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.action import Action, ALLOWED_CORE_ACTIONS, ALLOWED_SUB_ACTIONS
from helix.core.agent import Agent
from helix.core.environment import Environment
from helix.runtime.loop import run_loop
from helix.core.state import Turn
from helix.core.sandbox import docker_is_available, sandbox_executor
from helix.runtime.approval import ApprovalPolicy
from helix.runtime.display import TURN_SEPARATOR


def _docker_ready() -> bool:
    available, reason = docker_is_available()
    if not available:
        print(f"  Docker unavailable, skipping delegation exec test: {reason}")
        return False
    return True


# =========================================================================== #
# Mock models
# =========================================================================== #


class CoreAgentModel:
    """Mock model for core agent: delegates a task, then chats the result."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt, *, stream=False, chunk_callback=None):
        self.call_count += 1
        if self.call_count == 1:
            # First turn: delegate
            return (
                '<output>'
                '{"response": "I will delegate research to a sub-agent.", '
                '"action": "delegate", '
                '"action_input": {'
                '"role": "researcher", '
                '"objective": "Find the capital of France.", '
                '"context": "User asked a geography question."'
                '}}'
                '</output>'
            )
        # Second turn: report based on sub-agent result
        return (
            '<output>'
            '{"response": "The sub-agent reported the capital of France is Paris.", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        )


class SubAgentModel:
    """Mock model for sub-agent: does a simple chat response."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt, *, stream=False, chunk_callback=None):
        self.call_count += 1
        return (
            '<output>'
            '{"response": "The capital of France is Paris.", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        )


class SharedModel:
    """Model shared between core and sub-agent (realistic scenario).

    Distinguishes core vs sub-agent by checking allowed_actions context
    embedded in the prompt (sub-agents get a different system prompt).
    """

    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, stream=False, chunk_callback=None):
        self.calls.append(prompt[:100])  # track calls

        if "sub-agent" in prompt.lower():
            # Sub-agent call
            return (
                '<output>'
                '{"response": "Research complete: Python was created by Guido van Rossum.", '
                '"action": "chat", "action_input": {}}'
                '</output>'
            )

        # Core agent
        if len(self.calls) == 1:
            return (
                '<output>'
                '{"response": "Let me delegate this research.", '
                '"action": "delegate", '
                '"action_input": {'
                '"role": "researcher", '
                '"objective": "Who created Python?"'
                '}}'
                '</output>'
            )
        return (
            '<output>'
            '{"response": "According to my research sub-agent, Python was created by Guido van Rossum.", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        )


# =========================================================================== #
# Tests
# =========================================================================== #


def test_delegate_no_model_ref():
    """Delegation should fail gracefully if no model reference is set."""
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.set_loop_fn(run_loop)
        # Don't set model ref
        action = Action(
            response="Delegating...",
            type="delegate",
            payload={"role": "test", "objective": "test task"},
        )
        result = env.delegate(action)
        assert "no model reference" in result.lower()
        print("  Delegate without model ref OK")





def test_delegate_basic():
    """Test direct delegation: sub-agent runs and returns result."""
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), mode="auto")
        model = SubAgentModel()
        env.set_model_ref(model)
        env.set_loop_fn(run_loop)

        action = Action(
            response="Delegating...",
            type="delegate",
            payload={
                "role": "researcher",
                "objective": "What is 2+2?",
                "context": "Math question",
            },
        )
        result = env.delegate(action)

        # Sub-agent should have chatted back
        assert "chat" in result.lower() or len(result) > 0
        assert model.call_count >= 1

        # No child workspace should be created (sub-agent shares parent workspace)
        sub_agents_dir = Path(td) / "sub_agents"
        assert not sub_agents_dir.exists()
        print("  Delegate basic OK")


def test_delegate_shares_parent_workspace():
    """Verify sub-agent shares the parent workspace (no child dir created)."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        env = Environment(workspace=workspace, mode="auto", executor=sandbox_executor)
        policy = ApprovalPolicy(mode="auto")
        env.on_before_execute(policy)
        env.set_model_ref(SubAgentModel())
        env.set_loop_fn(run_loop)

        action = Action(
            response="Delegating...",
            type="delegate",
            payload={"role": "researcher", "objective": "test workspace sharing"},
        )
        env.delegate(action)

        # No sub_agents directory should be created
        assert not (workspace / "sub_agents").exists()
        print("  Delegate shares parent workspace OK")


def test_delegate_sub_agent_cannot_delegate():
    """Verify sub-agents don't have the delegate action."""
    assert "delegate" not in ALLOWED_SUB_ACTIONS
    assert "delegate" in ALLOWED_CORE_ACTIONS
    print("  Sub-agent action restriction OK")


def test_full_delegation_loop():
    """End-to-end: core-agent delegates, sub-agent runs, result flows back."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        model = SharedModel()

        env = Environment(workspace=workspace, mode="auto")
        env.set_model_ref(model)
        env.set_loop_fn(run_loop)
        env.record(Turn(role="user", content="Who created Python?"))

        agent = Agent(
            model,
            system_prompt="You are the core agent.",
        )

        output = StringIO()
        result = run_loop(agent, env, output=output)

        # Should get the final answer
        assert "Guido" in result
        assert len(model.calls) >= 3  # core(delegate) + sub(chat) + core(chat)
        assert f"{TURN_SEPARATOR}\nruntime> Delegating to sub-agent" in output.getvalue()
        assert TURN_SEPARATOR in output.getvalue()
        assert "sub-agent>" not in output.getvalue()

        # Verify sub_agent turn appears in history
        sub_turns = [t for t in env.full_history if t.role == "sub-agent"]
        assert len(sub_turns) == 1
        assert "Guido" in sub_turns[0].content

        print("  Full delegation loop OK")


def test_delegate_with_exec_in_sub_agent():
    """Test sub-agent that uses exec before chatting back."""
    if not _docker_ready():
        return

    class ExecSubModel:
        def __init__(self):
            self.count = 0

        def generate(self, prompt, **kwargs):
            self.count += 1
            if self.count == 1:
                return (
                    '<output>'
                    '{"response": "Let me run a script.", '
                    '"action": "exec", '
                    '"action_input": {"job_name": "sub-task", '
                    '"code_type": "bash", "script": "echo sub-agent-output"}}'
                    '</output>'
                )
            return (
                '<output>'
                '{"response": "Script ran successfully: sub-agent-output", '
                '"action": "chat", "action_input": {}}'
                '</output>'
            )

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        model = ExecSubModel()

        env = Environment(
            workspace=workspace,
            mode="auto",
            executor=sandbox_executor,
        )
        policy = ApprovalPolicy(mode="auto")
        env.on_before_execute(policy)
        env.set_model_ref(model)
        env.set_loop_fn(run_loop)

        action = Action(
            response="Delegating with exec...",
            type="delegate",
            payload={"role": "executor", "objective": "Run a test script"},
        )
        result = env.delegate(action)

        assert "sub-agent-output" in result
        assert model.count == 2
        print("  Delegate with exec in sub-agent OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Delegation Guards ===")
    test_delegate_no_model_ref()
    test_delegate_sub_agent_cannot_delegate()

    print("\n=== Basic Delegation ===")
    test_delegate_basic()
    test_delegate_shares_parent_workspace()

    print("\n=== Full Delegation Loop ===")
    test_full_delegation_loop()
    test_delegate_with_exec_in_sub_agent()

    print("\n✅ All delegation tests passed!")
