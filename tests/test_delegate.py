"""Sub-agent delegation integration tests.

Tests the delegate action flow: core-agent emits delegate → run_loop
spawns sub-agent with isolated environment → sub-agent runs → result
flows back into parent history.
"""

import json
import sys
import tempfile
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.action import Action, ALLOWED_CORE_ACTIONS, ALLOWED_SUB_ACTIONS
from helix.core.agent import Agent
from helix.core.environment import Environment
from helix.runtime.loop import run_loop, _delegate
from helix.runtime.sub_agent_meta import load as _load_sub_agents_meta
from helix.core.state import Turn
from helpers import sandbox_executor
from helix.runtime.approval import ApprovalPolicy


# =========================================================================== #
# Mock models
# =========================================================================== #


class CoreAgentModel:
    """Mock model for core agent: delegates a task, then chats the result."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, *, chunk_callback=None):
        self.call_count += 1
        if self.call_count == 1:
            # First turn: delegate
            return (
                '<output>'
                '{"response": "I will delegate research to a sub-agent.", '
                '"next_action": "delegate", '
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
            '"next_action": "chat", "action_input": {}}'
            '</output>'
        )


class SubAgentModel:
    """Mock model for sub-agent: does a simple chat response."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, *, chunk_callback=None):
        self.call_count += 1
        return (
            '<output>'
            '{"response": "The capital of France is Paris.", '
            '"next_action": "chat", "action_input": {}}'
            '</output>'
        )


class SharedModel:
    """Model shared between core and sub-agent (realistic scenario).

    Distinguishes core vs sub-agent by checking allowed_actions context
    embedded in the prompt (sub-agents get a different system prompt).
    """

    def __init__(self):
        self.calls = []

    def generate(self, messages, *, chunk_callback=None):
        full_text = " ".join(m.get("content", "") for m in messages)
        self.calls.append(full_text[:100])  # track calls
        system_msg = messages[0].get("content", "") if messages else ""

        if "You are a Sub-Agent" in system_msg:
            # Sub-agent call
            return (
                '<output>'
                '{"response": "Research complete: Python was created by Guido van Rossum.", '
                '"next_action": "chat", "action_input": {}}'
                '</output>'
            )

        # Core agent
        if len(self.calls) == 1:
            return (
                '<output>'
                '{"response": "Let me delegate this research.", '
                '"next_action": "delegate", '
                '"action_input": {'
                '"role": "researcher", '
                '"objective": "Who created Python?"'
                '}}'
                '</output>'
            )
        return (
            '<output>'
            '{"response": "According to my research sub-agent, Python was created by Guido van Rossum.", '
            '"next_action": "chat", "action_input": {}}'
            '</output>'
        )


# =========================================================================== #
# Tests
# =========================================================================== #


def test_delegate_no_model():
    """Delegation should fail gracefully if no model is provided."""
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        action = Action(
            response="Delegating...",
            type="delegate",
            payload={"role": "test", "objective": "test task"},
        )
        result = _delegate(action, env, model=None)
        assert "no model reference" in result.lower()
        print("  Delegate without model OK")


def test_delegate_basic():
    """Test direct delegation: sub-agent runs and returns result."""
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), mode="auto")
        model = SubAgentModel()

        action = Action(
            response="Delegating...",
            type="delegate",
            payload={
                "role": "researcher",
                "objective": "What is 2+2?",
                "context": "Math question",
            },
        )
        result = _delegate(action, env, model=model)

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

        action = Action(
            response="Delegating...",
            type="delegate",
            payload={"role": "researcher", "objective": "test workspace sharing"},
        )
        _delegate(action, env, model=SubAgentModel())

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
        env.record(Turn(role="user", content="Who created Python?"))

        agent = Agent(
            model,
            workspace=workspace,
        )

        output = StringIO()
        result = run_loop(agent, env, model=model, output=output)

        # Should get the final answer
        assert "Guido" in result
        assert len(model.calls) >= 3  # core(delegate) + sub(chat) + core(chat)
        assert "Delegating to sub-agent" in output.getvalue()
        assert "sub-agent>" not in output.getvalue()

        # Verify sub_agent turn appears in history
        sub_turns = [t for t in env.full_history if t.role == "sub_agent"]
        assert len(sub_turns) == 1
        assert "Guido" in sub_turns[0].content

        print("  Full delegation loop OK")


def test_delegate_with_exec_in_sub_agent():
    """Test sub-agent that uses exec before chatting back."""

    class ExecSubModel:
        def __init__(self):
            self.count = 0

        def generate(self, messages, **kwargs):
            self.count += 1
            if self.count == 1:
                return (
                    '<output>'
                    '{"response": "Let me run a script.", '
                    '"next_action": "exec", '
                    '"action_input": {"job_name": "sub-task", '
                    '"code_type": "bash", "script": "echo sub-agent-output"}}'
                    '</output>'
                )
            return (
                '<output>'
                '{"response": "Script ran successfully: sub-agent-output", '
                '"next_action": "chat", "action_input": {}}'
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

        action = Action(
            response="Delegating with exec...",
            type="delegate",
            payload={"role": "executor", "objective": "Run a test script"},
        )
        result = _delegate(action, env, model=model)

        assert "sub-agent-output" in result
        assert model.count == 2
        print("  Delegate with exec in sub-agent OK")


# =========================================================================== #
# Sub-agent state persistence tests
# =========================================================================== #


def test_delegate_persists_and_restores_state():
    """Delegate twice to same role — second sees first's history."""
    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None):
            call_count[0] += 1
            return (
                '<output>'
                '{"response": "Result from call ' + str(call_count[0]) + '.", '
                '"next_action": "chat", "action_input": {}}'
                '</output>'
            )

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        state_root = workspace / ".state"
        state_root.mkdir(parents=True)
        env = Environment(workspace=workspace, mode="auto", state_root=state_root)

        # First delegation
        action1 = Action(
            response="Delegating...", type="delegate",
            payload={"role": "researcher", "role_description": "Research assistant", "objective": "Find X"},
        )
        _delegate(action1, env, model=MockModel())

        # State file should exist
        sub_state = state_root / "sub_agents" / "researcher.json"
        assert sub_state.exists()
        saved = json.loads(sub_state.read_text())
        assert len(saved["full_history"]) >= 2  # core_agent seed + sub_agent chat

        # Second delegation to same role
        action2 = Action(
            response="Follow up...", type="delegate",
            payload={"role": "researcher", "objective": "Now find Y"},
        )
        _delegate(action2, env, model=MockModel())

        # Should have accumulated history from both delegations
        saved2 = json.loads(sub_state.read_text())
        assert len(saved2["full_history"]) > len(saved["full_history"])
        print("  Delegate persists and restores state OK")


def test_delegate_new_role_starts_fresh():
    """Delegation to a new role starts with empty history."""

    class MockModel:
        def generate(self, messages, *, chunk_callback=None):
            return '<output>{"response": "Done.", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        state_root = workspace / ".state"
        state_root.mkdir(parents=True)
        env = Environment(workspace=workspace, mode="auto", state_root=state_root)

        action = Action(
            response="Delegating...", type="delegate",
            payload={"role": "new-role", "objective": "Do something"},
        )
        _delegate(action, env, model=MockModel())

        saved = json.loads((state_root / "sub_agents" / "new-role.json").read_text())
        # Fresh: only core_agent seed + sub_agent chat
        assert len(saved["full_history"]) == 2
        print("  Delegate new role starts fresh OK")


def test_delegate_meta_registry_updated():
    """Meta registry is created and updated on delegation."""

    class MockModel:
        def generate(self, messages, *, chunk_callback=None):
            return '<output>{"response": "Done.", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        state_root = workspace / ".state"
        state_root.mkdir(parents=True)
        env = Environment(workspace=workspace, mode="auto", state_root=state_root)

        _delegate(Action(
            response="", type="delegate",
            payload={"role": "researcher", "role_description": "Research papers", "objective": "Find X"},
        ), env, model=MockModel())

        _delegate(Action(
            response="", type="delegate",
            payload={"role": "reviewer", "role_description": "Code review", "objective": "Review Y"},
        ), env, model=MockModel())

        meta = _load_sub_agents_meta(state_root)
        assert len(meta) == 2
        roles = {e["role"] for e in meta}
        assert roles == {"researcher", "reviewer"}
        print("  Delegate meta registry updated OK")


def test_delegate_role_description_updates_meta():
    """Re-delegating with a new description updates the meta."""

    class MockModel:
        def generate(self, messages, *, chunk_callback=None):
            return '<output>{"response": "Done.", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        state_root = workspace / ".state"
        state_root.mkdir(parents=True)
        env = Environment(workspace=workspace, mode="auto", state_root=state_root)

        _delegate(Action(
            response="", type="delegate",
            payload={"role": "researcher", "role_description": "v1", "objective": "X"},
        ), env, model=MockModel())

        _delegate(Action(
            response="", type="delegate",
            payload={"role": "researcher", "role_description": "v2", "objective": "Y"},
        ), env, model=MockModel())

        meta = _load_sub_agents_meta(state_root)
        assert len(meta) == 1
        assert meta[0]["description"] == "v2"
        print("  Delegate role_description updates meta OK")


def test_delegate_without_state_root_still_works():
    """Delegation works without state_root (no persistence)."""

    class MockModel:
        def generate(self, messages, *, chunk_callback=None):
            return '<output>{"response": "Done.", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), mode="auto")
        assert env.state_root is None
        result = _delegate(Action(
            response="", type="delegate",
            payload={"role": "test", "objective": "Do something"},
        ), env, model=MockModel())
        assert "Done." in result
        print("  Delegate without state_root OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Delegation Guards ===")
    test_delegate_no_model()
    test_delegate_sub_agent_cannot_delegate()

    print("\n=== Basic Delegation ===")
    test_delegate_basic()
    test_delegate_shares_parent_workspace()

    print("\n=== Full Delegation Loop ===")
    test_full_delegation_loop()
    test_delegate_with_exec_in_sub_agent()

    print("\n=== Sub-Agent State Persistence ===")
    test_delegate_persists_and_restores_state()
    test_delegate_new_role_starts_fresh()
    test_delegate_meta_registry_updated()
    test_delegate_role_description_updates_meta()
    test_delegate_without_state_root_still_works()

    print("\n✅ All delegation tests passed!")
