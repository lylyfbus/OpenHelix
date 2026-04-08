"""Phase 1 verification tests for the core RL-inspired abstractions."""

import re
import sys
import tempfile
from http.client import RemoteDisconnected
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.state import State, Turn
from helix.core.action import (
    Action,
    parse_action,
    ActionParseError,
    ALLOWED_CORE_ACTIONS,
    ALLOWED_SUB_ACTIONS,
)
from helix.core.agent import Agent
from helix.core.compactor import Compactor
from helix.core.environment import Environment
from helix.runtime.display import TURN_SEPARATOR
from helix.runtime.loop import run_loop
from helix.runtime.approval import ApprovalPolicy


def test_turn_creation():
    t = Turn(role="user", content="hello")
    assert t.role == "user"
    assert t.content == "hello"
    assert t.timestamp  # auto-generated
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", t.timestamp)
    print("  Turn creation OK")


def test_state_creation():
    t = Turn(role="user", content="hello")
    s = State(observation=[t], workflow_summary="")
    assert s.observation == [t]
    print("  State creation OK")


def test_parse_chat():
    raw = '<output>{"response": "Hi!", "action": "chat", "action_input": {}}</output>'
    a = parse_action(raw)
    assert a.type == "chat"
    assert a.response == "Hi!"
    assert a.payload == {}
    print("  Parse chat OK")


def test_parse_think():
    raw = '<output>{"response": "Thinking...", "action": "think", "action_input": {}}</output>'
    a = parse_action(raw)
    assert a.type == "think"
    print("  Parse think OK")


def test_parse_exec():
    raw = '<output>{"response": "Running", "action": "exec", "action_input": {"job_name": "test", "code_type": "bash", "script": "echo hello"}}</output>'
    a = parse_action(raw)
    assert a.type == "exec"
    assert a.payload["code_type"] == "bash"
    assert a.payload["script"] == "echo hello"
    print("  Parse exec OK")


def test_parse_exec_script_args_array():
    raw = '<output>{"response": "Running", "action": "exec", "action_input": {"job_name": "test", "code_type": "python", "script_path": "skills/x.py", "script_args": ["--query", "hello", "--limit", "5"]}}</output>'
    a = parse_action(raw)
    assert a.type == "exec"
    assert a.payload["script_args"] == ["--query", "hello", "--limit", "5"]
    print("  Parse exec script_args array OK")


def test_parse_exec_script_args_string():
    raw = '<output>{"response": "Running", "action": "exec", "action_input": {"job_name": "test", "code_type": "python", "script_path": "skills/x.py", "script_args": "--query \\"hello\\" --limit 5"}}</output>'
    a = parse_action(raw)
    assert a.type == "exec"
    assert a.payload["script_args"] == '--query "hello" --limit 5'
    print("  Parse exec script_args string OK")


def test_parse_delegate():
    raw = '<output>{"response": "Delegating", "action": "delegate", "action_input": {"role": "researcher", "objective": "Find info"}}</output>'
    a = parse_action(raw)
    assert a.type == "delegate"
    assert a.payload["role"] == "researcher"
    print("  Parse delegate OK")



def test_parse_error_missing_tags():
    try:
        parse_action("no tags here")
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (missing tags) OK")


def test_parse_error_invalid_json():
    try:
        parse_action("<output>not json</output>")
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (invalid JSON) OK")


def test_parse_error_invalid_action():
    raw = '<output>{"response": "Hi!", "action": "fly", "action_input": {}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (invalid action) OK")


def test_parse_error_exec_missing_script():
    raw = '<output>{"response": "R", "action": "exec", "action_input": {"code_type": "bash"}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec missing script) OK")


def test_parse_error_exec_both_script_and_path():
    raw = '<output>{"response": "R", "action": "exec", "action_input": {"code_type": "bash", "script": "echo x", "script_path": "/foo"}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec both script+path) OK")


def test_parse_error_exec_invalid_script_args_type():
    raw = '<output>{"response": "R", "action": "exec", "action_input": {"code_type": "python", "script_path": "skills/x.py", "script_args": {"bad": true}}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec invalid script_args type) OK")


def test_parse_error_exec_script_args_with_script():
    raw = '<output>{"response": "R", "action": "exec", "action_input": {"code_type": "bash", "script": "echo x", "script_args": ["--bad"]}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec script_args with script) OK")


def test_parse_error_delegate_missing_role():
    raw = '<output>{"response": "D", "action": "delegate", "action_input": {"objective": "o"}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (delegate missing role) OK")


def test_sub_agent_cannot_delegate():
    raw = '<output>{"response": "D", "action": "delegate", "action_input": {"role": "r", "objective": "o"}}</output>'
    try:
        parse_action(raw, allowed_actions=ALLOWED_SUB_ACTIONS)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Sub-agent delegate denied OK")



def test_environment_dual_history():
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hello"))
        env.record(Turn(role="agent", content="hi back"))
        assert len(env.full_history) == 2
        assert len(env.observation) == 2
        print("  Dual history OK")


def test_environment_build_state():
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hello"))
        state = env.build_state()
        assert len(state.observation) == 1
        print("  build_state OK")


def test_environment_compaction():
    class CompactorModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            return "## Session Goal\nTest session\n## Current Status\nCompacted."

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), token_limit=200, keep_last_k=2,
                          compactor=Compactor(CompactorModel()))
        # Add many turns to exceed budget
        for i in range(20):
            env.record(Turn(role="agent", content=f"Turn {i} " + "x" * 100))
        assert len(env.full_history) == 20  # full_history never compacted
        state = env.build_state()
        # After compaction: only the 2 recent turns remain in observation.
        assert len(state.observation) == 2
        assert env.workflow_summary  # summary was generated
        assert len(env.full_history) == 20  # full_history still intact
        print("  Compaction OK")


def test_agent_prompt_keeps_summary_separate_from_recent_history():
    state = State(
        observation=[
            Turn(role="runtime", content="Job succeeded."),
            Turn(role="user", content="what now?"),
        ],
        workflow_summary="## Summary\nCompacted",
    )
    agent = Agent(type("MockModel", (), {"generate": lambda self, *args, **kwargs: ""})(), system_prompt="test")

    prompt = agent._build_prompt(state)

    assert "<workflow_summary>\n## Summary\nCompacted\n</workflow_summary>" in prompt
    assert "runtime> Job succeeded." in prompt
    assert "what now?" in prompt
    print("  Prompt keeps summary separate from recent history OK")


def test_environment_compaction_error():
    """Compaction raises CompactionError when no compactor is available."""
    from helix.core.compactor import CompactionError

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), token_limit=200, keep_last_k=2)
        # No compactor set — compaction should fail
        for i in range(20):
            env.record(Turn(role="agent", content=f"Turn {i} " + "x" * 100))
        try:
            env.build_state()
            assert False, "Should have raised CompactionError"
        except CompactionError:
            pass
        print("  Compaction error (no model) OK")


def test_environment_persistence():
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hello"))
        env.record(Turn(role="agent", content="world"))
        env.workflow_summary = "test summary"

        sp = Path(td) / "session.json"
        env.save_session(sp)

        env2 = Environment(workspace=Path(td))
        loaded = env2.load_session(sp)
        assert loaded
        assert len(env2.full_history) == 2
        assert len(env2.observation) == 2
        assert env2.workflow_summary == "test summary"
        assert env2.full_history[0].content == "hello"
        print("  Persistence OK")


def test_run_loop_compaction_failure_is_ui_only():
    """Compaction failures should pause the session without recording a runtime turn."""

    class FailingCompactionModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, *, stream=False, chunk_callback=None):
            self.calls += 1
            raise RemoteDisconnected("compactor closed connection")

    class UnusedAgentModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            assert False, "Agent model should not be called when compaction fails"

    with tempfile.TemporaryDirectory() as td:
        compactor_model = FailingCompactionModel()
        env = Environment(workspace=Path(td), token_limit=10, keep_last_k=1,
                          compactor=Compactor(compactor_model))
        env.record(Turn(role="user", content="x" * 120))
        env.record(Turn(role="runtime", content="y" * 120))

        agent = Agent(UnusedAgentModel(), system_prompt="test")
        captured = StringIO()
        result = run_loop(agent, env, output=captured)

        assert "compaction failed" in result.lower()
        assert compactor_model.calls == 3
        assert f"{TURN_SEPARATOR}\nruntime> Session paused:" in captured.getvalue()
        assert TURN_SEPARATOR in captured.getvalue()
        assert len(env.full_history) == 2
        assert len(env.observation) == 2
        assert all("compaction failed" not in t.content.lower() for t in env.full_history)
        print("  run_loop (compaction failure UI-only) OK")



def test_run_loop_chat():
    """Test that run_loop correctly handles a mock agent returning chat."""

    class MockModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            return '<output>{"response": "Hello user!", "action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hi"))
        agent = Agent(MockModel(), system_prompt="You are helpful.")
        result = run_loop(agent, env, output=sys.stderr)
        assert result == "Hello user!"
        assert len(env.full_history) == 2  # user + agent
        print("  run_loop (chat) OK")


def test_run_loop_think_then_chat():
    """Test that run_loop handles think -> chat sequence."""
    call_count = [0]

    class MockModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return '<output>{"response": "Let me think...", "action": "think", "action_input": {}}</output>'
            return '<output>{"response": "Done thinking!", "action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="think first"))
        agent = Agent(MockModel(), system_prompt="test")
        result = run_loop(agent, env, output=sys.stderr)
        assert result == "Done thinking!"
        assert call_count[0] == 2
        # user + think_response + chat_response = 3
        assert len(env.full_history) == 3
        print("  run_loop (think→chat) OK")


def test_run_loop_parse_retry():
    """Test that run_loop retries on parse failure then succeeds."""
    call_count = [0]

    class MockModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "bad output no tags"
            return '<output>{"response": "Fixed!", "action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="test"))
        agent = Agent(MockModel(), system_prompt="test")
        result = run_loop(agent, env, output=sys.stderr)
        assert result == "Fixed!"
        assert call_count[0] == 2
        print("  run_loop (parse retry) OK")


def test_run_loop_max_retries():
    """Test that run_loop stops after max parse failures."""

    class MockModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            return "always bad"

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="test"))
        agent = Agent(MockModel(), system_prompt="test")
        result = run_loop(agent, env, max_retries=2, output=sys.stderr)
        assert "parse failures" in result.lower()
        print("  run_loop (max retries) OK")


def test_run_loop_exec_denied_returns_control():
    """Test that approval denial records evidence and returns to requester."""
    call_count = [0]

    class MockModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '<output>'
                    '{"response": "Checking status.", '
                    '"action": "exec", '
                    '"action_input": {"job_name": "check-status", "code_type": "bash", "script": "echo x"}}'
                    '</output>'
                )
            return (
                '<output>'
                '{"response": "This should not run.", "action": "chat", "action_input": {}}'
                '</output>'
            )

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), mode="controlled")
        env.on_before_execute(ApprovalPolicy(mode="controlled", prompt=lambda _prompt: "n"))
        env.record(Turn(role="user", content="check status"))
        agent = Agent(MockModel(), system_prompt="test")
        result = run_loop(agent, env, output=sys.stderr)

        assert "denied by requester" in result.lower()
        assert call_count[0] == 1
        runtime_turns = [t for t in env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "denied by requester" in runtime_turns[0].content.lower()
        print("  run_loop (exec denied returns control) OK")


def test_run_loop_exec_cancelled_returns_control():
    """Test that approval cancel records evidence and returns to requester."""
    call_count = [0]

    class MockModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '<output>'
                    '{"response": "Checking status.", '
                    '"action": "exec", '
                    '"action_input": {"job_name": "check-status", "code_type": "bash", "script": "echo x"}}'
                    '</output>'
                )
            return (
                '<output>'
                '{"response": "This should not run.", "action": "chat", "action_input": {}}'
                '</output>'
            )

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), mode="controlled")
        env.on_before_execute(ApprovalPolicy(
            mode="controlled",
            prompt=lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
        ))
        env.record(Turn(role="user", content="check status"))
        agent = Agent(MockModel(), system_prompt="test")
        result = run_loop(agent, env, output=sys.stderr)

        assert "cancelled during approval prompt" in result.lower()
        assert call_count[0] == 1
        runtime_turns = [t for t in env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "cancelled during approval prompt" in runtime_turns[0].content.lower()
        print("  run_loop (exec cancelled returns control) OK")


if __name__ == "__main__":
    print("=== State & Turn ===")
    test_turn_creation()
    test_state_creation()

    print("\n=== Action Parsing ===")
    test_parse_chat()
    test_parse_think()
    test_parse_exec()
    test_parse_delegate()

    print("\n=== Parse Errors ===")
    test_parse_error_missing_tags()
    test_parse_error_invalid_json()
    test_parse_error_invalid_action()
    test_parse_error_exec_missing_script()
    test_parse_error_exec_both_script_and_path()
    test_parse_error_delegate_missing_role()
    test_sub_agent_cannot_delegate()



    print("\n=== Environment ===")
    test_environment_dual_history()
    test_environment_build_state()
    test_environment_compaction()
    test_agent_prompt_keeps_summary_separate_from_recent_history()
    test_environment_compaction_error()
    test_environment_persistence()

    print("\n=== Universal Loop ===")
    test_run_loop_chat()
    test_run_loop_think_then_chat()
    test_run_loop_parse_retry()
    test_run_loop_max_retries()
    test_run_loop_exec_denied_returns_control()
    test_run_loop_exec_cancelled_returns_control()

    print("\n✅ All Phase 1 tests passed!")
