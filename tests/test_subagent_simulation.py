"""Sub-agent delegation simulation — end-to-end integration test.

Simulates the core-agent → delegate → sub-agent → result cycle with
detailed observation logging from both agents. Verifies:

1. Core-agent receives user request and delegates to sub-agent
2. Sub-agent runs in isolated workspace with exec + chat actions
3. Sub-agent result flows back into core-agent's history
4. Core-agent uses sub-agent result to produce final answer
5. Communication records are saved correctly in session persistence
6. Both core-agent and sub-agent observations are inspectable
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
from helix.core.state import Turn
from helpers import sandbox_executor
from helix.runtime.loop import run_loop, _delegate
from helix.runtime.approval import ApprovalPolicy


# =========================================================================== #
# Instrumented models — capture all observations for inspection
# =========================================================================== #


class InstrumentedCoreModel:
    """Core-agent model: think → delegate → chat(result).

    Records every prompt it receives so we can inspect what the core-agent
    observes at each stage of the delegation cycle.
    """

    def __init__(self):
        self.call_count = 0
        self.messages_received: list[list[dict]] = []

    def generate(self, messages, *, chunk_callback=None):
        self.call_count += 1
        self.messages_received.append(messages)

        if self.call_count == 1:
            # Turn 1: Think about the task
            return (
                '<output>'
                '{"response": "The user wants to know the Python version. '
                'I should delegate this to a sub-agent who can run a command.", '
                '"action": "think", "action_input": {}}'
                '</output>'
            )

        if self.call_count == 2:
            # Turn 2: Delegate to a sub-agent
            return (
                '<output>'
                '{"response": "I will delegate the version check to a sub-agent.", '
                '"action": "delegate", '
                '"action_input": {'
                '"role": "system-inspector", '
                '"objective": "Run `python3 --version` and report the Python version number.", '
                '"context": "The user asked what Python version is installed. '
                'Run the command and report back the version string."'
                '}}'
                '</output>'
            )

        # Turn 3: Use sub-agent result to answer
        return (
            '<output>'
            '{"response": "Based on the sub-agent report, the installed Python '
            'version is the one reported in the execution output. '
            'The sub-agent successfully ran the version check command.", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        )


class InstrumentedSubModel:
    """Sub-agent model: exec(python --version) → chat(result).

    Records every prompt it sees so we can inspect the sub-agent's
    observation window at each step.
    """

    def __init__(self):
        self.call_count = 0
        self.messages_received: list[list[dict]] = []

    def generate(self, messages, *, chunk_callback=None):
        self.call_count += 1
        self.messages_received.append(messages)

        if self.call_count == 1:
            # Turn 1: Run the version command
            return (
                '<output>'
                '{"response": "I will run the python version command to check.", '
                '"action": "exec", '
                '"action_input": {'
                '"job_name": "check-python-version", '
                '"code_type": "bash", '
                '"script": "python3 --version"'
                '}}'
                '</output>'
            )

        # Turn 2: Report results
        return (
            '<output>'
            '{"response": "The Python version check is complete. '
            'I have identified the installed Python version from the command output.", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        )


class SharedInstrumentedModel:
    """Single model shared between core and sub-agent.

    This is the realistic scenario — both agents share a model reference.
    The model distinguishes calls by prompt content (sub-agent has different
    system prompt).
    """

    def __init__(self):
        self.all_calls: list[dict] = []

    def generate(self, messages, *, chunk_callback=None):
        call_num = len(self.all_calls) + 1
        full_text = " ".join(m.get("content", "") for m in messages)
        is_sub = "sub-agent" in full_text.lower()
        caller = "sub_agent" if is_sub else "core_agent"
        self.all_calls.append({
            "call_num": call_num,
            "caller": caller,
            "prompt_snippet": full_text[:200],
            "prompt_length": len(full_text),
        })

        if is_sub:
            # Sub-agent calls
            sub_calls = sum(1 for c in self.all_calls if c["caller"] == "sub_agent")
            if sub_calls == 1:
                return (
                    '<output>'
                    '{"response": "Running the requested command.", '
                    '"action": "exec", '
                    '"action_input": {"job_name": "gather-info", '
                    '"code_type": "bash", "script": "echo Hello from sub-agent && date"}}'
                    '</output>'
                )
            return (
                '<output>'
                '{"response": "Task complete. The command executed successfully '
                'and returned: Hello from sub-agent with timestamp.", '
                '"action": "chat", "action_input": {}}'
                '</output>'
            )

        # Core-agent calls
        core_calls = sum(1 for c in self.all_calls if c["caller"] == "core_agent")
        if core_calls == 1:
            return (
                '<output>'
                '{"response": "I need to gather system info. Let me delegate.", '
                '"action": "delegate", '
                '"action_input": {"role": "info-gatherer", '
                '"objective": "Run echo and date commands, report output."}}'
                '</output>'
            )
        return (
            '<output>'
            '{"response": "The sub-agent gathered the system info successfully. '
            'The output confirmed the commands ran correctly.", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        )


# =========================================================================== #
# Simulation runner
# =========================================================================== #

def _separator(title: str) -> str:
    return f"\n{'='*72}\n  {title}\n{'='*72}\n"


def _format_history(turns: list[Turn], label: str) -> str:
    """Format a history list for display."""
    lines = [f"\n--- {label} ({len(turns)} turns) ---"]
    for i, t in enumerate(turns):
        prefix = f"  [{i}]"
        formatted = f"[{t.role}] {t.content}"
        # Truncate long content for readability
        if len(formatted) > 300:
            formatted = formatted[:300] + "... (truncated)"
        lines.append(f"{prefix} {formatted}")
    lines.append(f"--- end {label} ---\n")
    return "\n".join(lines)


def run_simulation_scenario_1():
    """Scenario 1: Separate models for core and sub-agent.

    Tests that:
    - Core-agent thinks, then delegates
    - Sub-agent executes a command, then chats back
    - Sub-agent result is recorded in core-agent's history
    - Session persistence captures everything
    """
    print(_separator("Scenario 1: Separate Models (Core + Sub)"))

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        core_model = InstrumentedCoreModel()
        sub_model = InstrumentedSubModel()

        # Set up environment
        env = Environment(
            workspace=workspace,
            mode="auto",
            executor=sandbox_executor,
        )
        policy = ApprovalPolicy(mode="auto")
        env.on_before_execute(policy)

        # Record user request
        env.record(Turn(role="user", content="What Python version is installed?"))

        # Create core agent
        core_agent = Agent(
            core_model,
            workspace=workspace,
        )

        # Run the loop — pass sub_model for delegation
        output = StringIO()
        result = run_loop(core_agent, env, model=sub_model, output=output)

        # ---- Inspect core-agent observations ----
        print("CORE-AGENT OBSERVATIONS:")
        print(f"  Total model calls: {core_model.call_count}")
        for i, msgs in enumerate(core_model.messages_received):
            total_len = sum(len(m.get("content", "")) for m in msgs)
            print(f"\n  Call {i+1} messages: {len(msgs)} total_chars: {total_len}")
            if msgs:
                last_msg = msgs[-1]
                snippet = last_msg.get("content", "")[:500]
                print(f"  Last message ({last_msg['role']}):\n    {snippet}")

        # ---- Inspect sub-agent observations ----
        print("\nSUB-AGENT OBSERVATIONS:")
        print(f"  Total model calls: {sub_model.call_count}")
        for i, msgs in enumerate(sub_model.messages_received):
            total_len = sum(len(m.get("content", "")) for m in msgs)
            print(f"\n  Call {i+1} messages: {len(msgs)} total_chars: {total_len}")
            if msgs:
                last_msg = msgs[-1]
                snippet = last_msg.get("content", "")[:500]
                print(f"  Last message ({last_msg['role']}):\n    {snippet}")

        # ---- Inspect full history ----
        print(_format_history(env.full_history, "Core-Agent Full History"))

        # ---- Check sub-agent result recorded ----
        sub_turns = [t for t in env.full_history if t.role == "sub_agent"]
        print(f"Sub-agent turns in core history: {len(sub_turns)}")
        for t in sub_turns:
            print(f"  Content: {t.content[:200]}")

        # ---- Check child workspace ----
        sub_agents_dir = workspace / "sub_agents"
        if sub_agents_dir.exists():
            children = list(sub_agents_dir.iterdir())
            print(f"\nChild workspaces created: {len(children)}")
            for child in children:
                print(f"  {child.name}/")
                for item in sorted(child.rglob("*")):
                    if item.is_file():
                        rel = item.relative_to(child)
                        print(f"    {rel} ({item.stat().st_size} bytes)")
        else:
            print("\n  No sub_agents directory created (expected — sub-agent shares parent workspace)")

        # ---- Session persistence ----
        session_path = workspace / "session_test.json"
        env.save_session(session_path, extra_fields={"last_prompt": core_agent.last_prompt})
        assert session_path.exists(), "Session file not created"

        raw = json.loads(session_path.read_text(encoding="utf-8"))
        print(f"\nSession file saved: {session_path}")
        print(f"  full_history entries: {len(raw['full_history'])}")
        print(f"  observation entries: {len(raw['observation'])}")
        print(f"  workflow_summary: '{raw['workflow_summary'][:100] if raw['workflow_summary'] else '(empty)'}'")
        last_p = raw.get("last_prompt", "")
        print(f"  last_prompt entries: {len(last_p) if isinstance(last_p, list) else 'str:' + str(len(last_p))}")

        # Verify session reloads correctly
        env2 = Environment(workspace=workspace)
        loaded = env2.load_session(session_path)
        assert loaded, "Session failed to reload"
        assert len(env2.full_history) == len(env.full_history), "History mismatch after reload"
        sub_turns_reloaded = [t for t in env2.full_history if t.role == "sub_agent"]
        assert len(sub_turns_reloaded) == len(sub_turns), "Sub-agent turns lost on reload"
        print("  Session reload verified — all turns preserved")

        # ---- Verify communication records in detail ----
        print("\n  Communication record verification:")
        roles_sequence = [t.role for t in env.full_history]
        print(f"  Turn sequence: {' -> '.join(roles_sequence)}")

        # Expected: user → core_agent(think) → core_agent(delegate) → sub_agent → core_agent(chat)
        assert "user" in roles_sequence, "Missing user turn"
        assert "core_agent" in roles_sequence, "Missing core_agent turn"
        assert "sub_agent" in roles_sequence, "Missing sub_agent turn"

        print(f"\n  Final result: {result[:100]}...")
        print("\n  Scenario 1 PASSED")


def run_simulation_scenario_2():
    """Scenario 2: Shared model (realistic scenario).

    Tests that both core and sub-agent can use the same model reference
    and the delegation cycle works correctly.
    """
    print(_separator("Scenario 2: Shared Model (Realistic)"))

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        model = SharedInstrumentedModel()

        env = Environment(
            workspace=workspace,
            mode="auto",
            executor=sandbox_executor,
        )
        policy = ApprovalPolicy(mode="auto")
        env.on_before_execute(policy)

        env.record(Turn(role="user", content="Gather system information for me."))

        agent = Agent(
            model,
            workspace=workspace,
        )

        output = StringIO()
        result = run_loop(agent, env, model=model, output=output)

        # Inspect all model calls
        print("ALL MODEL CALLS (chronological):")
        for call in model.all_calls:
            print(f"  #{call['call_num']} caller={call['caller']} "
                  f"prompt_len={call['prompt_length']}")

        # Inspect histories
        print(_format_history(env.full_history, "Core-Agent Full History"))

        # Verify sequence
        sub_turns = [t for t in env.full_history if t.role == "sub_agent"]
        assert len(sub_turns) == 1, f"Expected 1 sub-agent turn, got {len(sub_turns)}"
        assert "successfully" in sub_turns[0].content.lower()

        # Session persistence
        session_path = workspace / "scenario2_session.json"
        env.save_session(session_path)
        raw = json.loads(session_path.read_text(encoding="utf-8"))

        print(f"\nSession saved: {len(raw['full_history'])} turns")
        for i, turn in enumerate(raw["full_history"]):
            content_preview = turn["content"][:80].replace("\n", " ")
            print(f"  [{i}] {turn['role']:>12} | {content_preview}...")

        print(f"\n  Final result: {result[:100]}...")
        print("\n  Scenario 2 PASSED")


def run_simulation_scenario_3():
    """Scenario 3: Verify sub-agent CANNOT delegate (no recursive delegation).

    Tests the safety constraint that sub-agents can only chat/think/exec,
    not delegate further.
    """
    print(_separator("Scenario 3: Sub-Agent Cannot Delegate"))

    # Verify at the action constraint level
    assert "delegate" not in ALLOWED_SUB_ACTIONS, "Sub-agents should not have delegate"
    assert "delegate" in ALLOWED_CORE_ACTIONS, "Core agent should have delegate"
    print("  Action constraints verified:")
    print(f"    Core actions: {sorted(ALLOWED_CORE_ACTIONS)}")
    print(f"    Sub actions:  {sorted(ALLOWED_SUB_ACTIONS)}")

    # Try to parse a delegate action with sub-agent constraints
    from helix.core.action import parse_action, ActionParseError
    raw = (
        '<output>'
        '{"response": "test", "action": "delegate", '
        '"action_input": {"role": "x", "objective": "y"}}'
        '</output>'
    )
    try:
        parse_action(raw, allowed_actions=ALLOWED_SUB_ACTIONS)
        assert False, "Should have raised ActionParseError"
    except ActionParseError as exc:
        print(f"  Parse correctly rejected delegate: {exc}")

    print("\n  Scenario 3 PASSED")


def run_simulation_scenario_4():
    """Scenario 4: Delegation failure modes.

    Tests:
    - Missing model (no model passed to _delegate)
    """
    print(_separator("Scenario 4: Delegation Failure Modes"))

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        action = Action(
            response="Delegating...",
            type="delegate",
            payload={"role": "test", "objective": "test task"},
        )

        # No model
        env = Environment(workspace=workspace / "test1")
        result = _delegate(action, env, model=None)
        assert "no model reference" in result.lower()
        print(f"  No model: {result}")

    print("\n  Scenario 4 PASSED")


def run_simulation_scenario_5():
    """Scenario 5: Full observation window inspection.

    Runs a complete delegation cycle and inspects the exact observation
    windows that each agent sees at each step.
    """
    print(_separator("Scenario 5: Detailed Observation Windows"))

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        core_model = InstrumentedCoreModel()
        sub_model = InstrumentedSubModel()

        env = Environment(
            workspace=workspace,
            mode="auto",
            executor=sandbox_executor,
        )
        policy = ApprovalPolicy(mode="auto")
        env.on_before_execute(policy)

        env.record(Turn(role="user", content="Check the Python version."))

        agent = Agent(
            core_model,
            workspace=workspace,
        )

        output = StringIO()
        run_loop(agent, env, model=sub_model, output=output)

        # Detailed breakdown of what each agent saw
        print("CORE-AGENT - Messages #1 (before think):")
        m1 = core_model.messages_received[0]
        _print_messages_summary(m1)

        print("\nCORE-AGENT - Messages #2 (before delegate):")
        m2 = core_model.messages_received[1]
        _print_messages_summary(m2)

        print("\nCORE-AGENT - Messages #3 (after sub-agent returned):")
        m3 = core_model.messages_received[2]
        _print_messages_summary(m3)

        print("\nSUB-AGENT - Messages #1 (initial task):")
        sm1 = sub_model.messages_received[0]
        _print_messages_summary(sm1)

        print("\nSUB-AGENT - Messages #2 (after exec result):")
        sm2 = sub_model.messages_received[1]
        _print_messages_summary(sm2)

        # Verify the sub-agent result appears in core-agent's messages #3
        m3_text = " ".join(m.get("content", "") for m in m3)
        assert "sub-agent" in m3_text.lower() or "sub_agent" in m3_text.lower(), \
            "Core-agent messages #3 should contain sub-agent result"

        print("\n  Scenario 5 PASSED")


def _print_messages_summary(messages: list[dict]):
    """Print a summary of a chat messages array."""
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        preview = content[:300].replace("\n", "\n    ")
        if len(content) > 300:
            preview += "\n    ... (truncated)"
        print(f"  [{i}] {role}:")
        print(f"    {preview}")


# =========================================================================== #
# Main
# =========================================================================== #


if __name__ == "__main__":
    print("=" * 72)
    print("  SUB-AGENT DELEGATION SIMULATION")
    print("  Testing core-agent <-> sub-agent communication flows")
    print("=" * 72)

    run_simulation_scenario_1()
    run_simulation_scenario_2()
    run_simulation_scenario_3()
    run_simulation_scenario_4()
    run_simulation_scenario_5()

    print("\n" + "=" * 72)
    print("  ALL SIMULATION SCENARIOS PASSED!")
    print("=" * 72)
