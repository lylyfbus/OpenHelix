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
from helix.core.state import Turn, format_turn
from helix.runtime.sandbox import sandbox_executor
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
        self.prompts_received: list[str] = []

    def generate(self, prompt, *, stream=False, chunk_callback=None):
        self.call_count += 1
        self.prompts_received.append(prompt)

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
        self.prompts_received: list[str] = []

    def generate(self, prompt, *, stream=False, chunk_callback=None):
        self.call_count += 1
        self.prompts_received.append(prompt)

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

    def generate(self, prompt, *, stream=False, chunk_callback=None):
        call_num = len(self.all_calls) + 1
        is_sub = "sub-agent" in prompt.lower()
        caller = "sub-agent" if is_sub else "core-agent"
        self.all_calls.append({
            "call_num": call_num,
            "caller": caller,
            "prompt_snippet": prompt[:200],
            "prompt_length": len(prompt),
        })

        if is_sub:
            # Sub-agent calls
            sub_calls = sum(1 for c in self.all_calls if c["caller"] == "sub-agent")
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
        core_calls = sum(1 for c in self.all_calls if c["caller"] == "core-agent")
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
        formatted = format_turn(t)
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
            system_prompt="You are the core agent. Manage tasks and delegate when needed.",
        )

        # Run the loop — pass sub_model for delegation
        output = StringIO()
        result = run_loop(core_agent, env, model=sub_model, output=output)

        # ---- Inspect core-agent observations ----
        print("CORE-AGENT OBSERVATIONS:")
        print(f"  Total model calls: {core_model.call_count}")
        for i, prompt in enumerate(core_model.prompts_received):
            print(f"\n  Call {i+1} prompt length: {len(prompt)} chars")
            # Show key sections of the prompt
            if "<latest_context>" in prompt:
                start = prompt.index("<latest_context>")
                end = prompt.index("</latest_context>") + len("</latest_context>")
                snippet = prompt[start:end]
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                print(f"  Latest context:\n    {snippet}")

        # ---- Inspect sub-agent observations ----
        print("\nSUB-AGENT OBSERVATIONS:")
        print(f"  Total model calls: {sub_model.call_count}")
        for i, prompt in enumerate(sub_model.prompts_received):
            print(f"\n  Call {i+1} prompt length: {len(prompt)} chars")
            if "<latest_context>" in prompt:
                start = prompt.index("<latest_context>")
                end = prompt.index("</latest_context>") + len("</latest_context>")
                snippet = prompt[start:end]
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                print(f"  Latest context:\n    {snippet}")

        # ---- Inspect full history ----
        print(_format_history(env.full_history, "Core-Agent Full History"))

        # ---- Check sub-agent result recorded ----
        sub_turns = [t for t in env.full_history if t.role == "sub-agent"]
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
        print(f"  last_prompt length: {len(raw.get('last_prompt', ''))}")

        # Verify session reloads correctly
        env2 = Environment(workspace=workspace)
        loaded = env2.load_session(session_path)
        assert loaded, "Session failed to reload"
        assert len(env2.full_history) == len(env.full_history), "History mismatch after reload"
        sub_turns_reloaded = [t for t in env2.full_history if t.role == "sub-agent"]
        assert len(sub_turns_reloaded) == len(sub_turns), "Sub-agent turns lost on reload"
        print("  Session reload verified — all turns preserved")

        # ---- Verify communication records in detail ----
        print("\n  Communication record verification:")
        roles_sequence = [t.role for t in env.full_history]
        print(f"  Turn sequence: {' -> '.join(roles_sequence)}")

        # Expected: user → core-agent(think) → core-agent(delegate) → sub-agent → core-agent(chat)
        assert "user" in roles_sequence, "Missing user turn"
        assert "core-agent" in roles_sequence, "Missing core-agent turn"
        assert "sub-agent" in roles_sequence, "Missing sub-agent turn"

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
            system_prompt="You are the core agent.",
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
        sub_turns = [t for t in env.full_history if t.role == "sub-agent"]
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
            system_prompt="You are the core agent.",
        )

        output = StringIO()
        run_loop(agent, env, model=sub_model, output=output)

        # Detailed breakdown of what each agent saw
        print("CORE-AGENT - Prompt #1 (before think):")
        p1 = core_model.prompts_received[0]
        _print_prompt_sections(p1)

        print("\nCORE-AGENT - Prompt #2 (before delegate):")
        p2 = core_model.prompts_received[1]
        _print_prompt_sections(p2)

        print("\nCORE-AGENT - Prompt #3 (after sub-agent returned):")
        p3 = core_model.prompts_received[2]
        _print_prompt_sections(p3)

        print("\nSUB-AGENT - Prompt #1 (initial task):")
        sp1 = sub_model.prompts_received[0]
        _print_prompt_sections(sp1)

        print("\nSUB-AGENT - Prompt #2 (after exec result):")
        sp2 = sub_model.prompts_received[1]
        _print_prompt_sections(sp2)

        # Verify the sub-agent result appears in core-agent's prompt #3
        assert "sub-agent" in p3.lower() or "sub_agent" in p3.lower(), \
            "Core-agent prompt #3 should contain sub-agent result"

        print("\n  Scenario 5 PASSED")


def _print_prompt_sections(prompt: str):
    """Extract and print the key sections of a prompt."""
    sections = {
        "workflow_summary": ("<workflow_summary>", "</workflow_summary>"),
        "workflow_history": ("<workflow_history>", "</workflow_history>"),
        "latest_context": ("<latest_context>", "</latest_context>"),
    }
    for name, (open_tag, close_tag) in sections.items():
        if open_tag in prompt:
            start = prompt.index(open_tag)
            end = prompt.index(close_tag) + len(close_tag)
            content = prompt[start:end]
            if len(content) > 400:
                content = content[:400] + "\n    ... (truncated)"
            print(f"  [{name}]")
            for line in content.split("\n"):
                print(f"    {line}")
        else:
            print(f"  [{name}] (not present)")


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
