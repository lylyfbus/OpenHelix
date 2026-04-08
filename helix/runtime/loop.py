"""Universal agent loop — the heart of the framework.

This single loop is used by both core-agents and sub-agents.
The entire orchestration reduces to: state → agent → action → environment → observation.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO, Callable

from ..core.action import Action, ActionParseError, ALLOWED_SUB_ACTIONS
from ..core.agent import Agent
from ..core.environment import Environment, CompactionError, ExecutionInterrupted
from ..core.state import Turn
from .display import iter_exec_payload_items, write_framed_text


DEFAULT_MAX_TURNS = 9999999
DEFAULT_MAX_RETRIES = 10


def run_loop(
    agent: Agent,
    env: Environment,
    *,
    model: Any = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    output: TextIO = sys.stdout,
    on_turn_start: Callable[[str], None] | None = None,
    on_turn_end: Callable[[], None] | None = None,
    on_turn_error: Callable[[], None] | None = None,
    on_token_chunk: Callable[[str], None] | None = None,
) -> str:
    """Universal agent loop.

    Used identically by core-agents and sub-agents.
    Runs until the agent emits a "chat" action (returning control to the
    caller) or the turn limit is reached.

    Args:
        agent: The LLM-based agent.
        env: The sandbox environment.
        max_turns: Maximum loop iterations before forced stop.
        max_retries: Maximum consecutive parse failures before forced stop.
        output: Stream for runtime status messages.
        on_turn_start: Optional callback fired before the agent acts.
        on_turn_end: Optional callback fired after a valid agent output is finalized.
        on_turn_error: Optional callback fired after a parse-failed attempt.
        on_token_chunk: Optional callback for streaming agent responses.

    Returns:
        The agent's final response text.
    """
    consecutive_failures = 0

    for turn_num in range(max_turns):
        # 1. Build state from environment (compacts if needed)
        try:
            state = env.build_state()
        except CompactionError as exc:
            msg = (
                f"Session paused: context window is full and compaction failed ({exc}). "
                f"Please start a new session or reduce context."
            )
            _print(output, f"runtime> {msg}\n", add_separator=True)
            return msg

        # 2. Agent decides
        if on_turn_start:
            on_turn_start(agent.name)
        try:
            action = agent.act(
                state,
                stream=bool(on_token_chunk),
                chunk_callback=on_token_chunk,
            )
            consecutive_failures = 0
        except ActionParseError as exc:
            if on_turn_error:
                on_turn_error()
            consecutive_failures += 1
            env.record(Turn(
                role="runtime",
                content=(
                    f"Output parse error (attempt {consecutive_failures}/{max_retries}): "
                    f"{exc}. Please respond with valid <output>...</output> JSON."
                ),
            ))
            if consecutive_failures >= max_retries:
                msg = "Loop ended: too many consecutive parse failures."
                _print(output, f"runtime> {msg}\n", add_separator=True)
                return msg
            continue

        # 3. Finalize streaming display (adds trailing newline)
        if on_turn_end:
            on_turn_end()

        # 4. Print action details (response was already streamed)
        _print(output, action=action)

        # 5. Record agent turn with full action details
        record_content = _format_agent_record(action)
        env.record(Turn(
            role=agent.name,
            content=record_content,
        ))

        # 6. Execute action
        if action.type == "chat":
            # Done — return to caller
            return action.response

        if action.type == "think":
            # Loop continues — response already recorded
            pass

        elif action.type == "exec":
            _print(
                output,
                f"runtime> Executing: {action.payload.get('job_name', 'unnamed')}...\n",
                add_separator=True,
            )
            try:
                observation = env.execute(action)
            except ExecutionInterrupted as exc:
                env.record(exc.observation)
                _print(output, f"runtime> {exc.observation.content}\n", add_separator=True)
                return exc.observation.content
            env.record(observation)

        elif action.type == "delegate":
            _print(
                output,
                f"runtime> Delegating to sub-agent: {action.payload.get('role', 'unknown')}...\n",
                add_separator=True,
            )
            result = _delegate(action, env, model)
            env.record(Turn(
                role="sub-agent",
                content=result,
            ))

    # Turn limit reached
    msg = "Loop ended: maximum turns reached."
    _print(output, f"runtime> {msg}\n", add_separator=True)
    return msg

# --------------------------------------------------------------------------- #
# Sub-agent delegation
# --------------------------------------------------------------------------- #


def _delegate(action: Action, env: Environment, model: Any) -> str:
    """Spawn a sub-agent to handle a delegated task.

    Creates an isolated Environment sharing the parent's workspace, executor,
    compactor, and approval hook, then runs a recursive loop.
    """
    task = action.payload

    if model is None:
        return "Delegation failed: no model reference. Pass model= to run_loop()."

    # Build sub-environment sharing parent's infrastructure
    sub_env = Environment(
        workspace=env.workspace,
        mode=env.mode,
        token_limit=env.token_limit,
        keep_last_k=env.keep_last_k,
        executor=env._executor,
        compactor=env._compactor,
    )
    sub_env._on_before_execute = env._on_before_execute

    # Seed sub-agent with task objective
    role = task.get("role", "assistant")
    objective = task.get("objective", "")
    context = task.get("context", "")
    seed_content = objective
    if context:
        seed_content += f"\n\nContext:\n{context}"
    sub_env.record(Turn(role="core-agent", content=seed_content))

    # Build sub-agent (cannot delegate further)
    sub_agent = Agent(
        model,
        name="sub-agent",
        workspace=env.workspace,
        role="sub_agent",
        sub_agent_role=role,
        allowed_actions=ALLOWED_SUB_ACTIONS,
    )

    return run_loop(sub_agent, sub_env)


# --------------------------------------------------------------------------- #
def _print(output: TextIO, text: str = "", *, action: Action | None = None, add_separator: bool = False) -> None:
    """Print to the output stream immediately (unbuffered).

    Action metadata is not printed to the requester-facing UI.
    The full action details are still recorded into workflow history.
    """
    if "\n" not in text and action is None:
        pass
    elif text:
        if add_separator:
            write_framed_text(text, output)
        else:
            output.write(text)
            output.flush()

# --------------------------------------------------------------------------- #
# Agent record formatting
# --------------------------------------------------------------------------- #


def _format_agent_record(action: Action) -> str:
    """Format agent response + action details into a readable record.

    This is what the LLM will see in workflow_history, so it must be clear
    enough for the LLM to trace its own decisions.

    Examples:
        I'll search the project structure.
        [next_action] exec
        [action_input]
          job_name: list-project-files
          code_type: bash
          script: find . -type f

        Let me think about the best approach.
        [next_action] think

        Here are the results you asked for.
        [next_action] chat

        I'll delegate the research to a sub-agent.
        [next_action] delegate
        [action_input]
          role: researcher
          objective: Find papers on RLHF
    """
    parts = [action.response, f"[next_action] {action.type}"]

    if action.type == "exec" and action.payload:
        lines = ["[action_input]"]
        for key, value in iter_exec_payload_items(action.payload):
            text = str(value)
            if "\n" in text:
                lines.append(f"  {key}:")
                lines.extend(f"    {row}" for row in text.split("\n"))
            else:
                lines.append(f"  {key}: {text}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    elif action.type == "delegate" and action.payload:
        lines = ["[action_input]"]
        for key in ("role", "objective"):
            value = action.payload.get(key)
            if value:
                lines.append(f"  {key}: {value}")
        context = action.payload.get("context")
        if context:
            text = str(context)
            if "\n" in text:
                lines.append(f"  context:")
                lines.extend(f"    {row}" for row in text.split("\n"))
            else:
                lines.append(f"  context: {text}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n".join(parts)
