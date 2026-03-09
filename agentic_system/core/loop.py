"""Universal agent loop — the heart of the framework.

This single loop is used by both core-agents and sub-agents.
The entire orchestration reduces to: state → agent → action → environment → observation.
"""

from __future__ import annotations

import json
import sys
from typing import TextIO, Callable

from .action import Action, ActionParseError
from .agent import Agent
from .environment import Environment, CompactionError
from .state import Turn


DEFAULT_MAX_TURNS = 60
DEFAULT_MAX_RETRIES = 3


def run_loop(
    agent: Agent,
    env: Environment,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    output: TextIO = sys.stdout,
    on_turn_start: Callable[[str], None] | None = None,
    on_turn_end: Callable[[], None] | None = None,
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
        on_turn_end: Optional callback fired after agent output is finalized.
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
            _print(output, f"\nruntime> {msg}\n")
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
            if on_turn_end:
                on_turn_end()
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
                _print(output, f"\nruntime> {msg}\n")
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
            _print(output, f"runtime> Executing: {action.payload.get('job_name', 'unnamed')}...\n")
            observation = env.execute(action)
            env.record(observation)

        elif action.type == "delegate":
            _print(output, f"runtime> Delegating to sub-agent: {action.payload.get('role', 'unknown')}...\n")
            result = env.delegate(action)
            env.record(Turn(
                role="sub-agent",
                content=result,
            ))

    # Turn limit reached
    msg = "Loop ended: maximum turns reached."
    _print(output, f"\nruntime> {msg}\n")
    return msg

# --------------------------------------------------------------------------- #
def _print(output: TextIO, text: str = "", *, action: Action | None = None) -> None:
    """Print to the output stream immediately (unbuffered).
    
    If an action is provided, prints its action type and input details
    (the response text is assumed to be already streamed).
    """
    if "\n" not in text and action is None:
        pass
    elif text:
        output.write(text)
        output.flush()

    if action:
        output.write(f"[next_action] {action.type}\n")
        
        if action.type == "exec" and action.payload:
            lines = ["[action_input]"]
            for key in ("job_name", "code_type", "script_path", "script"):
                value = action.payload.get(key)
                if value:
                    text_val = str(value)
                    if key == "script" and len(text_val) > 200:
                        text_val = text_val[:200] + "..."
                    lines.append(f"  {key}: {text_val}")
            args = action.payload.get("script_args")
            if args:
                lines.append(f"  script_args: {args}")
            if len(lines) > 1:
                output.write("\n".join(lines) + "\n")
                
        elif action.type == "delegate" and action.payload:
            lines = ["[action_input]"]
            for key in ("role", "objective"):
                value = action.payload.get(key)
                if value:
                    lines.append(f"  {key}: {value}")
            if len(lines) > 1:
                output.write("\n".join(lines) + "\n")
        
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
        for key in ("job_name", "code_type", "script_path", "script"):
            value = action.payload.get(key)
            if value:
                text = str(value)
                if key == "script" and len(text) > 200:
                    text = text[:200] + "..."
                lines.append(f"  {key}: {text}")
        args = action.payload.get("script_args")
        if args:
            lines.append(f"  script_args: {args}")
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
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"  context: {text}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n".join(parts)
