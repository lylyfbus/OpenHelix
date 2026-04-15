"""Universal agent loop — the heart of the framework.

This single loop is used by both core-agents and sub-agents.
The entire orchestration reduces to: state → agent → action → environment → observation.
"""

from __future__ import annotations

import random
import sys
import time
from typing import Any, TextIO, Callable

from ..core.action import Action, ActionParseError, ALLOWED_SUB_ACTIONS
from ..core.agent import Agent
from ..core.environment import Environment, CompactionError, ExecutionInterrupted
from ..core.state import State, Turn
from ..providers.openai_compat import LLMTransientError
from . import sub_agent_meta
from .display import iter_exec_payload_items, write_runtime


DEFAULT_MAX_TURNS = 9999999
DEFAULT_LLM_RETRIES = 20
DEFAULT_PARSE_RETRIES = 20
DEFAULT_COMPACTION_RETRIES = 20
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 60.0


def run_loop(
    agent: Agent,
    env: Environment,
    *,
    model: Any = None,
    max_turns: int = DEFAULT_MAX_TURNS,
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

    Error handling contract — three error classes, each with its own policy:
      * ``LLMTransientError`` and ``CompactionError`` are *runtime-handled*:
        retried at the runtime level with exponential backoff; if retries are
        exhausted, the error is recorded as a ``runtime`` Turn and the loop
        returns to the requester (user or parent core-agent) without feeding
        the error back to the agent. The agent cannot influence infrastructure
        issues, so retry-and-return is the pragmatic default.
      * ``ActionParseError`` is *agent-handled*: the first failure is recorded
        so the agent sees it on the next turn and can adjust its output. After
        ``DEFAULT_PARSE_RETRIES`` consecutive failures the loop bails out
        (stuck-state detection, independent of ``max_turns``).
      * ``ExecutionInterrupted`` is *pass-through*: the observation is recorded
        and the loop returns immediately so the requester can react.

    Args:
        agent: The LLM-based agent.
        env: The sandbox environment.
        max_turns: Total iteration budget for this session.
        output: Stream for runtime status messages.
        on_turn_start: Optional callback fired before the agent acts.
        on_turn_end: Optional callback fired after a valid agent output is finalized.
        on_turn_error: Optional callback fired after a parse-failed attempt.
        on_token_chunk: Optional callback for streaming agent responses.

    Returns:
        The agent's final response text.
    """
    consecutive_parse_failures = 0

    for _ in range(max_turns):
        # 1. Build state from environment (retries compaction internally)
        try:
            state = _build_state_with_retry(env, output=output)
        except CompactionError as exc:
            msg = (
                f"Compaction failed after {DEFAULT_COMPACTION_RETRIES} attempts: {exc}. "
                "Returning to requester."
            )
            _print(output, f"runtime> {msg}\n")
            env.record(Turn(role="runtime", content=msg))
            return msg

        # 2. Agent decides (retries transient LLM errors internally)
        if on_turn_start:
            on_turn_start(agent.role)
        try:
            action = _act_with_retry(
                agent, state,
                chunk_callback=on_token_chunk,
                on_turn_error=on_turn_error,
                output=output,
            )
            consecutive_parse_failures = 0
        except LLMTransientError as exc:
            msg = (
                f"LLM provider error after {DEFAULT_LLM_RETRIES} attempts: {exc}. "
                "Returning to requester."
            )
            _print(output, f"runtime> {msg}\n")
            env.record(Turn(role="runtime", content=msg))
            return msg
        except ActionParseError as exc:
            if on_turn_error:
                on_turn_error()
            consecutive_parse_failures += 1
            _print(
                output,
                f"runtime> Output parse error (attempt "
                f"{consecutive_parse_failures}/{DEFAULT_PARSE_RETRIES}): {exc}\n",
            )
            # Record only the first parse error to avoid flooding the observation window.
            if consecutive_parse_failures == 1:
                env.record(Turn(
                    role="runtime",
                    content=f"Output parse error: {exc}. Please respond with valid <output>...</output> JSON.",
                ))
            if consecutive_parse_failures >= DEFAULT_PARSE_RETRIES:
                msg = (
                    f"Loop ended: {DEFAULT_PARSE_RETRIES} consecutive parse failures. "
                    "Returning to requester."
                )
                _print(output, f"runtime> {msg}\n")
                env.record(Turn(role="runtime", content=msg))
                return msg
            continue

        # 3. Finalize streaming display (adds trailing newline)
        if on_turn_end:
            on_turn_end()

        # 4. Gate: the loop only dispatches actions the current agent is
        # actually allowed to emit. This is the loop's own authoritative
        # check — anything unexpected gets recorded as a runtime error and
        # the loop continues so the agent can react on its next turn.
        if action.type not in agent.allowed_actions:
            failure = (
                f"Action {action.type!r} is not allowed for agent role "
                f"{agent.role!r} (allowed: {sorted(agent.allowed_actions)}). "
                "Dropping the action and continuing."
            )
            _print(output, f"runtime> {failure}\n")
            env.record(Turn(role="runtime", content=failure))
            continue

        # 5. Record agent turn with full action details
        record_content = _format_agent_record(action)
        env.record(Turn(
            role=agent.role,
            content=record_content,
        ))

        # 6. Execute action
        if action.type == "chat":
            # Done — return to caller
            return action.response
        elif action.type == "think":
            # Response already recorded; nothing more to do this iteration.
            continue
        elif action.type == "exec":
            _print(output, f"runtime> Executing: {action.payload.get('job_name', 'unnamed')}...\n")
            try:
                observation = env.execute(action)
            except ExecutionInterrupted as exc:
                env.record(exc.observation)
                _print(output, f"runtime> {exc.observation.content}\n")
                return exc.observation.content
            env.record(observation)
            continue
        elif action.type == "delegate":
            role_name = action.payload.get('role', 'unknown')
            _print(output, f"runtime> Delegating to sub-agent: {role_name}...\n")
            # The sub-agent's run_loop handles LLMTransientError, CompactionError,
            # and ActionParseError internally (recording runtime Turns and
            # returning a summary string), so only a truly unexpected exception
            # should reach this catch-all.
            try:
                result = _delegate(
                    action, env, model,
                    max_turns=max_turns,
                    output=output,
                    on_turn_start=on_turn_start,
                    on_turn_end=on_turn_end,
                    on_turn_error=on_turn_error,
                    on_token_chunk=on_token_chunk,
                )
                env.record(Turn(role="sub_agent", content=result))
            except Exception as exc:
                failure = (
                    f"Sub-agent {role_name!r} failed with an unexpected error: "
                    f"{type(exc).__name__}: {exc}"
                )
                _print(output, f"runtime> {failure}\n")
                env.record(Turn(role="runtime", content=failure))
            # Refresh the parent agent's cached sub_agents_meta so the very
            # next act() in this same run_loop iteration sees the updated
            # registry — important because the core agent may re-delegate to
            # the same (or a new) sub-agent multiple times within one user
            # message, and it needs fresh state on every turn. Runs whether
            # delegation succeeded or failed, because _delegate's finally
            # clause may have persisted partial sub-agent state even on error.
            if env.state_root is not None:
                agent.set_sub_agents_meta(
                    sub_agent_meta.format_for_prompt(sub_agent_meta.load(env.state_root))
                )
            continue
        else:
            # Reached only when action.type is in agent.allowed_actions (step
            # 4 gate already approved it) but no dispatch branch above matches
            # it. That means someone extended ALLOWED_*_ACTIONS with a new
            # type but forgot to add the corresponding elif here. Record the
            # mismatch as a runtime Turn so the agent and the user can see it.
            failure = (
                f"Action {action.type!r} is allowed for agent role "
                f"{agent.role!r} but has no dispatch handler in run_loop. "
                "This indicates a missing elif branch in the dispatch chain."
            )
            _print(output, f"runtime> {failure}\n")
            env.record(Turn(role="runtime", content=failure))
            continue

    # Turn limit reached
    msg = "Loop ended: maximum turns reached."
    _print(output, f"runtime> {msg}\n")
    return msg


# --------------------------------------------------------------------------- #
# Sub-agent delegation
# --------------------------------------------------------------------------- #


def _delegate(
    action: Action,
    env: Environment,
    model: Any,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    output: TextIO = sys.stdout,
    on_turn_start: Callable[[str], None] | None = None,
    on_turn_end: Callable[[], None] | None = None,
    on_turn_error: Callable[[], None] | None = None,
    on_token_chunk: Callable[[str], None] | None = None,
) -> str:
    """Spawn a sub-agent to handle a delegated task.

    Creates an isolated Environment sharing the parent's workspace, executor,
    compactor, and approval hook, then runs a recursive loop. Display
    callbacks and output stream are forwarded so the sub-agent's runtime
    messages and streamed responses render in the same UI flow as the core
    agent — the sub-agent's turns appear under the green `sub_agent` badge
    because StreamingDisplay.reset() is called with the sub_agent's role on
    each act.

    If state_root is set on env, sub-agent state (full history + observation
    + workflow_summary + last_prompt) is persisted and restored across
    delegations to the same role.

    Error handling note: the recursive ``run_loop`` call handles
    ``LLMTransientError``, ``CompactionError``, and ``ActionParseError``
    internally and returns a summary string in every case. The caller
    (the ``delegate`` branch in ``run_loop``) only needs a catch-all
    ``except Exception`` for truly unexpected failures.
    """
    task = action.payload

    if model is None:
        return "Delegation failed: no model reference. Pass model= to run_loop()."

    role = task.get("role", "assistant")
    objective = task.get("objective", "")
    context = task.get("context", "")
    role_description = task.get("role_description", "")

    # Build sub-environment sharing parent's infrastructure
    sub_env = Environment(
        workspace=env.workspace,
        mode=env.mode,
        token_limit=env.token_limit,
        keep_last_k=env.keep_last_k,
        executor=env._executor,
        compactor=env._compactor,
        state_root=env.state_root,
    )
    sub_env._on_before_execute = env._on_before_execute

    # Restore previous sub-agent state if available
    state_root = env.state_root
    if state_root is not None:
        sub_state_path = state_root / "sub_agents" / f"{role}.json"
        sub_env.load_session(sub_state_path)

    # Append new objective as a core_agent turn (like a new user message)
    seed_content = objective
    if context:
        seed_content += f"\n\nContext:\n{context}"
    sub_env.record(Turn(role="core_agent", content=seed_content))

    # Build sub-agent (cannot delegate further)
    sub_agent = Agent(
        model,
        workspace=env.workspace,
        role="sub_agent",
        sub_agent_role=role,
        sub_agent_description=role_description,
        allowed_actions=ALLOWED_SUB_ACTIONS,
    )

    try:
        result = run_loop(
            sub_agent,
            sub_env,
            model=model,
            max_turns=max_turns,
            output=output,
            on_turn_start=on_turn_start,
            on_turn_end=on_turn_end,
            on_turn_error=on_turn_error,
            on_token_chunk=on_token_chunk,
        )
    finally:
        # Always persist what the sub-agent saw, even on error, so /view
        # can still inspect the failed sub-agent's last state.
        if state_root is not None:
            sub_state_path = state_root / "sub_agents" / f"{role}.json"
            sub_env.save_session(
                sub_state_path,
                extra_fields={"last_prompt": getattr(sub_agent, "last_prompt", "")},
            )
            sub_agent_meta.update(state_root, role, role_description)

    return result


def _act_with_retry(
    agent: Agent,
    state: State,
    *,
    chunk_callback: Callable[[str], None] | None,
    on_turn_error: Callable[[], None] | None,
    output: TextIO,
) -> Action:
    """Call ``agent.act()`` with retry on transient LLM errors.

    Retries up to ``DEFAULT_LLM_RETRIES`` times with exponential backoff +
    jitter. Does NOT record anything to the environment — the agent never
    produced output. If all retries are exhausted, the final
    ``LLMTransientError`` is re-raised so the caller (``run_loop``) can
    record it as a ``runtime`` Turn and return to the requester.
    """
    last_exc: LLMTransientError | None = None
    for attempt in range(1, DEFAULT_LLM_RETRIES + 1):
        try:
            return agent.act(state, chunk_callback=chunk_callback)
        except LLMTransientError as exc:
            last_exc = exc
            if on_turn_error:
                on_turn_error()
            if attempt >= DEFAULT_LLM_RETRIES:
                raise
            delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY) + random.uniform(0, 1)
            if exc.retry_after is not None:
                delay = max(delay, exc.retry_after)
            _print(
                output,
                f"runtime> LLM provider error (attempt {attempt}/{DEFAULT_LLM_RETRIES}, "
                f"retrying in {delay:.1f}s): {exc}\n",
            )
            time.sleep(delay)
    raise last_exc  # all retries exhausted


def _build_state_with_retry(
    env: Environment,
    *,
    output: TextIO,
) -> State:
    """Call ``env.build_state()`` with retry on compaction failures.

    Compaction typically invokes the LLM to summarize older turns, so a
    ``CompactionError`` is usually downstream of a transient provider issue.
    Retries up to ``DEFAULT_COMPACTION_RETRIES`` times with exponential
    backoff + jitter. If all retries are exhausted, the final
    ``CompactionError`` is re-raised so the caller (``run_loop``) can record
    it as a ``runtime`` Turn and return to the requester.
    """
    last_exc: CompactionError | None = None
    for attempt in range(1, DEFAULT_COMPACTION_RETRIES + 1):
        try:
            return env.build_state()
        except CompactionError as exc:
            last_exc = exc
            if attempt >= DEFAULT_COMPACTION_RETRIES:
                raise
            delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY) + random.uniform(0, 1)
            _print(
                output,
                f"runtime> Compaction error (attempt {attempt}/{DEFAULT_COMPACTION_RETRIES}, "
                f"retrying in {delay:.1f}s): {exc}\n",
            )
            time.sleep(delay)
    raise last_exc  # all retries exhausted


def _print(output: TextIO, text: str) -> None:
    """Print runtime-styled output."""
    if text:
        write_runtime(text, output)

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
        for key in ("role", "role_description", "objective"):
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
