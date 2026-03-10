"""Agent — the LLM brain.

A pure function conceptually: (system_prompt, state) → Action.
The agent itself is stateless; all state lives in the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .action import Action, parse_action, ActionParseError, ALLOWED_CORE_ACTIONS, ALLOWED_SUB_ACTIONS
from .state import State


# --------------------------------------------------------------------------- #
# ModelProvider protocol
# --------------------------------------------------------------------------- #


class ModelProvider(Protocol):
    """Minimal contract for LLM providers."""

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text from the given prompt.

        Args:
            prompt: Full prompt text (system + context).
            stream: Whether to stream tokens.
            chunk_callback: Called with each token chunk when streaming.

        Returns:
            Complete generated text.
        """
        ...


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    """LLM-based agent that produces Actions from State.

    The agent is stateless — it reads the current State (built by the
    Environment) and produces an Action.  All persistent state lives
    in the Environment's history.

    Prompt building is owned by the agent:

    - **Core agents** pass ``workspace=`` — a ``PromptBuilder`` assembles the
      system prompt from the workspace's skills, knowledge, and prompt
      templates at construction time.
    - **Sub-agents** pass ``system_prompt=`` directly (simple role prompt,
      no workspace).

    Exactly one of ``workspace`` or ``system_prompt`` must be provided.

    Args:
        model: Any object satisfying ``ModelProvider`` protocol.
        workspace: Workspace directory (builds system prompt via PromptBuilder).
        system_prompt: Pre-built system prompt string (for sub-agents).
        allowed_actions: Which action types this agent can emit.
        on_token: Callback fired with each streamed token (for UI).
    """

    def __init__(
        self,
        model: ModelProvider,
        *,
        name: str = "core-agent",
        workspace: Optional[Path] = None,
        system_prompt: Optional[str] = None,
        allowed_actions: frozenset[str] = ALLOWED_CORE_ACTIONS,
    ) -> None:
        if workspace is not None and system_prompt is not None:
            raise ValueError("Provide workspace or system_prompt, not both.")
        if workspace is None and system_prompt is None:
            raise ValueError("Provide either workspace or system_prompt.")

        self.model = model
        self.name = name
        self.allowed_actions = allowed_actions
        self.last_prompt = ""

        if workspace is not None:
            from ..context.prompt_builder import PromptBuilder
            self._prompt_builder = PromptBuilder(workspace)
            self.system_prompt = self._prompt_builder.build("core_agent")
        else:
            self._prompt_builder = None
            self.system_prompt = system_prompt  # type: ignore[assignment]

    def act(
        self,
        state: State,
        *,
        stream: bool = True,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> Action:
        """Given current state, produce the next Action.

        Raises:
            ActionParseError: If the model output fails parsing/validation.
        """
        prompt = self._build_prompt(state)
        self.last_prompt = prompt
        raw_output = self.model.generate(
            prompt,
            stream=stream,
            chunk_callback=chunk_callback,
        )
        return parse_action(raw_output, allowed_actions=self.allowed_actions)

    # ----- prompt construction --------------------------------------------- #

    def _build_prompt(self, state: State) -> str:
        """Compose the final prompt from system prompt + state context.

        Layout (optimized for LLM attention):
            1. system_prompt     — identity, skills, constraints (beginning)
            2. workflow_summary  — compacted long-term memory
            3. workflow_history  — chronological turns (excluding latest)
            4. latest_context    — the immediate turn to respond to (end)
        """
        sections: list[str] = [self.system_prompt]

        if state.workflow_summary:
            sections.append(
                f"<workflow_summary>\n{state.workflow_summary}\n</workflow_summary>"
            )

        if state.observation:
            # Split: history (all but last) + latest (last turn)
            history_turns = state.observation[:-1]
            latest_turn = state.observation[-1]

            if history_turns:
                history_text = "\n".join(
                    f"[{t.timestamp}] {t.role}> {t.content}"
                    for t in history_turns
                )
                sections.append(
                    f"<workflow_history>\n{history_text}\n</workflow_history>"
                )

            latest_text = f"[{latest_turn.timestamp}] {latest_turn.role}> {latest_turn.content}"
            sections.append(
                f"<latest_context>\n{latest_text}\n</latest_context>"
            )

        return "\n\n".join(sections)
