"""Environment — the agent's sandbox (computer).

Manages dual history (full_history + observation), sandbox execution,
approval gates, state building with LLM-based compaction, and sub-agent delegation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from .action import Action
from .state import State, Turn


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class CompactionError(RuntimeError):
    """Raised when LLM-based context compaction fails after all retries.

    The caller (run_loop) should catch this and return control to the
    requester with a clear message.
    """


# --------------------------------------------------------------------------- #
# Hook signatures
# --------------------------------------------------------------------------- #

OnBeforeExecute = Callable[["Environment", Action], bool]

# Sandbox executor signature: (payload, workspace) -> Turn
SandboxExecutor = Callable[[dict, Path], Turn]


# --------------------------------------------------------------------------- #
# Compactor prompt
# --------------------------------------------------------------------------- #

COMPACTOR_PROMPT = "\n".join([
    "You are an objective observer maintaining long-term memory for an agentic session.",
    "Your task: merge the existing workflow_summary with the older workflow_history",
    "turns into ONE updated workflow_summary.",
    "",
    "Input:",
    "1) <workflow_summary>: existing long-term summary from previous compaction.",
    "2) <workflow_history>: older turns being compacted (chronological).",
    "",
    "Update policy:",
    "- Preserve durable key facts already in the summary.",
    "- Update facts when new evidence changes them.",
    "- Append newly confirmed key facts from the history turns.",
    "- Never forget the user's original objective, constraints, or key decisions.",
    "- Remove low-level execution detail, repetition, and speculation.",
    "- Keep it compact but complete enough for long-horizon continuity.",
    "",
    "Required sections (use exactly these headers):",
    "## Session Goal & Scope",
    "## Key Decisions",
    "## Progress & Milestones",
    "## Current Status",
    "## Open Loops & Next Actions",
    "",
    "Output: Return ONLY the updated summary text. No JSON wrapping, no tags.",
    "Keep total length under 1500 characters.",
])


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class Environment:
    """The agent's sandbox runtime.

    Responsibilities:
        - Dual history management (full_history + observation).
        - State building with LLM-based context compaction.
        - Sandbox execution via pluggable executor.
        - Approval gates for exec actions.
        - Sub-agent delegation.
        - Session persistence.

    Args:
        workspace: Working directory for the agent.
        mode: "auto" (no approval) or "controlled" (approval required for exec).
        token_limit: Maximum token budget for LLM context.
        keep_last_k: Number of recent turns to keep verbatim after compaction.
        executor: Sandbox execution function. Receives (payload, workspace) -> Turn.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        mode: str = "controlled",
        token_limit: int = int(256_000 * 0.75),
        keep_last_k: int = 10,
        executor: Optional[SandboxExecutor] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.token_limit = token_limit
        self.keep_last_k = keep_last_k

        # Pluggable executor
        self._executor = executor

        # Dual history
        self.full_history: list[Turn] = []
        self.observation: list[Turn] = []
        self.workflow_summary: str = ""

        # Model reference for delegation and LLM-based compaction
        self._model_ref: Any = None

        # Hooks
        self._on_before_execute: Optional[OnBeforeExecute] = None

    # ----- Hook registration ----------------------------------------------- #

    def on_before_execute(self, hook: OnBeforeExecute) -> None:
        """Register hook for pre-execution checks (approval, budget, etc.)."""
        self._on_before_execute = hook

    # ----- History --------------------------------------------------------- #

    def record(self, turn: Turn) -> None:
        """Append a turn to both full_history and observation."""
        self.full_history.append(turn)
        self.observation.append(turn)

    # ----- State building + compaction ------------------------------------- #

    def build_state(self) -> State:
        """Build the State that the agent sees.

        If observation fits the token budget, use it as-is.
        Otherwise, compact the old part into workflow_summary and keep recent K.
        """
        available = self.token_limit

        observation_tokens = _estimate_tokens_for_turns(self.observation)
        if observation_tokens <= available:
            return State(
                observation=list(self.observation),
                workflow_summary=self.workflow_summary,
            )

        # Compact: summarize old part, keep recent K
        split = max(0, len(self.observation) - self.keep_last_k)
        old_part = self.observation[:split]
        recent = self.observation[split:]

        if old_part:
            self.workflow_summary = self._compact(
                self.workflow_summary, old_part
            )

        summary_turn = Turn(
            role="compactor",
            content=self.workflow_summary,
        )
        self.observation = [summary_turn] + recent

        return State(
            observation=list(self.observation),
            workflow_summary=self.workflow_summary,
        )

    def _compact(self, current_summary: str, old_turns: list[Turn]) -> str:
        """Compact old turns into an updated workflow summary using the LLM.

        Raises:
            CompactionError: If no model is available or all retries fail.
        """
        if self._model_ref is None:
            raise CompactionError(
                "Context window exceeded but no model available for compaction."
            )

        # Build the compactor prompt
        history_text = "\n".join(
            f"[{t.timestamp}] {t.role}> {t.content}" for t in old_turns
        )
        prompt = (
            f"{COMPACTOR_PROMPT}\n\n"
            f"<workflow_summary>\n"
            f"{current_summary if current_summary else '(empty)'}\n"
            f"</workflow_summary>\n\n"
            f"<workflow_history>\n"
            f"{history_text}\n"
            f"</workflow_history>"
        )

        # Try LLM compaction with retry
        last_error = ""
        for attempt in range(3):
            try:
                result = self._model_ref.generate(prompt, stream=False)
                if isinstance(result, str) and result.strip():
                    return result.strip()
                last_error = "LLM returned empty compaction result"
            except Exception as exc:
                last_error = str(exc)

        raise CompactionError(
            f"Context compaction failed after 3 retries: {last_error}"
        )

    # ----- Execution ------------------------------------------------------- #

    def execute(self, action: Action) -> Turn:
        """Execute an action in the sandbox.

        Checks approval hooks before execution in controlled mode.

        Returns:
            Observation Turn with execution results.
        """
        if self._on_before_execute:
            allowed = self._on_before_execute(self, action)
            if not allowed:
                return Turn(
                    role="runtime",
                    content="Execution denied by approval policy.",
                )

        if self._executor is None:
            return Turn(
                role="runtime",
                content="No sandbox executor configured. Set executor on Environment.",
            )

        return self._executor(action.payload, self.workspace)

    # ----- Delegation ------------------------------------------------------ #

    def delegate(self, action: Action) -> str:
        """Spawn a sub-agent to handle a delegated task.

        Creates an isolated child workspace and runs the same universal loop.

        Returns:
            The sub-agent's final report text.
        """
        task = action.payload
        task_id = uuid4().hex[:12]

        # Isolated child workspace
        child_workspace = self.workspace / "sub_agents" / task_id
        child_workspace.mkdir(parents=True, exist_ok=True)

        # Import here to avoid circular imports
        from .loop import run_loop
        from .agent import Agent
        from .action import ALLOWED_SUB_ACTIONS

        if self._model_ref is None:
            return "Delegation failed: no model reference available. Call set_model_ref() first."

        sub_env = Environment(
            workspace=child_workspace,
            mode=self.mode,
            token_limit=self.token_limit,
            keep_last_k=self.keep_last_k,
            executor=self._executor,
        )
        # Propagate execution hook (sub-agents can't delegate)
        sub_env._on_before_execute = self._on_before_execute
        # Sub-agent gets model ref for its own compaction if needed
        sub_env.set_model_ref(self._model_ref)

        # Seed sub-agent with task objective
        sub_env.record(Turn(role="user", content=task.get("objective", "")))

        # Build sub-agent with role-specific prompt
        role = task.get("role", "assistant")
        context = task.get("context", "")
        sub_prompt = (
            f"You are a sub-agent with the role: {role}.\n"
            f"Complete the assigned task and report your results.\n"
        )
        if context:
            sub_prompt += f"\nAdditional context from the core agent:\n{context}\n"

        sub_agent = Agent(
            self._model_ref,
            name="sub-agent",
            system_prompt=sub_prompt,
            allowed_actions=ALLOWED_SUB_ACTIONS,
        )

        return run_loop(sub_agent, sub_env)

    def set_model_ref(self, model: Any) -> None:
        """Store a reference to the model provider for delegation and compaction."""
        self._model_ref = model

    # ----- Persistence ----------------------------------------------------- #

    def save_session(self, session_path: Path) -> None:
        """Persist session state to disk."""
        state = {
            "full_history": [_turn_to_dict(t) for t in self.full_history],
            "observation": [_turn_to_dict(t) for t in self.observation],
            "workflow_summary": self.workflow_summary,
        }
        session_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = session_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(session_path)

    def load_session(self, session_path: Path) -> bool:
        """Load session state from disk. Returns False if not found."""
        if not session_path.exists():
            return False
        try:
            raw = json.loads(session_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        self.full_history = [_dict_to_turn(d) for d in raw.get("full_history", [])]
        self.observation = [_dict_to_turn(d) for d in raw.get("observation", [])]
        self.workflow_summary = raw.get("workflow_summary", "")
        return True

# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #


def _turn_to_dict(turn: Turn) -> dict:
    return {
        "role": turn.role,
        "content": turn.content,
        "timestamp": turn.timestamp,
    }


def _dict_to_turn(d: dict) -> Turn:
    return Turn(
        role=d.get("role", "unknown"),
        content=d.get("content", ""),
        timestamp=d.get("timestamp", ""),
    )


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return len(text) // 4


def _estimate_tokens_for_turns(turns: list[Turn]) -> int:
    """Estimate tokens across a list of turns."""
    return sum(_estimate_tokens(t.content) + 20 for t in turns)  # 20 for role/ts overhead
