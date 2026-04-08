"""Environment — the agent's sandbox (computer).

Manages dual history (full_history + observation), sandbox execution,
approval gates, and state building with LLM-based compaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Union


from .action import Action
from .compactor import Compactor, CompactionError
from .state import State, Turn


class ExecutionInterrupted(RuntimeError):
    """Raised when execution should stop and control should return to the requester."""

    def __init__(self, observation: Turn) -> None:
        super().__init__(observation.content)
        self.observation = observation


# --------------------------------------------------------------------------- #
# Hook signatures
# --------------------------------------------------------------------------- #

ApprovalResult = Union[bool, Turn]
OnBeforeExecute = Callable[["Environment", Action], ApprovalResult]

# Sandbox executor signature: (payload, workspace) -> Turn
SandboxExecutor = Callable[[dict, Path], Turn]


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
        token_limit: int = int(100_000 * 0.75),
        keep_last_k: int = 10,
        executor: Optional[SandboxExecutor] = None,
        compactor: Optional[Compactor] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.token_limit = token_limit
        self.keep_last_k = keep_last_k
        self.approval_profile = "docker-online-rw-workspace-v1"

        # Pluggable executor
        self._executor = executor

        # Pluggable compactor for LLM-based context compaction
        self._compactor = compactor

        # Dual history
        self.full_history: list[Turn] = []
        self.observation: list[Turn] = []
        self.workflow_summary: str = ""

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
            if self._compactor is None:
                raise CompactionError(
                    "Context window exceeded but no compactor available."
                )
            self.workflow_summary = self._compactor.compact(
                self.workflow_summary, old_part
            )

        self.observation = recent

        return State(
            observation=list(self.observation),
            workflow_summary=self.workflow_summary,
        )

    # ----- Execution ------------------------------------------------------- #

    def execute(self, action: Action) -> Turn:
        """Execute an action in the sandbox.

        Checks approval hooks before execution in controlled mode.

        Returns:
            Observation Turn with execution results.
        """
        if self._on_before_execute:
            decision = self._on_before_execute(self, action)
            if isinstance(decision, Turn):
                raise ExecutionInterrupted(decision)
            if not decision:
                raise ExecutionInterrupted(
                    Turn(
                        role="runtime",
                        content="Execution denied by approval policy.",
                    )
                )

        if self._executor is None:
            return Turn(
                role="runtime",
                content="No sandbox executor configured. Set executor on Environment.",
            )

        return self._executor(action.payload, self.workspace)

    # ----- Persistence ----------------------------------------------------- #

    def save_session(self, session_path: Path, *, extra_fields: dict | None = None) -> None:
        """Persist session state to disk."""
        state = {
            "full_history": [_turn_to_dict(t) for t in self.full_history],
            "observation": [_turn_to_dict(t) for t in self.observation],
            "workflow_summary": self.workflow_summary,
        }
        if extra_fields:
            state.update(extra_fields)
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
        if not isinstance(raw, dict):
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
