"""State and Turn dataclasses — the agent's observation of the world."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now_timestamp() -> str:
    """Return current UTC time in a compact, human-readable form."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Turn:
    """One entry in the environment timeline.

    Roles:
        - "user"      — requester input
        - "agent"     — agent response / reasoning
        - "runtime"   — sandbox execution observation (stdout/stderr)
        - "sub_agent" — sub-agent result returned to parent
    """

    role: str
    content: str
    timestamp: str = field(default_factory=_utc_now_timestamp)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _utc_now_timestamp()


@dataclass
class State:
    """What the agent sees at turn t — built from the environment's observation window.

    Fields:
        observation:      LLM-facing window (compacted when token budget is exceeded).
        workflow_summary: Compacted summary of older history.
    """

    observation: list[Turn]
    workflow_summary: str = ""



