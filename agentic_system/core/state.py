"""State and Turn dataclasses — the agent's observation of the world."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Turn:
    """One entry in the environment timeline.

    Roles:
        - "user"      — requester input
        - "agent"     — agent response / reasoning
        - "runtime"   — sandbox execution observation (stdout/stderr)
        - "sub_agent" — sub-agent result returned to parent
        - "compactor" — compacted summary of older turns
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _utc_now_iso()


@dataclass
class State:
    """What the agent sees at turn t — built from the environment's observation window.

    Fields:
        observation:      LLM-facing window (compacted when token budget is exceeded).
        workflow_summary: Compacted summary of older history.
    """

    observation: list[Turn]
    workflow_summary: str = ""


def _utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
