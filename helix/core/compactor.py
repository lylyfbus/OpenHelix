"""Compactor — LLM-based context compaction.

Summarizes older conversation turns into a compact workflow summary
to keep the agent's observation window within the token budget.
"""

from __future__ import annotations

from typing import Any

from .state import Turn, format_turn


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
# Compactor
# --------------------------------------------------------------------------- #


class CompactionError(RuntimeError):
    """Raised when LLM-based context compaction fails after all retries.

    The caller (run_loop) should catch this and return control to the
    requester with a clear message.
    """


class Compactor:
    """LLM-based context compactor.

    Takes older turns and the existing workflow summary, and produces
    an updated compact summary using the LLM.

    Args:
        model: Any object satisfying the ModelProvider protocol.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def compact(self, current_summary: str, old_turns: list[Turn]) -> str:
        """Compact old turns into an updated workflow summary.

        Raises:
            CompactionError: If all retries fail.
        """
        history_text = "\n".join(format_turn(t) for t in old_turns)
        prompt = (
            f"{COMPACTOR_PROMPT}\n\n"
            f"<workflow_summary>\n"
            f"{current_summary if current_summary else '(empty)'}\n"
            f"</workflow_summary>\n\n"
            f"<workflow_history>\n"
            f"{history_text}\n"
            f"</workflow_history>"
        )

        last_error = ""
        for attempt in range(3):
            try:
                result = self._model.generate(prompt, stream=False)
                if isinstance(result, str) and result.strip():
                    return result.strip()
                last_error = "LLM returned empty compaction result"
            except Exception as exc:
                last_error = str(exc)

        raise CompactionError(
            f"Context compaction failed after 3 retries: {last_error}"
        )
