"""Action dataclass and output parser for LLM responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Action dataclass
# --------------------------------------------------------------------------- #

ALLOWED_CORE_ACTIONS = frozenset({"chat", "think", "exec", "delegate"})
ALLOWED_SUB_ACTIONS = frozenset({"chat", "think", "exec"})


@dataclass
class Action:
    """Agent output per turn.

    Attributes:
        response: Natural language reasoning/answer (always present).
        type:     One of "chat", "think", "exec", "delegate".
        payload:  {} for chat/think; exec or delegate details otherwise.
    """

    response: str
    type: str
    payload: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Parse errors
# --------------------------------------------------------------------------- #


class ActionParseError(Exception):
    """Raised when LLM output cannot be parsed into a valid Action."""

    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


# --------------------------------------------------------------------------- #
# Output parser
# --------------------------------------------------------------------------- #


def parse_action(
    raw_llm_output: str,
    *,
    allowed_actions: frozenset[str] = ALLOWED_CORE_ACTIONS,
) -> Action:
    """Parse raw LLM text into an Action.

    Expected format::

        <output>
        {
          "response": "...",
          "action": "chat" | "think" | "exec" | "delegate",
          "action_input": {}
        }
        </output>

    Raises:
        ActionParseError: on any parsing or validation failure.
    """
    # 1. Extract <output>...</output>
    match = re.search(r"<output>\s*(.*?)\s*</output>", raw_llm_output, re.DOTALL)
    if not match:
        raise ActionParseError(
            "Missing <output>...</output> tags in model response.",
            raw_text=raw_llm_output,
        )

    json_text = match.group(1)

    # 2. Parse JSON
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ActionParseError(
            f"Invalid JSON inside <output> tags: {exc}",
            raw_text=raw_llm_output,
        ) from exc

    if not isinstance(payload, dict):
        raise ActionParseError(
            f"Expected JSON object, got {type(payload).__name__}.",
            raw_text=raw_llm_output,
        )

    # 3. Extract required keys
    response = payload.get("response") or ""
    action_type = str(payload.get("action", "")).strip().lower()
    action_input = payload.get("action_input", {})

    if not response:
        raise ActionParseError(
            "Missing or empty 'response' key.",
            raw_text=raw_llm_output,
        )

    # 4. Validate action type
    if action_type not in allowed_actions:
        raise ActionParseError(
            f"Invalid action '{action_type}'. Must be one of: {sorted(allowed_actions)}.",
            raw_text=raw_llm_output,
        )

    # 5. Validate action_input shape
    if not isinstance(action_input, dict):
        action_input = {}

    if action_type in ("chat", "think") and action_input:
        # Silently clear — these actions require empty payload.
        action_input = {}

    if action_type == "exec":
        _validate_exec_payload(action_input, raw_llm_output)

    if action_type == "delegate":
        _validate_delegate_payload(action_input, raw_llm_output)

    return Action(response=response, type=action_type, payload=action_input)


# --------------------------------------------------------------------------- #
# Payload validators
# --------------------------------------------------------------------------- #


def _validate_exec_payload(payload: dict, raw_text: str) -> None:
    """Validate exec action_input has required fields."""
    code_type = str(payload.get("code_type", "")).strip().lower()
    if code_type not in ("bash", "python"):
        raise ActionParseError(
            f"exec action requires code_type 'bash' or 'python', got '{code_type}'.",
            raw_text=raw_text,
        )

    has_script = bool(str(payload.get("script", "")).strip())
    has_path = bool(str(payload.get("script_path", "")).strip())

    if not has_script and not has_path:
        raise ActionParseError(
            "exec action requires either 'script' or 'script_path'.",
            raw_text=raw_text,
        )
    if has_script and has_path:
        raise ActionParseError(
            "exec action must have exactly one of 'script' or 'script_path', not both.",
            raw_text=raw_text,
        )

    raw_args = payload.get("script_args")
    if raw_args is None:
        return

    if isinstance(raw_args, str):
        args_present = bool(raw_args.strip())
    elif isinstance(raw_args, (list, tuple)):
        args_present = bool(raw_args)
    else:
        raise ActionParseError(
            "exec action 'script_args' must be either an array of strings or a shell-style argument string.",
            raw_text=raw_text,
        )

    if not args_present:
        return

    if has_script:
        raise ActionParseError(
            "exec action 'script_args' is only allowed when using 'script_path'.",
            raw_text=raw_text,
        )

    if isinstance(raw_args, (list, tuple)):
        normalized = [str(item).strip() for item in raw_args]
        if not normalized or any(not item for item in normalized):
            raise ActionParseError(
                "exec action 'script_args' array must contain only non-empty argument strings.",
                raw_text=raw_text,
            )


def _validate_delegate_payload(payload: dict, raw_text: str) -> None:
    """Validate delegate action_input has required fields."""
    if not str(payload.get("role", "")).strip():
        raise ActionParseError(
            "delegate action requires a non-empty 'role' field.",
            raw_text=raw_text,
        )
    if not str(payload.get("objective", "")).strip():
        raise ActionParseError(
            "delegate action requires a non-empty 'objective' field.",
            raw_text=raw_text,
        )

