"""Streaming display callbacks and extraction for the agent loop."""

from __future__ import annotations

import re
import sys
from typing import Any, Optional, TextIO

_JSON_ESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

_EXEC_PAYLOAD_ORDER = (
    "job_name",
    "code_type",
    "script_path",
    "script",
    "script_args",
    "timeout_seconds",
)

# --------------------------------------------------------------------------- #
# ANSI color scheme
# --------------------------------------------------------------------------- #

# ANSI styles
_RESET = "\033[0m"
_BOLD = "\033[1m"

# Role prefix badges: bold + colored background + white text
_BADGE = {
    "user":       f"{_BOLD}\033[48;5;240m\033[38;5;255m",   # gray badge
    "core_agent": f"{_BOLD}\033[48;5;25m\033[38;5;255m",    # blue badge
    "runtime":    f"{_BOLD}\033[48;5;130m\033[38;5;255m",   # amber badge
    "sub_agent":  f"{_BOLD}\033[48;5;28m\033[38;5;255m",    # green badge
    "approval":   f"{_BOLD}\033[48;5;130m\033[38;5;255m",   # amber badge
}


def _write_role_block(role: str, text: str, output: TextIO) -> None:
    """Write a block with a colored role badge prefix and content."""
    if not text:
        return
    badge = _BADGE.get(role, f"{_BOLD}")

    # Split into prefix (role>) and content
    if "> " in text:
        prefix, content = text.split("> ", 1)
        prefix_text = f"{prefix}>"
    else:
        prefix_text = role
        content = text

    # Badge + content
    output.write(f"{badge} {prefix_text} {_RESET} {content}")
    if not content.endswith("\n"):
        output.write("\n")
    output.write("\n")
    output.flush()


def write_agent(text: str, output: Optional[TextIO] = None, *, role: str = "core_agent") -> None:
    """Write agent output with the role's badge prefix.

    The role argument selects the badge color — `core_agent` gets the blue
    badge, `sub_agent` gets the green badge. Any unknown role falls back to
    the default bold.
    """
    stream = output if output is not None else sys.stdout
    _write_role_block(role, text, stream)


def write_runtime(text: str, output: Optional[TextIO] = None) -> None:
    """Write runtime output with amber badge prefix."""
    stream = output if output is not None else sys.stdout
    _write_role_block("runtime", text, stream)


def write_approval(text: str, output: Optional[TextIO] = None) -> None:
    """Write approval prompt with amber badge prefix."""
    stream = output if output is not None else sys.stdout
    _write_role_block("approval", text, stream)


# --------------------------------------------------------------------------- #
# Exec payload formatting
# --------------------------------------------------------------------------- #


def _has_display_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def iter_exec_payload_items(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return non-empty exec payload items in stable display order."""
    seen: set[str] = set()
    items: list[tuple[str, Any]] = []

    for key in _EXEC_PAYLOAD_ORDER:
        if key in payload and _has_display_value(payload[key]):
            items.append((key, payload[key]))
            seen.add(key)

    for key in sorted(payload):
        if key in seen or not _has_display_value(payload[key]):
            continue
        items.append((key, payload[key]))

    return items


# --------------------------------------------------------------------------- #
# Streaming response extraction
# --------------------------------------------------------------------------- #


def extract_streaming_response(partial_text: str) -> Optional[str]:
    """Extract the 'response' value from a partial JSON stream.

    Used to stream tokens to the UI during generation. Returns the
    extracted response text so far, or None if the key hasn't appeared yet.
    """
    # Look for "response": "... pattern
    marker = '"response"'
    idx = partial_text.find(marker)
    if idx == -1:
        return None

    # Skip past the key and colon
    after_key = partial_text[idx + len(marker) :]
    colon_idx = after_key.find(":")
    if colon_idx == -1:
        return None

    after_colon = after_key[colon_idx + 1 :].lstrip()
    if not after_colon or after_colon[0] != '"':
        return None

    # Extract and decode the partial JSON string value.
    result_chars: list[str] = []
    i = 1  # skip opening quote
    while i < len(after_colon):
        ch = after_colon[i]
        if ch == "\\":
            if i + 1 >= len(after_colon):
                break
            esc = after_colon[i + 1]
            if esc in _JSON_ESCAPE_MAP:
                result_chars.append(_JSON_ESCAPE_MAP[esc])
                i += 2
                continue
            if esc == "u":
                if i + 6 > len(after_colon):
                    break
                hex_value = after_colon[i + 2 : i + 6]
                if not re.fullmatch(r"[0-9a-fA-F]{4}", hex_value):
                    break
                result_chars.append(chr(int(hex_value, 16)))
                i += 6
                continue
            # Unknown or malformed escape; stop until more text arrives.
            break
        if ch == '"':
            break
        result_chars.append(ch)
        i += 1
    return "".join(result_chars) if result_chars else None


# --------------------------------------------------------------------------- #
# Streaming display
# --------------------------------------------------------------------------- #


class StreamingDisplay:
    """Stateful streaming callback that buffers only the parsed response.

    Accumulates raw LLM tokens and uses extract_streaming_response() to
    track the latest response text. The text is only printed if the turn
    later passes parsing/validation. Raw JSON structure remains hidden.
    """

    def __init__(self, output: Optional[TextIO] = None) -> None:
        self._output = output
        self._accumulated = ""
        self._response_text = ""
        self._current_name = "agent"

    def __call__(self, token: str) -> None:
        """Called per token during model.generate()."""
        self._accumulated += token
        response = extract_streaming_response(self._accumulated)
        if response is None:
            return
        self._response_text = response

    def reset(self, name: str = "agent") -> None:
        """Reset state for a new turn."""
        self._accumulated = ""
        self._response_text = ""
        self._current_name = name

    def commit(self) -> None:
        """Print the buffered response as agent output."""
        if not self._response_text:
            return
        output = self._output if self._output is not None else sys.stdout
        write_agent(
            f"{self._current_name}> {self._response_text}",
            output,
            role=self._current_name,
        )

    def discard(self) -> None:
        """Drop any buffered response from a failed parse attempt."""
        self._accumulated = ""
        self._response_text = ""
