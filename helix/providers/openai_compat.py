"""Universal LLM provider — OpenAI-compatible /v1/chat/completions adapter.

Works with any server that speaks the OpenAI chat completions API:
Ollama, vLLM, LM Studio, DeepSeek, Together, OpenRouter, etc.

Environment variables:
    LLM_BASE_URL: Base URL (default: http://localhost:11434/v1 — Ollama)
    LLM_API_KEY: API key (default: empty — no auth)
    LLM_MODEL: Model name (default: llama3.1:8b)
    LLM_TIMEOUT_SECONDS: Request timeout (default: 300)
"""

from __future__ import annotations

import json
import re
import socket
import ssl
from http.client import RemoteDisconnected
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._http import post_json as _post_json, to_runtime_error as _to_runtime_error

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_TIMEOUT = 300


class LLMProvider:
    """Universal LLM provider using ``/v1/chat/completions``.

    Satisfies the ``ModelProvider`` protocol.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: float = 0.2,
    ) -> None:
        import os

        raw_base = (
            base_url
            or os.getenv("LLM_BASE_URL", "").strip()
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        if not re.search(r"/v\d+$", raw_base):
            raw_base = f"{raw_base}/v1"
        self.endpoint = f"{raw_base}/chat/completions"
        self.base_url = raw_base

        self.model = (
            model
            or os.getenv("LLM_MODEL", "").strip()
            or _DEFAULT_MODEL
        )
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY", "").strip()
            or ""
        )
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT)))
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text via chat completions; optionally stream via SSE."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": self.temperature,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if not stream:
            data = _post_json(
                self.endpoint,
                headers,
                payload,
                timeout=self.timeout,
                error_prefix="LLM",
            )
            return _extract_response_text(data)

        # SSE streaming
        req = Request(
            url=self.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        parts: list[str] = []
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    raw = line.decode("utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    if raw.startswith("data:"):
                        raw = raw[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    piece = _extract_stream_piece(item)
                    if piece:
                        parts.append(piece)
                        if chunk_callback is not None:
                            chunk_callback(piece)
            return "".join(parts)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            RemoteDisconnected,
            ssl.SSLError,
        ) as exc:
            raise _to_runtime_error("LLM", exc) from exc


# --------------------------------------------------------------------------- #
# Response extraction helpers
# --------------------------------------------------------------------------- #


def _content_to_text(content: Any) -> str:
    """Normalize content field that may be string or structured content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _extract_response_text(data: dict[str, Any]) -> str:
    """Extract text from a non-streaming chat completion response."""
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        text = _content_to_text(message.get("content"))
        if text:
            return text
    return str(first.get("text", ""))


def _extract_stream_piece(data: dict[str, Any]) -> str:
    """Extract text from one streaming SSE chunk."""
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        text = _content_to_text(delta.get("content"))
        if text:
            return text
    return str(first.get("text", ""))
