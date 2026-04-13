"""Universal LLM provider — OpenAI-compatible chat completions adapter.

Works with any server that speaks the OpenAI chat completions API:
Ollama, vLLM, LM Studio, DeepSeek, Together, OpenRouter, etc.
"""

from __future__ import annotations

import json
import socket
import ssl
from http.client import RemoteDisconnected
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

class LLMProvider:
    """Universal LLM provider for OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 300,
        temperature: float = 0.2,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text via streaming SSE chat completions."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = Request(
            url=self.endpoint_url,
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
                    piece = self._extract_stream_piece(item)
                    if piece:
                        parts.append(piece)
                        if chunk_callback is not None:
                            chunk_callback(piece)
            return "".join(parts)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc
        except (URLError, TimeoutError, socket.timeout, ConnectionError, RemoteDisconnected, ssl.SSLError) as exc:
            raise RuntimeError(f"LLM network error: {exc}") from exc

    @staticmethod
    def _extract_stream_piece(data: dict[str, Any]) -> str:
        """Extract text from one streaming SSE chunk."""
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        return ""
