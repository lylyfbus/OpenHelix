"""Ollama model provider — direct /api/generate adapter.

Satisfies the ``ModelProvider`` protocol defined in ``core/agent.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaProvider:
    """Ollama adapter using ``/api/generate`` with optional streaming.

    Environment variables:
        OLLAMA_BASE_URL: Base URL (default: http://localhost:11434)
        OLLAMA_MODEL: Model name (default: llama3.1:8b)
        OLLAMA_TIMEOUT_SECONDS: Request timeout (default: 300)
        OLLAMA_KEEP_ALIVE: Keep-alive duration for loaded model
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: float = 0.2,
    ) -> None:
        base = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.endpoint = f"{base}/api/generate"
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        self.temperature = temperature
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "").strip()
        self.keep_alive: Optional[str] = keep_alive or None

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text through Ollama; optionally stream chunks to callback."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": self.temperature},
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        headers = {"Content-Type": "application/json"}

        if not stream:
            data = _post_json(self.endpoint, headers, payload, timeout=self.timeout)
            text = str(data.get("response", "") or "")
            if not text and isinstance(data.get("message"), dict):
                text = str(data["message"].get("content", "") or "")
            return text

        # Streaming mode
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
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    piece = str(item.get("response", "") or "")
                    if piece:
                        parts.append(piece)
                        if chunk_callback is not None:
                            chunk_callback(piece)
            return "".join(parts)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama network error: {exc}") from exc


# --------------------------------------------------------------------------- #
# HTTP helper
# --------------------------------------------------------------------------- #


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 300,
) -> dict[str, Any]:
    """POST JSON and return parsed response."""
    req = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc
