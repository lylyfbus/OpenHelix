"""OpenAI-compatible model provider — /chat/completions adapter.

Supports any provider speaking the OpenAI chat completions API:
DeepSeek, LM Studio, Z.AI, vLLM, Together, etc.

Satisfies the ``ModelProvider`` protocol defined in ``core/agent.py``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenAICompatProvider:
    """OpenAI-compatible ``/chat/completions`` adapter with SSE streaming.

    Environment variables (fallback chain):
        OPENAI_COMPAT_BASE_URL / provider-specific URL env
        OPENAI_COMPAT_API_KEY / provider-specific key env
        OPENAI_COMPAT_MODEL / provider-specific model env
        OPENAI_COMPAT_TIMEOUT_SECONDS (default: 300)
    """

    # Pre-defined provider presets
    PRESETS: dict[str, dict[str, str]] = {
        "deepseek": {
            "base_url_env": "DEEPSEEK_BASE_URL",
            "api_key_env": "DEEPSEEK_API_KEY",
            "model_env": "DEEPSEEK_MODEL",
            "default_base_url": "https://api.deepseek.com",
            "default_model": "deepseek-chat",
        },
        "lmstudio": {
            "base_url_env": "LMSTUDIO_BASE_URL",
            "api_key_env": "LMSTUDIO_API_KEY",
            "model_env": "LMSTUDIO_MODEL",
            "default_base_url": "http://localhost:1234/v1",
            "default_model": "local-model",
        },
        "zai": {
            "base_url_env": "ZAI_BASE_URL",
            "api_key_env": "ZAI_API_KEY",
            "model_env": "ZAI_MODEL",
            "default_base_url": "https://api.z.ai/api/paas/v4",
            "default_model": "glm-5",
        },
    }

    def __init__(
        self,
        *,
        provider: str = "openai_compatible",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: float = 0.2,
    ) -> None:
        preset = self.PRESETS.get(provider.lower(), {})

        # Resolve base URL
        raw_base = (
            base_url
            or os.getenv(preset.get("base_url_env", ""), "").strip()
            or os.getenv("OPENAI_COMPAT_BASE_URL", "").strip()
            or preset.get("default_base_url", "http://localhost:1234/v1")
        ).rstrip("/")
        if not re.search(r"/v\d+$", raw_base):
            raw_base = f"{raw_base}/v1"
        self.endpoint = f"{raw_base}/chat/completions"

        # Resolve model
        self.model = (
            model
            or os.getenv(preset.get("model_env", ""), "").strip()
            or os.getenv("OPENAI_COMPAT_MODEL", "").strip()
            or preset.get("default_model", "local-model")
        )

        # Resolve API key
        self.api_key = (
            api_key
            or os.getenv(preset.get("api_key_env", ""), "").strip()
            or os.getenv("OPENAI_COMPAT_API_KEY", "").strip()
            or ""
        )

        self.timeout = timeout or int(os.getenv("OPENAI_COMPAT_TIMEOUT_SECONDS", "300"))
        self.temperature = temperature
        self.provider = provider

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
            data = _post_json(self.endpoint, headers, payload, timeout=self.timeout)
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
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider} HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"{self.provider} network error: {exc}") from exc


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
