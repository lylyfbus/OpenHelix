from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class ModelResponse:
    provider: str
    model: str
    text: str


def _first_env_value(keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


def _first_int_env_value(keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        value = os.getenv(key, "").strip()
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return int(default)


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
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


class OllamaAdapter:
    provider = "ollama"

    def __init__(self) -> None:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.endpoint = f"{base}/api/generate"
        self.timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "").strip()
        self.keep_alive = keep_alive or None

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": 0.2},
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        headers = {"Content-Type": "application/json"}
        if not stream:
            data = _post_json(self.endpoint, headers, payload, timeout=self.timeout_seconds)
            text = str(data.get("response", "") or "")
            if not text and isinstance(data.get("message"), dict):
                text = str(data["message"].get("content", "") or "")
            return ModelResponse(provider=self.provider, model=model, text=text)

        req = Request(
            url=self.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        parts: list[str] = []
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
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
            return ModelResponse(provider=self.provider, model=model, text="".join(parts))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc


class OpenAICompatibleAdapter:
    def __init__(
        self,
        *,
        provider: str,
        base_url_env_keys: tuple[str, ...],
        default_base_url: str,
        api_key_env_keys: tuple[str, ...],
        timeout_env_keys: tuple[str, ...],
        default_timeout_seconds: int = 300,
    ) -> None:
        self.provider = str(provider).strip().lower() or "openai_compatible"

        raw_base = _first_env_value(base_url_env_keys, default_base_url).strip()
        base = raw_base.rstrip("/")
        if not re.search(r"/v\d+$", base):
            base = f"{base}/v1"
        self.endpoint = f"{base}/chat/completions"
        self.timeout_seconds = _first_int_env_value(timeout_env_keys, default_timeout_seconds)
        self.api_token = _first_env_value(api_key_env_keys, "")

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)

    @classmethod
    def _extract_response_text(cls, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if isinstance(message, dict):
            text = cls._content_to_text(message.get("content"))
            if text:
                return text
        text_value = first.get("text")
        if isinstance(text_value, str):
            return text_value
        return ""

    @classmethod
    def _extract_stream_piece(cls, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if isinstance(delta, dict):
            text = cls._content_to_text(delta.get("content"))
            if text:
                return text
        text_value = first.get("text")
        if isinstance(text_value, str):
            return text_value
        return ""

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        if not stream:
            data = _post_json(self.endpoint, headers, payload, timeout=self.timeout_seconds)
            text = self._extract_response_text(data)
            return ModelResponse(provider=self.provider, model=model, text=text)

        req = Request(
            url=self.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        parts: list[str] = []
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
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
            return ModelResponse(provider=self.provider, model=model, text="".join(parts))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc


class ModelRouter:
    def __init__(self, provider: str = "ollama", model_name: str | None = None) -> None:
        provider_name = str(provider).strip().lower() or "ollama"
        if provider_name in {"openai-compatible", "openaicompat"}:
            provider_name = "openai_compatible"
        if provider_name == "z.ai":
            provider_name = "zai"
        self.provider = provider_name

        if provider_name == "ollama":
            self.adapter = OllamaAdapter()
            core_model = model_name or os.getenv(
                "OLLAMA_MODEL_CORE_AGENT",
                os.getenv("OLLAMA_MODEL_THINKING", "llama3.1:8b"),
            )
            summarizer_model = os.getenv("OLLAMA_MODEL_WORKFLOW_SUMMARIZER", core_model)
        elif provider_name == "lmstudio":
            self.adapter = OpenAICompatibleAdapter(
                provider="lmstudio",
                base_url_env_keys=("LMSTUDIO_BASE_URL", "OPENAI_COMPAT_BASE_URL"),
                default_base_url="http://localhost:1234/v1",
                api_key_env_keys=("LMSTUDIO_API_KEY", "LM_API_TOKEN", "OPENAI_COMPAT_API_KEY"),
                timeout_env_keys=("LMSTUDIO_TIMEOUT_SECONDS", "OPENAI_COMPAT_TIMEOUT_SECONDS"),
            )
            core_model = model_name or os.getenv(
                "LMSTUDIO_MODEL_CORE_AGENT",
                os.getenv("LMSTUDIO_MODEL_THINKING", "local-model"),
            )
            summarizer_model = os.getenv("LMSTUDIO_MODEL_WORKFLOW_SUMMARIZER", core_model)
        elif provider_name == "zai":
            self.adapter = OpenAICompatibleAdapter(
                provider="zai",
                base_url_env_keys=("ZAI_BASE_URL", "OPENAI_COMPAT_BASE_URL", "LMSTUDIO_BASE_URL"),
                default_base_url="https://api.z.ai/api/paas/v4",
                api_key_env_keys=("ZAI_API_KEY", "OPENAI_COMPAT_API_KEY", "LMSTUDIO_API_KEY", "LM_API_TOKEN"),
                timeout_env_keys=("ZAI_TIMEOUT_SECONDS", "OPENAI_COMPAT_TIMEOUT_SECONDS", "LMSTUDIO_TIMEOUT_SECONDS"),
            )
            core_model = model_name or os.getenv(
                "ZAI_MODEL_CORE_AGENT",
                os.getenv("ZAI_MODEL_THINKING", "glm-5"),
            )
            summarizer_model = os.getenv("ZAI_MODEL_WORKFLOW_SUMMARIZER", core_model)
        elif provider_name == "openai_compatible":
            self.adapter = OpenAICompatibleAdapter(
                provider="openai_compatible",
                base_url_env_keys=("OPENAI_COMPAT_BASE_URL",),
                default_base_url="http://localhost:1234/v1",
                api_key_env_keys=("OPENAI_COMPAT_API_KEY", "LM_API_TOKEN"),
                timeout_env_keys=("OPENAI_COMPAT_TIMEOUT_SECONDS",),
            )
            core_model = model_name or os.getenv(
                "OPENAI_COMPAT_MODEL_CORE_AGENT",
                os.getenv("OPENAI_COMPAT_MODEL_THINKING", "local-model"),
            )
            summarizer_model = os.getenv("OPENAI_COMPAT_MODEL_WORKFLOW_SUMMARIZER", core_model)
        else:
            raise NotImplementedError(
                "Provider "
                f"'{provider_name}' is not implemented yet. "
                "Use --provider ollama, lmstudio, zai, or openai_compatible."
            )
        self.models: dict[str, str] = {
            "core_agent": core_model,
            "workflow_summarizer": summarizer_model,
            "workflow_history_compactor": summarizer_model,
        }

    def _select_model(self, role: str) -> str:
        role_name = str(role).strip()
        if role_name in self.models:
            return self.models[role_name]
        return self.models["core_agent"]

    @staticmethod
    def _parse_json_payload(text: str) -> dict[str, Any] | None:
        raw = text.strip()
        if not raw:
            return None

        match = re.search(r"<output>(.*?)</output>", raw, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        block = match.group(1).strip()
        if not block:
            return None
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    @staticmethod
    def _stream_raw_response_from_chunk_factory(callback: Callable[[str], None]) -> Callable[[str], None]:
        key_pattern = re.compile(r'"raw_response"\s*:\s*"')
        search_buffer = ""
        capture = False
        done = False
        escape = False
        unicode_remaining = 0
        unicode_digits = ""

        def emit(token: str) -> None:
            if token:
                callback(token)

        def consume_string_chars(text: str) -> None:
            nonlocal done, escape, unicode_remaining, unicode_digits
            if done:
                return
            for ch in text:
                if done:
                    return
                if unicode_remaining > 0:
                    if ch.lower() in "0123456789abcdef":
                        unicode_digits += ch
                        unicode_remaining -= 1
                        if unicode_remaining == 0:
                            try:
                                emit(chr(int(unicode_digits, 16)))
                            except Exception:
                                emit("\\u" + unicode_digits)
                            unicode_digits = ""
                    else:
                        emit("\\u" + unicode_digits + ch)
                        unicode_remaining = 0
                        unicode_digits = ""
                    continue

                if escape:
                    mapping = {
                        '"': '"',
                        "\\": "\\",
                        "/": "/",
                        "b": "\b",
                        "f": "\f",
                        "n": "\n",
                        "r": "\r",
                        "t": "\t",
                    }
                    if ch == "u":
                        unicode_remaining = 4
                        unicode_digits = ""
                    else:
                        emit(mapping.get(ch, ch))
                    escape = False
                    continue

                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    done = True
                    return
                emit(ch)

        def on_chunk(chunk: str) -> None:
            nonlocal search_buffer, capture
            if done or not chunk:
                return
            if capture:
                consume_string_chars(chunk)
                return

            search_buffer += chunk
            match = key_pattern.search(search_buffer)
            if not match:
                if len(search_buffer) > 256:
                    search_buffer = search_buffer[-256:]
                return

            capture = True
            remainder = search_buffer[match.end() :]
            search_buffer = ""
            if remainder:
                consume_string_chars(remainder)

        return on_chunk

    def generate(
        self,
        role: str = "core_agent",
        final_prompt: str | None = None,
        raw_response_callback: Callable[[str], None] | None = None,
        stream_text_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        prompt = str(final_prompt or "").strip()
        if not prompt:
            return {}
        model = self._select_model(role)
        parsed_callback: Callable[[str], None] | None = None
        if raw_response_callback is not None:
            parsed_callback = self._stream_raw_response_from_chunk_factory(raw_response_callback)

        chunk_callback: Callable[[str], None] | None = None
        if stream_text_callback is not None or parsed_callback is not None:
            def _combined(piece: str) -> None:
                if stream_text_callback is not None:
                    stream_text_callback(piece)
                if parsed_callback is not None:
                    parsed_callback(piece)

            chunk_callback = _combined

        response = self.adapter.generate(
            model=model,
            prompt=prompt,
            stream=True,
            chunk_callback=chunk_callback,
        )
        payload = self._parse_json_payload(response.text or "")
        if isinstance(payload, dict):
            return payload
        return {}
