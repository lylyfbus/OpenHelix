from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class ModelResponse:
    provider: str
    model: str
    text: str


class ModelAdapter(Protocol):
    provider: str

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ModelResponse:
        ...


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
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


class OpenAIAdapter:
    provider = "openai"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.endpoint = f"{base}/chat/completions"

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "temperature": 0.2}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = _post_json(self.endpoint, headers, payload)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        else:
            text = str(content)
        return ModelResponse(provider=self.provider, model=model, text=text)


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.endpoint = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
        self.version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "Content-Type": "application/json",
        }
        data = _post_json(self.endpoint, headers, payload)
        blocks = data.get("content", [])
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        return ModelResponse(provider=self.provider, model=model, text=text)


class OllamaAdapter:
    provider = "ollama"

    def __init__(self) -> None:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.endpoint = f"{base}/api/generate"
        self.timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "").strip()
        self.keep_alive = keep_alive or None

    def generate(
        self,
        model: str,
        prompt: str,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        headers = {"Content-Type": "application/json"}
        data = _post_json(self.endpoint, headers, payload, timeout=self.timeout_seconds)
        text = str(data.get("response", "") or "")
        if not text and isinstance(data.get("message"), dict):
            text = str(data["message"].get("content", "") or "")
        return ModelResponse(provider=self.provider, model=model, text=text)


class ModelRouter:
    def __init__(self, provider: str | None = None, model_name: str | None = None) -> None:
        token = (provider or os.getenv("AGENTIC_MODEL_PROVIDER", "openai")).strip().lower()
        if token == "claude":
            token = "anthropic"
        if token not in {"openai", "anthropic", "ollama"}:
            raise ValueError("model_provider must be one of: openai, claude, ollama")

        self.provider = token
        self.model_name = model_name
        self.adapters: dict[str, ModelAdapter] = {
            "openai": OpenAIAdapter(),
            "anthropic": AnthropicAdapter(),
            "ollama": OllamaAdapter(),
        }
        self.provider_defaults: dict[str, dict[str, str]] = {
            "openai": {
                "core_agent": os.getenv("OPENAI_MODEL_CORE_AGENT", os.getenv("OPENAI_MODEL_THINKING", "gpt-4o-mini")),
                "workflow_summarizer": os.getenv(
                    "OPENAI_MODEL_WORKFLOW_SUMMARIZER",
                    os.getenv("OPENAI_MODEL_THINKING", "gpt-4o-mini"),
                ),
            },
            "anthropic": {
                "core_agent": os.getenv(
                    "ANTHROPIC_MODEL_CORE_AGENT",
                    os.getenv("ANTHROPIC_MODEL_THINKING", "claude-3-5-sonnet-latest"),
                ),
                "workflow_summarizer": os.getenv(
                    "ANTHROPIC_MODEL_WORKFLOW_SUMMARIZER",
                    os.getenv("ANTHROPIC_MODEL_THINKING", "claude-3-5-sonnet-latest"),
                ),
            },
            "ollama": {
                "core_agent": os.getenv("OLLAMA_MODEL_CORE_AGENT", os.getenv("OLLAMA_MODEL_THINKING", "llama3.1:8b")),
                "workflow_summarizer": os.getenv(
                    "OLLAMA_MODEL_WORKFLOW_SUMMARIZER",
                    os.getenv("OLLAMA_MODEL_THINKING", "llama3.1:8b"),
                ),
            },
        }
        if model_name:
            self.provider_defaults[self.provider]["core_agent"] = model_name
            self.provider_defaults[self.provider]["workflow_summarizer"] = model_name

    def _select_model(self, role: str) -> str:
        if self.model_name:
            return self.model_name
        role_name = str(role).strip()
        if role_name in self.provider_defaults[self.provider]:
            return self.provider_defaults[self.provider][role_name]
        return self.provider_defaults[self.provider]["core_agent"]

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

    def generate(
        self,
        role: str,
        state: Any,
        prompt_engine: Any,
    ) -> dict[str, Any]:
        role_name = str(role).strip()
        if not role_name:
            return {}
        if prompt_engine is None:
            return {}
        try:
            final_prompt = prompt_engine.build_prompt(
                agent_role=role_name,
                input_payload={
                    "workflow_summary": getattr(state, "workflow_summary", ""),
                    "workflow_history": getattr(state, "workflow_hist", []),
                },
            )
        except Exception:
            return {}
        if not isinstance(final_prompt, str) or not final_prompt.strip():
            return {}

        model = self._select_model(role_name)
        adapter = self.adapters[self.provider]
        response = adapter.generate(
            model=model,
            prompt=final_prompt,
        )
        payload = self._parse_json_payload(response.text or "")
        if isinstance(payload, dict):
            return payload
        return {}
