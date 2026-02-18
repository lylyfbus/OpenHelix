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


class ModelRouter:
    def __init__(self, provider: str = "ollama", model_name: str | None = None) -> None:
        provider_name = str(provider).strip().lower() or "ollama"
        self.provider = provider_name
        if provider_name == "ollama":
            self.adapter = OllamaAdapter()
        else:
            raise NotImplementedError(
                f"Provider '{provider_name}' is not implemented yet. Use --provider ollama for now."
            )
        core_model = model_name or os.getenv(
            "OLLAMA_MODEL_CORE_AGENT",
            os.getenv("OLLAMA_MODEL_THINKING", "llama3.1:8b"),
        )
        summarizer_model = os.getenv("OLLAMA_MODEL_WORKFLOW_SUMMARIZER", core_model)
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
        prefix = '"raw_response":"'
        prefix_idx = 0
        capture = False
        done = False
        escape = False
        unicode_remaining = 0
        unicode_digits = ""

        def emit(token: str) -> None:
            if token:
                callback(token)

        def on_chunk(chunk: str) -> None:
            nonlocal prefix_idx, capture, done, escape, unicode_remaining, unicode_digits
            if done:
                return
            for ch in chunk:
                if done:
                    return

                if not capture:
                    if ch == prefix[prefix_idx]:
                        prefix_idx += 1
                        if prefix_idx == len(prefix):
                            capture = True
                            prefix_idx = 0
                        continue
                    prefix_idx = 1 if ch == prefix[0] else 0
                    continue

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

        return on_chunk

    def generate(
        self,
        role: str = "core_agent",
        final_prompt: str | None = None,
        raw_response_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        prompt = str(final_prompt or "").strip()
        if not prompt:
            return {}
        model = self._select_model(role)
        chunk_callback: Callable[[str], None] | None = None
        if raw_response_callback is not None:
            chunk_callback = self._stream_raw_response_from_chunk_factory(raw_response_callback)
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
