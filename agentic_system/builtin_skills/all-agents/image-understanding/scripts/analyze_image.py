#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "image-understanding"


def _first_non_empty(*values: str) -> str:
    for value in values:
        token = str(value or "").strip()
        if token:
            return token
    return ""


def _ok(
    *,
    image_source: str,
    analysis: str,
    provider_used: str,
    model_used: str,
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "image_source": image_source,
        "analysis": analysis,
        "provider_used": provider_used,
        "model_used": model_used,
        "error_code": "",
    }


def _err(
    *,
    image_source: str,
    analysis: str,
    provider_used: str,
    model_used: str,
    error_code: str,
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "image_source": image_source,
        "analysis": analysis,
        "provider_used": provider_used,
        "model_used": model_used,
        "error_code": error_code,
    }


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if not body:
        return ""
    compact = re.sub(r"\s+", " ", body).strip()
    return compact[:800] + ("..." if len(compact) > 800 else "")


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset, errors="replace"))


def _normalize_provider(raw: str) -> str:
    provider = str(raw or "").strip().lower()
    alias_map = {
        "openai": "openai_compatible",
        "openai-compatible": "openai_compatible",
        "openaicompat": "openai_compatible",
        "z.ai": "zai",
        "deepseek-ai": "deepseek",
    }
    return alias_map.get(provider, provider)


def _resolve_image_analysis_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    provider = _normalize_provider(
        _first_non_empty(
            str(args.provider or "").strip(),
            os.getenv("IMAGE_ANALYSIS_PROVIDER", ""),
            "none",
        )
    )
    model = _first_non_empty(
        str(args.model or "").strip(),
        os.getenv("IMAGE_ANALYSIS_MODEL", ""),
        "none",
    )
    base_url = str(args.base_url or "").strip() or os.getenv("IMAGE_ANALYSIS_BASE_URL", "")
    api_key = str(args.api_key or "").strip() or os.getenv("IMAGE_ANALYSIS_API_KEY", "")

    if provider in {"none", ""} or model in {"none", ""}:
        return provider or "none", model or "none", base_url, api_key

    if provider == "zai":
        if not base_url:
            base_url = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
        if not api_key:
            api_key = os.getenv("ZAI_API_KEY", "")
    elif provider == "deepseek":
        if not base_url:
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
    elif provider == "lmstudio":
        if not base_url:
            base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        if not api_key:
            api_key = os.getenv("LMSTUDIO_API_KEY", os.getenv("LM_API_TOKEN", ""))
    elif provider == "openai_compatible":
        if not base_url:
            base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "")
        if not api_key:
            api_key = os.getenv("OPENAI_COMPAT_API_KEY", os.getenv("LM_API_TOKEN", ""))
    elif provider == "ollama":
        if not base_url:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return provider, model, base_url, api_key


def _ensure_chat_endpoint(base_url: str) -> str:
    base = str(base_url).strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if "/v" in base.rsplit("/", 1)[-1]:
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
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
    text_value = first.get("text")
    if isinstance(text_value, str):
        return text_value
    return ""


def _guess_mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    return "image/jpeg"


def _read_local_image(path_text: str) -> tuple[bytes | None, str, str]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        return None, "", f"image_input_error: image file not found: {path}"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, "", f"image_input_error: failed reading image file: {exc}"
    if not data:
        return None, "", "image_input_error: image file is empty"
    return data, _guess_mime(path), ""


def _download_image(url: str, timeout: int) -> tuple[bytes | None, str, str]:
    req = Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (AgenticSystemImageSkill/1.0)",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(10_000_000)
            content_type = str(resp.headers.get("Content-Type", "")).strip()
    except (HTTPError, URLError) as exc:
        return None, "", f"image_input_error: failed downloading image url: {exc}"
    if not raw:
        return None, "", "image_input_error: downloaded image is empty"
    mime = content_type.split(";")[0].strip() if content_type else "image/jpeg"
    return raw, mime or "image/jpeg", ""


def _bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    image_url: str,
    timeout: int,
) -> tuple[str, str]:
    endpoint = _ensure_chat_endpoint(base_url)
    if not endpoint:
        return "", "vision_config_missing: missing base url for openai-compatible provider"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0.1,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        raw = _post_json(endpoint, headers=headers, payload=payload, timeout=timeout)
    except HTTPError as exc:
        body = _read_http_error_body(exc)
        detail = f"; body={body}" if body else ""
        return "", f"vision_runtime_error: request failed: HTTP {exc.code}{detail}"
    except URLError as exc:
        return "", f"vision_runtime_error: request failed: {exc}"
    except Exception as exc:
        return "", f"vision_runtime_error: unexpected request error: {exc}"

    text = _extract_chat_text(raw).strip()
    if not text:
        return "", "vision_runtime_error: empty model response"
    return text, ""


def _call_ollama(
    *,
    base_url: str,
    model: str,
    query: str,
    image_bytes: bytes,
    timeout: int,
) -> tuple[str, str]:
    endpoint = f"{str(base_url).strip().rstrip('/')}/api/chat"
    if endpoint.startswith("/api/chat"):
        return "", "vision_config_missing: missing base url for ollama provider"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": query,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    headers = {"Content-Type": "application/json"}
    try:
        raw = _post_json(endpoint, headers=headers, payload=payload, timeout=timeout)
    except HTTPError as exc:
        body = _read_http_error_body(exc)
        detail = f"; body={body}" if body else ""
        return "", f"vision_runtime_error: request failed: HTTP {exc.code}{detail}"
    except URLError as exc:
        return "", f"vision_runtime_error: request failed: {exc}"
    except Exception as exc:
        return "", f"vision_runtime_error: unexpected request error: {exc}"

    message = raw.get("message", {})
    if isinstance(message, dict):
        text = str(message.get("content", "")).strip()
        if text:
            return text, ""
    return "", "vision_runtime_error: empty model response"


def run_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    query = str(args.query or "").strip()
    image_url_arg = str(args.image_url or "").strip()
    image_path_arg = str(args.image_path or "").strip()
    image_source = image_url_arg if image_url_arg else image_path_arg

    provider, model, base_url, api_key = _resolve_image_analysis_config(args)
    if provider in {"none", ""}:
        out = _err(
            image_source=image_source,
            analysis=(
                "vision provider/model not configured; cannot run image analysis in runtime. "
                "Please return to requester and ask for vision config."
            ),
            provider_used=provider or "none",
            model_used=model or "none",
            error_code="vision_config_missing",
        )
        return out, 1

    if model in {"none", ""}:
        out = _err(
            image_source=image_source,
            analysis=(
                "vision model not configured; cannot run image analysis in runtime. "
                "Please return to requester and ask for vision config."
            ),
            provider_used=provider or "none",
            model_used=model or "none",
            error_code="vision_config_missing",
        )
        return out, 1

    if provider in {"zai", "deepseek"} and not str(api_key).strip():
        out = _err(
            image_source=image_source,
            analysis=(
                f"{provider} vision requires api key; cannot run image analysis in runtime. "
                "Please return to requester and ask for vision config."
            ),
            provider_used=provider,
            model_used=model,
            error_code="vision_config_missing",
        )
        return out, 1

    if provider not in {"openai_compatible", "zai", "deepseek", "lmstudio", "ollama"}:
        out = _err(
            image_source=image_source,
            analysis=f"unsupported vision provider: {provider}",
            provider_used=provider,
            model_used=model,
            error_code="vision_provider_unsupported",
        )
        return out, 1

    if not query:
        out = _err(
            image_source=image_source,
            analysis="image query context is required; provide --query",
            provider_used=provider,
            model_used=model,
            error_code="image_query_missing",
        )
        return out, 1

    image_bytes: bytes | None = None
    mime_type = "image/jpeg"
    input_error = ""
    if image_path_arg:
        image_bytes, mime_type, input_error = _read_local_image(image_path_arg)
    elif image_url_arg:
        if provider == "ollama":
            image_bytes, mime_type, input_error = _download_image(image_url_arg, timeout=max(5, int(args.timeout)))
    if input_error:
        out = _err(
            image_source=image_source,
            analysis=input_error,
            provider_used=provider,
            model_used=model,
            error_code="image_input_error",
        )
        return out, 1

    timeout = max(5, int(args.timeout))
    if provider == "ollama":
        if image_bytes is None:
            out = _err(
                image_source=image_source,
                analysis="image_input_error: ollama requires image bytes, but none were resolved",
                provider_used=provider,
                model_used=model,
                error_code="image_input_error",
            )
            return out, 1
        analysis, err = _call_ollama(
            base_url=base_url,
            model=model,
            query=query,
            image_bytes=image_bytes,
            timeout=timeout,
        )
    else:
        if image_path_arg:
            if image_bytes is None:
                out = _err(
                    image_source=image_source,
                    analysis="image_input_error: local image data missing",
                    provider_used=provider,
                    model_used=model,
                    error_code="image_input_error",
                )
                return out, 1
            image_for_model = _bytes_to_data_url(image_bytes, mime_type)
        else:
            image_for_model = image_url_arg
        analysis, err = _call_openai_compatible(
            base_url=base_url,
            api_key=api_key,
            model=model,
            query=query,
            image_url=image_for_model,
            timeout=timeout,
        )

    if err:
        error_code = "vision_runtime_error"
        if err.startswith("vision_config_missing"):
            error_code = "vision_config_missing"
        out = _err(
            image_source=image_source,
            analysis=err,
            provider_used=provider,
            model_used=model,
            error_code=error_code,
        )
        return out, 1

    out = _ok(
        image_source=image_source,
        analysis=analysis,
        provider_used=provider,
        model_used=model,
    )
    return out, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze image content with vision-capable provider/model.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image-url", default="")
    source_group.add_argument("--image-path", default="")
    parser.add_argument("--query", required=True)
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--timeout",
        default=_first_non_empty(
            os.getenv("IMAGE_ANALYSIS_TIMEOUT_SECONDS", ""),
            "120",
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out, code = run_analysis(args)
        print(json.dumps(out, ensure_ascii=True))
        return code
    except Exception as exc:
        out = _err(
            image_source=str(args.image_url or args.image_path or ""),
            analysis=f"unexpected runtime exception: {exc}",
            provider_used=_first_non_empty(
                str(args.provider or "").strip(),
                os.getenv("IMAGE_ANALYSIS_PROVIDER", ""),
                "none",
            ),
            model_used=_first_non_empty(
                str(args.model or "").strip(),
                os.getenv("IMAGE_ANALYSIS_MODEL", ""),
                "none",
            ),
            error_code="vision_unexpected_exception",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
