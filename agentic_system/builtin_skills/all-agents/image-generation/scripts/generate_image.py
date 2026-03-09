#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_EXECUTED_SKILL = "image-generation"


def _first_non_empty(*values: str) -> str:
    for value in values:
        token = str(value or "").strip()
        if token:
            return token
    return ""


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _resolve_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    provider = _normalize_provider(
        _first_non_empty(
            str(args.provider or "").strip(),
            os.getenv("IMAGE_GENERATION_PROVIDER", ""),
            "ollama",
        )
    )
    model = _first_non_empty(
        str(args.model or "").strip(),
        os.getenv("IMAGE_GENERATION_MODEL", ""),
        "x/z-image-turbo",
    )

    base_url = _first_non_empty(
        str(args.base_url or "").strip(),
        os.getenv("IMAGE_GENERATION_BASE_URL", ""),
    )
    api_key = _first_non_empty(
        str(args.api_key or "").strip(),
        os.getenv("IMAGE_GENERATION_API_KEY", ""),
    )

    if provider == "ollama":
        if not base_url:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        if not api_key:
            api_key = os.getenv("OLLAMA_API_KEY", "")
    elif provider == "zai":
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

    return provider, model, base_url, api_key


def _ensure_images_endpoint(base_url: str) -> str:
    base = str(base_url).strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/v1/images/generations"):
        return base
    if base.endswith("/v1"):
        return f"{base}/images/generations"
    return f"{base}/v1/images/generations"


def _image_extension_from_bytes(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _write_output(
    *,
    image_bytes: bytes,
    output_path_arg: str,
    output_dir_arg: str,
    model: str,
) -> str:
    output_path_raw = str(output_path_arg).strip()
    if output_path_raw:
        path = Path(output_path_raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        out_dir = Path(output_dir_arg).expanduser()
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-") or "model"
        stamp = _utc_now_compact()
        path = out_dir / f"image_{safe_model}_{stamp}"

    if not path.suffix:
        path = path.with_suffix(_image_extension_from_bytes(image_bytes))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)

    cwd = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(cwd))
    except ValueError:
        return str(path.resolve())


def _ok(
    *,
    prompt: str,
    output_path: str,
    provider_used: str,
    model_used: str,
    generation_result: str,
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "prompt": prompt,
        "output_path": output_path,
        "provider_used": provider_used,
        "model_used": model_used,
        "error_code": "",
        "generation_result": generation_result,
    }


def _err(
    *,
    prompt: str,
    output_path: str,
    provider_used: str,
    model_used: str,
    error_code: str,
    generation_result: str,
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "prompt": prompt,
        "output_path": output_path,
        "provider_used": provider_used,
        "model_used": model_used,
        "error_code": error_code,
        "generation_result": generation_result,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    prompt = str(args.prompt or "").strip()
    if not prompt:
        return (
            _err(
                prompt="",
                output_path="",
                provider_used="none",
                model_used="none",
                error_code="image_prompt_missing",
                generation_result="image prompt is required; provide --prompt",
            ),
            1,
        )

    provider, model, base_url, api_key = _resolve_config(args)
    if provider in {"", "none"} or model in {"", "none"}:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider or "none",
                model_used=model or "none",
                error_code="image_config_missing",
                generation_result=(
                    "image generation provider/model is not configured; please return to requester for config"
                ),
            ),
            1,
        )

    if provider not in {"ollama", "openai_compatible", "zai", "deepseek", "lmstudio"}:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_provider_unsupported",
                generation_result=f"unsupported image generation provider: {provider}",
            ),
            1,
        )

    endpoint = _ensure_images_endpoint(base_url)
    if not endpoint:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_config_missing",
                generation_result="missing base url for image generation provider",
            ),
            1,
        )

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": max(1, int(args.n)),
        "size": str(args.size),
        "response_format": "b64_json",
    }
    if str(args.style).strip():
        payload["style"] = str(args.style).strip()
    if str(args.quality).strip():
        payload["quality"] = str(args.quality).strip()

    headers = {"Content-Type": "application/json"}
    if str(api_key).strip():
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = max(5, int(args.timeout))
    try:
        raw = _post_json(endpoint, headers=headers, payload=payload, timeout=timeout)
    except HTTPError as exc:
        body = _read_http_error_body(exc)
        detail = f"; body={body}" if body else ""
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_generation_unavailable",
                generation_result=f"request failed: HTTP {exc.code}{detail}",
            ),
            1,
        )
    except URLError as exc:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_generation_unavailable",
                generation_result=f"request failed: {exc}",
            ),
            1,
        )
    except Exception as exc:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_generation_unavailable",
                generation_result=f"unexpected request error: {exc}",
            ),
            1,
        )

    data = raw.get("data", [])
    if not isinstance(data, list) or not data:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_generation_empty",
                generation_result="empty image generation response: missing data[]",
            ),
            1,
        )

    first = data[0] if isinstance(data[0], dict) else {}
    b64 = str(first.get("b64_json", "")).strip()
    if not b64:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_generation_empty",
                generation_result="empty image generation response: missing b64_json",
            ),
            1,
        )

    try:
        image_bytes = base64.b64decode(b64)
    except Exception as exc:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_decode_error",
                generation_result=f"failed decoding b64 image: {exc}",
            ),
            1,
        )

    if not image_bytes:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_decode_error",
                generation_result="decoded image bytes are empty",
            ),
            1,
        )

    try:
        saved_path = _write_output(
            image_bytes=image_bytes,
            output_path_arg=str(args.output_path),
            output_dir_arg=str(args.output_dir),
            model=model,
        )
    except Exception as exc:
        return (
            _err(
                prompt=prompt,
                output_path="",
                provider_used=provider,
                model_used=model,
                error_code="image_output_write_error",
                generation_result=f"failed writing output image: {exc}",
            ),
            1,
        )

    return (
        _ok(
            prompt=prompt,
            output_path=saved_path,
            provider_used=provider,
            model_used=model,
            generation_result=f"generated 1 image and saved to {saved_path}",
        ),
        0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image from prompt using a model provider.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--style", default="")
    parser.add_argument("--quality", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out, code = run(args)
        print(json.dumps(out, ensure_ascii=True))
        return int(code)
    except Exception as exc:
        out = _err(
            prompt=str(args.prompt or ""),
            output_path="",
            provider_used=_first_non_empty(
                str(args.provider or "").strip(),
                os.getenv("IMAGE_GENERATION_PROVIDER", ""),
                "ollama",
            ),
            model_used=_first_non_empty(
                str(args.model or "").strip(),
                os.getenv("IMAGE_GENERATION_MODEL", ""),
                "x/z-image-turbo",
            ),
            error_code="image_unexpected_exception",
            generation_result=f"unexpected runtime exception: {exc}",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
