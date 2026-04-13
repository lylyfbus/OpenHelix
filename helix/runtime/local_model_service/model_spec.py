"""Model-spec validation helpers for local model service."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def _require_string(value: Any, *, name: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"{name} is required")
    return token


def _require_string_list(value: Any, *, name: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    return [
        _require_string(item, name=f"{name}[{i}]")
        for i, item in enumerate(value)
    ]


def normalize_model_spec(model_spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a model_spec for downloading."""
    if not isinstance(model_spec, dict):
        raise ValueError("model_spec must be a JSON object")

    backend = _require_string(model_spec.get("backend"), name="model_spec.backend").lower()

    source = model_spec.get("source")
    if not isinstance(source, dict):
        raise ValueError("model_spec.source must be a JSON object")
    repo_id = _require_string(source.get("repo_id"), name="model_spec.source.repo_id")

    download_manifest = model_spec.get("download_manifest")
    if not isinstance(download_manifest, dict):
        raise ValueError("model_spec.download_manifest must be a JSON object")
    include = _require_string_list(download_manifest.get("include"), name="model_spec.download_manifest.include")
    exclude = _require_string_list(download_manifest.get("exclude"), name="model_spec.download_manifest.exclude")
    required = _require_string_list(download_manifest.get("required"), name="model_spec.download_manifest.required")
    if not required:
        raise ValueError("model_spec.download_manifest.required must contain at least one path pattern")

    prerequisites = model_spec.get("prerequisites") or {}
    if not isinstance(prerequisites, dict):
        raise ValueError("model_spec.prerequisites must be a JSON object when provided")

    result: dict[str, Any] = {
        "backend": backend,
        "source": {"repo_id": repo_id},
        "download_manifest": {"include": include, "exclude": exclude, "required": required},
        "prerequisites": dict(prerequisites),
    }
    sources = model_spec.get("sources")
    if isinstance(sources, dict):
        result["sources"] = dict(sources)
    return result


def model_spec_signature(model_spec: dict[str, Any]) -> str:
    """Compute a short hash of the normalized model spec."""
    normalized = normalize_model_spec(model_spec)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def manifest_matches(model_root: Path, model_spec: dict[str, Any]) -> bool:
    """Check if all required files exist in the model root."""
    root = Path(model_root).expanduser().resolve()
    if not root.exists():
        return False
    for pattern in model_spec["download_manifest"]["required"]:
        if not any(root.glob(pattern)):
            return False
    return True
