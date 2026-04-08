"""Model-spec validation and cache-layout helpers for local model service."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .paths import _backend_cache_root, _safe_model_dir_name
from .protocol import _canonical_task_type


_PREPARED_MARKER_NAME = ".helix-model-prepared.json"

# Populated at startup by adapter_discovery.discover_and_register_builtins()
# and discover_and_register() for workspace skills.
_SUPPORTED_MODEL_FAMILIES: dict[str, dict[str, Any]] = {}


def register_family(family: str, *, backend: str, task_types: set[str]) -> None:
    """Register a new model family (e.g. from a skill adapter)."""
    _SUPPORTED_MODEL_FAMILIES[family] = {
        "backend": backend,
        "task_types": set(task_types),
    }


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
    result: list[str] = []
    for index, item in enumerate(value):
        token = str(item or "").strip()
        if not token:
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        result.append(token)
    return result


def normalize_model_spec(
    model_spec: dict[str, Any],
    *,
    override_task_type: str | None = None,
) -> dict[str, Any]:
    if not isinstance(model_spec, dict):
        raise ValueError("model_spec must be a JSON object")

    family = _require_string(model_spec.get("family"), name="model_spec.family")
    family_config = _SUPPORTED_MODEL_FAMILIES.get(family)
    if family_config is None:
        raise ValueError(f"unsupported model family: {family}")

    backend = _require_string(model_spec.get("backend"), name="model_spec.backend").lower()
    if backend != family_config["backend"]:
        raise ValueError(
            f"model_spec.backend must be {family_config['backend']} for family {family}"
        )

    raw_task_type = override_task_type if override_task_type not in (None, "") else model_spec.get("task_type")
    task_type = _canonical_task_type(str(raw_task_type or "").strip())
    if not task_type:
        raise ValueError("model_spec.task_type is required")
    if task_type not in family_config["task_types"]:
        raise ValueError(f"family {family} does not support task_type {task_type}")

    source = model_spec.get("source")
    if not isinstance(source, dict):
        raise ValueError("model_spec.source must be a JSON object")
    repo_id = _require_string(source.get("repo_id"), name="model_spec.source.repo_id")

    download_manifest = model_spec.get("download_manifest")
    if not isinstance(download_manifest, dict):
        raise ValueError("model_spec.download_manifest must be a JSON object")
    include = _require_string_list(
        download_manifest.get("include"),
        name="model_spec.download_manifest.include",
    )
    exclude = _require_string_list(
        download_manifest.get("exclude"),
        name="model_spec.download_manifest.exclude",
    )
    required = _require_string_list(
        download_manifest.get("required"),
        name="model_spec.download_manifest.required",
    )
    if not required:
        raise ValueError("model_spec.download_manifest.required must contain at least one path pattern")

    load_config = model_spec.get("load_config")
    if not isinstance(load_config, dict):
        raise ValueError("model_spec.load_config must be a JSON object")

    defaults = model_spec.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("model_spec.defaults must be a JSON object when provided")

    prerequisites = model_spec.get("prerequisites") or {}
    if not isinstance(prerequisites, dict):
        raise ValueError("model_spec.prerequisites must be a JSON object when provided")

    spec_id = _require_string(model_spec.get("id"), name="model_spec.id")
    return {
        "id": spec_id,
        "backend": backend,
        "task_type": task_type,
        "family": family,
        "source": {
            "repo_id": repo_id,
        },
        "download_manifest": {
            "include": include,
            "exclude": exclude,
            "required": required,
        },
        "load_config": dict(load_config),
        "defaults": dict(defaults),
        "prerequisites": dict(prerequisites),
    }


def model_spec_display_id(model_spec: dict[str, Any]) -> str:
    normalized = normalize_model_spec(model_spec)
    return normalized["id"]


def model_spec_signature(model_spec: dict[str, Any], *, override_task_type: str | None = None) -> str:
    normalized = normalize_model_spec(model_spec, override_task_type=override_task_type)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def model_spec_backend_cache_root(cache_root: Path, model_spec: dict[str, Any]) -> Path:
    normalized = normalize_model_spec(model_spec)
    return _backend_cache_root(Path(cache_root).expanduser().resolve(), normalized["backend"])


def model_spec_model_root(cache_root: Path, model_spec: dict[str, Any]) -> Path:
    normalized = normalize_model_spec(model_spec)
    backend_root = model_spec_backend_cache_root(cache_root, normalized)
    repo_id = normalized["source"]["repo_id"]
    return (backend_root / "models" / _safe_model_dir_name(repo_id)).resolve()


def prepared_marker_path(model_root: Path) -> Path:
    return Path(model_root).expanduser().resolve() / _PREPARED_MARKER_NAME


def manifest_matches(model_root: Path, model_spec: dict[str, Any]) -> bool:
    normalized = normalize_model_spec(model_spec)
    root = Path(model_root).expanduser().resolve()
    if not root.exists():
        return False
    for pattern in normalized["download_manifest"]["required"]:
        if not any(root.glob(pattern)):
            return False
    return True

