"""Model and source file preparation for local model inference."""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from .constants import MODELS_SUBDIR, SERVICE_ROOT, VENVS_SUBDIR, sources_path
from .model_spec import manifest_matches, normalize_model_spec
from .helpers import _ensure_worker_dependencies, _worker_python
_HF_CLI_DEPENDENCIES = ("huggingface_hub[cli]",)


def download_model(
    *,
    model_spec: dict[str, Any],
    backend_mode: str,
    timeout_seconds: int,
    progress_stream: Any,
) -> tuple[dict[str, Any], Path]:
    """Download model weights and optional source files.

    Skips downloading if all required files already exist.
    """
    normalized = normalize_model_spec(model_spec)
    _check_prerequisites(normalized)
    repo_id = normalized["source"]["repo_id"]
    model_root = SERVICE_ROOT / MODELS_SUBDIR / repo_id.replace("/", "--")

    if backend_mode == "fake":
        model_root.mkdir(parents=True, exist_ok=True)
        return normalized, model_root

    # Skip if already complete
    if model_root.exists() and manifest_matches(model_root, normalized):
        progress_stream.write(f"Model {repo_id} already downloaded, skipping.\n")
        _download_sources(normalized, progress_stream)
        return normalized, model_root

    # Download model weights from HuggingFace
    venv_root = SERVICE_ROOT / VENVS_SUBDIR / normalized["backend"]
    venv_root.mkdir(parents=True, exist_ok=True)
    python_bin = _worker_python(venv_root)
    _ensure_worker_dependencies(python_bin, _HF_CLI_DEPENDENCIES)

    env = os.environ.copy()
    hub_root = str(SERVICE_ROOT / MODELS_SUBDIR)
    env.setdefault("HF_HOME", hub_root)
    env.setdefault("TRANSFORMERS_CACHE", hub_root)
    env.setdefault("HF_HUB_CACHE", hub_root)
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    cmd = _hf_download_command(
        python_bin=python_bin,
        repo_id=repo_id,
        local_dir=model_root,
        include_patterns=list(normalized["download_manifest"]["include"]),
        exclude_patterns=list(normalized["download_manifest"]["exclude"]),
    )
    completed = subprocess.run(
        cmd, stdout=progress_stream, stderr=progress_stream,
        text=True, check=False, timeout=max(30, int(timeout_seconds)), env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"failed downloading {repo_id}; see terminal output above")
    if not manifest_matches(model_root, normalized):
        raise RuntimeError(f"prepared files are incomplete for {repo_id}")

    # Download source files if specified
    _download_sources(normalized, progress_stream)

    return normalized, model_root


def _download_sources(normalized: dict[str, Any], progress_stream: Any) -> None:
    """Download runtime source files if model_spec defines them."""
    sources = normalized.get("sources")
    if not isinstance(sources, dict):
        return
    skill_name = str(sources.get("skill_name", "")).strip()
    repo_url = str(sources.get("repo", "")).strip()
    commit = str(sources.get("commit", "")).strip()
    files = sources.get("files", [])
    if not all((skill_name, repo_url, commit, files)):
        return

    sources_root = sources_path(skill_name, commit)
    sources_root.mkdir(parents=True, exist_ok=True)

    for filename in files:
        target = sources_root / filename
        if target.exists():
            continue
        url = f"{repo_url.rstrip('/')}/{commit}/{filename}"
        progress_stream.write(f"Downloading source: {filename}\n")
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            target.write_bytes(resp.read())


def _hf_download_command(
    *, python_bin: Path, repo_id: str, local_dir: Path,
    include_patterns: list[str], exclude_patterns: list[str],
) -> list[str]:
    bin_dir = python_bin.parent
    hf_bin = bin_dir / "hf"
    legacy_bin = bin_dir / "huggingface-cli"
    if hf_bin.exists():
        cli = [str(hf_bin)]
    elif legacy_bin.exists():
        cli = [str(legacy_bin)]
    else:
        cli = [str(python_bin), "-m", "huggingface_hub.commands.huggingface_cli"]
    cmd = [*cli, "download", repo_id]
    if include_patterns and not exclude_patterns:
        cmd.extend(include_patterns)
    else:
        for pattern in include_patterns:
            cmd.extend(["--include", pattern])
        for pattern in exclude_patterns:
            cmd.extend(["--exclude", pattern])
    cmd.extend(["--local-dir", str(local_dir)])
    return cmd


def _check_prerequisites(model_spec: dict[str, Any]) -> None:
    prerequisites = model_spec.get("prerequisites") or {}
    binaries = prerequisites.get("host_binaries")
    if binaries in (None, ""):
        return
    if not isinstance(binaries, list):
        raise RuntimeError("model_spec.prerequisites.host_binaries must be a list of strings")
    missing = [str(name).strip() for name in binaries if str(name).strip() and shutil.which(str(name).strip()) is None]
    if missing:
        install_hint = str(prerequisites.get("install_hint", "")).strip()
        suffix = f" {install_hint}" if install_hint else ""
        raise RuntimeError(f"missing required host binaries: {', '.join(missing)}.{suffix}".strip())
