"""Persistent registry of sub-agents created during a session.

This module owns the index file at ``{state_root}/sub_agents_meta.json`` —
the per-session lookup of "which sub-agent personas exist and what do they
do". Per-sub-agent conversation state (full history, observation window,
workflow summary, last prompt) is a separate concern handled by
``Environment.save_session`` / ``Environment.load_session``, which write to
``{state_root}/sub_agents/{role}.json``.

Keeping the meta registry out of ``loop.py`` makes it clear that it's a
session-level artifact, not part of the agent loop mechanics — both the
``run_loop`` delegation path and the ``RuntimeHost`` command handlers can
import from here without circular dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path


_META_FILENAME = "sub_agents_meta.json"


def load(state_root: Path) -> list[dict]:
    """Read the sub-agents meta registry. Returns [] if missing or malformed."""
    meta_path = state_root / _META_FILENAME
    if not meta_path.exists():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save(state_root: Path, meta: list[dict]) -> None:
    """Atomically write the sub-agents meta registry to disk."""
    meta_path = state_root / _META_FILENAME
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)


def update(state_root: Path, role: str, description: str) -> None:
    """Add or update a sub-agent entry in the meta registry."""
    meta = load(state_root)
    for entry in meta:
        if entry.get("role") == role:
            if description:
                entry["description"] = description
            save(state_root, meta)
            return
    meta.append({
        "role": role,
        "description": description or f"Sub-agent: {role}",
    })
    save(state_root, meta)


def format_for_prompt(meta: list[dict]) -> str:
    """Render the meta registry as the bulleted text the system prompt expects.

    Empty string when no sub-agents exist yet — the prompt template falls
    back to "- (no sub-agents created yet)" via its own `or` clause.
    """
    if not meta:
        return ""
    return "\n".join(
        f"- {entry.get('role', '?')}: {entry.get('description', '')}"
        for entry in meta
    )
