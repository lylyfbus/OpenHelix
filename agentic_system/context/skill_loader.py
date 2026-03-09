"""Skill metadata loader — scans skills/ directories for SKILL.md frontmatter.

Produces a compact JSON summary of available skills for injection
into the agent's system prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse simple YAML-like frontmatter from a SKILL.md file.

    Expected format::

        ---
        name: my-skill
        description: Does something useful
        handler: scripts/run.py
        required_tools: bash
        ---
        ... rest of SKILL.md ...
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end == -1:
        return {}
    result: dict[str, str] = {}
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_csv(value: str) -> list[str]:
    """Split comma-separated field into trimmed list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_skills(skills_root: Path) -> list[dict[str, Any]]:
    """Scan skills directory tree and return metadata rows.

    Expected structure::

        skills/
          all-agents/
            skill-name/
              SKILL.md         <- frontmatter parsed
              scripts/
                handler.py
          core-agent/
            ...

    Returns:
        List of skill metadata dicts with keys:
        skill_id, scope, path, handler, name, description,
        required_tools, recommended_tools, forbidden_tools
    """
    skills_root = Path(skills_root)
    if not skills_root.exists():
        return []

    # Built-in loader skill IDs are excluded (they become
    # built-in reference loader entries in the prompt instead)
    builtin_ids = {"load-skill", "load-knowledge-docs"}

    rows: list[dict[str, Any]] = []
    for scope_dir in sorted(skills_root.iterdir()):
        if not scope_dir.is_dir():
            continue
        scope = scope_dir.name  # e.g. "all-agents", "core-agent"
        for skill_dir in sorted(scope_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_id = skill_dir.name
            if skill_id in builtin_ids:
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                text = skill_md.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            name = fm.get("name", skill_id).strip() or skill_id
            handler = fm.get("handler", "").strip()
            description = fm.get("description", "").strip()
            path = f"skills/{scope}/{skill_id}"
            handler_path = f"{path}/{handler}" if handler else ""

            rows.append({
                "skill_id": skill_id,
                "scope": scope,
                "path": path,
                "handler": handler_path,
                "name": name,
                "description": description,
                "required_tools": _parse_csv(fm.get("required_tools", "")),
                "recommended_tools": _parse_csv(fm.get("recommended_tools", "")),
                "forbidden_tools": _parse_csv(fm.get("forbidden_tools", "")),
            })

    rows.sort(key=lambda r: (r["scope"], r["skill_id"]))
    return rows


def format_skills_for_prompt(skills: list[dict[str, Any]]) -> str:
    """Format skill metadata rows into a prompt-injectable text block."""
    if not skills:
        return "- (no skills found)"
    return "\n".join("- " + json.dumps(row, ensure_ascii=True) for row in skills)
