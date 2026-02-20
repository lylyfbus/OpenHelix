from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_EXECUTED_SKILL = "skill-authorization"


def _ok(skill_target: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "skill_created/updated": skill_target,
    }


def _err(skill_target: str = "") -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "skill_created/updated": skill_target,
    }


def _read_skill_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

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

    out: dict[str, str] = {}
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _skill_template(skill_name: str, description: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {skill_name}",
            "handler:",
            f"description: {description}",
            "required_tools: exec",
            "recommended_tools: exec",
            "forbidden_tools:",
            "---",
            "",
            "# Purpose",
            "",
            "Describe what this skill does.",
            "",
            "# When To Use",
            "",
            "Describe clear trigger conditions.",
            "",
            "# Procedure",
            "",
            "List deterministic steps.",
            "",
            "# Action Input Templates",
            "",
            "Provide concrete action_input examples.",
            "",
            "# Notes",
            "",
            "Add constraints and caveats.",
            "",
        ]
    )


def _normalize_skill_name(skill_id: str) -> str:
    return " ".join(part.capitalize() for part in skill_id.split("-") if part)


def run_inspect(workspace: Path, skill_id: str, scope: str) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    skill_md = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"

    exists = skill_dir.exists() and skill_dir.is_dir()
    skill_md_exists = skill_md.exists()
    frontmatter = _read_skill_frontmatter(skill_md) if skill_md_exists else {}

    scripts: list[str] = []
    if scripts_dir.exists() and scripts_dir.is_dir():
        for p in sorted(scripts_dir.rglob("*")):
            if p.is_file():
                scripts.append(str(p.relative_to(workspace)))

    _ = (skill_md_exists, frontmatter, scripts, workspace)
    return _ok(skill_target=(skill_id if exists else ""))


def run_scaffold(workspace: Path, skill_id: str, scope: str, description: str, overwrite: bool) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    scripts_dir = skill_dir / "scripts"
    skill_md = skill_dir / "SKILL.md"

    created: list[str] = []
    updated: list[str] = []

    if not skill_dir.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(skill_dir.relative_to(workspace)))

    if not scripts_dir.exists():
        scripts_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(scripts_dir.relative_to(workspace)))

    skill_name = _normalize_skill_name(skill_id)
    default_description = description.strip() or f"Describe purpose for {skill_id}."
    template = _skill_template(skill_name, default_description)

    if not skill_md.exists():
        skill_md.write_text(template, encoding="utf-8")
        created.append(str(skill_md.relative_to(workspace)))
    elif overwrite:
        skill_md.write_text(template, encoding="utf-8")
        updated.append(str(skill_md.relative_to(workspace)))

    _ = (workspace, created, updated)
    return _ok(skill_target=skill_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or scaffold skill packages with structured JSON output.")
    parser.add_argument("--action", required=True, choices=["inspect", "scaffold"])
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--scope", required=True, choices=["all-agents", "core-agent"])
    parser.add_argument("--description", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workspace", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action = str(args.action)
    skill_id = str(args.skill_id).strip()
    scope = str(args.scope).strip()
    workspace = Path(args.workspace).expanduser().resolve()

    if not _SKILL_ID_RE.match(skill_id):
        out = _err()
        print(json.dumps(out, ensure_ascii=True))
        return 1

    try:
        if action == "inspect":
            out = run_inspect(workspace=workspace, skill_id=skill_id, scope=scope)
        else:
            out = run_scaffold(
                workspace=workspace,
                skill_id=skill_id,
                scope=scope,
                description=str(args.description),
                overwrite=bool(args.overwrite),
            )
        print(json.dumps(out, ensure_ascii=True))
        return 0 if out.get("status") == "ok" else 1
    except Exception:  # unexpected runtime failure
        out = _err()
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
