from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_EXECUTED_SKILL = "load-skill"

def _ok(skill_context: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "skill_context": skill_context,
    }


def _err(skill_context: str = "") -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "skill_context": skill_context,
    }


def _list_script_paths(skill_dir: Path) -> list[str]:
    out: list[str] = []
    for file_path in sorted(path for path in skill_dir.rglob("*") if path.is_file()):
        rel = file_path.relative_to(skill_dir)
        if rel.parts and rel.parts[0] != "scripts":
            continue
        if "__pycache__" in rel.parts or file_path.suffix == ".pyc":
            continue
        out.append(str(rel))
    return out


def run_understand(workspace: Path, skill_id: str, scope: str) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    skill_md_path = skill_dir / "SKILL.md"

    if not skill_dir.exists() or not skill_dir.is_dir():
        return _err()
    if not skill_md_path.exists():
        return _err()

    try:
        skill_md = skill_md_path.read_text(encoding="utf-8")
    except OSError:
        return _err()

    scripts = _list_script_paths(skill_dir)
    scripts_text = "\n".join(f"- {path}" for path in scripts) if scripts else "- (none)"
    skill_context = "\n".join(
        [
            "Skill #1",
            f"skill_id: {skill_id}",
            f"scope: {scope}",
            "skill_md:",
            skill_md.strip() if skill_md.strip() else "(empty)",
            "scripts:",
            scripts_text,
        ]
    )

    _ = (skill_md_path, workspace, scripts)
    return _ok(skill_context=skill_context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a skill's full context from runtime workspace.")
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--scope", required=True, choices=["all-agents", "core-agent"])
    parser.add_argument("--workspace", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skill_id = str(args.skill_id).strip()
    scope = str(args.scope).strip()
    workspace = Path(args.workspace).expanduser().resolve()

    if not skill_id:
        out = _err()
        print(json.dumps(out, ensure_ascii=True))
        return 1

    try:
        out = run_understand(workspace=workspace, skill_id=skill_id, scope=scope)
        print(json.dumps(out, ensure_ascii=True))
        return 0 if out.get("status") == "ok" else 1
    except Exception:
        out = _err()
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
