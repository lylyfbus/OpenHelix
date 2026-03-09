#!/usr/bin/env python3
"""Initialize planning files from templates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EXECUTED_SKILL = "file-based-planning"
_ACTION = "init"
_TEMPLATE_NAMES = ("task_plan.md", "findings.md", "progress.md")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_templates_dir(raw_templates_dir: str) -> Path:
    raw = str(raw_templates_dir or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    return (Path(__file__).resolve().parent.parent / "templates").resolve()


def _build_output(
    *,
    status: str,
    project_name: str,
    templates_dir: Path,
    created: list[str],
    skipped: list[str],
    errors: list[str],
    message: str,
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "action": _ACTION,
        "status": status,
        "project_name": project_name,
        "templates_dir": str(templates_dir),
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "message": message,
        "timestamp": _utc_now_iso(),
    }


def init_planning(*, project_name: str, templates_dir: Path) -> tuple[dict[str, Any], int]:
    cwd = Path.cwd()
    created: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for filename in _TEMPLATE_NAMES:
        template_path = templates_dir / filename
        target_path = cwd / filename

        if target_path.exists():
            skipped.append(filename)
            continue

        if not template_path.exists():
            errors.append(f"template_missing:{filename}")
            continue

        try:
            content = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"template_read_error:{filename}:{exc}")
            continue

        content = content.replace("{{DATE}}", datetime.now().strftime("%Y-%m-%d"))
        content = content.replace("{{PROJECT}}", project_name)

        try:
            target_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            errors.append(f"target_write_error:{filename}:{exc}")
            continue

        created.append(filename)

    if errors and not created:
        status = "error"
        code = 1
    elif errors:
        status = "partial"
        code = 0
    else:
        status = "ok"
        code = 0

    message = (
        f"initialized {len(created)} planning file(s); skipped {len(skipped)} existing; "
        f"errors {len(errors)}"
    )

    return (
        _build_output(
            status=status,
            project_name=project_name,
            templates_dir=templates_dir,
            created=created,
            skipped=skipped,
            errors=errors,
            message=message,
        ),
        code,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize planning files from skill templates.")
    parser.add_argument("--project-name", default="project", help="Name of the project/task")
    parser.add_argument("--templates-dir", default="", help="Optional templates directory override")
    parser.add_argument("--dry-run", action="store_true", help="Show resolved paths without writing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_name = str(args.project_name or "project").strip() or "project"

    try:
        templates_dir = _resolve_templates_dir(str(args.templates_dir))

        if bool(args.dry_run):
            out = _build_output(
                status="dry_run",
                project_name=project_name,
                templates_dir=templates_dir,
                created=[],
                skipped=[],
                errors=[],
                message="dry_run only; no files were written",
            )
            print(json.dumps(out, ensure_ascii=True))
            return 0

        out, code = init_planning(project_name=project_name, templates_dir=templates_dir)
        print(json.dumps(out, ensure_ascii=True))
        return int(code)
    except Exception as exc:  # unexpected runtime failure
        out = _build_output(
            status="error",
            project_name=project_name,
            templates_dir=Path("."),
            created=[],
            skipped=[],
            errors=[f"unexpected_exception:{exc}"],
            message="unexpected runtime failure",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
