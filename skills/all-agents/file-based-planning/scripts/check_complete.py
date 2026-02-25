#!/usr/bin/env python3
"""Check whether all phases in task_plan.md are complete."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_EXECUTED_SKILL = "file-based-planning"
_ACTION = "check_complete"


def _output(
    *,
    status: str,
    message: str,
    total: int,
    complete: int,
    in_progress: int,
    pending: int,
    error_code: str = "",
) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "action": _ACTION,
        "status": status,
        "message": message,
        "total": int(total),
        "complete": int(complete),
        "in_progress": int(in_progress),
        "pending": int(pending),
        "error_code": error_code,
    }


def check_complete(plan_file: str = "task_plan.md") -> tuple[dict[str, Any], int]:
    plan_path = Path.cwd() / str(plan_file)

    if not plan_path.exists():
        return (
            _output(
                status="no_plan",
                message="No task_plan.md found - no active planning session",
                total=0,
                complete=0,
                in_progress=0,
                pending=0,
            ),
            0,
        )

    try:
        content = plan_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return (
            _output(
                status="error",
                message=f"Failed to read plan file: {exc}",
                total=0,
                complete=0,
                in_progress=0,
                pending=0,
                error_code="plan_read_error",
            ),
            1,
        )

    complete = len(re.findall(r"\*\*Status:\*\*\s*complete", content, re.IGNORECASE))
    in_progress = len(re.findall(r"\*\*Status:\*\*\s*in_progress", content, re.IGNORECASE))
    pending = len(re.findall(r"\*\*Status:\*\*\s*pending", content, re.IGNORECASE))

    if complete == 0 and in_progress == 0 and pending == 0:
        complete = len(re.findall(r"\[complete\]", content, re.IGNORECASE))
        in_progress = len(re.findall(r"\[in_progress\]", content, re.IGNORECASE))
        pending = len(re.findall(r"\[pending\]", content, re.IGNORECASE))

    total = complete + in_progress + pending

    if total == 0:
        return (
            _output(
                status="no_phases",
                message="No phases found in task_plan.md",
                total=0,
                complete=0,
                in_progress=0,
                pending=0,
            ),
            0,
        )

    if complete == total:
        return (
            _output(
                status="complete",
                message=f"ALL PHASES COMPLETE ({complete}/{total})",
                total=total,
                complete=complete,
                in_progress=in_progress,
                pending=pending,
            ),
            0,
        )

    return (
        _output(
            status="in_progress",
            message=f"Task in progress ({complete}/{total} phases complete)",
            total=total,
            complete=complete,
            in_progress=in_progress,
            pending=pending,
        ),
        0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check planning completion status")
    parser.add_argument("--plan-file", default="task_plan.md", help="Path to task plan file")
    args = parser.parse_args()

    try:
        out, code = check_complete(args.plan_file)
        print(json.dumps(out, ensure_ascii=True))
        return int(code)
    except Exception as exc:  # unexpected runtime failure
        out = _output(
            status="error",
            message=f"unexpected runtime exception: {exc}",
            total=0,
            complete=0,
            in_progress=0,
            pending=0,
            error_code="unexpected_exception",
        )
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
