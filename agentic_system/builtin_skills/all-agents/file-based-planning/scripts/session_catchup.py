#!/usr/bin/env python3
"""Session catchup helper for file-based-planning skill."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_EXECUTED_SKILL = "file-based-planning"
_ACTION = "session_catchup"
_PLANNING_FILES = ("task_plan.md", "findings.md", "progress.md")


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "exists": True,
            "error": f"stat_error:{exc}",
        }

    info: dict[str, Any] = {
        "exists": True,
        "size": int(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "modified_epoch": float(stat.st_mtime),
        "lines": None,
    }

    try:
        info["lines"] = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError as exc:
        info["error"] = f"read_error:{exc}"

    return info


def _analyze_task_plan_content(content: str) -> dict[str, Any]:
    complete = len(re.findall(r"\*\*Status:\*\*\s*complete", content, re.IGNORECASE))
    in_progress = len(re.findall(r"\*\*Status:\*\*\s*in_progress", content, re.IGNORECASE))
    pending = len(re.findall(r"\*\*Status:\*\*\s*pending", content, re.IGNORECASE))

    if complete == 0 and in_progress == 0 and pending == 0:
        complete = len(re.findall(r"\[complete\]", content, re.IGNORECASE))
        in_progress = len(re.findall(r"\[in_progress\]", content, re.IGNORECASE))
        pending = len(re.findall(r"\[pending\]", content, re.IGNORECASE))

    total = complete + in_progress + pending

    current_phase = None
    phase_match = re.search(r"## Current Phase\s*\n\s*Phase\s*(\d+)", content, re.IGNORECASE)
    if phase_match:
        current_phase = int(phase_match.group(1))

    return {
        "total": total,
        "complete": complete,
        "in_progress": in_progress,
        "pending": pending,
        "current": current_phase,
    }


def analyze_session() -> tuple[dict[str, Any], int]:
    cwd = Path.cwd()
    files: dict[str, Any] = {}

    for filename in _PLANNING_FILES:
        files[filename] = _file_info(cwd / filename)

    existing_files = [name for name, info in files.items() if bool(info.get("exists"))]
    file_errors = [name for name, info in files.items() if str(info.get("error", "")).strip()]

    if not existing_files:
        return (
            {
                "executed_skill": _EXECUTED_SKILL,
                "action": _ACTION,
                "status": "no_session",
                "message": "No planning files found. Start a new session with init_planning.py",
                "files": files,
                "recommendation": "init",
                "file_errors": file_errors,
            },
            0,
        )

    latest_file = ""
    latest_epoch = -1.0
    latest_time = ""

    for filename, info in files.items():
        if not bool(info.get("exists")):
            continue
        epoch = float(info.get("modified_epoch", -1.0))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_file = filename
            latest_time = str(info.get("modified", ""))

    task_plan_info = files.get("task_plan.md", {})
    task_plan_path = cwd / "task_plan.md"

    if bool(task_plan_info.get("exists")):
        try:
            task_content = task_plan_path.read_text(encoding="utf-8", errors="replace")
            phases = _analyze_task_plan_content(task_content)
        except OSError as exc:
            return (
                {
                    "executed_skill": _EXECUTED_SKILL,
                    "action": _ACTION,
                    "status": "partial_session",
                    "message": f"task_plan.md exists but is unreadable: {exc}",
                    "files": files,
                    "recommendation": "recover",
                    "file_errors": sorted(set(file_errors + ["task_plan.md"])),
                },
                0,
            )

        return (
            {
                "executed_skill": _EXECUTED_SKILL,
                "action": _ACTION,
                "status": "active_session",
                "message": (
                    f"Active session found. Phase {phases.get('current') or '?'} "
                    f"of {phases.get('total') or '?'}"
                ),
                "files": files,
                "latest_update": latest_file,
                "latest_time": latest_time,
                "phases": phases,
                "recommendation": "resume",
                "file_errors": file_errors,
            },
            0,
        )

    return (
        {
            "executed_skill": _EXECUTED_SKILL,
            "action": _ACTION,
            "status": "partial_session",
            "message": f"Found {len(existing_files)} planning file(s), but task_plan.md is missing",
            "files": files,
            "latest_update": latest_file,
            "latest_time": latest_time,
            "recommendation": "recover",
            "file_errors": file_errors,
        },
        0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Session catchup helper")
    parser.add_argument("--json", action="store_true", help="Kept for compatibility; JSON is always stdout")
    args = parser.parse_args()

    _ = args.json

    try:
        payload, code = analyze_session()
        print(json.dumps(payload, ensure_ascii=True))
        return int(code)
    except Exception as exc:  # unexpected runtime failure
        payload = {
            "executed_skill": _EXECUTED_SKILL,
            "action": _ACTION,
            "status": "error",
            "message": f"unexpected runtime exception: {exc}",
            "files": {},
            "recommendation": "recover",
            "file_errors": ["unexpected_exception"],
        }
        print(json.dumps(payload, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
