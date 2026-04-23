"""Approval gates and policies for the Environment."""

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Optional

from helix.core.action import Action
from helix.core.environment import ApprovalResult, Environment
from helix.core.state import Turn
from helix.runtime.display import iter_exec_payload_items, write_approval


PromptFn = Callable[[str], str]

_APPROVAL_LABELS = {
    "job_name": "Job Name",
    "code_type": "Type",
    "script_path": "Script Path",
    "script": "Script",
    "script_args": "Args",
    "timeout_seconds": "Timeout Seconds",
}


# --------------------------------------------------------------------------- #
# Outside-workspace write detection
# --------------------------------------------------------------------------- #

# Heuristic: catch literal absolute paths used as write targets. False
# negatives are expected (variables, pipes, ``$(...)``); the goal is to
# surface obvious cases in the approval prompt, not to be a hard sandbox.

_BASH_WRITE_PATTERNS = (
    # Redirects: > /abs, >> /abs, &> /abs
    re.compile(r'(?:^|[\s;|&(`])(?:>{1,2}|&>)\s*(/[^\s;|&(`)]+)'),
    # tee [flags] /abs
    re.compile(r'(?:^|[\s;|&(`])tee\b(?:\s+-\S+)*\s+(/[^\s;|&(`)]+)'),
    # Destructive commands whose FIRST absolute-path argument is a target.
    re.compile(
        r'(?:^|[\s;|&(`])'
        r'(?:rm|mkdir|touch|chmod|chown|install|rename|truncate)\b'
        r'(?:\s+-\S+)*\s+(/[^\s;|&(`)]+)'
    ),
    # cp/mv/ln: the LAST argument is the destination; approximate by
    # matching any absolute path that appears after the command word.
    re.compile(
        r'(?:^|[\s;|&(`])(?:cp|mv|ln|rsync|scp)\b'
        r'(?:\s+-\S+)*[^\n;|&`]*?\s(/[^\s;|&(`)]+)'
    ),
)

_PY_WRITE_PATTERNS = (
    # open('/abs', 'w'|'a'|'x'|...)
    re.compile(r'open\s*\(\s*[\'"](/[^\'"]+)[\'"]\s*,\s*[\'"][wax+ab]+'),
    # os.remove/unlink/rmdir/mkdir/makedirs('/abs')
    re.compile(
        r'(?:os\.(?:remove|unlink|rmdir|mkdir|makedirs|rename|replace|truncate)|'
        r'shutil\.(?:rmtree|copyfile|copy2?|move|copytree))'
        r'\s*\(\s*[\'"](/[^\'"]+)[\'"]'
    ),
    # Path('/abs').write_text(...), .unlink(), .mkdir(), etc.
    re.compile(
        r'Path\s*\(\s*[\'"](/[^\'"]+)[\'"]\s*\)\s*\.'
        r'(?:write_text|write_bytes|unlink|mkdir|rmdir|touch|rename|replace)\b'
    ),
)


def detect_outside_workspace_writes(payload: dict, workspace_root: Path) -> list[str]:
    """Return literal absolute paths the script appears to write to outside workspace.

    Only inline ``script`` text is scanned — ``script_path`` executions are
    covered by the same-path (``k``) approval instead. Deduplicates results
    while preserving first-seen order.
    """
    script = str(payload.get("script", "") or "")
    if not script.strip():
        return []

    code_type = str(payload.get("code_type", "")).strip().lower()
    patterns = _BASH_WRITE_PATTERNS if code_type == "bash" else _PY_WRITE_PATTERNS

    try:
        workspace_str = str(Path(workspace_root).expanduser().resolve())
    except (OSError, RuntimeError):
        workspace_str = str(Path(workspace_root))

    seen: dict[str, None] = {}
    for pat in patterns:
        for m in pat.finditer(script):
            raw = m.group(1).rstrip('.,;:)]}"\'`')
            if not raw.startswith("/"):
                continue
            try:
                resolved = str(Path(raw).resolve())
            except (OSError, RuntimeError):
                resolved = raw
            if resolved == workspace_str or resolved.startswith(workspace_str + "/"):
                continue
            seen.setdefault(raw, None)
    return list(seen.keys())


class ApprovalPolicy:
    """Manages approval state for a single session.

    Approval modes:
        y: allow once
        s: allow same exact exec for this session
        p: allow same script pattern for this session
        k: allow same script_path for this session (ignore args)
    """

    def __init__(
        self,
        mode: str = "controlled",
        *,
        prompt: Optional[PromptFn] = None,
    ) -> None:
        self.mode = mode
        self._prompt = prompt or input
        self.approved_exact: set[str] = set()
        self.approved_patterns: set[str] = set()
        self.approved_paths: set[str] = set()

    def _hash_payload(self, payload: dict, *, profile: str = "") -> str:
        """Hash the full payload for exact-match approval."""
        content = (
            profile +
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
        ).encode("utf-8")
        return hashlib.md5(content).hexdigest()

    def _pattern_key(self, payload: dict, *, profile: str = "") -> Optional[str]:
        """Extract a normalized pattern key for inline scripts only.

        Pattern approvals are defined over script content, not script paths.
        For ``script_path`` executions the caller should use exact approval
        (``s``) or same-path approval (``k``) instead.
        """
        script = payload.get("script", "") or ""
        if not str(script).strip():
            return None

        # Normalize whitespace, remove quoted strings and numbers
        normalized = re.sub(r'"[^"]*"', '"..."', script)
        normalized = re.sub(r"'[^']*'", "'...'", normalized)
        normalized = re.sub(r"\b\d+\b", "N", normalized)
        return f"{profile}:{payload.get('code_type', 'bash')}:{normalized.strip()}"

    def __call__(self, env: Environment, action: Action) -> ApprovalResult:
        """Environment hook: OnBeforeExecute."""
        if action.type != "exec":
            return True

        if self.mode == "auto":
            return True

        # Check cached approvals
        if action.payload.get("script_path") in self.approved_paths:
            return True

        profile = str(getattr(env, "approval_profile", "") or "")
        payload_hash = self._hash_payload(action.payload, profile=profile)
        if payload_hash in self.approved_exact:
            return True

        pattern_key = self._pattern_key(action.payload, profile=profile)
        if pattern_key and pattern_key in self.approved_patterns:
            return True

        # Prompt user
        details = ["runtime> Action requires approval:"]

        outside_writes = detect_outside_workspace_writes(action.payload, env.workspace)
        if outside_writes:
            details.append("")
            details.append("WARNING: this exec writes to path(s) outside the workspace:")
            for path_str in outside_writes[:5]:
                details.append(f"    {path_str}")
            if len(outside_writes) > 5:
                details.append(f"    ... and {len(outside_writes) - 5} more")
            details.append("")

        for key, value in iter_exec_payload_items(action.payload):
            label = _APPROVAL_LABELS.get(key, key)
            text = str(value)
            if "\n" in text:
                details.append(f"{label}:")
                details.extend(text.splitlines())
            else:
                details.append(f"{label}: {text}")

        if pattern_key:
            details.extend([
                "Approve this execution? [y/N/s/p/k]",
                "  y: allow once",
                "  s: allow same exact exec for this session",
                "  p: allow same script pattern for this session",
                "  k: allow same script_path for this session (ignore args)",
            ])
        else:
            details.extend([
                "Approve this execution? [y/N/s/k]",
                "  y: allow once",
                "  s: allow same exact exec for this session",
                "  k: allow same script_path for this session (ignore args)",
                "  p: unavailable for script_path executions",
            ])
        write_approval("\n".join(details), None)

        try:
            choice = self._prompt("> ").strip().lower()
            print()  # blank line after approval input
        except EOFError:
            return Turn(
                role="runtime",
                content="Execution cancelled during approval prompt (input closed).",
            )
        except KeyboardInterrupt:
            return Turn(
                role="runtime",
                content="Execution cancelled during approval prompt by requester.",
            )

        if choice in {"y", "yes", "once"}:
            return True
        if choice in {"s", "session", "exact"}:
            self.approved_exact.add(payload_hash)
            return True
        if choice in {"p", "pattern"}:
            if not pattern_key:
                write_approval(
                    "runtime> 'p' requires an inline script. Use 's' or 'k' for script_path.",
                    None,
                )
                return Turn(
                    role="runtime",
                    content="Execution denied by requester during approval prompt.",
                )
            self.approved_patterns.add(pattern_key)
            return True
        if choice in {"k", "path", "skill"}:
            if "script_path" in action.payload:
                self.approved_paths.add(action.payload["script_path"])
                return True
            else:
                write_approval("runtime> 'k' requires a script_path. Denied.", None)
                return Turn(
                    role="runtime",
                    content="Execution denied by requester during approval prompt.",
                )

        return Turn(
            role="runtime",
            content="Execution denied by requester during approval prompt.",
        )
