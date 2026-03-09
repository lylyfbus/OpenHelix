"""Approval gates and policies for the Environment."""

import hashlib
import re
from typing import Any
from agentic_system.core.action import Action
from agentic_system.core.environment import Environment


class ApprovalPolicy:
    """Manages approval state for a single session.

    Approval modes:
        y: allow once
        s: allow same exact exec for this session
        p: allow same script pattern for this session
        k: allow same script_path for this session (ignore args)
    """

    def __init__(self, mode: str = "controlled"):
        self.mode = mode
        self.approved_exact: set[str] = set()
        self.approved_patterns: set[str] = set()
        self.approved_paths: set[str] = set()

    def _hash_payload(self, payload: dict) -> str:
        """Hash the full payload for exact-match approval."""
        content = (
            str(payload.get("code_type", "")) +
            str(payload.get("script", "")) +
            str(payload.get("script_path", "")) +
            str(payload.get("script_args", ""))
        ).encode("utf-8")
        return hashlib.md5(content).hexdigest()

    def _pattern_key(self, payload: dict) -> str:
        """Extract a normalized pattern key from the script content.

        Strips variable parts (quoted strings, numbers) to match
        structurally similar scripts.
        """
        script = payload.get("script", "") or ""
        # Normalize whitespace, remove quoted strings and numbers
        normalized = re.sub(r'"[^"]*"', '"..."', script)
        normalized = re.sub(r"'[^']*'", "'...'", normalized)
        normalized = re.sub(r"\b\d+\b", "N", normalized)
        return f"{payload.get('code_type', 'bash')}:{normalized.strip()}"

    def __call__(self, env: Environment, action: Action) -> bool:
        """Environment hook: OnBeforeExecute."""
        if action.type != "exec":
            return True

        if self.mode == "auto":
            return True

        # Check cached approvals
        if action.payload.get("script_path") in self.approved_paths:
            return True

        payload_hash = self._hash_payload(action.payload)
        if payload_hash in self.approved_exact:
            return True

        pattern_key = self._pattern_key(action.payload)
        if pattern_key in self.approved_patterns:
            return True

        # Prompt user
        print(f"\nruntime> Action requires approval:")
        print(f"Type: {action.payload.get('code_type', 'bash')}")
        if "script" in action.payload:
            print(f"Script:\n{action.payload['script']}\n")
        elif "script_path" in action.payload:
            print(f"Script Path: {action.payload['script_path']}")
            print(f"Args: {action.payload.get('script_args', [])}\n")

        print("Approve this execution? [y/N/s/p/k]")
        print("  y: allow once")
        print("  s: allow same exact exec for this session")
        print("  p: allow same script pattern for this session")
        print("  k: allow same script_path for this session (ignore args)")

        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return False

        if choice in {"y", "yes", "once"}:
            return True
        if choice in {"s", "session", "exact"}:
            self.approved_exact.add(payload_hash)
            return True
        if choice in {"p", "pattern"}:
            self.approved_patterns.add(pattern_key)
            return True
        if choice in {"k", "path", "skill"}:
            if "script_path" in action.payload:
                self.approved_paths.add(action.payload["script_path"])
                return True
            else:
                print("runtime> 'k' requires a script_path. Denied.")
                return False

        return False
