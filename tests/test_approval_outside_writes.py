"""Tests for outside-workspace write detection in ApprovalPolicy.

The detector is a heuristic — it only catches obvious literal absolute
paths in write positions. Its purpose is to surface suspicious ops in
the approval prompt, not to act as a hard sandbox.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.action import Action
from helix.core.environment import Environment
from helix.runtime.approval import ApprovalPolicy, detect_outside_workspace_writes


# ----- detector unit tests --------------------------------------------------- #


def test_detect_bash_redirect_outside_workspace():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {"code_type": "bash", "script": "echo hi > /tmp/outside.txt"}
        hits = detect_outside_workspace_writes(payload, workspace)
        assert any("/tmp/outside.txt" in h for h in hits), hits
        print("  detect bash redirect outside OK")


def test_detect_bash_rm_outside_workspace():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {"code_type": "bash", "script": "rm -rf /etc/something"}
        hits = detect_outside_workspace_writes(payload, workspace)
        assert any("/etc/something" in h for h in hits), hits
        print("  detect bash rm outside OK")


def test_detect_python_open_write_outside_workspace():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "python",
            "script": "open('/tmp/x.txt', 'w').write('hi')",
        }
        hits = detect_outside_workspace_writes(payload, workspace)
        assert any("/tmp/x.txt" in h for h in hits), hits
        print("  detect python open outside OK")


def test_detect_python_pathlib_write_outside_workspace():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "python",
            "script": "from pathlib import Path; Path('/tmp/y.txt').write_text('hi')",
        }
        hits = detect_outside_workspace_writes(payload, workspace)
        assert any("/tmp/y.txt" in h for h in hits), hits
        print("  detect python pathlib write outside OK")


def test_detect_ignores_relative_paths():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            "script": "echo hi > local.txt; rm -rf some/dir",
        }
        hits = detect_outside_workspace_writes(payload, workspace)
        assert hits == [], f"expected no hits, got {hits}"
        print("  ignore relative paths OK")


def test_detect_ignores_workspace_scoped_absolute_paths():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        payload = {
            "code_type": "bash",
            "script": f"echo hi > {workspace}/inside.txt",
        }
        hits = detect_outside_workspace_writes(payload, workspace)
        assert hits == [], f"expected no hits, got {hits}"
        print("  ignore workspace-scoped absolute paths OK")


def test_detect_read_only_commands_not_flagged():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        payload = {
            "code_type": "bash",
            "script": "cat /etc/passwd; ls /tmp; grep foo /var/log/syslog",
        }
        hits = detect_outside_workspace_writes(payload, workspace)
        assert hits == [], f"read-only commands should not flag: {hits}"
        print("  read-only commands not flagged OK")


# ----- prompt integration tests --------------------------------------------- #


def test_prompt_shows_outside_write_warning():
    captured_prompt: list[str] = []

    def capture(_prompt: str) -> str:
        return "n"  # deny

    policy = ApprovalPolicy(mode="controlled", prompt=capture)

    # Monkey-patch write_approval so we capture what the user would see.
    import helix.runtime.approval as approval_mod
    original_write_approval = approval_mod.write_approval
    approval_mod.write_approval = lambda text, _stream=None: captured_prompt.append(text)
    try:
        with tempfile.TemporaryDirectory() as td:
            env = Environment(workspace=Path(td))
            action = Action(
                response="writing outside",
                type="exec",
                payload={"code_type": "bash", "script": "rm -rf /etc/foo"},
            )
            result = policy(env, action)

        combined = "\n".join(captured_prompt)
        assert "outside the workspace" in combined.lower(), combined
        assert "/etc/foo" in combined, combined
        print("  prompt shows outside write warning OK")
    finally:
        approval_mod.write_approval = original_write_approval


if __name__ == "__main__":
    print("=== approval: outside-workspace write detection ===")
    test_detect_bash_redirect_outside_workspace()
    test_detect_bash_rm_outside_workspace()
    test_detect_python_open_write_outside_workspace()
    test_detect_python_pathlib_write_outside_workspace()
    test_detect_ignores_relative_paths()
    test_detect_ignores_workspace_scoped_absolute_paths()
    test_detect_read_only_commands_not_flagged()
    test_prompt_shows_outside_write_warning()
    print("\n✅ All outside-write detection tests passed!")
