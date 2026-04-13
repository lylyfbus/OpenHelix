"""Phase 4 verification — Skills + Knowledge integration for bootstrapped workspaces.

Tests that the new framework correctly loads, formats, and injects
the built-in skills and knowledge into the agent pipeline.

After the skills bundling change, the built-in skills live in
``helix/builtin_skills/`` and are bootstrapped into the
workspace at startup by RuntimeHost. These tests verify both the
raw loader against the package source and bootstrapped temp workspaces.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.agent import _load_skills as load_skills
from helix.core.agent import _build_system_prompt
from helix.core.action import Action, parse_action
from helix.core.environment import Environment
from helix.core.agent import Agent
from helix.core.state import Turn
from helix.runtime.loop import run_loop
from helix.runtime.host import docker_is_available
from helpers import sandbox_executor
from helix.runtime.approval import ApprovalPolicy
from helix.runtime.host import RuntimeHost

class _FakeDockerExecutor:
    approval_profile = "docker-online-rw-workspace-v1:test"

    def __init__(self, workspace: Path, *, session_id: str | None = None, **kwargs):
        self.workspace = workspace
        self.session_id = session_id

    def __call__(self, payload, workspace):
        return Turn(role="runtime", content="fake")

    def prepare_runtime(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


    def status_fields(self) -> dict[str, str]:
        return {"sandbox_backend": "docker", "docker_image": "fake-image"}

    def tool_environment(self) -> dict[str, str]:
        return {"SEARXNG_BASE_URL": "http://fake-searxng:8080"}


# Path to the package source and builtin skills
WORKSPACE = Path(__file__).resolve().parent.parent
BUILTIN_SKILLS = WORKSPACE / "helix" / "builtin_skills"


def _make_host(workspace: Path, **kwargs) -> RuntimeHost:
    params = {
        "workspace": workspace,
        "session_id": "skills-01",
        "endpoint_url": "http://localhost:11434/v1",
        "model": "llama3.1:8b",
    }
    params.update(kwargs)

    with patch("helix.runtime.host.docker_is_available", return_value=(True, "")):
        with patch("helix.services.searxng.discover", return_value=None):
            with patch("helix.services.local_model_service.discover", return_value=None):
                with patch("helix.runtime.host.DockerSandboxExecutor", _FakeDockerExecutor):
                    return RuntimeHost(**params)


def _docker_ready() -> bool:
    available, reason = docker_is_available()
    if not available:
        print(f"  Docker unavailable, skipping skills exec test: {reason}")
        return False
    return True


# =========================================================================== #
# Skills integration — loading from builtin_skills package directory
# =========================================================================== #


def test_real_skill_loading():
    """Verify that the skill loader correctly reads skills from builtin_skills."""
    skills = load_skills(BUILTIN_SKILLS)
    paths = {s["path"] for s in skills}

    # Should find all non-loader built-in skills.
    assert any("search-online-context" in p for p in paths), f"Missing search-online-context, got: {paths}"
    assert any("generate-image" in p for p in paths)
    assert any("generate-audio" in p for p in paths)
    assert any("generate-video" in p for p in paths)
    assert any("analyze-image" in p for p in paths)
    assert any("create-document" in p for p in paths)
    assert any("file-based-planning" in p for p in paths)
    assert any("create-skill" in p for p in paths)
    assert any("brainstorming" in p for p in paths)

    # retrieve-knowledge replaces load-knowledge-docs
    assert any("retrieve-knowledge" in p for p in paths)

    # Verify metadata quality
    for skill in skills:
        assert skill["name"], f"Empty name for {skill['path']}"
        assert skill["path"], "Empty path"
        assert skill["description"] is not None

    print(f"  Real skill loading OK ({len(skills)} skills found)")


def test_real_skill_metadata_fields():
    """Verify specific skill metadata is correctly extracted."""
    skills = load_skills(BUILTIN_SKILLS)
    skill_map = {s["path"].rsplit("/", 1)[-1]: s for s in skills}

    # search-online-context
    search = skill_map["search-online-context"]
    assert "search" in search["name"].lower()
    assert search["description"]

    # brainstorming
    bs = skill_map["brainstorming"]
    assert bs["description"]

    # file-based-planning
    fbp = skill_map["file-based-planning"]
    assert fbp["description"]

    print("  Real skill metadata fields OK")


# =========================================================================== #
# Bootstrap verification — RuntimeHost syncs skills into workspace
# =========================================================================== #


def test_bootstrap_skills():
    """Verify that RuntimeHost bootstraps all packaged skills into a fresh workspace."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        ws_skills = Path(td) / "skills" / "builtin_skills"
        skill_dirs = sorted(p.name for p in ws_skills.iterdir() if p.is_dir())
        assert len(skill_dirs) == 12, f"Expected 12, got {len(skill_dirs)}: {skill_dirs}"
        assert "search-online-context" in skill_dirs
        assert "generate-image" in skill_dirs
        assert "generate-audio" in skill_dirs
        assert "generate-video" in skill_dirs
        assert "analyze-image" in skill_dirs
        assert "retrieve-knowledge" in skill_dirs
        print(f"  Bootstrap skills OK ({len(skill_dirs)} skills synced)")


def test_bootstrapped_prompt_builder():
    """Verify PromptBuilder works with the bootstrapped workspace."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        prompt = _build_system_prompt(Path(td), "core_agent")

        assert prompt, "Empty system prompt"
        assert len(prompt) > 500, f"System prompt too short ({len(prompt)} chars)"
        assert "search-online-context" in prompt
        assert "generate-image" in prompt
        assert "generate-audio" in prompt
        assert "generate-video" in prompt
        assert "analyze-image" in prompt
        assert "retrieve-knowledge" in prompt

        print(f"  Bootstrapped prompt builder OK ({len(prompt)} chars)")


def test_bootstrap_prunes_renamed_packaged_skills_but_keeps_user_skills():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        skills_root = workspace / "skills" / "builtin_skills"
        legacy_generation_skill = skills_root / "generate-image-from-pytorch"
        legacy_analysis_skill = skills_root / "analyze-image-from-ollama"
        user_skill = skills_root / "user-custom-skill"
        legacy_generation_skill.mkdir(parents=True, exist_ok=True)
        legacy_analysis_skill.mkdir(parents=True, exist_ok=True)
        user_skill.mkdir(parents=True, exist_ok=True)
        (legacy_generation_skill / "SKILL.md").write_text("---\nname: Legacy Generation\n---\n", encoding="utf-8")
        (legacy_analysis_skill / "SKILL.md").write_text("---\nname: Legacy Analysis\n---\n", encoding="utf-8")
        (user_skill / "SKILL.md").write_text("---\nname: User\n---\n", encoding="utf-8")
        manifest_path = workspace / ".runtime" / "builtin_skills_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                [
                    "generate-image-from-pytorch",
                    "analyze-image-from-ollama",
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        _make_host(workspace)

        assert not legacy_generation_skill.exists()
        assert not legacy_analysis_skill.exists()
        assert (skills_root / "generate-image").exists()
        assert (skills_root / "analyze-image").exists()
        assert user_skill.exists()
        print("  Packaged skill rename prune OK")


# =========================================================================== #
# Full pipeline: Prompt → Agent → Loop → Sandbox
# =========================================================================== #


def test_full_pipeline_with_skill_exec():
    """End-to-end: PromptBuilder builds prompt, Agent acts, Loop executes."""
    if not _docker_ready():
        return
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        system_prompt = _build_system_prompt(Path(td), "core_agent")

        call_count = [0]

        class SkillAwareModel:
            def generate(self, messages, *, chunk_callback=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (
                        '<output>'
                        '{"response": "Let me check something.", '
                        '"action": "exec", '
                        '"action_input": {"job_name": "test-skill-pipeline", '
                        '"code_type": "bash", '
                        '"script": "echo Skills loaded successfully"}}'
                        '</output>'
                    )
                return (
                    '<output>'
                    '{"response": "Skills and knowledge are properly loaded!", '
                    '"action": "chat", "action_input": {}}'
                    '</output>'
                )

        workspace = Path(td)
        env = Environment(workspace=workspace, executor=sandbox_executor, mode="auto")
        policy = ApprovalPolicy(mode="auto")
        env.on_before_execute(policy)
        env.record(Turn(role="user", content="Test the skills pipeline."))

        agent = Agent(
            SkillAwareModel(),
            workspace=workspace,
        )

        result = run_loop(agent, env, output=sys.stderr)

        assert result == "Skills and knowledge are properly loaded!"
        assert call_count[0] == 2

        runtime_turns = [t for t in env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "Job 'test-skill-pipeline' succeeded." in runtime_turns[0].content
        assert "Skills loaded successfully" in runtime_turns[0].content
        assert "Exit code: 0" in runtime_turns[0].content

    print("  Full pipeline with skill exec OK")


def test_builtin_skill_files_exist():
    """Verify that key built-in skills exist."""
    retrieve_knowledge = BUILTIN_SKILLS / "retrieve-knowledge" / "SKILL.md"
    assert retrieve_knowledge.exists(), f"retrieve-knowledge SKILL.md not found at {retrieve_knowledge}"

    documentation = BUILTIN_SKILLS / "create-document" / "SKILL.md"
    assert documentation.exists(), f"documentation-distillation SKILL.md not found at {documentation}"
    print("  Built-in skill files exist OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Skills Integration (Builtin Skills) ===")
    test_real_skill_loading()
    test_real_skill_metadata_fields()

    print("\n=== Bootstrap ===")
    test_bootstrap_skills()
    test_bootstrapped_prompt_builder()

    print("\n=== Full Pipeline ===")
    test_full_pipeline_with_skill_exec()
    test_builtin_skill_files_exist()

    print("\n✅ All Phase 4 tests passed!")
