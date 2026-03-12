"""Phase 4 verification — Skills + Knowledge integration for bootstrapped workspaces.

Tests that the new framework correctly loads, formats, and injects
the built-in skills and knowledge into the agent pipeline.

After the skills bundling change, the built-in skills live in
``agentic_system/builtin_skills/`` and are bootstrapped into the
workspace at startup by RuntimeHost. These tests verify both the
raw loader against the package source and bootstrapped temp workspaces.
"""

import json
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_system.core.agent import _load_skills as load_skills
from agentic_system.core.agent import _load_knowledge_catalog as load_knowledge_catalog
from agentic_system.core.agent import _build_system_prompt
from agentic_system.core.action import Action, parse_action
from agentic_system.core.environment import Environment
from agentic_system.core.agent import Agent
from agentic_system.core.state import Turn
from agentic_system.runtime.loop import run_loop
from agentic_system.core.sandbox import sandbox_executor
from agentic_system.runtime.approval import ApprovalPolicy
from agentic_system.runtime.host import RuntimeHost

# Path to the package source and builtin skills
WORKSPACE = Path(__file__).resolve().parent.parent
BUILTIN_SKILLS = WORKSPACE / "agentic_system" / "builtin_skills"


# =========================================================================== #
# Skills integration — loading from builtin_skills package directory
# =========================================================================== #


def test_real_skill_loading():
    """Verify that the skill loader correctly reads skills from builtin_skills."""
    skills = load_skills(BUILTIN_SKILLS)
    skill_ids = {s["skill_id"] for s in skills}

    # Should find all 9 built-in skills (7 non-loader + 2 loaders)
    assert "search-online-context" in skill_ids, f"Missing search-online-context, got: {skill_ids}"
    assert "image-generation" in skill_ids
    assert "image-understanding" in skill_ids
    assert "documentation-distillation" in skill_ids
    assert "file-based-planning" in skill_ids
    assert "skill-creation" in skill_ids
    assert "brainstorming" in skill_ids

    # Loaders should be excluded by the skill loader (they're handled separately)
    assert "load-skill" not in skill_ids
    assert "load-knowledge-docs" not in skill_ids

    # Verify metadata quality
    for skill in skills:
        assert skill["skill_id"], "Empty skill_id"
        assert skill["scope"] == "all-agents", f"Unexpected scope: {skill['scope']}"
        assert skill["name"], f"Empty name for {skill['skill_id']}"

    print(f"  Real skill loading OK ({len(skills)} skills found)")


def test_real_skill_metadata_fields():
    """Verify specific skill metadata is correctly extracted."""
    skills = load_skills(BUILTIN_SKILLS)
    skill_map = {s["skill_id"]: s for s in skills}

    # search-online-context
    search = skill_map["search-online-context"]
    assert "search_searxng.py" in search["handler"]
    assert "search" in search["name"].lower()
    assert search["description"]

    # brainstorming — no handler
    bs = skill_map["brainstorming"]
    assert bs["handler"] == ""  # no handler specified
    assert bs["description"]

    # file-based-planning — multi-script skill
    fbp = skill_map["file-based-planning"]
    assert "init_planning.py" in fbp["handler"]

    print("  Real skill metadata fields OK")


def _create_workspace_knowledge(workspace: Path) -> None:
    """Create a minimal knowledge catalog inside a temp workspace."""
    docs_root = workspace / "knowledge" / "docs"
    index_root = workspace / "knowledge" / "index"
    docs_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    doc_id = "doc_c83ff4bad5ca"
    (docs_root / f"{doc_id}.md").write_text(
        "# Schema test\n\nKnowledge fixture for integration tests.\n",
        encoding="utf-8",
    )
    catalog = [{
        "doc_id": doc_id,
        "title": "Schema test",
        "path": f"knowledge/docs/{doc_id}.md",
        "tags": ["test", "schema"],
        "quality_score": 0.9,
        "confidence": 0.95,
    }]
    (index_root / "catalog.json").write_text(
        json.dumps(catalog, indent=2),
        encoding="utf-8",
    )


def test_workspace_knowledge_loading():
    """Verify knowledge catalog loads from a runtime workspace layout."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace_knowledge(workspace)
        catalog = load_knowledge_catalog(workspace / "knowledge")
        assert len(catalog) == 1
        assert catalog[0]["doc_id"] == "doc_c83ff4bad5ca"
        assert catalog[0]["title"] == "Schema test"
        print(f"  Workspace knowledge loading OK ({len(catalog)} docs)")


# =========================================================================== #
# Bootstrap verification — RuntimeHost syncs skills into workspace
# =========================================================================== #


def test_bootstrap_skills():
    """Verify that RuntimeHost bootstraps all 9 skills into a fresh workspace."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        ws_skills = Path(td) / "skills" / "all-agents"
        skill_dirs = sorted(p.name for p in ws_skills.iterdir() if p.is_dir())
        assert len(skill_dirs) == 9, f"Expected 9, got {len(skill_dirs)}: {skill_dirs}"
        assert "search-online-context" in skill_dirs
        assert "image-generation" in skill_dirs
        assert "load-skill" in skill_dirs
        print(f"  Bootstrap skills OK ({len(skill_dirs)} skills synced)")


def test_bootstrapped_prompt_builder():
    """Verify PromptBuilder works with the bootstrapped workspace."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        prompt = _build_system_prompt(Path(td), "core_agent")

        assert prompt, "Empty system prompt"
        assert len(prompt) > 500, f"System prompt too short ({len(prompt)} chars)"
        assert "search-online-context" in prompt
        assert "image-generation" in prompt
        assert "load-skill" in prompt
        assert "load-knowledge-docs" not in prompt

        print(f"  Bootstrapped prompt builder OK ({len(prompt)} chars)")


# =========================================================================== #
# Full pipeline: Prompt → Agent → Loop → Sandbox
# =========================================================================== #


def test_full_pipeline_with_skill_exec():
    """End-to-end: PromptBuilder builds prompt, Agent acts, Loop executes."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        system_prompt = _build_system_prompt(Path(td), "core_agent")

        call_count = [0]

        class SkillAwareModel:
            def generate(self, prompt, *, stream=False, chunk_callback=None):
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
            system_prompt=system_prompt,
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


def test_load_skill_script_exists():
    """Verify that built-in loader scripts exist in the builtin_skills tree."""
    load_skill_path = BUILTIN_SKILLS / "all-agents" / "load-skill" / "scripts" / "load_skill.py"
    assert load_skill_path.exists(), f"load-skill script not found at {load_skill_path}"

    load_knowledge_path = (
        BUILTIN_SKILLS / "all-agents" / "load-knowledge-docs"
        / "scripts" / "load_knowledge_docs.py"
    )
    assert load_knowledge_path.exists(), f"load-knowledge-docs script not found at {load_knowledge_path}"
    print("  Built-in loader scripts exist OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Skills Integration (Builtin Skills) ===")
    test_real_skill_loading()
    test_real_skill_metadata_fields()

    print("\n=== Knowledge Integration ===")
    test_workspace_knowledge_loading()

    print("\n=== Bootstrap ===")
    test_bootstrap_skills()
    test_bootstrapped_prompt_builder()

    print("\n=== Full Pipeline ===")
    test_full_pipeline_with_skill_exec()
    test_load_skill_script_exists()

    print("\n✅ All Phase 4 tests passed!")
