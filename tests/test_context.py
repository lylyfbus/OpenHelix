"""Phase 3 verification tests for providers, context loaders, and prompt builder."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_system.providers.ollama import OllamaProvider
from agentic_system.providers.openai_compat import OpenAICompatProvider
from agentic_system.context.skill_loader import load_skills, format_skills_for_prompt
from agentic_system.context.knowledge_loader import load_knowledge_catalog, format_knowledge_for_prompt
from agentic_system.context.prompt_builder import PromptBuilder


# =========================================================================== #
# Provider tests (structural — no real LLM calls)
# =========================================================================== #


def test_ollama_provider_init():
    """Verify OllamaProvider initializes with correct defaults."""
    provider = OllamaProvider()
    assert provider.model == "llama3.1:8b"
    assert "11434" in provider.endpoint
    assert provider.timeout == 300
    print("  OllamaProvider init OK")


def test_ollama_provider_custom_init():
    """Verify OllamaProvider respects custom parameters."""
    provider = OllamaProvider(
        model="deepseek-r1:14b",
        base_url="http://myhost:8080",
        timeout=60,
        temperature=0.5,
    )
    assert provider.model == "deepseek-r1:14b"
    assert "myhost:8080" in provider.endpoint
    assert provider.timeout == 60
    print("  OllamaProvider custom init OK")


def test_openai_provider_init():
    """Verify OpenAICompatProvider initializes with correct defaults."""
    provider = OpenAICompatProvider()
    assert provider.model == "local-model"
    assert "/chat/completions" in provider.endpoint
    print("  OpenAICompatProvider init OK")


def test_openai_provider_presets():
    """Verify preset resolution for known providers."""
    dp = OpenAICompatProvider(provider="deepseek")
    assert "deepseek.com" in dp.endpoint
    assert dp.model == "deepseek-chat"

    zai = OpenAICompatProvider(provider="zai")
    assert "z.ai" in zai.endpoint

    lm = OpenAICompatProvider(provider="lmstudio", model="my-model")
    assert lm.model == "my-model"
    print("  OpenAICompatProvider presets OK")


def test_provider_satisfies_protocol():
    """Verify both providers have the generate() interface matching ModelProvider."""
    import inspect
    for cls in [OllamaProvider, OpenAICompatProvider]:
        assert hasattr(cls, "generate"), f"{cls.__name__} missing generate()"
        sig = inspect.signature(cls.generate)
        params = list(sig.parameters.keys())
        assert "prompt" in params, f"{cls.__name__}.generate() missing prompt param"
        assert "stream" in params, f"{cls.__name__}.generate() missing stream param"
        assert "chunk_callback" in params, f"{cls.__name__}.generate() missing chunk_callback param"
    print("  Protocol compliance OK")


# =========================================================================== #
# Skill loader tests
# =========================================================================== #


def _create_skill_tree(root: Path) -> None:
    """Create a test skill directory tree."""
    # all-agents/search-web/SKILL.md
    skill_dir = root / "all-agents" / "search-web"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: search-web\n"
        "description: Search the web for information\n"
        "handler: scripts/search.py\n"
        "required_tools: bash\n"
        "---\n"
        "Full instructions here...\n",
        encoding="utf-8",
    )

    # all-agents/code-review/SKILL.md
    skill_dir2 = root / "all-agents" / "code-review"
    skill_dir2.mkdir(parents=True)
    (skill_dir2 / "SKILL.md").write_text(
        "---\n"
        "name: code-review\n"
        "description: Review code quality\n"
        "handler: scripts/review.py\n"
        "recommended_tools: python\n"
        "---\n",
        encoding="utf-8",
    )

    # all-agents/load-skill/ — should be excluded (builtin)
    builtin_dir = root / "all-agents" / "load-skill"
    builtin_dir.mkdir(parents=True)
    (builtin_dir / "SKILL.md").write_text("---\nname: load-skill\n---\n", encoding="utf-8")


def test_skill_loader():
    """Test skill loading and filtering."""
    with tempfile.TemporaryDirectory() as td:
        skills_root = Path(td) / "skills"
        _create_skill_tree(skills_root)

        skills = load_skills(skills_root)
        assert len(skills) == 2, f"Expected 2 skills, got {len(skills)}"
        ids = {s["skill_id"] for s in skills}
        assert "search-web" in ids
        assert "code-review" in ids
        assert "load-skill" not in ids  # builtin excluded
        print("  Skill loader OK")


def test_skill_loader_empty():
    """Test skill loading from non-existent directory."""
    skills = load_skills(Path("/nonexistent/path"))
    assert skills == []
    print("  Skill loader (empty) OK")


def test_skill_formatter():
    """Test skill formatting for prompt injection."""
    text = format_skills_for_prompt([])
    assert "no skills found" in text

    text2 = format_skills_for_prompt([{"skill_id": "test", "name": "Test"}])
    assert "test" in text2
    assert text2.startswith("- ")
    print("  Skill formatter OK")


# =========================================================================== #
# Knowledge loader tests
# =========================================================================== #


def _create_knowledge_catalog(root: Path) -> None:
    """Create a test knowledge catalog."""
    index_dir = root / "index"
    index_dir.mkdir(parents=True)
    catalog = [
        {
            "doc_id": "rl-overview",
            "title": "RL for LLM Post-Training",
            "path": "knowledge/docs/rl-overview.md",
            "tags": ["rl", "llm"],
            "quality_score": 0.9,
            "confidence": 0.85,
        },
        {
            "doc_id": "agent-design",
            "title": "Agent System Design",
            "path": "knowledge/docs/agent-design.md",
            "tags": "design, architecture",  # String tags
            "quality_score": 0.8,
        },
    ]
    (index_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2), encoding="utf-8"
    )


def test_knowledge_loader():
    """Test knowledge catalog loading."""
    with tempfile.TemporaryDirectory() as td:
        knowledge_root = Path(td)
        _create_knowledge_catalog(knowledge_root)

        catalog = load_knowledge_catalog(knowledge_root)
        assert len(catalog) == 2
        assert catalog[0]["doc_id"] == "agent-design"  # sorted by doc_id
        assert catalog[1]["doc_id"] == "rl-overview"
        assert isinstance(catalog[0]["tags"], list)  # string tags normalized
        print("  Knowledge loader OK")


def test_knowledge_loader_empty():
    """Test knowledge loading from non-existent directory."""
    catalog = load_knowledge_catalog(Path("/nonexistent/path"))
    assert catalog == []
    print("  Knowledge loader (empty) OK")


def test_knowledge_formatter():
    """Test knowledge formatting for prompt injection."""
    text = format_knowledge_for_prompt([])
    assert "no knowledge docs found" in text

    text2 = format_knowledge_for_prompt([{"doc_id": "test", "title": "Test"}])
    assert "test" in text2
    print("  Knowledge formatter OK")


# =========================================================================== #
# Prompt builder tests
# =========================================================================== #


def _create_workspace(root: Path) -> None:
    """Create a test workspace with skills and knowledge."""
    # Create skills
    _create_skill_tree(root / "skills")

    # Create knowledge
    _create_knowledge_catalog(root / "knowledge")


def test_prompt_builder():
    """Test full prompt assembly with placeholder injection."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)

        builder = PromptBuilder(workspace)
        prompt = builder.build("core_agent")

        assert "Core Agent" in prompt
        assert "search-web" in prompt  # skill injected
        assert "rl-overview" in prompt  # knowledge injected
        assert "load-skill" in prompt  # builtin loader injected
        assert str(workspace) in prompt  # workspace path injected
        assert "{{SKILLS_META_FROM_JSON}}" not in prompt  # placeholder replaced
        assert "{{KNOWLEDGE_META_FROM_JSON}}" not in prompt
        assert "{{BUILTIN_REFERENCE_LOADERS}}" not in prompt
        assert "{{RUNTIME_WORKSPACE}}" not in prompt
        print("  Prompt builder OK")


def test_prompt_builder_unknown_role():
    """Test prompt builder returns empty for unknown role."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)

        builder = PromptBuilder(workspace)
        prompt = builder.build("nonexistent_role")
        assert prompt == ""
        print("  Prompt builder (unknown role) OK")


def test_prompt_builder_no_prompts():
    """Test prompt builder with workspace that has no matching role."""
    with tempfile.TemporaryDirectory() as td:
        builder = PromptBuilder(Path(td))
        prompt = builder.build("nonexistent_role_xyz")
        assert prompt == ""
        print("  Prompt builder (no prompts) OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Provider Initialization ===")
    test_ollama_provider_init()
    test_ollama_provider_custom_init()
    test_openai_provider_init()
    test_openai_provider_presets()
    test_provider_satisfies_protocol()

    print("\n=== Skill Loader ===")
    test_skill_loader()
    test_skill_loader_empty()
    test_skill_formatter()

    print("\n=== Knowledge Loader ===")
    test_knowledge_loader()
    test_knowledge_loader_empty()
    test_knowledge_formatter()

    print("\n=== Prompt Builder ===")
    test_prompt_builder()
    test_prompt_builder_unknown_role()
    test_prompt_builder_no_prompts()

    print("\n✅ All Phase 3 tests passed!")
