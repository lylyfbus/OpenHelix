"""Phase 3 verification tests for providers, context loaders, and prompt builder."""

import json
import sys
import tempfile
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.providers.openai_compat import LLMProvider
from helix.core.agent import Agent
from helix.core.agent import _load_skills as load_skills
from helix.core.agent import _load_knowledge_catalog as load_knowledge_catalog
from helix.core.agent import _build_system_prompt
from helix.core.state import State, Turn


# =========================================================================== #
# Provider tests (structural — no real LLM calls)
# =========================================================================== #


def test_llm_provider_default_init():
    """Verify LLMProvider initializes with correct defaults."""
    provider = LLMProvider()
    assert provider.model == "llama3.1:8b"
    assert "11434" in provider.endpoint
    assert provider.timeout == 300
    assert "/v1/chat/completions" in provider.endpoint
    print("  LLMProvider default init OK")


def test_llm_provider_custom_init():
    """Verify LLMProvider respects custom parameters."""
    provider = LLMProvider(
        base_url="http://myhost:8080/v1",
        api_key="test-key",
        model="deepseek-r1:14b",
        timeout=60,
        temperature=0.5,
    )
    assert provider.model == "deepseek-r1:14b"
    assert "myhost:8080" in provider.endpoint
    assert provider.timeout == 60
    assert provider.api_key == "test-key"
    print("  LLMProvider custom init OK")


def test_llm_provider_auto_appends_v1():
    """Verify base URL without /v1 suffix gets it appended."""
    provider = LLMProvider(base_url="https://api.deepseek.com")
    assert provider.endpoint == "https://api.deepseek.com/v1/chat/completions"
    print("  LLMProvider auto-appends /v1 OK")


def test_llm_provider_preserves_existing_v1():
    """Verify base URL with /v1 suffix is not doubled."""
    provider = LLMProvider(base_url="http://localhost:1234/v1")
    assert provider.endpoint == "http://localhost:1234/v1/chat/completions"
    print("  LLMProvider preserves /v1 OK")


def test_llm_provider_env_vars():
    """Verify LLMProvider reads from environment variables."""
    env = {
        "LLM_BASE_URL": "http://envhost:9999/v1",
        "LLM_API_KEY": "env-key",
        "LLM_MODEL": "env-model",
    }
    with patch.dict("os.environ", env, clear=True):
        provider = LLMProvider()
    assert "envhost:9999" in provider.endpoint
    assert provider.api_key == "env-key"
    assert provider.model == "env-model"
    print("  LLMProvider env vars OK")


def test_llm_provider_explicit_overrides_env():
    """Verify explicit constructor args override env vars."""
    env = {"LLM_BASE_URL": "http://envhost:9999", "LLM_MODEL": "env-model"}
    with patch.dict("os.environ", env, clear=True):
        provider = LLMProvider(base_url="http://explicit:1234/v1", model="explicit-model")
    assert "explicit:1234" in provider.endpoint
    assert provider.model == "explicit-model"
    print("  LLMProvider explicit overrides env OK")


def test_provider_satisfies_protocol():
    """Verify LLMProvider has the generate() interface matching ModelProvider."""
    import inspect
    assert hasattr(LLMProvider, "generate"), "LLMProvider missing generate()"
    sig = inspect.signature(LLMProvider.generate)
    params = list(sig.parameters.keys())
    assert "prompt" in params
    assert "stream" in params
    assert "chunk_callback" in params
    print("  Protocol compliance OK")


class _MockHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_MockHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_llm_provider_stream_timeout_wrapped_as_runtime_error():
    provider = LLMProvider()

    with patch(
        "helix.providers.openai_compat.urlopen",
        side_effect=TimeoutError("read timed out"),
    ):
        try:
            provider.generate("hello", stream=True)
            assert False, "Expected streaming timeout to raise RuntimeError"
        except RuntimeError as exc:
            assert "LLM network error" in str(exc)
            assert "read timed out" in str(exc)
    print("  LLMProvider stream timeout wrapping OK")


def test_llm_provider_stream_disconnect_wrapped_as_runtime_error():
    provider = LLMProvider()

    with patch(
        "helix.providers.openai_compat.urlopen",
        side_effect=RemoteDisconnected("closed"),
    ):
        try:
            provider.generate("hello", stream=True)
            assert False, "Expected stream disconnect to raise RuntimeError"
        except RuntimeError as exc:
            assert "LLM network error" in str(exc)
            assert "closed" in str(exc)
    print("  LLMProvider stream disconnect wrapping OK")


def test_llm_provider_non_stream_invalid_json_wrapped_as_runtime_error():
    provider = LLMProvider()

    with patch(
        "helix.providers._http.urlopen",
        return_value=_MockHTTPResponse(b"not-json"),
    ):
        try:
            provider.generate("hello", stream=False)
            assert False, "Expected invalid JSON response to raise RuntimeError"
        except RuntimeError as exc:
            assert "LLM invalid JSON response" in str(exc)
    print("  LLMProvider invalid JSON wrapping OK")


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

def test_skill_helpers():
    """Test helper parsing used by the skill loader."""
    from helix.core.agent import _parse_csv, _parse_frontmatter

    assert len(_parse_csv("a, b, c")) == 3
    assert _parse_frontmatter("---\nname: demo\n---\nbody\n") == {"name": "demo"}
    print("  Skill helper parsing OK")


# =========================================================================== #
# Knowledge loader tests
# =========================================================================== #


def _create_knowledge_catalog(root: Path) -> None:
    """Create a test knowledge catalog."""
    index_dir = root / "index"
    index_dir.mkdir(parents=True)
    catalog = [
        {
            "title": "RL for LLM Post-Training",
            "summary": "Overview of reinforcement learning methods commonly used in LLM post-training.",
            "path": "knowledge/docs/rl-overview.md",
            "tags": ["rl", "llm"],
        },
        {
            "title": "Agent System Design",
            "summary": "Notes on runtime architecture, orchestration, and component boundaries.",
            "path": "knowledge/docs/agent-design.md",
            "tags": "design, architecture",  # String tags
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
        assert catalog[0]["title"] == "Agent System Design"  # sorted by title
        assert catalog[1]["title"] == "RL for LLM Post-Training"
        assert catalog[0]["summary"].startswith("Notes on runtime architecture")
        assert isinstance(catalog[0]["tags"], list)  # string tags normalized
        print("  Knowledge loader OK")


def test_knowledge_loader_empty():
    """Test knowledge loading from non-existent directory."""
    catalog = load_knowledge_catalog(Path("/nonexistent/path"))
    assert catalog == []
    print("  Knowledge loader (empty) OK")

def test_knowledge_helpers():
    """Test helper normalization used by the knowledge loader."""
    from helix.core.agent import _normalize_tags

    assert len(_normalize_tags("a, b, c")) == 3
    assert _normalize_tags(["a", " ", "b"]) == ["a", "b"]
    print("  Knowledge helper normalization OK")


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
        session_root = workspace / "sessions" / "demo-01"
        project_root = session_root / "project"
        docs_root = session_root / "docs"
        state_root = session_root / ".state"

        prompt = _build_system_prompt(
            workspace,
            "core_agent",
            session_id="demo-01",
            session_root=session_root,
            project_root=project_root,
            docs_root=docs_root,
            state_root=state_root,
        )

        assert "Core Agent" in prompt
        assert "search-web" in prompt  # skill injected
        assert "rl-overview" in prompt  # knowledge injected
        assert "load-skill" in prompt  # builtin loader injected
        assert str(workspace) in prompt  # workspace path injected
        assert "demo-01" in prompt
        assert str(session_root) in prompt
        assert str(project_root) in prompt
        assert str(docs_root) in prompt
        assert str(state_root) in prompt
        assert "{{SKILLS_META_FROM_JSON}}" not in prompt  # placeholder replaced
        assert "{{KNOWLEDGE_META_FROM_JSON}}" not in prompt
        assert "{{BUILTIN_REFERENCE_LOADERS}}" not in prompt
        assert "{{WORKSPACE_ROOT}}" not in prompt
        assert "{{SESSION_ID}}" not in prompt
        assert "{{SESSION_ROOT}}" not in prompt
        assert "{{PROJECT_ROOT}}" not in prompt
        assert "{{DOCS_ROOT}}" not in prompt
        assert "{{STATE_ROOT}}" not in prompt
        assert "{{RUNTIME_WORKSPACE}}" not in prompt
        print("  Prompt builder OK")


def test_agent_rebuilds_prompt_from_updated_workspace_skills():
    """Workspace-backed agents should pick up skill metadata changes without restart."""

    class _DummyModel:
        def generate(self, prompt, *, stream=False, chunk_callback=None):
            return ""

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)
        session_root = workspace / "sessions" / "demo-01"
        project_root = session_root / "project"
        docs_root = session_root / "docs"
        state_root = session_root / ".state"

        agent = Agent(
            _DummyModel(),
            workspace=workspace,
            session_id="demo-01",
            session_root=session_root,
            project_root=project_root,
            docs_root=docs_root,
            state_root=state_root,
        )

        old_skill_dir = workspace / "skills" / "all-agents" / "search-web"
        old_skill_dir.rename(workspace / "skills" / "all-agents" / "search-live")
        (workspace / "skills" / "all-agents" / "search-live" / "SKILL.md").write_text(
            "---\n"
            "name: search-live\n"
            "description: Search live data sources\n"
            "handler: scripts/search_live.py\n"
            "required_tools: bash\n"
            "---\n"
            "Updated instructions here...\n",
            encoding="utf-8",
        )

        prompt = agent._build_prompt(
            State(observation=[Turn(role="user", content="What skills do you have?")])
        )

        assert '"skill_id": "search-live"' in prompt
        assert '"skill_id": "search-web"' not in prompt
        print("  Agent prompt rebuild picks up workspace skill changes OK")


def test_prompt_builder_unknown_role():
    """Test prompt builder returns empty for unknown role."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)

        prompt = _build_system_prompt(workspace, "nonexistent_role")
        assert prompt == ""
        print("  Prompt builder (unknown role) OK")


def test_prompt_builder_no_prompts():
    """Test prompt builder with workspace that has no matching role."""
    with tempfile.TemporaryDirectory() as td:
        prompt = _build_system_prompt(Path(td), "nonexistent_role_xyz")
        assert prompt == ""
        print("  Prompt builder (no prompts) OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Provider Initialization ===")
    test_llm_provider_default_init()
    test_llm_provider_custom_init()
    test_llm_provider_auto_appends_v1()
    test_llm_provider_preserves_existing_v1()
    test_llm_provider_env_vars()
    test_llm_provider_explicit_overrides_env()
    test_provider_satisfies_protocol()

    print("\n=== Skill Loader ===")
    test_skill_loader()
    test_skill_loader_empty()
    test_skill_helpers()

    print("\n=== Knowledge Loader ===")
    test_knowledge_loader()
    test_knowledge_loader_empty()
    test_knowledge_helpers()

    print("\n=== Prompt Builder ===")
    test_prompt_builder()
    test_prompt_builder_unknown_role()
    test_prompt_builder_no_prompts()

    print("\n✅ All Phase 3 tests passed!")
