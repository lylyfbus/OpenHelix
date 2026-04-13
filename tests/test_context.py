"""Phase 3 verification tests for providers, context loaders, and prompt builder."""

import sys
import tempfile
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.providers.openai_compat import LLMProvider
from helix.core.agent import Agent
from helix.core.agent import _load_skills as load_skills
from helix.core.agent import _build_system_prompt
from helix.core.state import State, Turn


# =========================================================================== #
# Provider tests (structural — no real LLM calls)
# =========================================================================== #


def test_llm_provider_default_init():
    """Verify LLMProvider initializes with correct defaults."""
    provider = LLMProvider(endpoint_url="http://localhost:11434/v1", model="test")
    assert provider.model == "test"
    assert "11434" in provider.endpoint_url
    assert provider.timeout == 300
    assert "/v1" in provider.endpoint_url
    print("  LLMProvider default init OK")


def test_llm_provider_custom_init():
    """Verify LLMProvider respects custom parameters."""
    provider = LLMProvider(
        endpoint_url="http://myhost:8080/v1",
        api_key="test-key",
        model="deepseek-r1:14b",
        timeout=60,
        temperature=0.5,
    )
    assert provider.model == "deepseek-r1:14b"
    assert "myhost:8080" in provider.endpoint_url
    assert provider.timeout == 60
    assert provider.api_key == "test-key"
    print("  LLMProvider custom init OK")


def test_llm_provider_builds_endpoint():
    """Verify LLMProvider constructs the chat completions endpoint."""
    provider = LLMProvider(endpoint_url="http://localhost:1234/v1", model="test")
    assert provider.endpoint_url == "http://localhost:1234/v1"
    print("  LLMProvider builds endpoint OK")


def test_provider_satisfies_protocol():
    """Verify LLMProvider has the expected generate() interface."""
    import inspect
    assert hasattr(LLMProvider, "generate"), "LLMProvider missing generate()"
    sig = inspect.signature(LLMProvider.generate)
    params = list(sig.parameters.keys())
    assert "messages" in params
    assert "chunk_callback" in params
    assert "stream" not in params
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
    provider = LLMProvider(endpoint_url="http://localhost:11434/v1", model="test")

    with patch(
        "helix.providers.openai_compat.urlopen",
        side_effect=TimeoutError("read timed out"),
    ):
        try:
            provider.generate([{"role": "user", "content": "hello"}])
            assert False, "Expected streaming timeout to raise RuntimeError"
        except RuntimeError as exc:
            assert "LLM network error" in str(exc)
            assert "read timed out" in str(exc)
    print("  LLMProvider stream timeout wrapping OK")


def test_llm_provider_stream_disconnect_wrapped_as_runtime_error():
    provider = LLMProvider(endpoint_url="http://localhost:11434/v1", model="test")

    with patch(
        "helix.providers.openai_compat.urlopen",
        side_effect=RemoteDisconnected("closed"),
    ):
        try:
            provider.generate([{"role": "user", "content": "hello"}])
            assert False, "Expected stream disconnect to raise RuntimeError"
        except RuntimeError as exc:
            assert "LLM network error" in str(exc)
            assert "closed" in str(exc)
    print("  LLMProvider stream disconnect wrapping OK")



# =========================================================================== #
# Skill loader tests
# =========================================================================== #


def _create_skill_tree(root: Path) -> None:
    """Create a test skill directory tree."""
    # builtin_skills/search-web/SKILL.md
    skill_dir = root / "builtin_skills" / "search-web"
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

    # builtin_skills/code-review/SKILL.md
    skill_dir2 = root / "builtin_skills" / "code-review"
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



def test_skill_loader():
    """Test skill loading and filtering."""
    with tempfile.TemporaryDirectory() as td:
        skills_root = Path(td) / "skills"
        _create_skill_tree(skills_root)

        skills = load_skills(skills_root)
        assert len(skills) == 2, f"Expected 2 skills, got {len(skills)}"
        paths = {s["path"] for s in skills}
        assert any("search-web" in p for p in paths)
        assert any("code-review" in p for p in paths)
        print("  Skill loader OK")


def test_skill_loader_empty():
    """Test skill loading from non-existent directory."""
    skills = load_skills(Path("/nonexistent/path"))
    assert skills == []
    print("  Skill loader (empty) OK")

def test_skill_helpers():
    """Test helper parsing used by the skill loader."""
    from helix.core.agent import _parse_frontmatter

    assert _parse_frontmatter("---\nname: demo\n---\nbody\n") == {"name": "demo"}
    print("  Skill helper parsing OK")


# ===========================================================================
# Prompt builder tests
# =========================================================================== #


def _create_workspace(root: Path) -> None:
    """Create a test workspace with skills."""
    # Create skills
    _create_skill_tree(root / "skills")


def test_prompt_builder():
    """Test full prompt assembly with placeholder injection."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)
        session_root = workspace / "sessions" / "demo-01"
        project_root = session_root / "project"
        docs_root = session_root / "docs"

        prompt = _build_system_prompt(
            workspace,
            "core_agent",
            session_root=session_root,
            project_root=project_root,
            docs_root=docs_root,
        )

        assert "Core Agent" in prompt
        assert "search-web" in prompt  # skill injected
        assert str(workspace) in prompt  # workspace path injected
        assert str(session_root) in prompt
        assert str(project_root) in prompt
        assert str(docs_root) in prompt
        assert "{{SKILLS_META_FROM_JSON}}" not in prompt  # placeholder replaced
        assert "{{WORKSPACE_ROOT}}" not in prompt
        assert "{{SESSION_ROOT}}" not in prompt
        assert "{{PROJECT_ROOT}}" not in prompt
        assert "{{DOCS_ROOT}}" not in prompt
        print("  Prompt builder OK")


def test_agent_rebuilds_prompt_from_updated_workspace_skills():
    """Workspace-backed agents should pick up skill metadata changes without restart."""

    class _DummyModel:
        def generate(self, messages, *, chunk_callback=None):
            return ""

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)
        session_root = workspace / "sessions" / "demo-01"
        project_root = session_root / "project"
        docs_root = session_root / "docs"

        agent = Agent(
            _DummyModel(),
            workspace=workspace,
            session_root=session_root,
            project_root=project_root,
            docs_root=docs_root,
        )

        old_skill_dir = workspace / "skills" / "builtin_skills" / "search-web"
        old_skill_dir.rename(workspace / "skills" / "builtin_skills" / "search-live")
        (workspace / "skills" / "builtin_skills" / "search-live" / "SKILL.md").write_text(
            "---\n"
            "name: search-live\n"
            "description: Search live data sources\n"
            "handler: scripts/search_live.py\n"
            "required_tools: bash\n"
            "---\n"
            "Updated instructions here...\n",
            encoding="utf-8",
        )

        messages = agent._build_messages(
            State(observation=[Turn(role="user", content="What skills do you have?")])
        )

        # Skills metadata lives in the system message
        system_content = messages[0]["content"]
        assert "search-live" in system_content
        assert "search-web" not in system_content
        print("  Agent messages rebuild picks up workspace skill changes OK")


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

    print("\n=== Prompt Builder ===")
    test_prompt_builder()
    test_prompt_builder_unknown_role()
    test_prompt_builder_no_prompts()

    print("\n✅ All Phase 3 tests passed!")
