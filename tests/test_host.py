"""Phase 5 verification — RuntimeHost + CLI integration tests.

Tests the RuntimeHost initialization, provider factory, slash commands,
and the full message processing pipeline (without a real LLM).
"""

import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.runtime.host import RuntimeHost
from helix.providers import create_provider as _create_provider
from helix.runtime.display import TURN_SEPARATOR, extract_streaming_response
from helix.runtime.cli import build_parser, main as cli_main
from helix.core.environment import Environment
from helix.core.state import Turn
from helix.providers.ollama import OllamaProvider
from helix.providers.openai_compat import OpenAICompatProvider


# Path to the real workspace
WORKSPACE = Path(__file__).resolve().parent.parent


def _make_host(workspace: Path, **kwargs) -> RuntimeHost:
    params = {"workspace": workspace, "session_id": "session-01", "sandbox_backend": "host"}
    params.update(kwargs)
    return RuntimeHost(**params)


# =========================================================================== #
# Provider factory tests
# =========================================================================== #


def test_provider_factory_ollama():
    """Verify provider factory creates OllamaProvider for 'ollama'."""
    provider = _create_provider("ollama")
    assert isinstance(provider, OllamaProvider)
    print("  Provider factory (ollama) OK")


def test_provider_factory_deepseek():
    """Verify provider factory creates OpenAICompatProvider for 'deepseek'."""
    provider = _create_provider("deepseek")
    assert isinstance(provider, OpenAICompatProvider)
    assert "deepseek" in provider.endpoint
    print("  Provider factory (deepseek) OK")


def test_provider_factory_lmstudio():
    """Verify provider factory creates OpenAICompatProvider for 'lmstudio'."""
    provider = _create_provider("lmstudio")
    assert isinstance(provider, OpenAICompatProvider)
    print("  Provider factory (lmstudio) OK")


def test_provider_factory_with_model():
    """Verify provider factory passes model override."""
    provider = _create_provider("ollama", model="custom-model")
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "custom-model"
    print("  Provider factory (with model) OK")


# =========================================================================== #
# RuntimeHost initialization tests
# =========================================================================== #


def test_host_init():
    """Verify RuntimeHost initializes with correct configuration."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(
            workspace=Path(td),
            provider="ollama",
            mode="auto",
        )
        assert host.provider_name == "ollama"
        assert host.mode == "auto"
        assert host.workspace == Path(td).resolve()
        assert host.session_id == "session-01"
        assert host.session_root == (Path(td) / "sessions" / "session-01").resolve()
        assert host.project_root == host.session_root / "project"
        assert host.docs_root == host.session_root / "docs"
        assert host.project_root.exists()
        assert host.docs_root.exists()
        assert host.state_root.exists()
        assert host.session_path == host.state_root / "session.json"
        assert len(host._agent.system_prompt) > 0  # loaded from package prompts
        assert str(host.project_root) in host._agent.system_prompt
        assert str(host.docs_root) in host._agent.system_prompt
        assert str(host.state_root) in host._agent.system_prompt
        print("  RuntimeHost init OK")


def test_host_init_controlled():
    """Verify RuntimeHost in controlled mode."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(
            workspace=Path(td),
            provider="deepseek",
            mode="controlled",
        )
        assert host.provider_name == "deepseek"
        assert host.mode == "controlled"
        print("  RuntimeHost init (controlled) OK")


def test_host_init_with_session_id_loads_existing_state():
    """Verify RuntimeHost resumes a named session from the current session path."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        session_path = workspace / "sessions" / "review-01" / ".state" / "session.json"

        env = Environment(workspace=workspace)
        env.record(Turn(role="user", content="Previous request"))
        env.record(Turn(role="core-agent", content="Previous response"))
        env.workflow_summary = "Prior summary"
        env.save_session(session_path)
        raw = json.loads(session_path.read_text(encoding="utf-8"))
        raw["last_prompt"] = "Prior prompt"
        session_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        host = RuntimeHost(
            workspace=workspace,
            provider="ollama",
            mode="auto",
            session_id="review-01",
            sandbox_backend="host",
        )

        assert host.session_id == "review-01"
        assert host.session_path == session_path.resolve()
        assert host._session_loaded is True
        assert len(host._env.full_history) == 2
        assert host._env.workflow_summary == "Prior summary"
        assert host._agent.last_prompt == "Prior prompt"
        print("  RuntimeHost named session resume OK")


def test_host_auto_prefers_docker_when_available():
    """Verify auto backend selects Docker when Docker is available."""
    calls: dict[str, object] = {}

    class FakeDockerExecutor:
        approval_profile = "docker-online-rw-workspace-v1:test"

        def __init__(
            self,
            workspace: Path,
            *,
            session_id: str | None = None,
        ):
            self.workspace = workspace
            self.session_id = session_id
            calls["session_id"] = session_id

        def __call__(self, payload, workspace):
            return Turn(role="runtime", content="fake")

        def prepare_runtime(self) -> None:
            calls["prepared"] = True

        def shutdown(self) -> None:
            calls["shutdown"] = True

        def status_fields(self) -> dict[str, str]:
            return {"sandbox_backend": "docker", "docker_image": "fake-image"}

        def tool_environment(self) -> dict[str, str]:
            return {"SEARXNG_BASE_URL": "http://fake-searxng:8080"}

    with tempfile.TemporaryDirectory() as td:
        with patch("helix.runtime.host.docker_is_available", return_value=(True, "")):
            with patch("helix.runtime.host.DockerSandboxExecutor", FakeDockerExecutor):
                host = RuntimeHost(
                    workspace=Path(td),
                    session_id="session-01",
                    sandbox_backend="auto",
                )
        assert host.resolved_sandbox_backend == "docker(auto)"
        assert host._env.approval_profile == "docker-online-rw-workspace-v1:test"
        assert os.environ["SEARXNG_BASE_URL"] == "http://fake-searxng:8080"
        assert calls["session_id"] == "session-01"
        assert calls["prepared"] is True
        host._shutdown()
        assert calls["shutdown"] is True
        print("  RuntimeHost auto docker selection OK")


def test_host_auto_falls_back_to_host_when_docker_unavailable():
    """Verify auto backend falls back to the host executor when Docker is unavailable."""
    with tempfile.TemporaryDirectory() as td:
        with patch(
            "helix.runtime.host.docker_is_available",
            return_value=(False, "docker unavailable"),
        ):
            host = RuntimeHost(
                workspace=Path(td),
                session_id="session-01",
                sandbox_backend="auto",
            )
        assert host.resolved_sandbox_backend == "host(auto-fallback)"
        assert host._env.approval_profile == "host-subprocess-v1"
        assert "docker unavailable" in host._status_text()
        print("  RuntimeHost auto host fallback OK")


# =========================================================================== #
# Slash command tests
# =========================================================================== #


def test_host_command_help():
    """Verify /help returns help text."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        result = host._handle_command("/help")
        assert "Commands:" in result
        assert "/exit" in result
        print("  /help command OK")


def test_host_command_status():
    """Verify /status returns status text."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        result = host._handle_command("/status")
        assert "provider=" in result
        assert "mode=" in result
        assert "session_state=new" in result
        print("  /status command OK")


def test_host_command_full_history():
    """Verify /full_history opens a raw timeline view."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td), session_id="debug-01", sandbox_backend="host")
        host._env.record(Turn(
            role="user",
            content="Hello",
            timestamp="2026-03-10 00:41:55",
        ))
        with patch("helix.runtime.host.open_file_in_viewer", return_value=True):
            result = host._handle_command("/full_history")
        path = Path(td) / "sessions" / "debug-01" / ".state" / "views" / "debug-01.full_history.html"
        assert result == f"Opened session view: {path.resolve()}"
        text = path.read_text(encoding="utf-8")
        assert "Agentic System Timeline View" in text
        assert '"value"' not in text
        assert "[2026-03-10 00:41:55] user&gt; Hello" in text
        assert "Hello" in text
        print("  /full_history command OK")


def test_host_command_observation():
    """Verify /observation opens a raw timeline view."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td), session_id="debug-01", sandbox_backend="host")
        host._env.record(Turn(role="user", content="Hello"))
        with patch("helix.runtime.host.open_file_in_viewer", return_value=True):
            result = host._handle_command("/observation")
        path = Path(td) / "sessions" / "debug-01" / ".state" / "views" / "debug-01.observation.html"
        assert result == f"Opened session view: {path.resolve()}"
        text = path.read_text(encoding="utf-8")
        assert "Agentic System Timeline View" in text
        assert '"value"' not in text
        assert "] user&gt; Hello" in text
        assert "Hello" in text
        print("  /observation command OK")


def test_host_command_workflow_summary():
    """Verify /workflow_summary opens a session-scoped HTML view."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td), session_id="debug-01", sandbox_backend="host")
        host._env.workflow_summary = "## Current Status\nWorking"
        with patch("helix.runtime.host.open_file_in_viewer", return_value=True):
            result = host._handle_command("/workflow_summary")
        path = Path(td) / "sessions" / "debug-01" / ".state" / "views" / "debug-01.workflow_summary.html"
        assert result == f"Opened session view: {path.resolve()}"
        text = path.read_text(encoding="utf-8")
        assert "workflow_summary" in text
        assert '## Current Status' in text
        print("  /workflow_summary command OK")


def test_host_command_last_prompt():
    """Verify /last_prompt opens the raw prompt text exactly as sent to the model."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td), session_id="debug-01", sandbox_backend="host")
        host._agent.last_prompt = "system\n\n<latest_context>\n[user] Hello\n</latest_context>"
        with patch("helix.runtime.host.open_file_in_viewer", return_value=True):
            result = host._handle_command("/last_prompt")
        path = Path(td) / "sessions" / "debug-01" / ".state" / "views" / "debug-01.last_prompt.html"
        assert result == f"Opened session view: {path.resolve()}"
        text = path.read_text(encoding="utf-8")
        assert "Agentic System Prompt View" in text
        assert '"value"' not in text
        assert '&lt;latest_context&gt;' in text
        assert "system\n\n&lt;latest_context&gt;" in text
        assert '[user] Hello' in text
        print("  /last_prompt command OK")


def test_host_command_last_prompt_empty():
    """Verify /last_prompt writes a placeholder HTML view before first use."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td), session_id="debug-01", sandbox_backend="host")
        with patch("helix.runtime.host.open_file_in_viewer", return_value=False):
            result = host._handle_command("/last_prompt")
        path = Path(td) / "sessions" / "debug-01" / ".state" / "views" / "debug-01.last_prompt.html"
        assert result == f"Session view written: {path.resolve()}"
        text = path.read_text(encoding="utf-8")
        assert "last_prompt" in text
        assert "(none yet)" in text
        print("  /last_prompt empty OK")


def test_host_requires_session_id():
    """Verify RuntimeHost requires a named session."""
    with tempfile.TemporaryDirectory() as td:
        try:
            RuntimeHost(workspace=Path(td))
            assert False, "Expected missing session_id to raise"
        except ValueError:
            print("  Session id required OK")


def test_host_command_exit():
    """Verify /exit returns None (exit signal)."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        result = host._handle_command("/exit")
        assert result is None
        print("  /exit command OK")


def test_host_command_unknown():
    """Verify unknown command returns error message."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        result = host._handle_command("/foobar")
        assert "Unknown" in result
        print("  Unknown command OK")


# =========================================================================== #
# Message processing test
# =========================================================================== #


def test_host_process_message():
    """Test that _process_message records turns and runs the agent loop."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))

        # Mock the model to return a simple chat response
        def mock_generate(prompt, *, stream=False, chunk_callback=None):
            if stream and chunk_callback is not None:
                for piece in ('<output>{"response": "Hello ', 'from the agent!", '):
                    chunk_callback(piece)
            return (
                '<output>'
                '{"response": "Hello from the agent!", '
                '"action": "chat", "action_input": {}}'
                '</output>'
            )

        mock_generate = MagicMock(side_effect=mock_generate)
        host._model.generate = mock_generate

        # Capture stdout
        captured = StringIO()
        with patch("sys.stdout", captured):
            host._process_message("Hello agent!")
        output = captured.getvalue()

        # Verify user turn was recorded
        user_turns = [t for t in host._env.full_history if t.role == "user"]
        assert len(user_turns) == 1
        assert user_turns[0].content == "Hello agent!"

        # Verify agent responded
        agent_turns = [t for t in host._env.full_history if t.role == "core-agent"]
        assert len(agent_turns) == 1
        assert "Hello from the agent!" in agent_turns[0].content
        assert "[next_action] chat" in agent_turns[0].content

        # Requester-facing UI should not include action metadata
        assert "Hello agent!" not in output
        assert f"{TURN_SEPARATOR}\ncore-agent> Hello from the agent!\n{TURN_SEPARATOR}" in output
        assert "[next_action]" not in output
        assert "[action_input]" not in output

        # Verify model was called
        assert mock_generate.called
        assert not (Path(td) / ".session").exists()

        print("  Message processing OK")


def test_host_process_message_saves_named_session():
    """Verify named sessions persist after each interaction."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        host = RuntimeHost(workspace=workspace, session_id="active-1", sandbox_backend="host")

        mock_generate = MagicMock(return_value=(
            '<output>'
            '{"response": "Saved response", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        ))
        host._model.generate = mock_generate

        captured = StringIO()
        with patch("sys.stdout", captured):
            host._process_message("Persist this")

        session_path = workspace / "sessions" / "active-1" / ".state" / "session.json"
        assert session_path.exists()
        reloaded = Environment(workspace=workspace)
        assert reloaded.load_session(session_path) is True
        assert any(t.content == "Persist this" for t in reloaded.full_history)
        raw = json.loads(session_path.read_text(encoding="utf-8"))
        assert raw["last_prompt"]
        print("  Named session persistence OK")

def test_host_process_exec():
    """Test that _process_message handles exec actions correctly."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td), mode="auto")

        call_count = [0]
        def mock_generate(prompt, *, stream=False, chunk_callback=None):
            call_count[0] += 1
            if call_count[0] == 1:
                if stream and chunk_callback is not None:
                    for piece in ('<output>{"response": "Let ', 'me check.", '):
                        chunk_callback(piece)
                return (
                    '<output>'
                    '{"response": "Let me check.", '
                    '"action": "exec", "action_input": {'
                    '"job_name": "test-exec", "code_type": "bash", '
                    '"script": "echo test-output"}}'
                    '</output>'
                )
            if stream and chunk_callback is not None:
                for piece in ('<output>{"response": "Do', 'ne!", '):
                    chunk_callback(piece)
            return (
                '<output>'
                '{"response": "Done!", "action": "chat", "action_input": {}}'
                '</output>'
            )

        host._model.generate = mock_generate

        captured = StringIO()
        with patch("sys.stdout", captured):
            host._process_message("Run a test")
        output = captured.getvalue()

        # Should have user, agent (exec), runtime (result), agent (chat)
        assert len(host._env.full_history) >= 4
        runtime_turns = [t for t in host._env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "test-output" in runtime_turns[0].content
        assert "Run a test" not in output
        assert f"{TURN_SEPARATOR}\ncore-agent> Let me check.\n{TURN_SEPARATOR}" in output
        assert f"{TURN_SEPARATOR}\ncore-agent> Done!\n{TURN_SEPARATOR}" in output

        print("  Exec processing OK")


def test_host_process_message_runtime_error():
    """Verify runtime-facing provider errors are surfaced without crashing."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))
        host._model.generate = MagicMock(
            side_effect=RuntimeError(
                "Missing API key for provider 'zai'. Set ZAI_API_KEY or OPENAI_COMPAT_API_KEY."
            )
        )

        captured = StringIO()
        with patch("sys.stdout", captured):
            host._process_message("Who are you?")

        output = captured.getvalue()
        assert f"{TURN_SEPARATOR}\nruntime> Agent error: Missing API key for provider 'zai'" in output
        assert TURN_SEPARATOR in output
        runtime_turns = [t for t in host._env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "Missing API key for provider 'zai'" in runtime_turns[0].content
        print("  Message runtime error handling OK")


def test_host_process_message_discards_parse_failed_preview():
    """Verify parse-failed streamed text is not shown to the requester."""
    with tempfile.TemporaryDirectory() as td:
        host = _make_host(Path(td))

        call_count = [0]

        def mock_generate(prompt, *, stream=False, chunk_callback=None):
            call_count[0] += 1
            if call_count[0] == 1:
                bad = (
                    '<output>'
                    '{"response": "Discard this preview", '
                    '"action": "chat", "action_input": {"bad": "\n"}}'
                    '</output>'
                )
                if stream and chunk_callback is not None:
                    for piece in ('<output>{"response": "Discard ', 'this preview", '):
                        chunk_callback(piece)
                return bad

            good = (
                '<output>'
                '{"response": "Keep this final answer", '
                '"action": "chat", "action_input": {}}'
                '</output>'
            )
            if stream and chunk_callback is not None:
                for piece in ('<output>{"response": "Keep ', 'this final answer", '):
                    chunk_callback(piece)
            return good

        host._model.generate = mock_generate

        captured = StringIO()
        with patch("sys.stdout", captured):
            host._process_message("Retry if needed")

        output = captured.getvalue()
        assert "Discard this preview" not in output
        assert "Keep this final answer" in output
        assert TURN_SEPARATOR in output

        runtime_turns = [t for t in host._env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "Output parse error" in runtime_turns[0].content

        agent_turns = [t for t in host._env.full_history if t.role == "core-agent"]
        assert len(agent_turns) == 1
        assert "Keep this final answer" in agent_turns[0].content
        print("  Parse-failed preview discarded OK")


# =========================================================================== #
# CLI parser tests
# =========================================================================== #


def test_cli_parser():
    """Verify CLI parser handles all arguments correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--provider", "deepseek",
        "--mode", "auto",
        "--model", "deepseek-chat",
        "--workspace", "/tmp/test",
        "--session-id", "design-01",
    ])
    assert args.provider == "deepseek"
    assert args.mode == "auto"
    assert args.model == "deepseek-chat"
    assert args.workspace == "/tmp/test"
    assert args.session_id == "design-01"
    print("  CLI parser OK")


def test_cli_parser_requires_session_id():
    """Verify CLI parser requires a session id."""
    parser = build_parser()
    try:
        parser.parse_args(["--workspace", "."])
        assert False, "Expected missing session_id to raise"
    except SystemExit:
        print("  CLI parser requires session id OK")


def test_host_invalid_session_id():
    """Verify RuntimeHost rejects unsafe session identifiers."""
    with tempfile.TemporaryDirectory() as td:
        try:
            RuntimeHost(workspace=Path(td), session_id="../bad")
            assert False, "Expected invalid session_id to raise"
        except ValueError:
            print("  Invalid session id rejected OK")


# =========================================================================== #
# Streaming Extractor tests
# =========================================================================== #


def test_streaming_extractor():
    partial = '{"response": "Hello wor'
    result = extract_streaming_response(partial)
    assert result == "Hello wor"
    print("  Streaming extractor OK")


def test_streaming_extractor_decodes_newlines():
    partial = '{"response": "Hello\\nworld"}'
    result = extract_streaming_response(partial)
    assert result == "Hello\nworld"
    print("  Streaming extractor newline decode OK")


def test_streaming_extractor_handles_partial_escape():
    partial = '{"response": "Hello\\'
    result = extract_streaming_response(partial)
    assert result == "Hello"
    print("  Streaming extractor partial escape OK")


def test_streaming_extractor_not_yet():
    result = extract_streaming_response('{"action": ')
    assert result is None
    print("  Streaming extractor (not yet) OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Provider Factory ===")
    test_provider_factory_ollama()
    test_provider_factory_deepseek()
    test_provider_factory_lmstudio()
    test_provider_factory_with_model()

    print("\n=== RuntimeHost Init ===")
    test_host_init()
    test_host_init_controlled()
    test_host_init_with_session_id_loads_existing_state()

    print("\n=== Slash Commands ===")
    test_host_command_help()
    test_host_command_status()
    test_host_command_full_history()
    test_host_command_observation()
    test_host_command_workflow_summary()
    test_host_command_last_prompt()
    test_host_command_last_prompt_empty()
    test_host_requires_session_id()
    test_host_command_exit()
    test_host_command_unknown()

    print("\n=== Message Processing ===")
    test_host_process_message()
    test_host_process_message_saves_named_session()
    test_host_process_exec()
    test_host_process_message_runtime_error()
    test_host_process_message_discards_parse_failed_preview()

    print("\n=== CLI Parser ===")
    test_cli_parser()
    test_cli_parser_requires_session_id()
    test_host_invalid_session_id()

    print("\n=== Streaming ===")
    test_streaming_extractor()
    test_streaming_extractor_decodes_newlines()
    test_streaming_extractor_handles_partial_escape()
    test_streaming_extractor_not_yet()

    print("\n✅ All Phase 5 tests passed!")
