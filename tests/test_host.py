"""Phase 5 verification — RuntimeHost + CLI integration tests.

Tests the RuntimeHost initialization, provider factory, slash commands,
and the full message processing pipeline (without a real LLM).
"""

import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_system.runtime.host import RuntimeHost, _create_provider, extract_streaming_response
from agentic_system.runtime.cli import build_parser, main as cli_main
from agentic_system.core.environment import Environment
from agentic_system.core.state import Turn
from agentic_system.providers.ollama import OllamaProvider
from agentic_system.providers.openai_compat import OpenAICompatProvider


# Path to the real workspace
WORKSPACE = Path(__file__).resolve().parent.parent


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
        host = RuntimeHost(
            workspace=Path(td),
            provider="ollama",
            mode="auto",
        )
        assert host.provider_name == "ollama"
        assert host.mode == "auto"
        assert host.workspace == Path(td).resolve()
        assert host.session_id is None
        assert host.session_path is None
        assert len(host._agent.system_prompt) > 0  # loaded from package prompts
        print("  RuntimeHost init OK")


def test_host_init_controlled():
    """Verify RuntimeHost in controlled mode."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(
            workspace=Path(td),
            provider="deepseek",
            mode="controlled",
        )
        assert host.provider_name == "deepseek"
        assert host.mode == "controlled"
        print("  RuntimeHost init (controlled) OK")


def test_host_init_with_session_id_loads_existing_state():
    """Verify RuntimeHost resumes a named session when it already exists."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        session_path = workspace / ".sessions" / "review-01.json"

        env = Environment(workspace=workspace)
        env.record(Turn(role="user", content="Previous request"))
        env.record(Turn(role="core-agent", content="Previous response"))
        env.workflow_summary = "Prior summary"
        env.save_session(session_path)

        host = RuntimeHost(
            workspace=workspace,
            provider="ollama",
            mode="auto",
            session_id="review-01",
        )

        assert host.session_id == "review-01"
        assert host.session_path == session_path.resolve()
        assert host._session_loaded is True
        assert len(host._env.full_history) == 2
        assert host._env.workflow_summary == "Prior summary"
        print("  RuntimeHost named session resume OK")


# =========================================================================== #
# Slash command tests
# =========================================================================== #


def test_host_command_help():
    """Verify /help returns help text."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        result = host._handle_command("/help")
        assert "Commands:" in result
        assert "/exit" in result
        print("  /help command OK")


def test_host_command_status():
    """Verify /status returns status text."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        result = host._handle_command("/status")
        assert "provider=" in result
        assert "mode=" in result
        assert "session_state=ephemeral" in result
        print("  /status command OK")


def test_host_command_full_history():
    """Verify /full_history returns the in-memory full history content."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        host._env.record(Turn(
            role="user",
            content="Hello",
            timestamp="2026-03-10 00:41:55",
        ))
        result = host._handle_command("/full_history")
        assert "<full_history>" in result
        assert "[2026-03-10 00:41:55] user> Hello" in result
        assert "user> Hello" in result
        print("  /full_history command OK")


def test_host_command_observation():
    """Verify /observation returns the current observation content."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        host._env.record(Turn(role="user", content="Hello"))
        result = host._handle_command("/observation")
        assert "<observation>" in result
        assert "user> Hello" in result
        print("  /observation command OK")


def test_host_command_workflow_summary():
    """Verify /workflow_summary returns the current summary content."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        host._env.workflow_summary = "## Current Status\nWorking"
        result = host._handle_command("/workflow_summary")
        assert "<workflow_summary>" in result
        assert "## Current Status" in result
        print("  /workflow_summary command OK")


def test_host_command_last_prompt():
    """Verify /last_prompt returns the last prompt sent to the core agent."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        host._agent.last_prompt = "system\n\n<latest_context>\n[user] Hello\n</latest_context>"
        result = host._handle_command("/last_prompt")
        assert "<last_prompt>" in result
        assert "<latest_context>" in result
        assert "[user] Hello" in result
        print("  /last_prompt command OK")


def test_host_command_last_prompt_empty():
    """Verify /last_prompt is explicit before any prompt is sent."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        result = host._handle_command("/last_prompt")
        assert "<last_prompt>" in result
        assert "(none yet)" in result
        print("  /last_prompt empty OK")


def test_host_command_exit():
    """Verify /exit returns None (exit signal)."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        result = host._handle_command("/exit")
        assert result is None
        print("  /exit command OK")


def test_host_command_unknown():
    """Verify unknown command returns error message."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        result = host._handle_command("/foobar")
        assert "Unknown" in result
        print("  Unknown command OK")


# =========================================================================== #
# Message processing test
# =========================================================================== #


def test_host_process_message():
    """Test that _process_message records turns and runs the agent loop."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))

        # Mock the model to return a simple chat response
        mock_generate = MagicMock(return_value=(
            '<output>'
            '{"response": "Hello from the agent!", '
            '"action": "chat", "action_input": {}}'
            '</output>'
        ))
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
        host = RuntimeHost(workspace=workspace, session_id="active-1")

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

        session_path = workspace / ".sessions" / "active-1.json"
        assert session_path.exists()
        reloaded = Environment(workspace=workspace)
        assert reloaded.load_session(session_path) is True
        assert any(t.content == "Persist this" for t in reloaded.full_history)
        print("  Named session persistence OK")


def test_host_process_exec():
    """Test that _process_message handles exec actions correctly."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td), mode="auto")

        call_count = [0]
        def mock_generate(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '<output>'
                    '{"response": "Let me check.", '
                    '"action": "exec", "action_input": {'
                    '"job_name": "test-exec", "code_type": "bash", '
                    '"script": "echo test-output"}}'
                    '</output>'
                )
            return (
                '<output>'
                '{"response": "Done!", "action": "chat", "action_input": {}}'
                '</output>'
            )

        host._model.generate = mock_generate

        captured = StringIO()
        with patch("sys.stdout", captured):
            host._process_message("Run a test")

        # Should have user, agent (exec), runtime (result), agent (chat)
        assert len(host._env.full_history) >= 4
        runtime_turns = [t for t in host._env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "test-output" in runtime_turns[0].content

        print("  Exec processing OK")


def test_host_process_message_runtime_error():
    """Verify runtime-facing provider errors are surfaced without crashing."""
    with tempfile.TemporaryDirectory() as td:
        host = RuntimeHost(workspace=Path(td))
        host._model.generate = MagicMock(
            side_effect=RuntimeError(
                "Missing API key for provider 'zai'. Set ZAI_API_KEY or OPENAI_COMPAT_API_KEY."
            )
        )

        captured = StringIO()
        with patch("sys.stdout", captured):
            host._process_message("Who are you?")

        output = captured.getvalue()
        assert "runtime> Agent error: Missing API key for provider 'zai'" in output
        runtime_turns = [t for t in host._env.full_history if t.role == "runtime"]
        assert len(runtime_turns) == 1
        assert "Missing API key for provider 'zai'" in runtime_turns[0].content
        print("  Message runtime error handling OK")


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


def test_cli_parser_defaults():
    """Verify CLI parser defaults."""
    parser = build_parser()
    args = parser.parse_args(["--workspace", "."])
    assert args.provider == "ollama"
    assert args.mode == "controlled"
    assert args.model is None
    assert args.session_id is None
    print("  CLI parser defaults OK")


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
    test_host_command_exit()
    test_host_command_unknown()

    print("\n=== Message Processing ===")
    test_host_process_message()
    test_host_process_message_saves_named_session()
    test_host_process_exec()
    test_host_process_message_runtime_error()

    print("\n=== CLI Parser ===")
    test_cli_parser()
    test_cli_parser_defaults()
    test_host_invalid_session_id()

    print("\n=== Streaming ===")
    test_streaming_extractor()
    test_streaming_extractor_decodes_newlines()
    test_streaming_extractor_handles_partial_escape()
    test_streaming_extractor_not_yet()

    print("\n✅ All Phase 5 tests passed!")
