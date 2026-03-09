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
        print("  /status command OK")


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

        # Verify user turn was recorded
        user_turns = [t for t in host._env.full_history if t.role == "user"]
        assert len(user_turns) == 1
        assert user_turns[0].content == "Hello agent!"

        # Verify agent responded
        agent_turns = [t for t in host._env.full_history if t.role == "core-agent"]
        assert len(agent_turns) == 1
        assert "Hello from the agent!" in agent_turns[0].content

        # Verify model was called
        assert mock_generate.called

        print("  Message processing OK")


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
    ])
    assert args.provider == "deepseek"
    assert args.mode == "auto"
    assert args.model == "deepseek-chat"
    assert args.workspace == "/tmp/test"
    print("  CLI parser OK")


def test_cli_parser_defaults():
    """Verify CLI parser defaults."""
    parser = build_parser()
    args = parser.parse_args(["--workspace", "."])
    assert args.provider == "ollama"
    assert args.mode == "controlled"
    assert args.model is None
    print("  CLI parser defaults OK")


# =========================================================================== #
# Streaming Extractor tests
# =========================================================================== #


def test_streaming_extractor():
    partial = '{"response": "Hello wor'
    result = extract_streaming_response(partial)
    assert result == "Hello wor"
    print("  Streaming extractor OK")


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

    print("\n=== Slash Commands ===")
    test_host_command_help()
    test_host_command_status()
    test_host_command_exit()
    test_host_command_unknown()

    print("\n=== Message Processing ===")
    test_host_process_message()
    test_host_process_exec()

    print("\n=== CLI Parser ===")
    test_cli_parser()
    test_cli_parser_defaults()

    print("\n=== Streaming ===")
    test_streaming_extractor()
    test_streaming_extractor_not_yet()

    print("\n✅ All Phase 5 tests passed!")
