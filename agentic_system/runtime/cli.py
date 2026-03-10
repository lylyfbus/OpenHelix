"""CLI entrypoint for the agentic framework.

Launches the RuntimeHost with provider, mode, model, workspace,
and tool configuration from command-line arguments.

Usage::

    python -m agentic_system.runtime.cli --workspace /path/to/workspace
    python -m agentic_system.runtime.cli --provider deepseek --mode auto --workspace .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .host import RuntimeHost


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Agentic System — RL-inspired agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --workspace .\n"
            "  %(prog)s --provider deepseek --model deepseek-chat --workspace ~/agent\n"
            "  %(prog)s --provider ollama --mode auto --workspace /tmp/sandbox\n"
        ),
    )

    # Core settings
    parser.add_argument(
        "--provider",
        default="ollama",
        help="LLM provider: ollama, deepseek, lmstudio, zai, openai_compatible (default: ollama)",
    )
    parser.add_argument(
        "--mode",
        default="controlled",
        choices=["auto", "controlled"],
        help="Execution mode: auto (no confirmation) or controlled (default: controlled)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (uses provider defaults if not specified)",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Runtime workspace path",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session identifier for loading and persisting conversation state",
    )

    # Tool configuration
    tool_group = parser.add_argument_group(
        "tool configuration",
        "Override default models and backends for built-in skill tools",
    )
    tool_group.add_argument(
        "--image-analysis-provider",
        default=None,
        help="Image analysis provider (default: ollama)",
    )
    tool_group.add_argument(
        "--image-analysis-model",
        default=None,
        help="Image analysis model (default: glm-ocr)",
    )
    tool_group.add_argument(
        "--image-generation-provider",
        default=None,
        help="Image generation provider (default: ollama)",
    )
    tool_group.add_argument(
        "--image-generation-model",
        default=None,
        help="Image generation model (default: x/z-image-turbo)",
    )
    tool_group.add_argument(
        "--searxng-base-url",
        default=None,
        help="SearXNG base URL for web search (default: http://127.0.0.1:8888)",
    )

    return parser


def main(argv: list[str] = None) -> int:
    """Parse CLI args and start the RuntimeHost."""
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()

    host = RuntimeHost(
        workspace=workspace,
        session_id=args.session_id,
        provider=args.provider,
        mode=args.mode,
        model=args.model,
        image_analysis_provider=args.image_analysis_provider,
        image_analysis_model=args.image_analysis_model,
        image_generation_provider=args.image_generation_provider,
        image_generation_model=args.image_generation_model,
        searxng_base_url=args.searxng_base_url,
    )
    return host.start()


if __name__ == "__main__":
    sys.exit(main())
