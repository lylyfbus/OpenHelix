"""CLI entrypoint for the agentic framework.

Launches the RuntimeHost with provider, mode, model, and workspace
from command-line arguments.

Usage::

    python -m helix.runtime.cli --workspace /path/to/workspace --session-id website-01
    python -m helix.runtime.cli --provider deepseek --mode auto --workspace . --session-id news-site
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
            "  %(prog)s --workspace . --session-id website-01\n"
            "  %(prog)s --provider deepseek --model deepseek-chat --workspace ~/agent --session-id research-01\n"
            "  %(prog)s --provider ollama --mode auto --workspace /tmp/sandbox --session-id sandbox-01\n"
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
        required=True,
        help="Session identifier for loading and persisting conversation state",
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
    )
    return host.start()


if __name__ == "__main__":
    sys.exit(main())
