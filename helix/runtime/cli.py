"""CLI entrypoint for the agentic framework.

Launches the RuntimeHost with LLM provider settings, mode, and workspace
from command-line arguments.

Usage::

    python -m helix.runtime.cli --workspace /path/to/workspace --session-id website-01
    python -m helix.runtime.cli --base-url https://api.deepseek.com/v1 --api-key $KEY --workspace . --session-id news-site
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .host import RuntimeHost
from .local_model_service.paths import default_cache_root
from .local_model_service.preparer import ModelPreparationError, prepare_model_spec


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Agentic System — RL-inspired agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --workspace . --session-id website-01\n"
            "  %(prog)s --base-url https://api.deepseek.com/v1 --api-key $DEEPSEEK_API_KEY --workspace ~/agent --session-id research-01\n"
            "  %(prog)s --model llama3.1:8b --mode auto --workspace /tmp/sandbox --session-id sandbox-01\n"
        ),
    )

    # LLM provider settings
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM API base URL (default: http://localhost:11434/v1 — Ollama). Env: LLM_BASE_URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key (default: none). Env: LLM_API_KEY",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: llama3.1:8b). Env: LLM_MODEL",
    )

    # Runtime settings
    parser.add_argument(
        "--mode",
        default="controlled",
        choices=["auto", "controlled"],
        help="Execution mode: auto (no confirmation) or controlled (default: controlled)",
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


def _build_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helix model management")
    subparsers = parser.add_subparsers(dest="model_command", required=True)

    download = subparsers.add_parser("download", help="Download and validate a local model spec")
    download.add_argument("--spec", required=True, help="Absolute path to a model_spec.json file")
    download.add_argument("--timeout", type=int, default=3600, help="Preparation timeout in seconds")
    return parser


def _run_model_command(argv: list[str]) -> int:
    parser = _build_model_parser()
    args = parser.parse_args(argv)
    if args.model_command != "download":
        raise ValueError(f"unsupported model command: {args.model_command}")
    spec_path = Path(args.spec).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        normalized, model_root = prepare_model_spec(
            cache_root=default_cache_root(),
            model_spec=payload,
            backend_mode="real",
            timeout_seconds=max(30, int(args.timeout)),
            progress_stream=sys.stderr,
        )
    except ModelPreparationError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "backend": "",
                    "task_type": "",
                    "model_root": "",
                    "error_code": exc.error_code,
                    "message": exc.message,
                },
                ensure_ascii=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "backend": normalized["backend"],
                "task_type": normalized["task_type"],
                "model_root": str(model_root),
                "error_code": "",
                "message": f"prepared model {normalized['id']}",
            },
            ensure_ascii=True,
        )
    )
    return 0


def main(argv: list[str] = None) -> int:
    """Parse CLI args and start the RuntimeHost or model-management command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["model"]:
        return _run_model_command(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()

    host = RuntimeHost(
        workspace=workspace,
        session_id=args.session_id,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        mode=args.mode,
    )
    return host.start()


if __name__ == "__main__":
    sys.exit(main())
