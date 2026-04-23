"""CLI entrypoint for the agentic framework.

Usage::

    helix start searxng
    helix start local-model-service
    helix stop searxng
    helix status
    helix model download --skill generate-image
    helix --endpoint-url http://localhost:11434/v1 --model llama3.1:8b --workspace ~/agent --session-id project-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .host import RuntimeHost
from helix.services import searxng as searxng_service
from helix.services import local_model_service as lms_service
from helix.runtime.local_model_service.download import download_model


# --------------------------------------------------------------------------- #
# Subcommand: start
# --------------------------------------------------------------------------- #


def _run_start(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Start a helix service")
    parser.add_argument("service", choices=["searxng", "local-model-service"])
    args = parser.parse_args(argv)

    if args.service == "searxng":
        state = searxng_service.start()
        print(f"SearXNG started: {state['base_url']}")
        print(f"PID: {state['pid']}")
        return 0

    if args.service == "local-model-service":
        state = lms_service.start()
        print(f"Local model service started on port {state['port']}")
        return 0
    return 1


# --------------------------------------------------------------------------- #
# Subcommand: stop
# --------------------------------------------------------------------------- #


def _run_stop(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stop a helix service")
    parser.add_argument("service", choices=["searxng", "local-model-service"])
    args = parser.parse_args(argv)

    if args.service == "searxng":
        searxng_service.stop()
        print("SearXNG stopped.")
        return 0
    if args.service == "local-model-service":
        lms_service.stop()
        print("Local model service stopped.")
        return 0
    return 1


# --------------------------------------------------------------------------- #
# Subcommand: status
# --------------------------------------------------------------------------- #


def _run_status() -> int:
    searxng = searxng_service.discover()
    lms = lms_service.discover()

    print("Services:")
    if searxng:
        print(f"  searxng: running ({searxng['base_url']})")
    else:
        print("  searxng: not running")
    if lms:
        print(f"  local-model-service: running (port {lms['port']})")
    else:
        print("  local-model-service: not running")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: model download
# --------------------------------------------------------------------------- #


_PACKAGE_BUILTIN_SKILLS = Path(__file__).resolve().parent.parent / "builtin_skills"


def _find_model_spec(skill_name: str, workspace: str | None) -> Path:
    """Find model_spec.json for a skill by name.

    Searches workspace skills first (if provided), then package builtins.
    """
    candidates: list[Path] = []
    if workspace:
        ws = Path(workspace).expanduser().resolve()
        # User skills: skills/{skill}/
        candidates.append(ws / "skills" / skill_name / "model_spec.json")
        # Bootstrapped builtins: skills/builtin_skills/{skill}/
        candidates.append(ws / "skills" / "builtin_skills" / skill_name / "model_spec.json")
    # Package builtins
    candidates.append(_PACKAGE_BUILTIN_SKILLS / skill_name / "model_spec.json")
    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(p.parent) for p in candidates)
    raise FileNotFoundError(f"No model_spec.json found for skill '{skill_name}'. Searched: {searched}")


def _run_model_download(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Helix model management")
    subparsers = parser.add_subparsers(dest="model_command", required=True)
    download = subparsers.add_parser("download", help="Download model weights for a skill")
    download.add_argument("--skill", required=True, help="Skill name (e.g. generate-image)")
    download.add_argument("--workspace", default=None, help="Workspace path (for user-created skills)")
    download.add_argument("--timeout", type=int, default=3600, help="Download timeout in seconds")
    args = parser.parse_args(argv)
    try:
        spec_path = _find_model_spec(args.skill, args.workspace)
    except FileNotFoundError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 1
    print(f"Using model spec: {spec_path}", file=sys.stderr)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        normalized, model_root = download_model(
            model_spec=payload,
            backend_mode="real",
            timeout_seconds=max(30, int(args.timeout)),
            progress_stream=sys.stderr,
        )
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 1

    repo_id = normalized["source"]["repo_id"]
    print(json.dumps({
        "status": "ok",
        "backend": normalized["backend"],
        "model_root": str(model_root),
        "message": f"prepared model {repo_id}",
    }))
    return 0


# --------------------------------------------------------------------------- #
# Main runtime parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the main runtime."""
    parser = argparse.ArgumentParser(
        description="Agentic System — RL-inspired agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  %(prog)s --endpoint-url https://api.deepseek.com/v1 --api-key $DEEPSEEK_API_KEY --model deepseek-chat --mode auto --workspace ~/agent --session-id research-01\n"
        ),
    )
    parser.add_argument("--endpoint-url", required=True, help="LLM API endpoint URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", default="", help="LLM API key (default: empty)")
    parser.add_argument("--mode", default="controlled", choices=["auto", "controlled"], help="Execution mode")
    parser.add_argument("--workspace", required=True, help="Runtime workspace path")
    parser.add_argument("--session-id", required=True, help="Session identifier")
    return parser


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main(argv: list[str] = None) -> int:
    """Parse CLI args and dispatch to the appropriate handler."""
    argv = list(sys.argv[1:] if argv is None else argv)

    cmd = argv[:1]
    if cmd == ["start"]:
        return _run_start(argv[1:])
    if cmd == ["stop"]:
        return _run_stop(argv[1:])
    if cmd == ["status"]:
        return _run_status()
    if cmd == ["model"]:
        return _run_model_download(argv[1:])

    # Default: start the RuntimeHost
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    host = RuntimeHost(
        workspace=workspace,
        session_id=args.session_id,
        endpoint_url=args.endpoint_url,
        model=args.model,
        api_key=args.api_key,
        mode=args.mode,
    )
    return host.start()


if __name__ == "__main__":
    sys.exit(main())
