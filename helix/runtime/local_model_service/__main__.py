"""CLI entrypoint for local model service."""

from __future__ import annotations

import argparse

from .server import _coordinator_main
from .constants import DEFAULT_BACKEND_MODE, DEFAULT_IDLE_SECONDS
from .worker import _worker_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helix local model service")
    subparsers = parser.add_subparsers(dest="role", required=True)

    coordinator = subparsers.add_parser("coordinator")
    coordinator.add_argument("--service-root", required=True)
    coordinator.add_argument("--host", required=True)
    coordinator.add_argument("--port", required=True)
    coordinator.add_argument("--token", required=True)
    coordinator.add_argument("--idle-seconds", default=str(DEFAULT_IDLE_SECONDS))
    coordinator.add_argument("--backend-mode", default=DEFAULT_BACKEND_MODE)
    coordinator.add_argument("--skills-root", default="")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--skill-name", required=True)
    worker.add_argument("--service-root", required=True)
    worker.add_argument("--task-type", required=True)
    worker.add_argument("--backend", required=True)
    worker.add_argument("--model-id", required=True)
    worker.add_argument("--model-spec-json", default="")
    worker.add_argument("--model-root", default="")
    worker.add_argument("--backend-mode", default=DEFAULT_BACKEND_MODE)
    worker.add_argument("--skills-root", default="")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.role == "coordinator":
        return _coordinator_main(args)
    if args.role == "worker":
        return _worker_main(args)
    raise ValueError(f"unsupported role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
