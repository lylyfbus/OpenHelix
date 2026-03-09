"""Python module entrypoint (`python -m agentic_system`)."""

from .runtime.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
