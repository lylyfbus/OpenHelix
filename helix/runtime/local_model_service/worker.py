"""Worker process entrypoint for local model service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .adapters import AdapterRegistry
from .constants import FAKE_BACKEND
from .helpers import _error_response


# --------------------------------------------------------------------------- #
# Worker main
# --------------------------------------------------------------------------- #


def _worker_main(args) -> int:
    registry = AdapterRegistry()
    skills_root = str(getattr(args, "skills_root", "") or "").strip()
    if skills_root:
        registry.discover(Path(skills_root))

    service_root = Path(args.service_root).expanduser().resolve()
    python_bin = Path(sys.executable).resolve()
    raw_model_spec = str(getattr(args, "model_spec_json", "") or "").strip()
    model_spec = json.loads(raw_model_spec) if raw_model_spec else None
    model_root = str(getattr(args, "model_root", "") or "").strip()

    if str(args.backend_mode) == FAKE_BACKEND:
        from .fake_backend import FakeBackend
        backend = FakeBackend(
            task_type=str(args.task_type),
            backend=str(args.backend),
            model_id=str(args.model_id),
        )
    else:
        backend = registry.build_backend(
            skill_name=str(args.skill_name),
            task_type=str(args.task_type),
            backend=str(args.backend),
            cache_root=service_root,
            model_id=str(args.model_id),
            python_bin=python_bin,
            model_spec=model_spec,
            model_root=Path(model_root).expanduser().resolve() if model_root else None,
        )

    print(
        json.dumps({
            "status": "ready",
            "backend": str(args.backend),
            "model_id": str(args.model_id),
        }),
        flush=True,
    )

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(
                json.dumps(_error_response(
                    task_type=str(args.task_type),
                    backend=str(args.backend),
                    model_id=str(args.model_id),
                    error_code="invalid_json",
                    message="request must be a JSON object",
                )),
                flush=True,
            )
            continue
        if not isinstance(payload, dict):
            print(
                json.dumps(_error_response(
                    task_type=str(args.task_type),
                    backend=str(args.backend),
                    model_id=str(args.model_id),
                    error_code="invalid_json",
                    message="request must be a JSON object",
                )),
                flush=True,
            )
            continue
        try:
            result = backend.handle(payload)
        except Exception as exc:
            result = _error_response(
                task_type=str(args.task_type),
                backend=str(args.backend),
                model_id=str(args.model_id),
                error_code="worker_runtime_error",
                message=str(exc),
            )
        print(json.dumps(result, ensure_ascii=True), flush=True)

    return 0
