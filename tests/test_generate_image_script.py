"""Tests for the generate-image inference handler."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = (
    ROOT
    / "helix"
    / "builtin_skills"
    / "generate-image"
    / "scripts"
    / "generate_image.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_image_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_module()


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self, _size: int = -1) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_generate_image_generation_success(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        def fake_urlopen(req, timeout=0):
            assert timeout == 30
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["model_spec"]["id"] == "builtin.generate-image.mlx-z-image"
            assert payload["model_spec"]["family"] == "mlx.z_image"
            assert payload["request_timeout_seconds"] == 30
            assert payload["inputs"]["prompt"] == "A poster concept"
            assert payload["inputs"]["size"] == "1536x1024"
            assert payload["inputs"]["output_path"] == "generated/poster.png"
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "ok",
                        "task_type": "text_to_image",
                        "backend": "mlx",
                        "model_id": "builtin.generate-image.mlx-z-image",
                        "outputs": {"output_path": "generated/poster.png"},
                        "error_code": "",
                        "message": "generated image at generated/poster.png",
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "prompt": "A poster concept",
                    "size": "1536x1024",
                    "output_path": "generated/poster.png",
                    "output_dir": "",
                    "timeout": 30,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["phase"] == "generate"
        assert out["output_path"] == "generated/poster.png"
        assert out["model_used"] == "uqer1244/MLX-z-image"
