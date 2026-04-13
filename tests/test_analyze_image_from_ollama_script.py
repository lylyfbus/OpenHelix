"""Tests for the analyze-image skill handler."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = (
    ROOT
    / "helix"
    / "builtin_skills"
    / "analyze-image"
    / "scripts"
    / "analyze_image.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_image_script", SCRIPT_PATH)
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


def test_analyze_image_local_path_success(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example:11434")
        image_path = workspace / "assets" / "sample.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")

        def fake_urlopen(req, timeout=0):
            assert timeout == 45
            assert req.full_url == "http://ollama.example:11434/api/generate"
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["model"] == "glm-ocr"
            assert payload["prompt"] == "Extract the visible title"
            assert isinstance(payload["images"], list) and len(payload["images"]) == 1
            return _FakeResponse(json.dumps({"response": "Visible title: Helix"}).encode("utf-8"))

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "image_path": "assets/sample.png",
                    "image_url": "",
                    "query": "Extract the visible title",
                    "timeout": 45,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["executed_skill"] == "analyze-image"
        assert out["analysis"] == "Visible title: Helix"
        assert out["model_used"] == "glm-ocr"


def test_analyze_image_downloads_remote_image(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example:11434")

        def fake_urlopen(req, timeout=0):
            if req.full_url == "https://example.com/image.png":
                return _FakeResponse(b"remote-image-bytes")
            assert req.full_url == "http://ollama.example:11434/api/generate"
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["images"]
            return _FakeResponse(json.dumps({"response": "Remote image analyzed"}).encode("utf-8"))

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "image_path": "",
                    "image_url": "https://example.com/image.png",
                    "query": "Describe the image",
                    "timeout": 30,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        downloads = list((workspace / ".runtime" / "ollama-image-analysis" / "downloads").glob("download_*"))
        assert downloads, "expected downloaded image in workspace"


def test_analyze_image_reports_unavailable_service(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        image_path = workspace / "sample.png"
        image_path.write_bytes(b"fake-image")

        def fake_urlopen(req, timeout=0):
            raise URLError("connection refused")

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "image_path": "sample.png",
                    "image_url": "",
                    "query": "Describe the image",
                    "timeout": 15,
                },
            )()
        )

        assert code == 1
        assert out["status"] == "error"
        assert out["error_code"] == "ollama_unavailable"
        assert "ollama serve" in out["message"]


def test_analyze_image_rejects_empty_response(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        image_path = workspace / "sample.png"
        image_path.write_bytes(b"fake-image")

        def fake_urlopen(req, timeout=0):
            return _FakeResponse(json.dumps({"response": ""}).encode("utf-8"))

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "image_path": "sample.png",
                    "image_url": "",
                    "query": "Describe the image",
                    "timeout": 15,
                },
            )()
        )

        assert code == 1
        assert out["status"] == "error"
        assert out["error_code"] == "image_analysis_failed"


if __name__ == "__main__":
    test_analyze_image_local_path_success()  # type: ignore[misc]
    test_analyze_image_downloads_remote_image()  # type: ignore[misc]
    test_analyze_image_reports_unavailable_service()  # type: ignore[misc]
    test_analyze_image_rejects_empty_response()  # type: ignore[misc]
    print("\n✅ All analyze-image script tests passed!")
