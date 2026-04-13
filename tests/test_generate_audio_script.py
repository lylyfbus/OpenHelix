"""Tests for the generate-audio inference handler."""

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
    / "generate-audio"
    / "scripts"
    / "generate_audio.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_audio_script", SCRIPT_PATH)
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


def test_generate_audio_success(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        def fake_urlopen(req, timeout=0):
            assert timeout == 1200
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["task_type"] == "text_to_audio"
            assert payload["model_spec"]["id"] == "builtin.generate-audio.qwen3-tts-customvoice"
            assert payload["model_spec"]["family"] == "pytorch.qwen_tts_custom_voice"
            assert payload["request_timeout_seconds"] == 1200
            assert payload["inputs"]["text"] == "Welcome home."
            assert payload["inputs"]["language"] == "English"
            assert payload["inputs"]["speaker"] == "Ryan"
            assert payload["inputs"]["instruct"] == "Speak warmly."
            assert payload["inputs"]["do_sample"] is True
            assert payload["inputs"]["top_k"] == 64
            assert payload["inputs"]["top_p"] == 0.95
            assert payload["inputs"]["temperature"] == 0.8
            assert payload["inputs"]["repetition_penalty"] == 1.1
            assert payload["inputs"]["max_new_tokens"] == 2048
            assert payload["inputs"]["non_streaming_mode"] is True
            assert payload["inputs"]["seed"] == 7
            assert payload["inputs"]["output_path"] == "generated_audio/welcome.wav"
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "ok",
                        "task_type": "text_to_audio",
                        "backend": "pytorch",
                        "model_id": "builtin.generate-audio.qwen3-tts-customvoice",
                        "outputs": {
                            "output_path": "generated_audio/welcome.wav",
                            "sample_rate": 24000,
                        },
                        "error_code": "",
                        "message": "generated audio at generated_audio/welcome.wav",
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "text": "Welcome home.",
                    "language": "English",
                    "speaker": "Ryan",
                    "instruct": "Speak warmly.",
                    "do_sample": "true",
                    "top_k": 64,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repetition_penalty": 1.1,
                    "max_new_tokens": 2048,
                    "non_streaming_mode": "true",
                    "seed": 7,
                    "output_path": "generated_audio/welcome.wav",
                    "output_dir": "",
                    "timeout": 1200,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["phase"] == "generate"
        assert out["output_path"] == "generated_audio/welcome.wav"
        assert out["sample_rate"] == 24000
        assert out["model_used"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def test_generate_audio_invalid_top_p(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "text": "Welcome home.",
                    "language": "English",
                    "speaker": "Ryan",
                    "instruct": "",
                    "do_sample": "true",
                    "top_k": 50,
                    "top_p": 1.5,
                    "temperature": 0.9,
                    "repetition_penalty": 1.05,
                    "max_new_tokens": 4096,
                    "non_streaming_mode": "true",
                    "seed": 42,
                    "output_path": "generated_audio/welcome.wav",
                    "output_dir": "",
                    "timeout": 1200,
                },
            )()
        )

        assert code == 1
        assert out["status"] == "error"
        assert out["error_code"] == "audio_parameter_invalid"
        assert "top_p" in out["message"]
