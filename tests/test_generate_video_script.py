"""Tests for the generate-video inference handler."""

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
    / "generate-video"
    / "scripts"
    / "generate_video.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_video_script", SCRIPT_PATH)
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


def test_generate_video_text_only_success(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        def fake_urlopen(req, timeout=0):
            assert timeout == 1200
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["task_type"] == "text_to_video"
            assert payload["model_spec"]["id"] == "builtin.generate-video.ltx-video"
            assert payload["model_spec"]["family"] == "pytorch.diffusers_ltx_video"
            assert payload["request_timeout_seconds"] == 1200
            assert payload["inputs"]["prompt"] == "A cinematic beach scene"
            assert payload["inputs"]["size"] == "704x512"
            assert payload["inputs"]["output_path"] == "generated/video.mp4"
            assert payload["inputs"]["guidance_rescale"] == 0.0
            assert payload["inputs"]["decode_timestep"] == 0.03
            assert payload["inputs"]["decode_noise_scale"] == 0.025
            assert payload["inputs"]["max_sequence_length"] == 128
            assert "image_path" not in payload["inputs"]
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "ok",
                        "task_type": "text_to_video",
                        "backend": "pytorch",
                        "model_id": "builtin.generate-video.ltx-video",
                        "outputs": {
                            "output_path": "generated/video.mp4",
                            "fps": 25,
                            "num_frames": 161,
                        },
                        "error_code": "",
                        "message": "generated video at generated/video.mp4",
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "prompt": "A cinematic beach scene",
                    "image_path": "",
                    "size": "704x512",
                    "negative_prompt": "",
                    "num_frames": 161,
                    "fps": 25,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.0,
                    "guidance_rescale": 0.0,
                    "decode_timestep": 0.03,
                    "decode_noise_scale": 0.025,
                    "max_sequence_length": 128,
                    "seed": 42,
                    "output_path": "generated/video.mp4",
                    "output_dir": "",
                    "timeout": 1200,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["phase"] == "generate"
        assert out["task_type"] == "text_to_video"
        assert out["output_path"] == "generated/video.mp4"
        assert out["fps"] == 25
        assert out["num_frames"] == 161
        assert out["model_used"] == "Lightricks/LTX-Video"


def test_generate_video_image_conditioned_success(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        image_path = workspace / "assets" / "frame.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"png")
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        def fake_urlopen(req, timeout=0):
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["task_type"] == "text_image_to_video"
            assert payload["model_spec"]["id"] == "builtin.generate-video.ltx-video"
            assert payload["request_timeout_seconds"] == 1200
            assert payload["inputs"]["image_path"] == "assets/frame.png"
            assert payload["inputs"]["decode_timestep"] == 0.03
            assert payload["inputs"]["decode_noise_scale"] == 0.025
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "ok",
                        "task_type": "text_image_to_video",
                        "backend": "pytorch",
                        "model_id": "builtin.generate-video.ltx-video",
                        "outputs": {
                            "output_path": "generated/conditioned.mp4",
                            "fps": 25,
                            "num_frames": 121,
                        },
                        "error_code": "",
                        "message": "generated video at generated/conditioned.mp4",
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "prompt": "Animate this frame gently.",
                    "image_path": "assets/frame.png",
                    "size": "704x512",
                    "negative_prompt": "",
                    "num_frames": 121,
                    "fps": 25,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.0,
                    "guidance_rescale": 0.0,
                    "decode_timestep": 0.03,
                    "decode_noise_scale": 0.025,
                    "max_sequence_length": 128,
                    "seed": 42,
                    "output_path": "generated/conditioned.mp4",
                    "output_dir": "",
                    "timeout": 1200,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["task_type"] == "text_image_to_video"
        assert out["image_path"] == "assets/frame.png"
        assert out["output_path"] == "generated/conditioned.mp4"


def test_generate_video_passes_custom_ltx_parameters(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        def fake_urlopen(req, timeout=0):
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["model_spec"]["id"] == "builtin.generate-video.ltx-video"
            assert payload["inputs"]["guidance_scale"] == 3.5
            assert payload["inputs"]["guidance_rescale"] == 0.4
            assert payload["inputs"]["decode_timestep"] == 0.04
            assert payload["inputs"]["decode_noise_scale"] == 0.03
            assert payload["inputs"]["max_sequence_length"] == 192
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "ok",
                        "task_type": "text_to_video",
                        "backend": "pytorch",
                        "model_id": "builtin.generate-video.ltx-video",
                        "outputs": {
                            "output_path": "generated/custom.mp4",
                            "fps": 25,
                            "num_frames": 161,
                        },
                        "error_code": "",
                        "message": "generated video at generated/custom.mp4",
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "prompt": "A slow cinematic dolly around a modern living room.",
                    "image_path": "",
                    "size": "704x512",
                    "negative_prompt": "",
                    "num_frames": 161,
                    "fps": 25,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "guidance_rescale": 0.4,
                    "decode_timestep": 0.04,
                    "decode_noise_scale": 0.03,
                    "max_sequence_length": 192,
                    "seed": 42,
                    "output_path": "generated/custom.mp4",
                    "output_dir": "",
                    "timeout": 1200,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["output_path"] == "generated/custom.mp4"
