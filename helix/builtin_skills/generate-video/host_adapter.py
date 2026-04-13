"""Host adapter for the generate-video skill (Wan video backend)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from helix.runtime.local_model_service.adapters import _BaseBackend
from helix.runtime.local_model_service.helpers import _ensure_worker_dependencies
from helix.runtime.local_model_service.helpers import (
    _parse_float,
    _parse_int,
    _parse_size,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

_DEPENDENCIES = (
    "accelerate",
    "git+https://github.com/huggingface/diffusers",
    "huggingface_hub",
    "imageio",
    "imageio-ffmpeg",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)


class _WanVideoBackend(_BaseBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipeline = None
        self.device = None
        self.torch = None
        self.export_to_video = None
        self.load_image = None
        self.call_params: set[str] = set()

    def _load(self) -> None:
        assert self.python_bin is not None
        assert self.model_root is not None
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.utils import export_to_video, load_image
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _DEPENDENCIES)
            import torch
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.utils import export_to_video, load_image

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32

        try:
            vae = AutoencoderKLWan.from_pretrained(
                str(self.model_root), subfolder="vae",
                torch_dtype=torch.float32, local_files_only=True,
            )
            self.pipeline = WanPipeline.from_pretrained(
                str(self.model_root), vae=vae,
                torch_dtype=dtype, local_files_only=True,
            )
        except TypeError:
            vae = AutoencoderKLWan.from_pretrained(
                str(self.model_root), subfolder="vae",
                torch_dtype=torch.float32,
            )
            self.pipeline = WanPipeline.from_pretrained(
                str(self.model_root), vae=vae, torch_dtype=dtype,
            )

        self.pipeline.to(device)
        self.device = device
        self.torch = torch
        self.export_to_video = export_to_video
        self.load_image = load_image
        self.call_params = set(inspect.signature(self.pipeline.__call__).parameters)

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="video_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        output_path = _resolve_workspace_path(
            workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False,
        )
        image_path_text = str(inputs.get("image_path", "")).strip()
        if self.task_type == "text_image_to_video" and not image_path_text:
            return self._error(error_code="video_image_missing", message="image_path is required for text_image_to_video")

        width, height = _parse_size(str(inputs.get("size", "")).strip() or "704x512")
        params = {
            "num_frames": _parse_int(inputs.get("num_frames"), default=161, minimum=1),
            "fps": _parse_int(inputs.get("fps"), default=25, minimum=1),
            "num_inference_steps": _parse_int(inputs.get("num_inference_steps"), default=50, minimum=1),
            "guidance_scale": _parse_float(inputs.get("guidance_scale"), default=3.0, minimum=0.0),
            "decode_timestep": _parse_float(inputs.get("decode_timestep"), default=0.03, minimum=0.0),
            "decode_noise_scale": _parse_float(inputs.get("decode_noise_scale"), default=0.025, minimum=0.0),
            "guidance_rescale": _parse_float(inputs.get("guidance_rescale"), default=0.0, minimum=0.0),
            "max_sequence_length": _parse_int(inputs.get("max_sequence_length"), default=128, minimum=1),
            "seed": _parse_int(inputs.get("seed"), default=42, minimum=0),
        }
        negative_prompt = str(inputs.get("negative_prompt", "")).strip()
        fps = params["fps"]

        try:
            if self.pipeline is None:
                self._load()
            assert self.pipeline is not None
            assert self.torch is not None
            assert self.export_to_video is not None

            # Build call kwargs — only pass params the pipeline accepts
            call_kwargs: dict[str, object] = {"prompt": prompt, "width": width, "height": height}
            for key in ("num_frames", "num_inference_steps", "guidance_scale", "decode_timestep",
                        "decode_noise_scale", "guidance_rescale", "max_sequence_length"):
                if key in self.call_params:
                    call_kwargs[key] = params[key]
            if "frame_rate" in self.call_params:
                call_kwargs["frame_rate"] = fps
            if negative_prompt and "negative_prompt" in self.call_params:
                call_kwargs["negative_prompt"] = negative_prompt
            if "generator" in self.call_params:
                call_kwargs["generator"] = self.torch.manual_seed(params["seed"])
            if image_path_text:
                if "image" not in self.call_params:
                    return self._error(error_code="video_conditioning_unsupported",
                                       message="this video model does not accept image conditioning")
                assert self.load_image is not None
                call_kwargs["image"] = self.load_image(str(
                    _resolve_workspace_path(workspace_root, image_path_text, expect_exists=True)
                ))

            result = self.pipeline(**call_kwargs)
            frames = getattr(result, "frames", None)
            if isinstance(frames, list) and frames:
                clip_frames = frames[0] if isinstance(frames[0], list) else frames
            else:
                raise RuntimeError("video pipeline did not return frames")
            self.export_to_video(clip_frames, str(output_path), fps=fps)
        except Exception as exc:
            return self._error(error_code="video_runtime_error", message=str(exc))

        rel = str(output_path.relative_to(workspace_root))
        return self._ok(
            outputs={"output_path": rel, "fps": fps, "num_frames": len(clip_frames)},
            message=f"generated video at {rel}",
        )


def create_adapter(**kwargs):
    return _WanVideoBackend(**kwargs)
