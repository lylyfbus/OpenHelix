"""Host adapter for the generate-video skill (LTX and Wan video families)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from helix.runtime.local_model_service.registry import _BaseBackend
from helix.runtime.local_model_service.paths import _ensure_worker_dependencies
from helix.runtime.local_model_service.protocol import (
    _parse_float,
    _parse_int,
    _parse_size,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

FAMILY = "pytorch.diffusers_ltx_video"
BACKEND = "pytorch"
TASK_TYPES = ["text_to_video", "text_image_to_video"]

_PYTORCH_VIDEO_DEPENDENCIES = (
    "accelerate",
    "git+https://github.com/huggingface/diffusers",
    "huggingface_hub",
    "imageio",
    "imageio-ffmpeg",
    "numpy",
    "pillow",
    "protobuf",
    "safetensors",
    "sentencepiece",
    "torch",
    "transformers",
)


def _ensure_ltx_tokenizer_dependencies(python_bin: Path) -> None:
    try:
        import sentencepiece  # noqa: F401
        import google.protobuf  # noqa: F401
    except ImportError:
        _ensure_worker_dependencies(python_bin, ("sentencepiece", "protobuf"))


class _SpecPyTorchVideoBackend(_BaseBackend):
    def __init__(
        self,
        *,
        task_type: str,
        backend: str,
        model_id: str,
        model_spec: dict[str, Any],
        model_root: Path,
        cache_root: Path,
        python_bin: Path,
    ) -> None:
        super().__init__(
            task_type=task_type,
            backend=backend,
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
            model_spec=model_spec,
            model_root=model_root,
        )
        self.pipeline = None
        self.device = None
        self.torch = None
        self.export_to_video = None
        self.load_image = None
        self.call_params: set[str] = set()

    def _load_pipeline(self, torch, dtype):  # pragma: no cover - subclass hook
        raise NotImplementedError

    def _load(self) -> None:
        assert self.python_bin is not None
        try:
            import torch
            from diffusers.utils import export_to_video, load_image
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _PYTORCH_VIDEO_DEPENDENCIES)
            import torch
            from diffusers.utils import export_to_video, load_image

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        self.pipeline = self._load_pipeline(torch, dtype)
        self.pipeline.to(device)
        self.device = device
        self.torch = torch
        self.export_to_video = export_to_video
        self.load_image = load_image
        self.call_params = set(inspect.signature(self.pipeline.__call__).parameters)

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            self._load()

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="video_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        output_path = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
        )
        image_path_text = str(inputs.get("image_path", "")).strip()
        if self.task_type == "text_image_to_video" and not image_path_text:
            return self._error(
                error_code="video_image_missing",
                message="image_path is required for text_image_to_video",
            )

        size_text = str(inputs.get("size", "")).strip() or "704x512"
        width, height = _parse_size(size_text)
        num_frames = _parse_int(inputs.get("num_frames"), default=161, minimum=1)
        fps = _parse_int(inputs.get("fps"), default=25, minimum=1)
        num_inference_steps = _parse_int(inputs.get("num_inference_steps"), default=50, minimum=1)
        guidance_scale = _parse_float(inputs.get("guidance_scale"), default=3.0, minimum=0.0)
        decode_timestep = _parse_float(inputs.get("decode_timestep"), default=0.03, minimum=0.0)
        decode_noise_scale = _parse_float(inputs.get("decode_noise_scale"), default=0.025, minimum=0.0)
        guidance_rescale = _parse_float(inputs.get("guidance_rescale"), default=0.0, minimum=0.0)
        max_sequence_length = _parse_int(inputs.get("max_sequence_length"), default=128, minimum=1)
        negative_prompt = str(inputs.get("negative_prompt", "")).strip()
        seed = _parse_int(inputs.get("seed"), default=42, minimum=0)

        try:
            self._ensure_loaded()
            assert self.pipeline is not None
            assert self.torch is not None
            assert self.export_to_video is not None
            call_kwargs: dict[str, object] = {}
            if "prompt" in self.call_params:
                call_kwargs["prompt"] = prompt
            if "width" in self.call_params:
                call_kwargs["width"] = width
            if "height" in self.call_params:
                call_kwargs["height"] = height
            if "num_frames" in self.call_params:
                call_kwargs["num_frames"] = num_frames
            if "num_inference_steps" in self.call_params:
                call_kwargs["num_inference_steps"] = num_inference_steps
            if "frame_rate" in self.call_params:
                call_kwargs["frame_rate"] = fps
            if "guidance_scale" in self.call_params:
                call_kwargs["guidance_scale"] = guidance_scale
            if "decode_timestep" in self.call_params:
                call_kwargs["decode_timestep"] = decode_timestep
            if "decode_noise_scale" in self.call_params:
                call_kwargs["decode_noise_scale"] = decode_noise_scale
            if "guidance_rescale" in self.call_params:
                call_kwargs["guidance_rescale"] = guidance_rescale
            if "max_sequence_length" in self.call_params:
                call_kwargs["max_sequence_length"] = max_sequence_length
            if negative_prompt and "negative_prompt" in self.call_params:
                call_kwargs["negative_prompt"] = negative_prompt
            if "generator" in self.call_params:
                call_kwargs["generator"] = self.torch.manual_seed(seed)
            if image_path_text:
                if "image" not in self.call_params:
                    return self._error(
                        error_code="video_conditioning_unsupported",
                        message="this video model does not accept image conditioning",
                    )
                assert self.load_image is not None
                image_path = _resolve_workspace_path(
                    workspace_root,
                    image_path_text,
                    expect_exists=True,
                )
                call_kwargs["image"] = self.load_image(str(image_path))
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


class _SpecLTXVideoBackend(_SpecPyTorchVideoBackend):
    def _load_pipeline(self, torch, dtype):
        assert self.model_root is not None
        assert self.model_spec is not None
        assert self.python_bin is not None
        _ensure_ltx_tokenizer_dependencies(self.python_bin)
        from diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline, LTXPipeline
        from transformers import T5EncoderModel, T5TokenizerFast

        pipeline_cls = LTXImageToVideoPipeline if self.task_type == "text_image_to_video" else LTXPipeline
        checkpoint_glob = str(
            self.model_spec.get("load_config", {}).get("checkpoint_glob", "") or "**/*.safetensors"
        ).strip()
        checkpoint_matches = sorted(self.model_root.glob(checkpoint_glob))
        if not checkpoint_matches:
            raise RuntimeError(f"missing prepared checkpoint for {self.model_id}")
        checkpoint_path = checkpoint_matches[0]
        text_encoder = T5EncoderModel.from_pretrained(
            str(self.model_root),
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=True,
        )
        tokenizer = T5TokenizerFast.from_pretrained(
            str(self.model_root),
            subfolder="tokenizer",
            local_files_only=True,
        )
        vae = AutoencoderKLLTXVideo.from_pretrained(
            str(self.model_root),
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
        try:
            return pipeline_cls.from_single_file(
                str(checkpoint_path),
                config=str(self.model_root),
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                torch_dtype=dtype,
                local_files_only=True,
            )
        except TypeError:
            return pipeline_cls.from_single_file(
                str(checkpoint_path),
                config=str(self.model_root),
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                torch_dtype=dtype,
            )


class _SpecWanVideoBackend(_SpecPyTorchVideoBackend):
    def _load_pipeline(self, torch, dtype):
        assert self.model_root is not None
        from diffusers import AutoencoderKLWan, WanPipeline

        try:
            vae = AutoencoderKLWan.from_pretrained(
                str(self.model_root),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            return WanPipeline.from_pretrained(
                str(self.model_root),
                vae=vae,
                torch_dtype=dtype,
                local_files_only=True,
            )
        except TypeError:
            vae = AutoencoderKLWan.from_pretrained(
                str(self.model_root),
                subfolder="vae",
                torch_dtype=torch.float32,
            )
            return WanPipeline.from_pretrained(str(self.model_root), vae=vae, torch_dtype=dtype)


def create_adapter(*, task_type, backend, model_id, cache_root, python_bin, model_spec, model_root):
    return _SpecLTXVideoBackend(
        task_type=task_type,
        backend=backend,
        model_id=model_id,
        model_spec=model_spec,
        model_root=model_root,
        cache_root=cache_root,
        python_bin=python_bin,
    )


# Second family: Wan Video (same skill, different model architecture)
FAMILY_WAN = "pytorch.diffusers_wan_video"
BACKEND_WAN = "pytorch"
TASK_TYPES_WAN = ["text_to_video", "text_image_to_video"]


def create_wan_adapter(*, task_type, backend, model_id, cache_root, python_bin, model_spec, model_root):
    return _SpecWanVideoBackend(
        task_type=task_type,
        backend=backend,
        model_id=model_id,
        model_spec=model_spec,
        model_root=model_root,
        cache_root=cache_root,
        python_bin=python_bin,
    )
