"""Host adapter for the generate-audio skill (Qwen custom-voice TTS)."""

from __future__ import annotations

import contextlib
import shutil
import sys
from pathlib import Path
from typing import Any

from helix.runtime.local_model_service.adapters import _BaseBackend
from helix.runtime.local_model_service.helpers import _ensure_worker_dependencies
from helix.runtime.local_model_service.constants import DEFAULT_AUDIO_SAMPLE_RATE
from helix.runtime.local_model_service.helpers import (
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

_PYTORCH_TEXT_TO_AUDIO_DEPENDENCIES = (
    "huggingface_hub",
    "numpy",
    "qwen-tts",
    "soundfile",
    "torch",
    "transformers",
)


class _MissingHostDependencyError(RuntimeError):
    """Raised when a required host binary is missing."""


class _SpecQwenTTSCustomVoiceBackend(_BaseBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.audio_model = None
        self.soundfile = None
        self.torch = None

    def _load(self) -> None:
        assert self.python_bin is not None
        assert self.model_root is not None
        if shutil.which("sox") is None:
            raise _MissingHostDependencyError(
                "SoX is required for Qwen3-TTS host inference. Install it on the host with `brew install sox`."
            )
        try:
            import torch
            import soundfile as sf
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _PYTORCH_TEXT_TO_AUDIO_DEPENDENCIES)
            import torch
            import soundfile as sf
            from qwen_tts import Qwen3TTSModel

        self.soundfile = sf
        self.torch = torch
        candidate_devices = (
            ["mps", "cpu"]
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else ["cpu"]
        )
        last_error = None
        for device_name in candidate_devices:
            dtype = torch.float16 if device_name == "mps" else torch.float32
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    self.audio_model = Qwen3TTSModel.from_pretrained(
                        str(self.model_root),
                        device_map=device_name,
                        dtype=dtype,
                    )
                return
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"failed to load text-to-audio model {self.model_id}")

    def _ensure_loaded(self) -> None:
        if self.audio_model is None:
            self._load()

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        text = str(inputs.get("text", "")).strip()
        if not text:
            return self._error(error_code="audio_text_missing", message="text is required")
        workspace_root = _resolve_service_workspace_root(payload)
        output_path = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
        )
        try:
            self._ensure_loaded()
            assert self.audio_model is not None
            assert self.soundfile is not None
            language = str(inputs.get("language", "")).strip() or "Auto"
            speaker = str(inputs.get("speaker", "")).strip() or "Vivian"
            instruct = str(inputs.get("instruct", "")).strip()
            if self.torch is not None and inputs.get("seed") not in (None, ""):
                self.torch.manual_seed(int(inputs.get("seed")))
            generation_kwargs = {
                key: value
                for key, value in inputs.items()
                if key not in {"text", "output_path", "language", "speaker", "instruct", "seed"}
                and value not in (None, "")
            }
            call_kwargs: dict[str, Any] = {
                "text": text,
                "language": language,
                "speaker": speaker,
            }
            if instruct:
                call_kwargs["instruct"] = instruct
            call_kwargs.update(generation_kwargs)
            with contextlib.redirect_stdout(sys.stderr):
                wavs, sample_rate = self.audio_model.generate_custom_voice(**call_kwargs)
        except _MissingHostDependencyError as exc:
            return self._error(error_code="missing_host_dependency", message=str(exc))
        except Exception as exc:
            return self._error(error_code="audio_runtime_error", message=str(exc))

        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        sample_rate = int(sample_rate or DEFAULT_AUDIO_SAMPLE_RATE)
        self.soundfile.write(str(output_path), audio, sample_rate)
        rel = str(output_path.relative_to(workspace_root))
        return self._ok(
            outputs={"output_path": rel, "sample_rate": sample_rate},
            message=f"generated audio at {rel}",
        )


def create_adapter(**kwargs):
    return _SpecQwenTTSCustomVoiceBackend(**kwargs)
