---
name: Generate Video
description: Prepare the built-in video model, then generate a video from text alone or from text plus an input image with the host-native local model service using the built-in PyTorch LTX-Video backend.
---

# Purpose

Use this skill when you need to prepare the built-in local video model and then generate a video artifact from a text prompt, optionally conditioned on a workspace-local image.

# When To Use

Use when:
- the user wants a new video generated from a text prompt
- the user wants image-conditioned video generation from a text prompt plus a workspace-local image
- the task should stay inside the built-in local video capability path
- the resulting video should be saved into the runtime workspace for later reuse

Skip when:
- the task is video analysis rather than generation
- the user explicitly wants a different backend or service not provided by this skill
- the task is better handled by a remote API video service rather than a local open-source model

# Supported LTX Surface

This skill currently supports these LTX-Video generation paths:

- `text_to_video`
  - one text prompt
  - one generated output video
- `text_image_to_video`
  - one text prompt
  - one workspace-local input image used as the first-frame condition
  - one generated output video

This skill does **not** currently expose the wider upstream LTX surface, including:

- multi-image or multi-video conditioning
- long-duration sliding-window pipelines such as `LTXI2VLongMultiPromptPipeline`
- prompt segment scheduling or per-window prompt changes
- video-to-video editing
- LoRA loading
- latent upsampling or multi-stage upscale flows
- GGUF checkpoints
- direct low-level scheduler/timestep arrays

Core Agent should stay inside the supported surface above and should not invent extra LTX arguments that this skill does not define.

# Skill Mode

- `script_mode: multi`
- Preferred for video generation because the agent should reason between preparation and inference.
- Default phase scripts:
  - `scripts/prepare_model.py`: prepare/download/warm the built-in video model before generation
  - `scripts/generate_video.py`: run the actual video inference and save the output artifact
- Core Agent should call this skill directly instead of passing provider/model config through Helix CLI.
- Default handler path: `skills/all-agents/generate-video/scripts/prepare_model.py`

# Procedure

1. Gather context:
   - confirm the exact prompt and desired scene
   - determine whether generation is text-only or image-conditioned
   - if image-conditioned, confirm the input image is already inside the workspace
2. Prepare the model first:
   - run `scripts/prepare_model.py`
   - treat this as the model download / warmup phase
   - wait for `status=ok` before moving on to inference
   - this step is idempotent; it is safe even if the model is already cached
   - `--timeout` controls both the script HTTP wait and the local model service request budget for this call
   - if the job may run long, set exec `timeout_seconds` larger than `--timeout`
3. Plan output:
   - choose one workspace-local target using `--output-path` or `--output-dir`
   - keep generated artifacts in deterministic, reusable locations
4. Infer:
   - run `scripts/generate_video.py` with `--prompt` and one output target option
   - pass `--image-path` only when image-conditioned video is actually required
   - pass generation knobs only when they materially improve the requested output
   - let the skill choose its built-in backend/model; do not add provider/model args
5. Verify:
   - inspect runtime stdout for `status`, `task_type`, `output_path`, `fps`, `num_frames`, and any service/runtime error details
   - confirm the generated video file exists in the workspace
6. Report:
   - return the resulting artifact path and any important generation constraints or failures

# Runtime Contract

All scripts in this skill must:
1. print one final JSON object to stdout
2. use stderr only for unexpected runtime failures
3. keep stdout concise but informative so workflow history remains readable

The next reasoning step should inspect runtime stdout/stderr before deciding the next phase.

# Action Input Templates

## Phase 1: Prepare Model

```json
{
  "code_type": "python",
  "timeout_seconds": 1800,
  "script_path": "skills/all-agents/generate-video/scripts/prepare_model.py",
  "script_args": [
    "--timeout", "1200"
  ]
}
```

## Phase 2: Generate Video

- required:
  - `--prompt`
  - one of:
    - `--output-dir`
    - `--output-path`

Recommended argument rules:
- use workspace-local paths only
- pass `--image-path` only when you want text-image-to-video rather than text-to-video
- pass only one `--image-path`; this skill does not support multiple conditions
- if `--output-dir` is used, let the script choose a deterministic file name under that directory
- keep `--size`, `--fps`, `--num-frames`, and `--num-inference-steps` close to defaults unless the user requests a specific output shape
- prefer a known safe preset over inventing unusual video dimensions
- `--timeout` controls both the script HTTP wait and the local model service request budget for this call
- when generation may run long, set exec `timeout_seconds` larger than `--timeout`
- do not pass provider, model, or API settings; the skill owns its backend and model choice

### LTX Prompt Guidance

- For text-to-video:
  - describe the subject, motion, camera movement, lighting, and mood in 1-4 focused sentences
  - avoid cramming multiple unrelated scene changes into one clip
- For text-image-to-video:
  - treat the input image as frame 0
  - describe how that scene should animate rather than describing a completely different scene
- Use `--negative-prompt` only when you need to suppress a specific artifact or unwanted style.

### Parameter Reference

- `--prompt`
  - required
  - best practice: 1-4 sentences focused on subject, motion, camera, lighting, and mood
- `--image-path`
  - optional
  - only use when the user wants image-conditioned video
  - must point to a single workspace-local image
- `--size`
  - default: `704x512`
  - recommended safe presets:
    - landscape: `704x512`
    - portrait: `512x704`
    - larger preview: `960x544`
  - best practice: keep dimensions divisible by `32`
- `--num-frames`
  - default: `161`
  - recommended working range: `81-161`
  - practical duration at `25 fps`:
    - `81` frames: about `3.2s`
    - `121` frames: about `4.8s`
    - `161` frames: about `6.4s`
- `--fps`
  - default: `25`
  - recommended working range: `16-25`
  - use `16` for quicker previews and `25` for default LTX-Video motion
- `--num-inference-steps`
  - default: `50`
  - recommended working range: `30-50`
  - lower values are faster; higher values are usually cleaner
- `--guidance-scale`
  - default: `3.0`
  - recommended working range: `2.0-4.0`
  - higher values follow the prompt more strictly but can over-constrain motion
- `--guidance-rescale`
  - default: `0.0`
  - recommended working range: `0.0-0.7`
  - keep at `0.0` unless high guidance is producing overexposed or overly rigid results
- `--decode-timestep`
  - default: `0.03`
  - recommended working range: `0.0-0.05`
  - keep the default unless you have a concrete LTX quality reason to change it
- `--decode-noise-scale`
  - default: `0.025`
  - recommended working range: `0.0-0.05`
  - pairs with `--decode-timestep`; leave it alone unless you are deliberately tuning decode behavior
- `--max-sequence-length`
  - default: `128`
  - best practice: keep at `128` unless the prompt is clearly being truncated
  - larger values can increase memory pressure
- `--negative-prompt`
  - default: empty
  - use only when artifacts or unwanted styles need correction
- `--seed`
  - default: `42`
  - keep fixed for reproducibility; change only when you want a different take

### Best-Practice Defaults

- Quick preview
  - `--size 704x512 --num-frames 81 --fps 16 --num-inference-steps 30 --guidance-scale 3.0 --guidance-rescale 0.0 --decode-timestep 0.03 --decode-noise-scale 0.025 --seed 42`
- Standard final clip
  - `--size 704x512 --num-frames 161 --fps 25 --num-inference-steps 50 --guidance-scale 3.0 --guidance-rescale 0.0 --decode-timestep 0.03 --decode-noise-scale 0.025 --max-sequence-length 128 --seed 42`
- Image-conditioned architectural reveal
  - `--image-path <workspace image> --size 704x512 --num-frames 121 --fps 25 --num-inference-steps 40 --guidance-scale 3.0 --guidance-rescale 0.0 --decode-timestep 0.03 --decode-noise-scale 0.025 --seed 42`

Example text-to-video:

```json
{
  "code_type": "python",
  "timeout_seconds": 1800,
  "script_path": "skills/all-agents/generate-video/scripts/generate_video.py",
  "script_args": [
    "--prompt", "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "--size", "704x512",
    "--num-frames", "161",
    "--fps", "25",
    "--num-inference-steps", "50",
    "--guidance-scale", "3.0",
    "--guidance-rescale", "0.0",
    "--decode-timestep", "0.03",
    "--decode-noise-scale", "0.025",
    "--max-sequence-length", "128",
    "--seed", "42",
    "--output-dir", "generated_videos/cats-boxing"
  ]
}
```

Example text-image-to-video:

```json
{
  "code_type": "python",
  "timeout_seconds": 1800,
  "script_path": "skills/all-agents/generate-video/scripts/generate_video.py",
  "script_args": [
    "--prompt", "A calm cinematic aerial reveal of this house at sunset.",
    "--image-path", "sessions/test_1/project/reference-house.png",
    "--size", "704x512",
    "--num-frames", "121",
    "--fps", "25",
    "--num-inference-steps", "40",
    "--guidance-scale", "3.0",
    "--guidance-rescale", "0.0",
    "--decode-timestep", "0.03",
    "--decode-noise-scale", "0.025",
    "--max-sequence-length", "128",
    "--seed", "42",
    "--output-path", "sessions/test_1/project/generated-house-reveal.mp4"
  ]
}
```

# Output JSON Shape

## Prepare Phase

```json
{
  "executed_skill": "generate-video",
  "phase": "prepare",
  "status": "ok|error",
  "model_used": "...",
  "error_code": "...",
  "message": "..."
}
```

## Generate Phase

```json
{
  "executed_skill": "generate-video",
  "phase": "generate",
  "status": "ok|error",
  "task_type": "text_to_video|text_image_to_video",
  "prompt": "...",
  "image_path": "...",
  "output_path": "...",
  "fps": 24,
  "num_frames": 121,
  "model_used": "...",
  "error_code": "...",
  "message": "..."
}
```

# Error Handling Rule

1. If the local model service variables are missing, stop internal retries and return control to requester or runtime with the configuration failure.
2. If model preparation fails, do not continue to inference until preparation succeeds.
3. If output path validation fails, do not retry with the same invalid path; choose a new workspace-local path first.
4. If image-conditioned generation is requested, do not proceed without a valid workspace-local `--image-path`.
5. If generation fails due to service/model runtime issues, surface the exact `error_code` and `message` rather than masking them.

# Skill Dependencies

- (none)

This skill is self-contained for local video generation and does not require another skill for normal execution.

# Notes

- This skill calls the runtime-managed local inference host through `HELIX_LOCAL_MODEL_SERVICE_URL`.
- The skill ships a `model_spec.json` next to `SKILL.md` and sends that full spec payload to the host.
- `scripts/prepare_model.py` calls `/models/prepare`; `scripts/generate_video.py` calls `/infer`.
- The built-in backend is `pytorch`.
- The built-in model repo is `Lightricks/LTX-Video`.
- The prepare phase downloads and validates the chosen single-file LTX checkpoint plus the local tokenizer, text encoder, scheduler, and VAE files it needs before inference.
- This built-in LTX family supports both:
  - text-to-video
  - text-image-to-video
- Upstream LTX supports more advanced multi-condition and long-duration pipelines, but this skill intentionally stays on the simpler `LTXPipeline` and `LTXImageToVideoPipeline` paths.
- The script chooses `task_type` automatically:
  - no `--image-path` -> `text_to_video`
  - with `--image-path` -> `text_image_to_video`
