---
name: Generate Image
description: Prepare the built-in image model, then generate a new image with the host-native local model service using the built-in MLX Z-Image backend.
---

# Purpose

Use this skill when you need to prepare the built-in image model and then generate a new image from a text prompt with the built-in local MLX backend.

# When To Use

Use when:
- the user wants a brand-new image generated from a text prompt
- the task should stay inside the built-in local-image capability path
- the resulting image should be saved into the runtime workspace for later reuse

Skip when:
- the task is image analysis rather than generation
- the user explicitly wants a different backend or service not provided by this skill
- the task is better handled by editing an existing asset instead of generating a new one

# Skill Mode

- `script_mode: multi`
- Preferred for image generation because the agent should reason between preparation and inference.
- Default phase scripts:
  - `scripts/prepare_model.py`: prepare/download/warm the built-in image model before generation
  - `scripts/generate_image.py`: run the actual image inference and save the output artifact
- Core Agent should call this skill directly instead of passing provider/model config through Helix CLI.
- Default handler path: `skills/all-agents/generate-image/scripts/prepare_model.py`

# Procedure

1. Gather context:
   - confirm the requested subject, style, and intended use
   - derive the output size from task intent
2. Prepare the model first:
   - run `scripts/prepare_model.py`
   - treat this as the model download / warmup phase
   - wait for `status=ok` before moving on to inference
   - this step is idempotent; it is safe even if the model is already cached
   - `--timeout` controls both the script HTTP wait and the local model service request budget for this call
   - if the job may run long, set exec `timeout_seconds` larger than `--timeout`
3. Plan output:
   - choose one workspace-local target using `--output-path` or `--output-dir`
   - keep generated assets in deterministic, reusable locations
4. Infer:
   - run `scripts/generate_image.py` with `--prompt` and one output target option
   - let the skill choose its built-in backend/model; do not add provider/model args
5. Verify:
   - inspect runtime stdout for `status`, `output_path`, and any service/runtime error details
   - confirm the generated file exists in the workspace
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
  "script_path": "skills/all-agents/generate-image/scripts/prepare_model.py",
  "script_args": [
    "--timeout", "1200"
  ]
}
```

## Phase 2: Generate Image

- required:
  - `--prompt`
  - one of:
    - `--output-dir`
    - `--output-path`

Recommended path rules:
- use workspace-local paths only
- if `--output-dir` is used, let the script choose a deterministic file name under that directory
- `--timeout` controls both the script HTTP wait and the local model service request budget for this call
- when generation may run long, set exec `timeout_seconds` larger than `--timeout`
- do not pass provider, model, or API settings; the skill owns its backend and model choice

Example inference using `--output-dir`:

```json
{
  "code_type": "python",
  "timeout_seconds": 1800,
  "script_path": "skills/all-agents/generate-image/scripts/generate_image.py",
  "script_args": [
    "--prompt", "A clean product hero banner with warm afternoon light",
    "--output-dir", "generated_images/hero-banner",
    "--size", "1536x1024"
  ]
}
```

Example inference using `--output-path`:

```json
{
  "code_type": "python",
  "timeout_seconds": 1800,
  "script_path": "skills/all-agents/generate-image/scripts/generate_image.py",
  "script_args": [
    "--prompt", "A minimal monochrome icon set for analytics dashboards",
    "--output-path", "generated_images/icons/analytics-set.png",
    "--size", "1024x1024"
  ]
}
```

# Output JSON Shape

## Prepare Phase

```json
{
  "executed_skill": "generate-image",
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
  "executed_skill": "generate-image",
  "phase": "generate",
  "status": "ok|error",
  "prompt": "...",
  "output_path": "...",
  "model_used": "...",
  "error_code": "...",
  "message": "..."
}
```

# Error Handling Rule

1. If the local model service variables are missing, stop internal retries and return control to requester or runtime with the configuration failure.
2. If model preparation fails, do not continue to inference until preparation succeeds.
3. If output path validation fails, do not retry with the same invalid path; choose a new workspace-local path first.
4. If generation fails due to service/model runtime issues, surface the exact `error_code` and `message` rather than masking them.
5. Retry only when a path or request shape can be corrected deterministically; otherwise stop and report.

# Skill Dependencies

- (none)

This skill is self-contained for image generation and does not require another skill for normal execution.

# Notes

- This skill calls the runtime-managed local inference host through `HELIX_LOCAL_MODEL_SERVICE_URL`.
- The skill ships a `model_spec.json` next to `SKILL.md` and sends that full spec payload to the host.
- `scripts/prepare_model.py` calls `/models/prepare`; `scripts/generate_image.py` calls `/infer`.
- The built-in backend is `mlx`.
- The built-in model repo is `uqer1244/MLX-z-image`.
- The script expects the Docker runtime to inject both `HELIX_LOCAL_MODEL_SERVICE_URL` and `HELIX_LOCAL_MODEL_SERVICE_TOKEN`.
