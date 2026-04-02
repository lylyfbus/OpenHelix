---
name: Generate Image From PyTorch
handler: scripts/generate_image_from_pytorch.py
description: Generate a new image with the host-native local model service using the built-in Z-Image-Turbo backend.
required_tools: exec
recommended_tools: exec
forbidden_tools:
script_mode: single
---

# Purpose

Use this skill when you need to generate a new image from a text prompt with the built-in local PyTorch backend.

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

- `script_mode: single`
- This is a backend-owned capability skill with one primary deterministic runtime script.
- Core Agent should call this skill directly instead of passing provider/model config through Helix CLI.
- Script path: `skills/all-agents/generate-image-from-pytorch/scripts/generate_image_from_pytorch.py`

# Procedure

1. Gather context:
   - confirm the requested subject, style, and intended use
   - derive the output size from task intent
2. Plan output:
   - choose one workspace-local target using `--output-path` or `--output-dir`
   - keep generated assets in deterministic, reusable locations
3. Act:
   - run the handler script with `--prompt` and one output target option
   - let the skill choose its built-in backend/model; do not add provider/model args
4. Verify:
   - inspect runtime stdout for `status`, `output_path`, and any service/runtime error details
   - confirm the generated file exists in the workspace
5. Report:
   - return the resulting artifact path and any important generation constraints or failures

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. `stderr` should be used only for unexpected runtime failures.
3. Keep output concise and structured so runtime history is readable.
4. The final JSON should expose `status`, `output_path`, `model_used`, `error_code`, and `message`.
5. The script must keep all file writes inside the runtime workspace.

# Action Input Templates

Required input:
- `--prompt`
- one of:
  - `--output-dir`
  - `--output-path`

Recommended path rules:
- use workspace-local paths only
- if `--output-dir` is used, let the script choose a deterministic file name under that directory
- do not pass provider, model, or API settings; the skill owns its backend and model choice

Example using `--output-dir`:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/generate-image-from-pytorch/scripts/generate_image_from_pytorch.py",
  "script_args": [
    "--prompt", "A clean product hero banner with warm afternoon light",
    "--output-dir", "generated_images/hero-banner",
    "--size", "1536x1024"
  ]
}
```

Example using `--output-path`:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/generate-image-from-pytorch/scripts/generate_image_from_pytorch.py",
  "script_args": [
    "--prompt", "A minimal monochrome icon set for analytics dashboards",
    "--output-path", "generated_images/icons/analytics-set.png",
    "--size", "1024x1024"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "generate-image-from-pytorch",
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
2. If output path validation fails, do not retry with the same invalid path; choose a new workspace-local path first.
3. If generation fails due to service/model runtime issues, surface the exact `error_code` and `message` rather than masking them.
4. Retry only when a path or request shape can be corrected deterministically; otherwise stop and report.

# Skill Dependencies

- (none)

This skill is self-contained for image generation and does not require another skill for normal execution.

# Notes

- This skill calls the runtime-managed local model service through `HELIX_LOCAL_MODEL_SERVICE_URL`.
- The built-in model ID is `Tongyi-MAI/Z-Image-Turbo`.
- The script expects the Docker runtime to inject both `HELIX_LOCAL_MODEL_SERVICE_URL` and `HELIX_LOCAL_MODEL_SERVICE_TOKEN`.
