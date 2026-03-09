---
name: Image Generation
handler: scripts/generate_image.py
description: Generate images from text prompts using a configured image-capable model provider.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill when you need to generate a new image from a text prompt.

# Runtime Script

- Script path: `skills/all-agents/image-generation/scripts/generate_image.py`
- Executor: `python` via `script_path` + `script_args`

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. `stderr` should be used only for unexpected runtime failures.
3. Keep output concise and structured so runtime history is readable.

# Output Location Rule

Core Agent should always set one of:

- `--output-dir` (recommended for task batches), or
- `--output-path` (recommended for deterministic single-file output).

This keeps image artifacts easy to locate and reuse in later steps.

Path policy:

- Use runtime-workspace-local paths only.
- Do not write outside runtime workspace.
- Recommended directory pattern: `generated_images/<task_slug>/`.
- Keep path names simple and deterministic for later reuse.

# Size Selection Rule

Core Agent should always set `--size` based on task intent:

- Square/object/icon/avatar style requests: `1024x1024`
- Portrait/poster style requests: `1024x1536`
- Landscape/hero/banner style requests: `1536x1024`

If provider/model rejects requested size, retry once with `1024x1024`.

# Config Priority

1. `script_args` values (`--provider`, `--model`, `--base-url`, `--api-key`)
2. Runtime env injected by UI startup:
   - `IMAGE_GENERATION_PROVIDER` from `--image-generation-provider`
   - `IMAGE_GENERATION_MODEL` from `--image-generation-model`
3. Provider-specific env fallbacks (for example `OLLAMA_BASE_URL`)

If provider/model resolves to `none` or empty, the script returns `image_config_missing`.

# Action Input Template

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/image-generation/scripts/generate_image.py",
  "script_args": [
    "--prompt", "Two colorful budgies perched on a branch, shallow depth of field, natural lighting",
    "--output-dir", "generated_images",
    "--size", "1024x1024"
  ]
}
```

If provider/model are not pre-configured at runtime, include them explicitly:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/image-generation/scripts/generate_image.py",
  "script_args": [
    "--prompt", "Two colorful budgies perched on a branch, shallow depth of field, natural lighting",
    "--provider", "ollama",
    "--model", "x/z-image-turbo",
    "--output-dir", "generated_images",
    "--size", "1024x1024"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "image-generation",
  "status": "ok|error",
  "prompt": "...",
  "output_path": "...",
  "provider_used": "...",
  "model_used": "...",
  "error_code": "...",
  "generation_result": "..."
}
```

# Error Handling Rule

If this skill returns `error_code = "image_generation_unavailable"`, Core Agent should stop internal retries and return to requester with a brief configuration/status request.

# Notes

- This script uses OpenAI-compatible `POST /v1/images/generations`.
- For Ollama, ensure your local runtime supports image generation APIs and the target model is available.
- If neither `--output-dir` nor `--output-path` is provided, script defaults to `generated_images/`.
- Recommended startup when using runtime defaults:
  - `--image-generation-provider ollama`
  - `--image-generation-model x/z-image-turbo`
