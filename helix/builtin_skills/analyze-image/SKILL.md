---
name: Analyze Image
description: Analyze a workspace-local image with the built-in Ollama GLM-OCR backend.
---

# Purpose

Use this skill when you need objective image-content analysis with the built-in Ollama GLM-OCR backend.

# When To Use

Use when:
- the task depends on visual content or OCR-style extraction from an image
- the image should be analyzed with the built-in Ollama capability path
- the result should be grounded in an explicit query rather than guesswork

Skip when:
- the task is image generation rather than understanding
- the answer can be obtained from adjacent text or metadata without image inspection
- the user explicitly wants a different backend or external vision service

# Skill Mode

- `script_mode: single`
- This is a backend-owned capability skill with one primary deterministic runtime script.
- Core Agent should call this skill directly instead of passing provider or model config through Helix CLI.
- Script path: `skills/all-agents/analyze-image/scripts/analyze_image.py`

# Procedure

1. Gather context:
   - identify exactly what the requester needs from the image
   - rewrite the request into a short, explicit analysis query
2. Prepare input:
   - use `--image-path` for a workspace-local file when available
   - if starting from a remote image, use `--image-url` and let the script download it into the workspace first
3. Act:
   - run the handler script with one image source plus `--query`
   - let the skill choose its built-in backend/model; do not add provider/model args
4. Verify:
   - inspect runtime stdout for `status`, `analysis`, `model_used`, and any failure detail
   - ensure the analysis actually answers the requested query
5. Report:
   - return the extracted analysis and any caveats or blockers clearly

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. `stderr` should be used only for unexpected runtime failures.
3. Keep output concise and structured so runtime history is readable.
4. The final JSON should expose `status`, `analysis`, `model_used`, `error_code`, and `message`.
5. Remote images must be downloaded into the runtime workspace before they are sent to Ollama.

# Action Input Templates

Required query context:
- `--query` is required
- Core Agent should prepare a short, explicit query describing what should be extracted or evaluated from the image

Use exactly one image source:
- `--image-path` for a workspace-local file
- `--image-url` for a remote image; the script will download it into the workspace before calling Ollama

Example using a local file:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/analyze-image/scripts/analyze_image.py",
  "script_args": [
    "--image-path", "assets/banner.jpg",
    "--query", "Extract the visible title text and key layout regions"
  ]
}
```

Example using a remote image:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/analyze-image/scripts/analyze_image.py",
  "script_args": [
    "--image-url", "https://example.com/sample-doc.png",
    "--query", "Extract the visible text and describe the document structure"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "analyze-image",
  "status": "ok|error",
  "image_source": "...",
  "analysis": "...",
  "model_used": "...",
  "error_code": "...",
  "message": "..."
}
```

# Error Handling Rule

1. If Ollama is unavailable, stop internal retries and return control with the connection failure.
2. If image-path validation fails or the downloaded image is unusable, do not retry until the input source is corrected.
3. If the analysis query is vague or missing, refine the query before rerunning instead of repeating the same request.
4. If Ollama returns a model/runtime failure, surface the exact `error_code` and `message` and stop unless a deterministic input fix is available.

# Skill Dependencies

- (none)

This skill is self-contained for image analysis and does not require another skill for normal execution.

# Notes

- This skill calls the local Ollama API directly from inside the Docker sandbox.
- The built-in model ID is `glm-ocr`.
- Helix does not start or stop Ollama for this skill; make sure `ollama serve` is running and `glm-ocr` is installed.
