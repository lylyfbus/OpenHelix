---
name: Generate Audio
description: Prepare the built-in text-to-audio model, then generate speech audio with the host-native local model service using the built-in PyTorch Qwen3-TTS backend.
---

# Purpose

Use this skill when you need to prepare the built-in text-to-audio model and then generate a spoken audio file from text with the built-in PyTorch Qwen3-TTS backend.

# When To Use

Use when:
- the user wants a new spoken audio file generated from text
- the task should stay inside the built-in local audio capability path
- the resulting audio should be saved into the runtime workspace for later reuse

Skip when:
- the task is audio transcription rather than generation
- the user explicitly wants a different backend or service not provided by this skill
- the task is better handled by a remote API voice rather than a local open-source model

# Skill Mode

- `script_mode: multi`
- Preferred for audio generation because the agent should reason between preparation and inference.
- Default phase scripts:
  - `scripts/prepare_model.py`: prepare/download/warm the built-in audio model before generation
  - `scripts/generate_audio.py`: run the actual speech inference and save the output artifact
- Core Agent should call this skill directly instead of passing provider/model config through Helix CLI.
- Default handler path: `skills/all-agents/generate-audio/scripts/prepare_model.py`

# Procedure

1. Gather context:
   - confirm the exact text that should be spoken
   - identify the target language, speaker style, and any optional instruction
2. Prepare the model first:
   - run `scripts/prepare_model.py`
   - treat this as the model download / warmup phase
   - wait for `status=ok` before moving on to inference
   - this step is idempotent; it is safe even if the model is already cached
   - if preparation returns `error_code=missing_host_dependency`, stop and install the missing host binary before retrying
   - `--timeout` controls both the script HTTP wait and the local model service request budget for this call
   - if the job may run long, set exec `timeout_seconds` larger than `--timeout`
3. Plan output:
   - choose one workspace-local target using `--output-path` or `--output-dir`
   - keep generated assets in deterministic, reusable locations
4. Infer:
   - run `scripts/generate_audio.py` with `--text` and one output target option
   - pass `--language`, `--speaker`, and `--instruct` only when needed
   - let the skill choose its built-in backend/model; do not add provider/model args
5. Verify:
   - inspect runtime stdout for `status`, `output_path`, `sample_rate`, and any service/runtime error details
   - confirm the generated audio file exists in the workspace
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
  "script_path": "skills/all-agents/generate-audio/scripts/prepare_model.py",
  "script_args": [
    "--timeout", "1200"
  ]
}
```

## Phase 2: Generate Audio

- required:
  - `--text`
  - one of:
    - `--output-dir`
    - `--output-path`

Recommended argument rules:
- use workspace-local paths only
- if `--output-dir` is used, let the script choose a deterministic file name under that directory
- pass `--language` when the target language is known; otherwise the default `Auto` is acceptable
- pass `--speaker` when a specific timbre is desired; otherwise the default built-in speaker is acceptable
- use `--instruct` only when explicit style control is needed
- keep sampling knobs at defaults unless the user asks for a noticeably steadier or more expressive voice
- `--timeout` controls both the script HTTP wait and the local model service request budget for this call
- when generation may run long, set exec `timeout_seconds` larger than `--timeout`
- do not pass provider, model, or API settings; the skill owns its backend and model choice

### Parameter Reference

- `--text`
  - required
  - use 1-3 sentences per clip for best latency; longer narration is supported but can be much slower
- `--language`
  - default: `Auto`
  - best practice: pass an explicit language when known
- `--speaker`
  - default: `Vivian`
  - common alternatives: `Ryan`, `Serena`, `Aiden`, `Ono_Anna`, `Sohee`
- `--instruct`
  - default: empty
  - best practice: keep this short, usually one sentence or phrase
- `--do-sample`
  - default: `true`
  - recommended: keep `true` for most cases
- `--top-k`
  - default: `50`
  - recommended working range: `20-80`
- `--top-p`
  - default: `1.0`
  - recommended working range: `0.85-1.0`
- `--temperature`
  - default: `0.9`
  - recommended working range: `0.7-1.0`
  - lower values are steadier; higher values are more expressive
- `--repetition-penalty`
  - default: `1.05`
  - recommended working range: `1.0-1.15`
- `--max-new-tokens`
  - default: `4096`
  - recommended working range: `1024-4096`
  - raise only for longer clips that genuinely need it
- `--non-streaming-mode`
  - default: `true`
  - best practice: keep `true`
- `--seed`
  - default: `42`
  - use a fixed seed for reproducibility; change it only when you want a different sampled take

### Best-Practice Defaults

- Standard narration
  - `--language English --speaker Ryan --temperature 0.9 --top-k 50 --top-p 1.0 --repetition-penalty 1.05 --max-new-tokens 4096 --seed 42`
- Steadier enterprise or explainer voice
  - `--temperature 0.75 --top-k 40 --top-p 0.95 --repetition-penalty 1.08`
- More expressive marketing or dramatic read
  - `--temperature 0.95 --top-k 60 --top-p 1.0 --repetition-penalty 1.03`

Example inference using `--output-dir`:

```json
{
  "code_type": "python",
  "timeout_seconds": 1800,
  "script_path": "skills/all-agents/generate-audio/scripts/generate_audio.py",
  "script_args": [
    "--text", "Welcome home. Dinner is ready when you are.",
    "--language", "English",
    "--speaker", "Ryan",
    "--temperature", "0.9",
    "--top-k", "50",
    "--top-p", "1.0",
    "--repetition-penalty", "1.05",
    "--max-new-tokens", "4096",
    "--seed", "42",
    "--output-dir", "generated_audio/welcome-home"
  ]
}
```

Example inference using `--output-path`:

```json
{
  "code_type": "python",
  "timeout_seconds": 1800,
  "script_path": "skills/all-agents/generate-audio/scripts/generate_audio.py",
  "script_args": [
    "--text", "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "--language", "Chinese",
    "--speaker", "Vivian",
    "--instruct", "用特别愤怒的语气说",
    "--temperature", "0.95",
    "--top-k", "60",
    "--top-p", "1.0",
    "--repetition-penalty", "1.03",
    "--seed", "42",
    "--output-path", "sessions/test_1/project/generated-voice.wav"
  ]
}
```

# Output JSON Shape

## Prepare Phase

```json
{
  "executed_skill": "generate-audio",
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
  "executed_skill": "generate-audio",
  "phase": "generate",
  "status": "ok|error",
  "text": "...",
  "output_path": "...",
  "sample_rate": 24000,
  "model_used": "...",
  "error_code": "...",
  "message": "..."
}
```

# Error Handling Rule

1. If the local model service variables are missing, stop internal retries and return control to requester or runtime with the configuration failure.
2. If model preparation fails, do not continue to inference until preparation succeeds.
3. If preparation or generation returns `error_code=missing_host_dependency`, stop and install the named host dependency first.
4. If output path validation fails, do not retry with the same invalid path; choose a new workspace-local path first.
5. If generation fails due to service/model runtime issues, surface the exact `error_code` and `message` rather than masking them.
6. Retry only when a path or request shape can be corrected deterministically; otherwise stop and report.

# Skill Dependencies

- (none)

This skill is self-contained for audio generation and does not require another skill for normal execution.

# Notes

- This skill calls the runtime-managed local inference host through `HELIX_LOCAL_MODEL_SERVICE_URL`.
- The skill ships a `model_spec.json` next to `SKILL.md` and sends that full spec payload to the host.
- `scripts/prepare_model.py` calls `/models/prepare`; `scripts/generate_audio.py` calls `/infer`.
- The built-in backend is `pytorch`.
- The built-in model repo is `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.
- The built-in default speaker is `Vivian`; common alternatives include `Ryan`, `Serena`, `Aiden`, `Ono_Anna`, and `Sohee`.
- The script expects the Docker runtime to inject both `HELIX_LOCAL_MODEL_SERVICE_URL` and `HELIX_LOCAL_MODEL_SERVICE_TOKEN`.
- This model also requires the host machine running the local model service to have `sox` installed and available on `PATH`.
- On macOS, install it with `brew install sox` before using this skill.
