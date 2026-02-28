# Agentic System

Terminal-first agentic runtime with:
- stateful workflow history + summary,
- strict action contract (`chat_with_requester|keep_reasoning|exec`),
- runtime exec control with approval modes and cancellation,
- pluggable model providers (`ollama`, `lmstudio`, `zai`, `deepseek`, `openai_compatible`),
- built-in skill bootstrapping into workspace.

## Current Package Layout

```text
agentic_system/
  cli.py
  runtime.py
  kernel/
    agent_loop.py
    prompts.py
    model_router.py
    executors.py
    storage.py
    history_utils.py
  prompts/
    agent_system_prompt.json
    agent_role_description.json
skills/
tests/
```

## Install

```bash
python -m pip install -e .
```

## Start Runtime UI

```bash
python -m agentic_system \
  --workspace /absolute/or/relative/workspace \
  --provider ollama \
  --model llama3.1:8b \
  --mode controlled
```

### Image skill runtime config

Use canonical names only:

```bash
python -m agentic_system \
  --workspace ./runtime_workspace \
  --provider zai \
  --model glm-5 \
  --image-analysis-provider ollama \
  --image-analysis-model llava:latest \
  --image-generation-provider ollama \
  --image-generation-model x/z-image-turbo
```

## Runtime Commands

- `/help`
- `/status`
- `/status workflow_summary`
- `/status workflow_hist`
- `/status full_proc_hist`
- `/status action_hist`
- `/status core_agent_prompt`
- `/refresh`
- `/exit`

## Production Notes

1. Prefer `--mode controlled` for human-in-the-loop execution.
2. Use `--mode auto` only when workspace write-policy behavior is acceptable for your use case.
3. Set provider credentials through environment variables (for example `ZAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_COMPAT_API_KEY`).
4. Keep `skills/` and `prompts/` under version control; runtime bootstraps these into the workspace each session.

## Tests

```bash
python -m unittest -v tests/test_runtime_kernel.py
```

## Migration

Legacy image config names were removed. See:
- `/Users/yangliu/Projects/Business/AgenticSystem/docs/migrations/2026-02-image-config-rename.md`
