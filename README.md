# Agentic System

RL-inspired agent framework built on a universal loop:

```
state → agent(state) → action → environment(action) → observation → state
```

## Package Layout

```
agentic_system/
  core/           → state, action, agent, environment, loop
  runtime/        → sandbox, approval, host, cli
  providers/      → ollama, openai_compat (deepseek, lmstudio, zai)
  context/        → skill_loader, knowledge_loader, prompt_builder
  builtin_skills/ → 9 built-in skills (bootstrapped into workspace)
  prompts/        → system prompt templates
tests/
knowledge/
```

## Install

```bash
python -m pip install -e .
```

## Usage

```bash
# Default: Ollama, controlled mode
python -m agentic_system --workspace .

# With DeepSeek
python -m agentic_system --provider deepseek --workspace ~/agent

# Auto mode (no confirmation prompts)
python -m agentic_system --mode auto --workspace .

# Custom tool models
python -m agentic_system --workspace . \
  --image-analysis-model glm-ocr \
  --image-generation-model x/z-image-turbo \
  --searxng-base-url http://127.0.0.1:8888
```

## Runtime Commands

- `/help` — show available commands
- `/status` — session overview (provider, mode, tool config, history)
- `/exit` — quit

## Architecture

**4 action types** cover everything:
- `chat` — respond to user
- `think` — internal reasoning (loop continues)
- `exec` — run bash/python in sandbox
- `delegate` — spawn sub-agent with isolated workspace

**Skills** are drop-in folders with `SKILL.md` + scripts. 9 built-in skills ship with the package and are synced into the workspace on startup.

**Providers**: `ollama`, `deepseek`, `lmstudio`, `zai`, `openai_compatible`.

## Tests

```bash
python tests/test_core.py      # 21 tests — state, action, agent, env, loop
python tests/test_runtime.py   # 11 tests — sandbox, approval
python tests/test_context.py   # 15 tests — providers, skills, knowledge, prompts
python tests/test_skills.py    #  8 tests — real workspace, bootstrap
python tests/test_host.py      # 16 tests — RuntimeHost, CLI
python tests/test_delegate.py  #  7 tests — sub-agent delegation
```

## Configuration

| Env Var | Default | Purpose |
|---|---|---|
| `IMAGE_ANALYSIS_PROVIDER` | `ollama` | Image understanding provider |
| `IMAGE_ANALYSIS_MODEL` | `glm-ocr` | Image understanding model |
| `IMAGE_GENERATION_PROVIDER` | `ollama` | Image generation provider |
| `IMAGE_GENERATION_MODEL` | `x/z-image-turbo` | Image generation model |
| `SEARXNG_BASE_URL` | `http://127.0.0.1:8888` | SearXNG for web search |

CLI flags (`--image-analysis-model`, etc.) override env vars.
