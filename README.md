# Helix

Helix is an RL-inspired agent framework built around one small control law:

```text
state -> agent -> action -> environment -> state
```

The LLM is the `Agent`. The `Sandbox` is the equipped computer that lets the agent take real actions through bash and python. The `Environment` is everything the agent can inspect or affect through its `bash` and `python` hands. `State` is structured context, not a vague chat log. The loop stays grounded by runtime evidence.

## Motivation

Popular agentic systems have demonstrated that this interaction pattern is powerful, but many production-grade systems, including tools like Codex and Claude Code, have not been available as fully public end-to-end reference implementations.

At the same time, the open community has produced many useful agent design patterns, but not one widely accepted standard architecture.

This project exists to push toward that standardization. Its goal is to express agentic systems in a clean reinforcement-learning-style setting, where `state`, `agent`, `action`, and `environment` are explicit primitives rather than implicit framework conventions. The intent is to make agent design easier to reason about, compare, extend, and teach.

## Elegant Agentic Loop

![Agentic System working loop](design.png)

The illustration above captures the design used throughout the repo:

- `State` is built from `workflow_summary` and `observation`.
- `Agent` reasons over that state and emits exactly one action.
- `Action` can `chat`, `think`, `exec`, or `delegate`.
- `Sandbox` is the agent's computer: it can run bash and python, and it can use skills as installable software.
- `Environment` is everything the agent can inspect or affect through its `bash` and `python` hands: local files, OS resources, databases, APIs, and other external systems.
- `Runtime Evidence` flows back into state through stdout/stderr observations.
- `User` sits outside the loop.
- `Sub-agent Loop` mirrors the same pattern in an isolated delegated workspace.

## Install

```bash
python -m pip install -e .
```

This installs the `Helix` CLI and its required runtime dependencies, including `prompt_toolkit`.
Running the package directly from source without installing dependencies is not a supported setup for the interactive host.

You can also run:

```bash
python -m helix
```

## Quick Start

You need four things:

1. a workspace directory
2. a session id
3. an LLM provider
4. Docker available if you want the managed sandbox and tool services

### Full-Featured Example

```bash
helix \
  --workspace ../test_workspace \
  --provider zai \
  --model glm-5 \
  --mode auto \
  --session-id test_session
```

This launches an agent session with:

- `--workspace` — the project directory the agent works in
- `--provider` / `--model` — the core LLM that drives reasoning
- `--mode auto` — the agent executes without asking for approval (`controlled` prompts before each action)
- `--session-id` — required; names the session so conversation state persists across restarts

### Minimal Local Setup: Ollama

If you want to run the reasoning model locally with Ollama:

```bash
ollama serve
ollama pull llama3.1:8b
```

```bash
helix --workspace . --session-id my-session
```

### Other Providers

```bash
# Z.AI
export ZAI_API_KEY="your-zai-api-key"
helix --workspace . --provider zai --model glm-5 --session-id my-session

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"
helix --workspace . --provider deepseek --model deepseek-chat --session-id my-session

# LM Studio
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"   # only if not using the default
helix --workspace . --provider lmstudio --session-id my-session
```

## Sessions

- `workspace` is the shared project and artifact directory.
- `session_id` is the conversation memory key.
- `--session-id` is required by the current CLI.
- state is saved to `WORKSPACE/sessions/<session_id>/.state/session.json`

Examples:

```bash
# start or resume a named session
helix --workspace . --session-id design-review-01

# start a different session against the same workspace
helix --workspace . --session-id bugfix-02
```

## CLI Essentials

| Argument | Default | Purpose |
|---|---|---|
| `--workspace` | required | workspace root |
| `--session-id` | required | persistent conversation key |
| `--provider` | `ollama` | core model provider |
| `--mode` | `controlled` | `controlled` asks before exec, `auto` does not |
| `--model` | provider default | override the provider's default model |

Provider model defaults when `--model` is omitted:

- `ollama` -> `llama3.1:8b`
- `deepseek` -> `deepseek-chat`
- `zai` -> `glm-5`
- `lmstudio` -> `local-model`
- `openai_compatible` -> `local-model`

## Environment Variables

Precedence is:

1. CLI flags
2. provider-specific environment variables
3. generic OpenAI-compatible environment variables
4. built-in defaults

### Core Model Providers

| Variable | Used by | Default / Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | core provider | `http://localhost:11434` for the core provider |
| `OLLAMA_MODEL` | core provider | `llama3.1:8b` |
| `OLLAMA_TIMEOUT_SECONDS` | core provider | `300` |
| `OLLAMA_KEEP_ALIVE` | core provider | optional Ollama keep-alive duration |
| `DEEPSEEK_BASE_URL` | deepseek provider | `https://api.deepseek.com` |
| `DEEPSEEK_API_KEY` | deepseek provider | required for `--provider deepseek` unless `OPENAI_COMPAT_API_KEY` is set |
| `DEEPSEEK_MODEL` | deepseek provider | `deepseek-chat` |
| `LMSTUDIO_BASE_URL` | lmstudio provider | `http://localhost:1234/v1` |
| `LMSTUDIO_API_KEY` | lmstudio provider | optional |
| `LMSTUDIO_MODEL` | lmstudio provider | `local-model` |
| `ZAI_BASE_URL` | zai provider | `https://api.z.ai/api/paas/v4` |
| `ZAI_API_KEY` | zai provider | required for `--provider zai` unless `OPENAI_COMPAT_API_KEY` is set |
| `ZAI_MODEL` | zai provider | `glm-5` |
| `OPENAI_COMPAT_BASE_URL` | generic OpenAI-compatible provider | generic fallback base URL |
| `OPENAI_COMPAT_API_KEY` | generic OpenAI-compatible provider, zai/deepseek fallback | generic fallback API key |
| `OPENAI_COMPAT_MODEL` | generic OpenAI-compatible provider | `local-model` |
| `OPENAI_COMPAT_TIMEOUT_SECONDS` | generic OpenAI-compatible provider | `300` |

### Tool Backends

| Variable | Used by | Default / Notes |
|---|---|---|
| `SEARXNG_BASE_URL` | search-online-context skill | injected automatically by the Docker runtime for sandboxed search execs |
| `HELIX_LOCAL_MODEL_SERVICE_URL` | local PyTorch image skills | injected automatically by the Docker runtime on macOS Apple Silicon |
| `HELIX_LOCAL_MODEL_SERVICE_TOKEN` | local PyTorch image skills | injected automatically by the Docker runtime on macOS Apple Silicon |

### Runtime Controls

| Variable | Used by | Default / Notes |
|---|---|---|
| `AGENTIC_SANDBOX_TIMEOUT` | sandbox executor | `600` seconds |

### Example: Z.AI General API vs Coding API

General API:

```bash
export ZAI_API_KEY="your-zai-api-key"
export ZAI_BASE_URL="https://api.z.ai/api/paas/v4"
helix --workspace . --provider zai --model glm-5 --session-id my-session
```

Coding API:

```bash
export ZAI_API_KEY="your-zai-api-key"
export ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"
helix --workspace . --provider zai --model glm-5 --session-id my-session
```

The runtime appends `/chat/completions`, so the coding setup above targets:

```text
https://api.z.ai/api/coding/paas/v4/chat/completions
```

## Runtime Commands

- `/help` — show commands
- `/status` — show runtime configuration and session status
- `/full_history` — open the saved full history view for the current named session
- `/observation` — open the current observation window view
- `/workflow_summary` — open the saved compact summary view
- `/last_prompt` — open the exact last prompt sent to the model
- `/exit` — quit

The inspection commands operate on the active named session.

## Optional Tool Backends

### Image Skills

The built-in image capabilities are skill-owned rather than CLI-configured:

- `analyze-image-from-pytorch` uses `zai-org/GLM-OCR`
- `generate-image-from-pytorch` uses `Tongyi-MAI/Z-Image-Turbo`

On macOS Apple Silicon, the Docker runtime starts a narrow host-native local model service and injects the service URL/token into Docker execs automatically. The core Helix CLI does not expose separate image provider/model flags.

### Web Search: SearXNG

The app manages SearXNG for you through Docker when the Docker sandbox is active.

For normal app usage you do not need to:

- pass a search backend URL
- start a separate local SearXNG container
- wire search execs to a host port

At startup the runtime prepares the Docker sandbox image, Docker network, cache volume, and the managed SearXNG sidecar. Search execs then receive the correct internal `SEARXNG_BASE_URL` automatically.

Manual `SEARXNG_BASE_URL` setup only matters if you run the search scripts outside the Helix runtime.

## Troubleshooting

- `Missing API key for provider 'zai'`
  Set `ZAI_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- `Missing API key for provider 'deepseek'`
  Set `DEEPSEEK_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- Ollama connection failures
  Make sure `ollama serve` is running and the model exists locally.
- Search returns `403 Forbidden`
  Your SearXNG `settings.yml` is likely missing `json` under `search.formats`.
- Search returns connection refused
  SearXNG is not running or the configured URL is wrong.

## Repo Layout

```text
helix/
  core/           -> state, action, agent, environment, sandbox
  runtime/        -> loop, approval, host, cli, display, debug
  providers/      -> ollama and OpenAI-compatible providers
  builtin_skills/ -> shipped skills bootstrapped into each workspace
  prompts/        -> system prompt templates
```
