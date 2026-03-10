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

This installs the `agentic-system` CLI. You can also run the module form with `python -m agentic_system`.

## UI Setup

The UI is the interactive terminal REPL started by `agentic-system` or `python -m agentic_system`.

Before launching it, make sure these pieces are ready:

1. Install the package in your current Python environment.
2. Choose a `workspace` directory for project files and generated artifacts.
3. Configure an LLM provider:
   - `ollama`: local service, no API key required
   - `zai`: requires `ZAI_API_KEY` or `OPENAI_COMPAT_API_KEY`
   - `deepseek`: requires `DEEPSEEK_API_KEY` or `OPENAI_COMPAT_API_KEY`
   - `lmstudio`: local OpenAI-compatible server, API key usually not required
4. Optional built-in tool backends:
   - image analysis defaults to `ollama` + `glm-ocr`
   - image generation defaults to `ollama` + `x/z-image-turbo`
   - web search defaults to `http://127.0.0.1:8888` for SearXNG

## Provider Setup

### Default Local Setup: Ollama

Install and start Ollama, then make sure the model you want is available:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull glm-ocr
ollama pull x/z-image-turbo
```

Then launch the UI:

```bash
agentic-system --workspace .
```

If you want a different Ollama model:

```bash
agentic-system --workspace . --provider ollama --model qwen2.5:14b
```

### Remote Setup: Z.AI

Export your API key before starting the UI:

```bash
export ZAI_API_KEY="your-zai-api-key"
```

Then launch:

```bash
agentic-system --workspace . --provider zai --model glm-5
```

If you want to use Z.AI's Coding API instead of the general API, set `ZAI_BASE_URL` before launch:

```bash
export ZAI_API_KEY="your-zai-api-key"
export ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"

agentic-system --workspace . --provider zai --model glm-5
```

The runtime will append `/chat/completions`, so this targets:

```text
https://api.z.ai/api/coding/paas/v4/chat/completions
```

### Remote Setup: DeepSeek

Export your API key before starting the UI:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

Then launch:

```bash
agentic-system --workspace . --provider deepseek --model deepseek-chat
```

### Local OpenAI-Compatible Setup: LM Studio

Start LM Studio's local server, then launch:

```bash
agentic-system --workspace . --provider lmstudio
```

If your local server uses a different base URL:

```bash
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
agentic-system --workspace . --provider lmstudio
```

## SearXNG Setup

The search skill uses an external SearXNG server. The UI does not start it automatically.

Recommended local setup uses the official SearXNG Docker image and the default runtime URL:

```text
http://127.0.0.1:8888
```

### Quick Local Setup

1. Install Docker.
2. Pull the official SearXNG image:

```bash
docker pull docker.io/searxng/searxng:latest
```

3. Create local config and data directories:

```bash
mkdir -p ./searxng/config ./searxng/data
```

4. Create `./searxng/config/settings.yml` and enable JSON search results:

```yaml
use_default_settings: true

server:
  secret_key: "change-this-to-a-random-string"
  limiter: false

search:
  safe_search: 0
  formats:
    - html
    - json
```

Why this is required:
- the search skill calls SearXNG with `format=json`
- SearXNG returns `403 Forbidden` if `json` is not enabled in `settings.yml`

5. Start SearXNG on port `8888`:

```bash
docker run --name searxng -d \
  -p 8888:8080 \
  -v "./searxng/config:/etc/searxng" \
  -v "./searxng/data:/var/cache/searxng" \
  docker.io/searxng/searxng:latest
```

6. Verify the service is reachable and JSON output works:

```bash
curl http://127.0.0.1:8888
curl 'http://127.0.0.1:8888/search?q=test&format=json'
```

If the second command returns `403 Forbidden`, your `settings.yml` is still not enabling `json`.

7. Start the UI:

```bash
agentic-system --workspace .
```

### Use a Different SearXNG Endpoint

If your SearXNG server is already running elsewhere:

```bash
agentic-system --workspace . --searxng-base-url http://your-host:port
```

or:

```bash
export SEARXNG_BASE_URL="http://your-host:port"
agentic-system --workspace .
```

### Stop the Local Container

```bash
docker container stop searxng
docker container rm searxng
```

## Quick Start

```bash
# Default startup: Ollama + controlled mode
agentic-system --workspace .

# Resume a named session in the current workspace
agentic-system --workspace . --session-id design-review-01

# Start a separate named session in the same workspace
agentic-system --workspace . --session-id bugfix-02

# Auto mode (no confirmation prompts)
agentic-system --workspace . --mode auto

# With DeepSeek
agentic-system --workspace ~/agent --provider deepseek

# With Z.AI
agentic-system --workspace ~/agent --provider zai --model glm-5

# Custom tool models
agentic-system --workspace . \
  --image-analysis-model glm-ocr \
  --image-generation-model x/z-image-turbo \
  --searxng-base-url http://127.0.0.1:8888
```

## How Sessions Work

- `workspace` is the shared project/artifact directory.
- `session_id` is the conversation memory key.
- If you omit `--session-id`, the run is ephemeral: it starts fresh and does not persist session memory on exit.
- If you provide `--session-id`, session state is loaded from `WORKSPACE/.sessions/<session_id>.json` if it already exists, otherwise a new named session is created there.
- Files created by the agent stay in the workspace regardless of session mode.

Typical pattern:

```bash
# First run
agentic-system --workspace ~/projects/my-app --session-id feature-plan

# Later, resume the same conversation
agentic-system --workspace ~/projects/my-app --session-id feature-plan

# Start a different conversation against the same project files
agentic-system --workspace ~/projects/my-app --session-id release-check
```

## Usage Notes

- Default provider: `ollama`
- Default mode: `controlled`
- `--workspace` is required
- `--model` is optional; provider defaults are used when omitted
- `--session-id` is optional; omit it for an ephemeral run
- Built-in skills are synced into `WORKSPACE/skills/` on startup
- The startup banner and `/status` show whether the current session is `ephemeral`, `new`, or `loaded`
- Module form is equivalent: `python -m agentic_system --workspace .`
- SearXNG is external; the runtime does not auto-start it

## CLI Reference

| Argument | Default | Notes |
|---|---|---|
| `--workspace` | required | Workspace directory for files, artifacts, and skills |
| `--session-id` | none | If omitted, the run is ephemeral |
| `--provider` | `ollama` | One of `ollama`, `deepseek`, `lmstudio`, `zai`, `openai_compatible` |
| `--mode` | `controlled` | `controlled` asks before exec; `auto` does not |
| `--model` | provider-specific | If omitted, the provider default model is used |
| `--image-analysis-provider` | `ollama` | Effective runtime default |
| `--image-analysis-model` | `glm-ocr` | Effective runtime default |
| `--image-generation-provider` | `ollama` | Effective runtime default |
| `--image-generation-model` | `x/z-image-turbo` | Effective runtime default |
| `--searxng-base-url` | `http://127.0.0.1:8888` | Effective runtime default for web search |

Provider model defaults when `--model` is omitted:

- `ollama` → `llama3.1:8b`
- `deepseek` → `deepseek-chat`
- `zai` → `glm-5`
- `lmstudio` → `local-model`
- `openai_compatible` → `local-model`

## Runtime Commands

- `/help` — show available commands
- `/status` — runtime overview (provider, mode, session, tool config, history)
- `/full_history` — show the full in-memory history from `Environment.full_history`
- `/observation` — show the current observation window from `Environment.observation`
- `/workflow_summary` — show the current `Environment.workflow_summary`
- `/last_prompt` — show the last prompt sent to the core agent
- `/exit` — quit

## Architecture

**4 action types** cover everything:
- `chat` — respond to user
- `think` — continue the loop without handing control back yet
- `exec` — run bash/python in the workspace sandbox
- `delegate` — spawn a sub-agent with an isolated child workspace

**Skills** are drop-in folders with `SKILL.md` + scripts. 9 built-in skills ship with the package and are synced into the workspace on startup.

**Providers**: `ollama`, `deepseek`, `lmstudio`, `zai`, `openai_compatible`.

## Configuration

| Env Var | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Default core-agent model when provider is `ollama` |
| `DEEPSEEK_API_KEY` | none | API key for `--provider deepseek` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API base URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Default model for `--provider deepseek` |
| `ZAI_API_KEY` | none | API key for `--provider zai` |
| `ZAI_BASE_URL` | `https://api.z.ai/api/paas/v4` | Z.AI base URL; can also point to the Coding API such as `https://api.z.ai/api/coding/paas/v4` |
| `ZAI_MODEL` | `glm-5` | Default model for `--provider zai` |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API base URL |
| `LMSTUDIO_MODEL` | `local-model` | Default model for `--provider lmstudio` |
| `OPENAI_COMPAT_API_KEY` | none | Generic fallback API key for OpenAI-compatible providers |
| `OPENAI_COMPAT_BASE_URL` | none | Generic fallback base URL for OpenAI-compatible providers |
| `OPENAI_COMPAT_MODEL` | none | Generic fallback model for OpenAI-compatible providers |
| `IMAGE_ANALYSIS_PROVIDER` | `ollama` | Image understanding provider |
| `IMAGE_ANALYSIS_MODEL` | `glm-ocr` | Image understanding model |
| `IMAGE_GENERATION_PROVIDER` | `ollama` | Image generation provider |
| `IMAGE_GENERATION_MODEL` | `x/z-image-turbo` | Image generation model |
| `SEARXNG_BASE_URL` | `http://127.0.0.1:8888` | SearXNG for web search |

CLI flags (`--image-analysis-model`, etc.) override env vars.

## Troubleshooting

- `zai HTTP 401` or `Authentication parameter not received in Header`
  Set `ZAI_API_KEY` before starting the UI.
- `Missing API key for provider 'zai'`
  The runtime did not find `ZAI_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- `Missing API key for provider 'deepseek'`
  The runtime did not find `DEEPSEEK_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- Ollama connection failures
  Make sure `ollama serve` is running and the selected model is available locally.
- Search skill cannot reach SearXNG
  Start a local SearXNG instance or override `--searxng-base-url` to a reachable endpoint.
