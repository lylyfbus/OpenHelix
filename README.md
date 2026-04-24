# OpenHelix

**An open, transparent, fully-local agentic system that evolves with you.**

OpenHelix gives an LLM a real computer — a host-shell sandbox where it writes and runs bash and python to get things done, gated by an approval prompt you answer in the loop. Everything runs locally by default; no data leaves your machine unless you choose to connect a hosted LLM. The agent learns over time by creating reusable skills and documenting knowledge as it works.

## Highlights

- **Local by default.** Local LLM via Ollama or any OpenAI-compatible endpoint, local web search via SearXNG, local image/audio/video generation via the built-in model service (MLX and PyTorch). No API keys required for the default setup.
- **Fully transparent.** Inspect the full conversation, the exact system prompt, the skills the agent has, the knowledge library, and the exact text sent to the LLM at any moment (`/view last_prompt`). Nothing is hidden.
- **Extensible through skills.** A skill is just a `SKILL.md` file the agent reads and follows — no code required for most skills. For complex tasks, add scripts alongside. The agent creates new skills when it discovers reusable patterns.
- **Self-evolving.** The agent documents what it learns into a library (global index → category catalog → document), so past learnings are reusable without vector databases or embeddings. Fork it, change it, extend it — every component has one job and every file is readable.

## The Control Law

Everything follows one loop:

```
state → agent → action → environment → state
```

![The OpenHelix agentic loop](design.png)

The LLM is the **Agent**. The host-shell sandbox is its computer — the hands that affect the **Environment**, with an approval gate you control. **Skills** are reusable procedures. **Knowledge** is documented experience. Every step is grounded by real stdout/stderr evidence from execution.

## Quick Start

### 1. Install

```bash
pip install -e .
```

Requires Python 3.10+. No Docker dependency — the exec sandbox runs on your host shell, and the bundled services (SearXNG, local model service) are managed as pip/venv subprocesses.

### 2. Your First Session

The fastest happy path is a local LLM via Ollama:

```bash
# One-time: pull a small local model
ollama serve && ollama pull llama3.1:8b

# Start OpenHelix
helix \
  --endpoint-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --workspace ~/agent \
  --session-id my-first-session
```

You land in an interactive prompt. Type a task in plain English and the agent will plan and execute it in the host-shell sandbox. Type `/help` for commands, `/exit` to quit.

**About approval mode.** By default the agent runs in `--mode controlled`, which prompts you to approve every bash/python execution before it runs. You see the job name, the script, and an `[y/N/s/p/k]` menu — this is how OpenHelix keeps you in the loop on every concrete action the agent takes. If you trust the task and want the agent to run autonomously without prompts, start it with `--mode auto`. The two valid values are `controlled` (default) and `auto`.

### 3. Optional: Add Local Services

```bash
# Local web search
helix start searxng

# Local image / audio / video generation
helix start local-model-service
helix model download --skill generate-image
helix model download --skill generate-audio
helix model download --skill generate-video
```

`helix model download` fetches model weights from **[HuggingFace Hub](https://huggingface.co)** — this is currently the only supported source. Each generative skill's `model_spec.json` points at a HuggingFace repo slug like `author/model-name`. Set `HF_TOKEN` in your environment first if a model is gated or private.

### 4. Alternative: Use a Hosted LLM

Any OpenAI-compatible endpoint works:

```bash
helix \
  --endpoint-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model deepseek-chat \
  --workspace ~/agent \
  --session-id research-01
```

## What a Session Looks Like

When you send a task, the agent:

1. **Plans** — decides which skills apply and loads their `SKILL.md`.
2. **Acts** — writes bash or python and runs it in the host-shell sandbox (subject to your approval in controlled mode).
3. **Observes** — reads stdout/stderr, updates its plan, and continues.
4. **Learns** — documents anything reusable into the knowledge library, optionally creates a new skill.
5. **Resumes** — saves session state so you can pick up later with the same `--session-id`.

Everything is inspectable. `/view last_prompt` shows the exact text sent to the LLM on the most recent turn. `/view observation` shows the recent turn trace. `/view workflow_summary` shows the compacted long-term memory. `/status` shows the session config. Nothing happens in secret.

## Built-in Skills

### Knowledge & planning

| Skill | Purpose |
|---|---|
| `retrieve-knowledge` | Search and load knowledge documents |
| `create-document` | Create a knowledge document |
| `update-document` | Update a knowledge document |
| `file-based-planning` | File-based task planning |
| `brainstorming` | Structured ideation and design |

### Skill authoring

| Skill | Purpose |
|---|---|
| `create-skill` | Create a new procedural skill (SKILL.md + optional scripts) |
| `update-skill` | Update an existing procedural skill |
| `create-generative-skill` | Create a new ML-backed skill (model_spec + host adapter + scripts) |
| `update-generative-skill` | Update an existing generative skill (with re-download/restart flow) |

### Web & media generation

| Skill | Purpose |
|---|---|
| `search-online-context` | Search the web via SearXNG |
| `analyze-image` | Analyze images via an Ollama vision model |
| `generate-image` | Text-to-image (local MLX, Z-Image) |
| `generate-audio` | Text-to-speech (local PyTorch, Qwen3-TTS) |
| `generate-video` | Text-to-video and image-to-video (local MLX, LTX-2.3) |

## CLI Reference

| Command | Purpose |
|---|---|
| `helix --endpoint-url URL --model MODEL --workspace PATH --session-id ID [--mode auto\|controlled] [--think enable\|disable] [--effort minimal\|low\|medium\|high]` | Start a session |
| `helix start searxng` | Start the SearXNG search service |
| `helix start local-model-service` | Start the local model service |
| `helix stop searxng \| local-model-service` | Stop a running service |
| `helix status` | Show running services |
| `helix model download --skill NAME` | Download model weights for a media-generation skill |

`--mode` controls the approval policy: `controlled` (default) prompts you before every bash/python execution; `auto` runs without prompts.

`--think` and `--effort` shape how hard the model reasons. They're optional and independent; omit either to fall back to the server's default.

- **`--think enable|disable`** — binary thinking-mode toggle. Maps to the three common OpenAI-compatible field conventions so a single flag works across servers: `thinking.type` (DeepSeek, Z.ai/GLM), `think` (Ollama), and `chat_template_kwargs.enable_thinking` (vLLM/SGLang Qwen3). Providers that don't recognize a field ignore it.
- **`--effort minimal|low|medium|high`** — reasoning-effort level, forwarded as `reasoning_effort`. Recognized by OpenAI (GPT-5/o-series), DeepSeek, and Gemini's OpenAI-compatible endpoint. Ignored by providers that don't support effort levels.

Example — DeepSeek with thinking enabled at medium effort:

```bash
helix \
  --endpoint-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model deepseek-chat \
  --think enable --effort medium \
  --workspace ~/agent --session-id research-01
```

## Runtime Commands

| Command | Purpose |
|---|---|
| `/help` | Show all commands |
| `/status` | Show the session configuration |
| `/view <field>` | Inspect core-agent state: `full_history`, `observation`, `workflow_summary`, or `last_prompt` |
| `/view sub_agents` | List sub-agents created in this session |
| `/view <field> <role>` | Inspect a specific sub-agent's state by role |
| `/exit` | Quit |

Sub-agents are spawned when the core-agent chooses a `delegate` action. Each sub-agent persists its full history, observation window, workflow summary, and last prompt across delegations to the same role, so you can drill into exactly what the sub-agent saw on any past turn.

## Documentation

- [Introduction](docs/introduction.md) — core concepts and design philosophy
- [Quick Start](docs/quickstart.md) — detailed first session walkthrough
- [Skills](docs/skills.md) — built-in skills, and how to create your own
- [Knowledge](docs/knowledge.md) — the hierarchical knowledge system
- [Storage](docs/storage.md) — workspace and global file layout
