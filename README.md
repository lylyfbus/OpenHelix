# OpenHelix

**An open, transparent, fully-local agentic system that evolves with you.**

OpenHelix gives an LLM a real computer — a Docker sandbox where it writes and runs bash and python to get things done. Everything runs locally by default; no data leaves your machine unless you choose to connect a hosted LLM. The agent learns over time by creating reusable skills and documenting knowledge as it works.

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

The LLM is the **Agent**. The Docker sandbox is its computer — the hands that affect the **Environment**. **Skills** are reusable procedures. **Knowledge** is documented experience. Every step is grounded by real stdout/stderr evidence from execution.

## Quick Start

### 1. Install

```bash
pip install -e .
```

Requires Docker.

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

You land in an interactive prompt. Type a task in plain English and the agent will plan and execute it inside the Docker sandbox. Type `/help` for commands, `/exit` to quit.

### 3. Optional: Add Local Services

```bash
# Local web search
helix start searxng

# Local image / audio / video generation
helix start local-model-service --workspace ~/agent
helix model download --skill generate-image
helix model download --skill generate-audio
helix model download --skill generate-video
```

`helix model download` downloads only the model weights — each skill's runtime code comes along with the package. Set `HF_TOKEN` in your environment first if a model requires HuggingFace authentication.

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
2. **Acts** — writes bash or python and runs it inside the Docker sandbox.
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
| `create-skill` | Create a new reusable skill |
| `update-skill` | Update an existing skill |

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
| `helix --endpoint-url URL --model MODEL --workspace PATH --session-id ID` | Start a session |
| `helix start searxng` | Start the SearXNG search service |
| `helix start local-model-service --workspace PATH` | Start the local model service |
| `helix stop searxng \| local-model-service` | Stop a running service |
| `helix status` | Show running services |
| `helix model download --skill NAME` | Download model weights for a media-generation skill |

## Runtime Commands

| Command | Purpose |
|---|---|
| `/help` | Show all commands |
| `/status` | Show the session configuration |
| `/view <field>` | Inspect `full_history`, `observation`, `workflow_summary`, or `last_prompt` |
| `/exit` | Quit |

## Documentation

- [Introduction](docs/introduction.md) — core concepts and design philosophy
- [Quick Start](docs/quickstart.md) — detailed first session walkthrough
- [Skills](docs/skills.md) — built-in skills, and how to create your own
- [Knowledge](docs/knowledge.md) — the hierarchical knowledge system
- [Storage](docs/storage.md) — workspace and global file layout
