# Helix

**An open, transparent, fully-local agentic system that evolves with you.**

Helix gives an LLM a real computer — a Docker sandbox where it writes and runs bash and python to get things done. Everything runs locally. No data leaves your machine unless you choose to connect an external LLM. The agent learns over time by creating reusable skills and documenting knowledge.

## Why Helix?

### Open and Transparent

Every part of Helix is visible and inspectable. The system prompt, the skills, the knowledge library, the full conversation history — nothing is hidden. You can view the exact prompt sent to the LLM at any time (`/view last_prompt`). No hidden tokens, no invisible instructions, no black boxes.

### Fully Local

Run everything on your own machine:
- **Local LLM** via Ollama or any OpenAI-compatible endpoint
- **Local web search** via SearXNG (self-hosted, privacy-respecting)
- **Local image/audio/video generation** via the built-in model service (MLX, PyTorch)

Your data stays on your machine. No API keys required for the default local setup.

### Yours to Modify

Helix is a framework, not a service. Fork it, change it, extend it. The codebase follows a simple, clean architecture — every component has one job, every file is readable, every decision is documented.

### Extensible Through Skills

Skills are reusable procedures the agent can discover and follow. Creating a new skill is as simple as writing a SKILL.md file — no code required for most skills. For complex tasks, add scripts. The agent uses existing skills and creates new ones as it works.

### Self-Evolving

The agent gets better over time:
- **Creates skills** when it discovers reusable task patterns
- **Documents knowledge** when it learns something worth remembering
- **Retrieves knowledge** from a library-style system to apply past learnings

### Library-Style Knowledge

Knowledge is organized like a real library: global index → category catalog → individual documents. The agent searches from broad classification to specific documents — no vector databases, no embeddings, just structured files you can read and edit yourself.

## The Control Law

Everything follows one loop:

```
state → agent → action → environment → state
```

The LLM is the **Agent**. The Docker sandbox is the agent's computer — its hands for affecting the **Environment**. **Skills** are reusable procedures. **Knowledge** is documented experience. The loop stays grounded by real execution evidence — stdout and stderr from every script.

## Quick Start

### 1. Install

```bash
pip install -e .
```

Requires Docker.

### 2. Start a Session (Local)

```bash
# Start Ollama
ollama serve && ollama pull llama3.1:8b

# Start Helix
helix \
  --endpoint-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --workspace ~/agent \
  --session-id my-project
```

### 3. Optional Services

```bash
helix start searxng                                    # Local web search
helix start local-model-service --workspace ~/agent    # Local ML models
helix model download --spec helix/builtin_skills/generate-image/model_spec.json
```

### 4. Or Use a Cloud LLM

```bash
helix \
  --endpoint-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model deepseek-chat \
  --workspace ~/agent \
  --session-id research-01
```

Any OpenAI-compatible endpoint works.

## CLI Reference

| Command | Purpose |
|---|---|
| `helix --endpoint-url URL --model MODEL --workspace PATH --session-id ID` | Start a session |
| `helix start searxng` | Start SearXNG search service |
| `helix start local-model-service --workspace PATH` | Start local model service |
| `helix stop searxng \| local-model-service` | Stop a service |
| `helix status` | Show running services |
| `helix model download --spec PATH` | Download model weights |

### Runtime Commands

| Command | Purpose |
|---|---|
| `/help` | Show commands |
| `/status` | Show session configuration |
| `/view <field>` | Inspect: full_history, observation, workflow_summary, last_prompt |
| `/exit` | Quit |

## Built-in Skills

| Skill | Purpose |
|---|---|
| `retrieve-knowledge` | Search and load knowledge documents |
| `create-document` | Create a knowledge document |
| `update-document` | Update a knowledge document |
| `create-skill` | Create a new skill |
| `update-skill` | Update an existing skill |
| `search-online-context` | Search the web via SearXNG |
| `brainstorming` | Structured ideation and design |
| `file-based-planning` | File-based task planning |
| `analyze-image` | Analyze images via Ollama |
| `generate-image` | Generate images (local MLX) |
| `generate-audio` | Generate audio (local PyTorch) |
| `generate-video` | Generate video (local PyTorch) |

## Documentation

- [Introduction](docs/introduction.md) — core concepts and design philosophy
- [Quick Start](docs/quickstart.md) — detailed first session walkthrough
- [Skills](docs/skills.md) — built-in skills, creating and extending skills
- [Knowledge](docs/knowledge.md) — the hierarchical knowledge system
- [Storage](docs/storage.md) — workspace and global file layout
