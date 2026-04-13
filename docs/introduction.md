# Introduction

## What is OpenHelix?

OpenHelix is an open, transparent, fully-local agentic system. It gives an LLM a real computer — a Docker sandbox where it writes and runs bash and python scripts to accomplish tasks. The agent learns over time by creating reusable skills and documenting knowledge in a structured library.

Everything is designed to run on your own machine. No data leaves your environment unless you choose to connect an external LLM endpoint.

## Key Features

### Open and Transparent

There are no hidden prompts, no invisible system instructions, no black-box tool calls. You can inspect:

- **The full system prompt** — see exactly what the agent is told, including all skills and workspace paths
- **The exact LLM request** — use `/view last_prompt` to see the complete messages array sent to the model
- **The full conversation history** — use `/view full_history` to see every turn, including runtime observations
- **All skills and knowledge** — plain markdown files in your workspace, readable and editable

The code itself is clean and documented. Every component has one job. You can read any source file and understand what it does.

### Fully Local

The default setup uses no external services:

| Component | Local Option |
|---|---|
| LLM reasoning | Ollama (llama3.1, deepseek, etc.) |
| Web search | SearXNG (self-hosted Docker container) |
| Image generation | MLX Z-Image (Apple Silicon) |
| Audio generation | Qwen3-TTS (PyTorch) |
| Video generation | Wan Video (PyTorch) |
| Script execution | Docker sandbox (ephemeral, isolated) |

Your workspace files, conversation history, knowledge documents, and generated artifacts all stay on your machine.

### Yours to Modify

OpenHelix is a framework you own, not a service you rent. The architecture is intentionally simple:

- `core/` — State, Action, Agent, Environment (~1000 lines total)
- `runtime/` — Loop, Host, Sandbox, Display (~1500 lines total)
- `providers/` — One LLM provider (~120 lines)
- `services/` — SearXNG and Local Model Service management
- `builtin_skills/` — 12 skills, each a SKILL.md file

No dependency injection, no plugin registries, no abstract base classes beyond what's needed. If you want to change how the agent works, you change `agent.py`. If you want to change how scripts run, you change `sandbox.py`.

### Extensible Through Skills

Skills are the agent's learned procedures. Creating one is as simple as writing a markdown file:

```markdown
---
name: My Skill
description: What it does.
---

# Procedure
1. Do this
2. Then this
3. Check the result
```

No Python decorators, no function schemas, no tool registration. The agent reads the SKILL.md and follows the steps. For complex logic, add scripts — but most skills don't need them.

### Self-Evolving

The agent improves through use:

1. **Creates skills** — when it notices a reusable pattern, it writes a SKILL.md for next time
2. **Updates skills** — when a procedure can be improved based on new experience
3. **Documents knowledge** — when it learns something worth preserving across sessions
4. **Retrieves knowledge** — when a new task overlaps with past work

This happens naturally during conversations. The user can also explicitly ask the agent to document or create skills.

### Library-Style Knowledge

Knowledge is organized like a physical library:

```
index.json                    → Global classification (like the Dewey Decimal system)
  category/subcategory/
    catalog.json              → Document titles and summaries (like catalog cards)
      docs/document.md        → The actual document (like the book)
```

The agent searches layer by layer: index → catalog → document. No vector databases, no embeddings — just structured files. You can browse, edit, and organize knowledge yourself using any text editor.

## The Control Law

Every interaction follows one reinforcement-learning-inspired loop:

```
state → agent → action → environment → state
```

| Step | What Happens |
|---|---|
| **State** | Built from the observation window + workflow summary (compacted long-term memory) |
| **Agent** | LLM reads state, produces one action |
| **Action** | `chat` (respond), `think` (reason), `exec` (run script), or `delegate` (spawn sub-agent) |
| **Environment** | Executes the action in the Docker sandbox; stdout/stderr flow back into state |

The loop repeats until the agent returns control to the user via `chat`. Every decision is grounded by real execution evidence — not simulated tool calls, but actual script output.

## Design Philosophy

- **Explicit over implicit** — state, action, and environment are visible primitives, not hidden framework internals
- **Evidence over assumption** — every agent decision is followed by real execution with observable results
- **Simple over clever** — each component has one job, each file is readable, no magic
- **Local over cloud** — default to running everything on the user's machine
- **Evolving over static** — the agent builds capability through skills and knowledge, not just conversation
