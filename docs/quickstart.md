# Quick Start

## Prerequisites

- Python 3.11+
- Docker (running)
- An LLM endpoint (local Ollama or cloud API)

## Install

```bash
pip install -e .
```

## Option A: Local LLM with Ollama

### 1. Start Ollama

```bash
ollama serve
ollama pull llama3.1:8b
```

### 2. Start OpenHelix

```bash
helix \
  --endpoint-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --workspace ~/my-agent \
  --session-id first-session
```

### 3. Try It

```
user> What files are in the current directory?
```

The agent will write a bash script (`ls -la`), execute it in the Docker sandbox, read the output, and report back.

## Option B: Cloud LLM (DeepSeek)

```bash
helix \
  --endpoint-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model deepseek-chat \
  --workspace ~/my-agent \
  --session-id first-session
```

Any OpenAI-compatible endpoint works — just provide the URL and model name.

## What Happened

When you started OpenHelix:

1. Docker was checked (required for the sandbox).
2. Built-in skills were copied into your workspace under `skills/builtin_skills/`.
3. The Docker sandbox image was built (first time only).
4. The REPL started, waiting for your input.

When you sent a message:

1. Your message was recorded as a `[user]` turn.
2. The agent read its system prompt (identity + available skills) and your message.
3. The agent decided on an action (e.g., `exec` with a bash script).
4. The script ran inside a Docker container.
5. The stdout/stderr came back as a `[runtime]` turn.
6. The agent read the result and responded with `chat`.

## Optional: Start Services

### Web Search (SearXNG)

```bash
helix start searxng
```

Now the agent can use the `search-online-context` skill to search the web.

### Local Model Service (for image/audio/video generation)

```bash
# Download a model first
helix model download --spec helix/builtin_skills/generate-image/model_spec.json

# Start the service
helix start local-model-service --workspace ~/my-agent
```

Now the agent can use generative skills like `generate-image`.

### Check Service Status

```bash
helix status
```

## Next Steps

- [Architecture](architecture.md) — understand how the system works
- [Skills Guide](skills-guide.md) — learn about skills and create your own
- [Knowledge Guide](knowledge-guide.md) — document and retrieve knowledge
- [CLI Reference](cli-reference.md) — all commands and options
