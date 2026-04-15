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

By default the agent starts in `--mode controlled`, which means **every bash or python execution pauses for your approval first**. You'll see a prompt showing the job name, the script, and an `[y/N/s/p/k]` menu (`y` = allow once, `s` = allow same exact exec for the session, `p` = allow same script pattern, `k` = allow same script_path ignoring args, `N` = deny). If you'd rather let the agent run autonomously without any interruptions, add `--mode auto` to the `helix` command. The two valid values are `controlled` and `auto`.

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
# Download model weights from HuggingFace Hub
helix model download --skill generate-image

# Start the service
helix start local-model-service --workspace ~/my-agent
```

All generative-skill model weights are fetched from **[HuggingFace Hub](https://huggingface.co)** — this is currently the only supported source. If a model is gated or private, set `HF_TOKEN=hf_xxx...` in your shell before running `helix model download`.

Now the agent can use generative skills like `generate-image`.

### Check Service Status

```bash
helix status
```

## Letting the Agent Push to GitHub

OpenHelix runs agent scripts in a Docker sandbox with strong isolation — `--read-only` root filesystem, capabilities dropped, no new privileges, network restricted to a dedicated bridge, runs as your host user. By default the container has **no credentials** for any external service, so `git push` against a GitHub URL fails. If you want the agent to be able to commit and push on your behalf, here are the two supported workflows.

### Option A: SSH (shares your host ~/.ssh)

If you already have SSH keys set up for GitHub on the host, the sandbox can use them directly. On startup, the sandbox:

1. Installs `openssh-client` in the image.
2. Copies your `~/.ssh` directory and `~/.gitconfig` file into the container's `$HOME` (`/helix-cache/home`), stripping macOS-only config options like `UseKeychain` that Linux OpenSSH rejects.
3. Generates a minimal `/etc/passwd` and `/etc/group` that map your host UID/GID to a valid user entry so OpenSSH can look up `getpwuid()`.

All of this happens automatically on first `helix` launch after the sandbox rebuild — no flags to pass. Inside a session, the agent can then run:

```bash
git clone git@github.com:you/your-repo.git
cd your-repo
# ...edit files...
git commit -m "..."
git push origin master
```

If you have `Host` aliases in your `~/.ssh/config` (e.g. `github-work` with a per-account `IdentityFile`), those work too:

```bash
git push git@github-work:org/repo.git master
```

**Security posture.** This gives the container full git authority as you — the agent can in principle push to **any** GitHub repo your SSH keys can reach, not just the one you're currently working on. The `--mode controlled` approval system is the primary safety net: with it enabled (the default), you see and approve every `git push` command before it runs. **Do not run in `--mode auto` with SSH keys accessible.**

### Option B: HTTPS token (scoped, per-repo)

If you want a narrower blast radius, use a GitHub personal access token instead. The sandbox already forwards any host environment variable prefixed with `SANDBOX_` into the container (stripping the prefix), so no sandbox config changes are needed.

1. Go to https://github.com/settings/tokens?type=beta → "Generate new token" → **fine-grained personal access token**.
2. Set **Repository access** → **Only select repositories** → choose the exact repo(s) you want the agent to touch.
3. Set **Repository permissions** → **Contents: Read and write** (and any other minimal scopes you need).
4. Set an **expiration date** (30 days is a sensible default).
5. Copy the generated token.

On your host, before launching OpenHelix:

```bash
export SANDBOX_GH_TOKEN=github_pat_11AB...
helix --endpoint-url ... --model ... --workspace ~/agent --session-id ...
```

Inside a session, the agent can then push over HTTPS:

```bash
git push https://x-access-token:$GH_TOKEN@github.com/owner/repo.git master
```

Or configure a remote once per repo:

```bash
git remote set-url origin https://github.com/owner/repo.git
git -c "http.extraheader=Authorization: Bearer $GH_TOKEN" push origin master
```

**Security posture.** Much narrower than SSH: if the token leaks (e.g. through a prompt-injected agent dump), the blast radius is exactly the repos and permissions you scoped it to, and you can revoke a single token without touching any other credentials. You still need `--mode controlled` as the first line of defence, but the token scope is a real second line.

### Which one to pick

| Goal | Pick |
|---|---|
| "I just want it to work, I trust the approval prompts" | **SSH** (Option A) — zero setup, reuses your existing keys |
| "Agent should only touch one specific repo" | **HTTPS token** (Option B) — fine-grained scope |
| "Agent needs to push to both `lylyf1987` and `lylyfbus` accounts in the same session" | **SSH** — one `Host` alias per account in your `~/.ssh/config` |
| "This is running unattended (CI-style) and I can't interactively approve" | **HTTPS token in a scoped, short-lived token** — never SSH, and never `--mode auto` with SSH keys in scope |
| "I want defence in depth" | Both — SSH for interactive sessions, HTTPS token for unattended |

Regardless of which you pick, **`--mode controlled` (the default) is the single most important safety control.** It gives you a human-readable preview of every bash/python command the agent wants to run before it executes, and you can deny anything that looks wrong.

## Next Steps

- [Introduction](introduction.md) — core concepts and design philosophy
- [Skills](skills.md) — built-in skills and how to create your own
- [Knowledge](knowledge.md) — the hierarchical knowledge system
- [Storage](storage.md) — where OpenHelix puts every file on your machine
