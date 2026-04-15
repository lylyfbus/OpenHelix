# Skills

## What is a Skill?

A skill is a directory with a `SKILL.md` file that describes a reusable procedure. The agent discovers skills by their metadata (name + description) in its system prompt, reads the full SKILL.md when needed, and follows the procedure.

```
my-skill/
  SKILL.md              Required: metadata + procedure
  scripts/              Optional: for complex steps
    my_script.py
```

### SKILL.md Format

```markdown
---
name: My Skill Name
description: One-line description.
---

# Purpose
# When To Use
# Procedure
# Rules
```

Frontmatter has exactly two fields: `name` and `description`.

## Built-in Skills

| Skill | Type | Purpose |
|---|---|---|
| `retrieve-knowledge` | no-script | Search and load knowledge documents |
| `create-document` | no-script | Create a knowledge document |
| `update-document` | no-script | Update a knowledge document |
| `create-skill` | no-script | Create a new skill |
| `update-skill` | no-script | Update an existing skill |
| `search-online-context` | multi-script | Search the web via SearXNG |
| `brainstorming` | no-script | Structured ideation and design |
| `file-based-planning` | multi-script | File-based task planning |
| `analyze-image` | single-script | Analyze images via Ollama |
| `generate-image` | multi-script + adapter | Text-to-image (MLX, Z-Image) |
| `generate-audio` | multi-script + adapter | Text-to-speech (PyTorch, Qwen3-TTS) |
| `generate-video` | multi-script + adapter | Text/image-to-video (MLX, LTX-2.3) |

Built-in skills live under `skills/builtin_skills/` and are synced from the package on every startup. User-created skills go directly under `skills/`.

## Creating a Skill

### Script Mode Decision

Choose based on **step complexity**:

- **No scripts** — every step is simple (read/write files, standard commands). Most skills.
- **Single script** — one step is complex (API with auth/retry, binary parsing).
- **Multiple scripts** — multiple steps are independently complex.

### Example: No-Script Skill

```
skills/check-formatting/
  SKILL.md
```

```markdown
---
name: Check Formatting
description: Check Python code formatting with black.
---

# Purpose
Run black formatter and report issues.

# When To Use
Before committing code or when asked about code quality.

# Procedure

## Step 1: Check formatting
\```json
{"job_name": "check-format", "code_type": "bash", "script": "pip install black && black --check --diff ."}
\```

## Step 2: Report
If passed, report success. If failed, show the diff.

# Rules
- Only check Python files.
- Do not auto-fix without user approval.
```

### Example: Script-Based Skill

```
skills/fetch-weather/
  SKILL.md
  scripts/
    fetch_weather.py
```

The SKILL.md procedure references the script:

```json
{"job_name": "fetch-weather", "code_type": "python",
 "script_path": "skills/fetch-weather/scripts/fetch_weather.py",
 "script_args": ["--city", "San Francisco"]}
```

Use a script when the logic is complex enough that writing it inline every time would be error-prone.

## Creating a Generative Skill

Generative skills run ML models on the host machine (not in the Docker sandbox, because models need GPU/MPS access and tens of gigabytes of weights). The local model service hosts these models as isolated worker subprocesses, and your skill talks to them via HTTP.

### Directory Structure

```
skills/my-gen-skill/
  SKILL.md                  Procedure the agent follows
  model_spec.json           Which model weights to download
  host_adapter.py           Host-side plugin that loads and runs the model
  scripts/
    prepare_model.py        Docker-side script → POSTs /models/prepare
    generate.py             Docker-side script → POSTs /infer
```

Three moving parts: the skill procedure (`SKILL.md`), the model spec (`model_spec.json`), and the host adapter (`host_adapter.py`). The scripts under `scripts/` are thin HTTP clients.

### model_spec.json

Describes what `helix model download` should fetch. This file is *only* about model weights — runtime Python code should come with a pip package or be fetched by the adapter itself (see below).

```json
{
  "backend": "mlx",
  "source": {"repo_id": "author/model-name"},
  "download_manifest": {
    "include": ["*.safetensors", "*.json"],
    "exclude": [],
    "required": [
      "config.json",
      "model.safetensors"
    ]
  }
}
```

**Important for new users — where the weights come from:**

OpenHelix always downloads model weights from **[HuggingFace Hub](https://huggingface.co)**. There is currently no other supported source: you cannot point `repo_id` at a direct URL, an S3 bucket, a local directory, or a different model hub like ModelScope. Under the hood, the runtime shells out to the `huggingface_hub` CLI (`hf download <repo_id> --local-dir ...`), which is what understands how to resolve the slug, fetch files, and cache them.

The `repo_id` field is therefore a **HuggingFace Hub repo slug** — the same `author/model-name` string you'd see in the URL bar at huggingface.co. For example, `notapalindrome/ltx23-mlx-av-q4` corresponds to `https://huggingface.co/notapalindrome/ltx23-mlx-av-q4`. If the model is gated or private, set `HF_TOKEN=hf_xxx...` in your shell before running `helix model download --skill my-gen-skill`.

The `download_manifest` gives you fine-grained control over what gets fetched:

- `include` — glob patterns of files to fetch (e.g. `*.safetensors`, `*.json`). If non-empty and `exclude` is empty, only these patterns are downloaded.
- `exclude` — glob patterns of files to skip (e.g. `transformer-dev.safetensors` if you only want the distilled variant).
- `required` — glob patterns that must resolve to at least one file after the download completes, otherwise the prepare phase fails with "prepared files are incomplete". This is your safety net against partial downloads.

**Choosing a backend name** decides which Python venv your skill uses:

| Backend | When to use |
|---|---|
| `mlx` | Standard MLX-native skill, shares venv with `generate-image` and `generate-video` |
| `pytorch` | Standard PyTorch skill, shares venv with `generate-audio` |
| *custom name* (e.g. `my-skill`) | Isolate your deps — get a fresh per-backend venv, safe from version conflicts with other skills |

Use a custom backend name when your pip deps conflict with an existing backend (e.g. your code needs `numpy<2` but another skill needs `numpy>=2`). Otherwise share — it saves ~5GB of disk per extra venv.

### host_adapter.py

The adapter is a host-side Python plugin that runs inside a dedicated worker subprocess (one per active skill). It must export `create_adapter(**kwargs)` that returns a subclass of `_BaseBackend`. Two real patterns in the codebase:

#### Pattern A: In-process backend (keep the model in memory)

Used by `generate-image`. Best when your model takes seconds-to-minutes to load and is called repeatedly during a session. The worker subprocess keeps the loaded pipeline in memory and reuses it across `/infer` calls. Memory is released when the worker is evicted (10 min idle timeout by default) or on `helix stop local-model-service`.

```python
from helix.runtime.local_model_service.adapters import _BaseBackend
from helix.runtime.local_model_service.helpers import (
    _ensure_worker_dependencies,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

_DEPENDENCIES = (
    "mlx>=0.20.0",
    "numpy",
    "pillow",
    # ... whatever your model runner needs
)


class _MySkillBackend(_BaseBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = None

    def _load(self):
        _ensure_worker_dependencies(self.python_bin, _DEPENDENCIES)
        from my_model_lib import MyPipeline   # your runner, installed into the venv
        self.pipeline = MyPipeline(model_path=str(self.model_root))

    def handle(self, payload):
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        workspace_root = _resolve_service_workspace_root(payload)
        output_path = _resolve_workspace_path(
            workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False,
        )
        try:
            if self.pipeline is None:
                self._load()
            image = self.pipeline.generate(prompt=prompt)
            image.save(output_path)
        except Exception as exc:
            return self._error(error_code="generation_runtime_error", message=str(exc))
        rel = str(output_path.relative_to(workspace_root))
        return self._ok(outputs={"output_path": rel}, message=f"generated at {rel}")


def create_adapter(**kwargs):
    return _MySkillBackend(**kwargs)
```

Key pieces:
- `_DEPENDENCIES` lists the pip packages the runner needs. `_ensure_worker_dependencies` pip-installs them into the backend's venv on first load.
- `self.model_root` is the directory where `helix model download` placed the weights.
- `self.python_bin` is the venv's Python binary.
- `_request_inputs(payload)` extracts the `inputs` dict from the `/infer` request. `_resolve_workspace_path(...)` safely resolves and validates paths that the Docker-side script passed through.
- Return shape uses `self._ok(...)` or `self._error(...)` — these build the JSON the coordinator sends back.

#### Pattern B: Subprocess-per-call backend (shell out to a CLI)

Used by `generate-video`. Best when the upstream runner ships a nice command-line interface (e.g. `python -m mlx_video.generate_av --prompt ... --output-path ...`), or when you want each inference to release its memory immediately on completion rather than holding it across calls.

```python
import os
import subprocess
from helix.runtime.local_model_service.adapters import _BaseBackend
from helix.runtime.local_model_service.helpers import (
    _ensure_worker_dependencies,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

_DEPENDENCIES = ("mlx-video-with-audio",)


class _MyCliBackend(_BaseBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ready = False

    def _load(self):
        _ensure_worker_dependencies(self.python_bin, _DEPENDENCIES)
        self._ready = True

    def handle(self, payload):
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        workspace_root = _resolve_service_workspace_root(payload)
        output_path = _resolve_workspace_path(
            workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False,
        )
        try:
            if not self._ready:
                self._load()
            cmd = [
                str(self.python_bin), "-m", "my_runner.generate",
                "--prompt", prompt,
                "--model-repo", str(self.model_root),
                "--output-path", str(output_path),
            ]
            env = os.environ.copy()
            completed = subprocess.run(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout or "").strip())
        except Exception as exc:
            return self._error(error_code="runtime_error", message=str(exc))
        rel = str(output_path.relative_to(workspace_root))
        return self._ok(outputs={"output_path": rel}, message=f"generated at {rel}")


def create_adapter(**kwargs):
    return _MyCliBackend(**kwargs)
```

Each `/infer` call spawns a fresh child process, loads the model, runs inference, writes the output, and exits. Memory is released immediately after the subprocess dies, without waiting for idle eviction.

**When to pick which:**

| | Pattern A (in-process) | Pattern B (subprocess) |
|---|---|---|
| Model load time | Amortized across calls (cheap) | Paid every call (expensive) |
| Peak memory during idle | High (model resident) | None |
| Complexity | Must integrate with a Python API | Just shells out |
| Crash recovery | Kills the whole worker | Kills only the child |

If your upstream ships a good Python API, prefer **A**. If it ships a CLI or if you care about aggressive memory release, prefer **B**.

### Upstream Code That Isn't a Pip Package

Sometimes the model you want to run was published as loose Python files on GitHub without a `setup.py` or `pyproject.toml`. You have three options:

1. **Install from git** — if the repo is installable but just isn't on PyPI, add `"git+https://github.com/author/repo@<commit>"` to `_DEPENDENCIES`. Pip will clone and install it into the venv.
2. **Vendor it into your repo** — copy the files under `skills/my-gen-skill/runner/` and `import` them directly. Simple but you take over maintenance.
3. **Fetch lazily on first load** — download individual files in `_load()` to a pinned-commit directory. This is what `generate-image` does for the MLX Z-Image runner:

   ```python
   _RUNNER_REPO = "https://raw.githubusercontent.com/author/repo"
   _RUNNER_COMMIT = "abc123..."
   _RUNNER_FILES = ("pipeline.py", "model.py", "utils.py")

   def _ensure_runner_sources(self):
       runner_root = self.model_root / "_runner" / _RUNNER_COMMIT
       runner_root.mkdir(parents=True, exist_ok=True)
       for filename in _RUNNER_FILES:
           target = runner_root / filename
           if target.exists():
               continue
           url = f"{_RUNNER_REPO}/{_RUNNER_COMMIT}/{filename}"
           with urllib.request.urlopen(url, timeout=60) as resp:
               target.write_bytes(resp.read())
       return runner_root
   ```

   Then in `_load()`: `sys.path.insert(0, str(self._ensure_runner_sources()))` and `import pipeline`.

The co-located `_runner/<commit>/` subfolder lives next to the model weights, so "download the model" semantically stays crisp — `helix model download` only fetches weights, and the adapter handles its own runtime code.

### Skill Scripts (the Docker-side half)

`scripts/prepare_model.py` and `scripts/generate.py` run inside the Docker sandbox. They're thin HTTP clients to the coordinator:

- Read `HELIX_LOCAL_MODEL_SERVICE_URL` and `HELIX_LOCAL_MODEL_SERVICE_TOKEN` from environment.
- POST the model_spec and inputs to `/models/prepare` or `/infer` respectively.
- Include `"skill_name"` in every request payload — the coordinator uses it to route to your adapter.
- Include `"task_type"` (e.g. `"text_to_image"`) in `/infer` requests.
- Include `"workspace_root": str(Path.cwd().resolve())` in **both** `/models/prepare` and `/infer` requests. The coordinator is a workspace-agnostic global service: it derives `skills_root` from each request's `workspace_root` so a single running coordinator can serve clients from any workspace. The Docker sandbox bind-mounts the workspace at the same host path, so `Path.cwd().resolve()` inside the container is already a path the host-side coordinator can resolve directly.
- Print exactly one JSON object to stdout as the final result — the agent reads stdout to know what happened.

The built-in `generate-image` and `generate-video` skills both have working script examples you can copy and adapt.

### Setup Walkthrough

```bash
# 1. Create the skill directory and files (see structure above).

# 2. Download the model weights (adapter, venv, and weights all get created).
helix model download --skill my-gen-skill

# 3. Start the coordinator so the agent can use the skill.
helix start local-model-service
```

On the first `/infer` call, the adapter's `_load()` runs: pip-installs `_DEPENDENCIES`, optionally fetches runner files, and loads the model. Subsequent calls reuse the warm worker.

## Best Practices

- Default to no-script skills.
- User skills go under `skills/`, not `skills/builtin_skills/`.
- Use kebab-case directory names.
- Reference other skills by name instead of duplicating logic.
- Include concrete exec action JSON in procedures.
