---
name: Create Generative Skill
description: Create a new skill that runs an ML model via the local model service (image, audio, video, or any other heavy-ML task).
---

# Purpose

Use this skill to author a new **generative skill** — one that runs an ML model in a dedicated long-lived worker managed by the local model service, separate from the ordinary host-shell exec sandbox. The built-in examples are `generate-image`, `generate-audio`, and `generate-video`.

# When To Use

- The task needs downloaded model weights (from HuggingFace Hub).
- The task needs GPU/MPS access or heavy Python dependencies (MLX, PyTorch, diffusion pipelines) that don't belong in the exec sandbox.
- You want the model kept warm across multiple calls in a session (or want each call to release memory immediately via a subprocess).
- You explicitly do NOT want this skill if the task can be done with plain code in the exec sandbox — use `create-skill` for that.

# Directory Structure

A generative skill has four required files under `skills/{skill-name}/`:

```
skills/my-gen-skill/
  SKILL.md                 Procedure the agent follows (like any skill)
  model_spec.json          Which model weights to download (HF repo + manifest)
  host_adapter.py          Host-side worker: loads the model, serves /infer
  scripts/
    prepare_model.py       Sandbox-side HTTP client → POSTs /models/prepare
    generate_{task}.py     Sandbox-side HTTP client → POSTs /infer
```

Global, machine-wide: downloaded weights land in `~/.helix/services/local-model-service/models/{repo_id}/`, and per-backend Python venvs in `~/.helix/services/local-model-service/venvs/{backend}/`. You never touch these directly — `helix model download` and the coordinator manage them.

# model_spec.json Contract

Declares which HuggingFace repo to fetch and which files matter. Weights are the only thing this file downloads; Python code that runs the model comes from either a pip package or the adapter itself (see "Upstream Code Options" below).

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

**Fields:**

- `backend` (string, required): selects the Python venv this skill's adapter runs in. See the backend table below.
- `source.repo_id` (string, required): HuggingFace Hub slug, e.g. `author/model-name` — matches the URL path at huggingface.co. Only HuggingFace Hub is supported; you cannot point at direct URLs, S3, local paths, or ModelScope. Gated/private repos need `HF_TOKEN=hf_xxx` in the shell launching `helix model download`.
- `download_manifest.include` (list of globs): files to fetch. If non-empty and `exclude` is empty, only these are downloaded.
- `download_manifest.exclude` (list of globs): files to skip (e.g. bulky dev variants you don't need).
- `download_manifest.required` (list of globs): every pattern must match at least one file after download, otherwise prepare fails with "prepared files are incomplete". This is the safety net against partial downloads.

**Backend selection table:**

| Backend | When to use |
|---|---|
| `mlx` | MLX-native skill on Apple Silicon; shares venv with `generate-image` and `generate-video`. |
| `pytorch` | PyTorch-based skill; shares venv with `generate-audio`. |
| *custom name* (e.g. `my-skill`) | Dependency isolation — get a fresh per-backend venv, safe from version conflicts with other skills. Each custom backend costs ~5GB of disk for a separate venv. |

Share when you can. Only pick a custom backend name when your pip deps strictly conflict with an existing backend (e.g. your runner needs `numpy<2` but another skill needs `numpy>=2`).

# host_adapter.py Contract

Runs inside the coordinator's worker subprocess, one worker per active skill. Must export a `create_adapter(**kwargs)` factory that returns a subclass of `_BaseBackend` (from `helix.runtime.local_model_service.adapters`). The subclass implements:

- `_load(self)` — called once before the first `handle()`; pip-installs deps, loads the model. Keep expensive setup here.
- `handle(self, payload) -> dict` — called per `/infer` request; returns `self._ok(outputs={...}, message=str)` on success or `self._error(error_code=str, message=str)` on failure.

Instance attributes provided by `_BaseBackend`:
- `self.model_root` — directory where `helix model download` placed the weights.
- `self.python_bin` — the venv's Python binary (useful for subprocess-based adapters).

Two patterns to choose from:

## Pattern A — in-process (keep the model warm)

Best when model load takes seconds-to-minutes and the user makes many calls in a session (e.g. text-to-image). The worker subprocess keeps the pipeline in memory between calls and releases it only on idle eviction (default 10 min) or `helix stop local-model-service`.

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
    # ... whatever your runner needs
)


class _MySkillBackend(_BaseBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = None

    def _load(self):
        _ensure_worker_dependencies(self.python_bin, _DEPENDENCIES)
        from my_model_lib import MyPipeline   # installed into the venv
        self.pipeline = MyPipeline(model_path=str(self.model_root))

    def handle(self, payload):
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        workspace_root = _resolve_service_workspace_root(payload)
        output_path = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
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

Key helpers:
- `_ensure_worker_dependencies(self.python_bin, _DEPENDENCIES)` — pip-installs into the backend's venv on first load. Idempotent: subsequent loads are fast no-ops.
- `_request_inputs(payload)` — extracts the `inputs` dict from the request.
- `_resolve_service_workspace_root(payload)` — returns the absolute workspace path the caller sent. The coordinator is workspace-agnostic, so every call carries its own root.
- `_resolve_workspace_path(root, relative, expect_exists=False)` — safely resolves and validates a path, preventing escapes outside the workspace.

## Pattern B — subprocess per call (shell out to a CLI)

Best when the upstream runner ships a good CLI (`python -m foo.generate ...`), or when you want each inference to release memory immediately on completion.

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

_DEPENDENCIES = ("my-runner-package",)


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
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
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
            completed = subprocess.run(
                cmd,
                env=os.environ.copy(),
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

Each `/infer` call spawns a fresh child process, loads the model, writes output, exits. Memory is reclaimed immediately — no idle-eviction wait.

## Choosing between Pattern A and Pattern B

| | Pattern A (in-process) | Pattern B (subprocess) |
|---|---|---|
| Model load time | Amortized across calls (cheap) | Paid every call (expensive) |
| Peak memory during idle | High (model resident) | None |
| Integration complexity | Must use a Python API | Just shells out to a CLI |
| Crash recovery | Kills the whole worker | Kills only the child |

Prefer Pattern A if the upstream has a clean Python API. Prefer Pattern B if it ships a CLI or if aggressive memory release matters.

# Upstream Code Options

Sometimes the model's runner isn't on PyPI. Three options:

1. **Install from git** — add `"git+https://github.com/author/repo@<commit>"` to `_DEPENDENCIES`. Pip clones and installs it into the venv. Works when the repo has `setup.py` / `pyproject.toml`.
2. **Vendor into the skill** — copy the runner's Python files under `skills/my-gen-skill/runner/` and import them directly. Simple, but you own maintenance.
3. **Fetch lazily on first `_load()`** — download files to a pinned-commit directory co-located with the weights. Keeps `model_spec.json` strictly about weights, and the adapter handles its own runtime code:

   ```python
   import urllib.request

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

   Then in `_load()`: `sys.path.insert(0, str(self._ensure_runner_sources())); import pipeline`.

# Scripts Contract

`scripts/prepare_model.py` and `scripts/generate_{task}.py` run in the host-shell exec sandbox. They're thin HTTP clients to the coordinator. Every request must:

- Read `HELIX_LOCAL_MODEL_SERVICE_URL` and `HELIX_LOCAL_MODEL_SERVICE_TOKEN` from env. The coordinator injects these; the skill never hardcodes them.
- POST to `{URL}/models/prepare` or `{URL}/infer` with `Authorization: Bearer {TOKEN}` and `Content-Type: application/json`.
- Include these payload keys in **every** request:
  - `skill_name` (string) — matches the skill directory name; coordinator uses this to route to your adapter.
  - `model_spec` (object) — the full JSON loaded from the skill's `model_spec.json`.
  - `workspace_root` (string) — `str(Path.cwd().resolve())`. The coordinator is workspace-agnostic and uses this to locate the skill's `host_adapter.py`.
- For `/infer` calls, also include:
  - `task_type` (string) — e.g. `"text_to_image"`. Purely for logging/categorization today; still required.
  - `inputs` (object) — task-specific keys consumed by `handle()` in the adapter (e.g. `{"prompt": "...", "output_path": "..."}`).
- Print exactly **one JSON object to stdout** as the final result. The agent reads stdout as the execution result.
- On failure, exit with non-zero status and write the error summary to stderr.

# Procedure

Create the skill end-to-end. Follow steps in order — skipping the model download or the coordinator restart are the two most common breakage modes.

## Step 1: Review an existing generative skill

Pick the closest match and read every file:

```json
{
  "job_name": "read-reference-skill",
  "code_type": "bash",
  "script": "ls skills/builtin_skills/generate-image && echo '--- SKILL.md ---' && cat skills/builtin_skills/generate-image/SKILL.md && echo '--- model_spec.json ---' && cat skills/builtin_skills/generate-image/model_spec.json && echo '--- host_adapter.py ---' && cat skills/builtin_skills/generate-image/host_adapter.py"
}
```

(Replace `generate-image` with `generate-audio` for PyTorch-based reference, or `generate-video` for a Pattern B subprocess reference.)

## Step 2: Create the skill directory

```json
{
  "job_name": "create-gen-skill-dir",
  "code_type": "bash",
  "script": "mkdir -p skills/{skill-name}/scripts"
}
```

User-created skills go under `skills/`, not `skills/builtin_skills/`.

## Step 3: Write the four required files

Write each file with `Path.write_text`. Example for one file — repeat the pattern for all four:

```json
{
  "job_name": "write-model-spec",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('skills/{skill-name}/model_spec.json')\npath.write_text('''{model_spec_json_content}''', encoding='utf-8')\nprint(f'wrote {path}')"
}
```

Do the same for `SKILL.md`, `host_adapter.py`, `scripts/prepare_model.py`, `scripts/generate_{task}.py`.

## Step 4: Download the model weights

This validates `model_spec.json`, provisions the per-backend venv at `~/.helix/services/local-model-service/venvs/{backend}/`, and fetches weights to `~/.helix/services/local-model-service/models/{repo_id}/`.

```json
{
  "job_name": "download-gen-skill-model",
  "code_type": "bash",
  "script": "helix model download --skill {skill-name}"
}
```

If the HF repo is gated, `HF_TOKEN=hf_xxx` must already be in the shell environment. Any error here means `model_spec.json` is malformed or the required files didn't match.

## Step 5: Restart the local model service

The coordinator discovers skills on startup; a new skill isn't served until restart.

```json
{
  "job_name": "restart-local-model-service",
  "code_type": "bash",
  "script": "helix stop local-model-service && helix start local-model-service"
}
```

## Step 6: Smoke-test

Run `prepare_model.py` (should complete without error on subsequent calls — the first call triggers the adapter's `_load()`), then one inference call via `generate_{task}.py`. Both should print exactly one JSON object with `status: "ok"`.

```json
{
  "job_name": "smoke-test-gen-skill",
  "code_type": "bash",
  "script": "python skills/{skill-name}/scripts/prepare_model.py && python skills/{skill-name}/scripts/generate_{task}.py --prompt 'hello world' --output-path out/test-output.{ext}"
}
```

# Rules

- Use lowercase kebab-case for the skill directory name.
- User-created generative skills go under `skills/`, not `skills/builtin_skills/`.
- Never edit a file under `skills/builtin_skills/` directly — that tree is resynced from the package on every startup, so your edits will be erased. To customize a built-in skill, copy its directory up one level into `skills/{new-name}/` and edit the copy.
- Frontmatter must have exactly `name` and `description` — nothing else.
- `model_spec.json` must list every file needed in `download_manifest.required` — partial downloads are the most common deployment bug.
- `host_adapter.py` must subclass `_BaseBackend`, export `create_adapter(**kwargs)`, and return `_ok(...)` or `_error(...)` from `handle()`. Any other return shape is an interface violation.
- Sandbox scripts must print exactly one JSON object to stdout on success; stderr is for failures only.
- After creating the skill, always run `helix model download --skill {name}` then restart `local-model-service` before using it.
