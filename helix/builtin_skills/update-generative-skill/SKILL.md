---
name: Update Generative Skill
description: Modify an existing generative skill's model_spec.json, host_adapter.py, scripts, or procedure — including the re-download and coordinator-restart follow-up.
---

# Purpose

Use this skill to evolve an existing generative skill (one with `model_spec.json` + `host_adapter.py` alongside its `SKILL.md`). The follow-up actions required after an edit depend on which file changed — this skill walks through both the edit and the correct follow-up.

# When To Use

- The skill's model_spec needs updating (e.g. bumping to a new model revision, narrowing the file manifest, switching backend).
- The host adapter has a bug, needs new dependencies, or needs to accept new `inputs` keys.
- The sandbox-side scripts need new args or better error handling.
- The agent-facing `SKILL.md` procedure needs refinement.

If the skill is **not** generative (no `model_spec.json` / `host_adapter.py`), use `update-skill` instead.

# Change → Follow-up Matrix

Match what you edited to what has to be re-run afterwards:

| What changed | Re-run `helix model download` | Restart coordinator |
|---|---|---|
| `model_spec.json` (repo_id, backend, include/exclude/required) | yes | yes |
| `host_adapter.py` (deps, `_load`, `handle`) | no | yes |
| `scripts/*.py` (sandbox-side HTTP clients) | no | no |
| `SKILL.md` (procedure, rules, inputs) | no | no |

The rules come from where each file lives at runtime:

- **Weights** live under `~/.helix/services/local-model-service/models/`. A `model_spec.json` edit doesn't move them; only `helix model download` does. A backend switch also rebuilds the venv.
- **Host adapter** runs inside a long-lived worker subprocess managed by the coordinator. Code changes are only picked up when the coordinator restarts (which recycles the workers).
- **Sandbox scripts** run in a fresh host-shell process per `exec`. Edits take effect on the very next call automatically.

# Contract Recap (so you can edit safely)

Compact reminder of what each file must satisfy. For full authoring details see `create-generative-skill`.

**`model_spec.json`**:
```json
{
  "backend": "mlx|pytorch|<custom>",
  "source": {"repo_id": "author/model-name"},
  "download_manifest": {
    "include": ["*.safetensors", "*.json"],
    "exclude": [],
    "required": ["config.json", "model.safetensors"]
  }
}
```
- `required` globs must all resolve to ≥1 file after download, or prepare fails.
- Only HuggingFace Hub is a supported source — `repo_id` is the hub slug.
- Switching `backend` changes which venv is used (each custom backend gets its own).

**`host_adapter.py`** exports `create_adapter(**kwargs)` returning a `_BaseBackend` subclass. That class must implement:
- `_load(self)` — called once before the first `handle()`; install deps via `_ensure_worker_dependencies`, load the model.
- `handle(self, payload) -> dict` — returns `self._ok(outputs={...}, message=...)` or `self._error(error_code=..., message=...)`.

Useful helpers from `helix.runtime.local_model_service.helpers`: `_request_inputs(payload)`, `_resolve_service_workspace_root(payload)`, `_resolve_workspace_path(root, rel, expect_exists=False)`.

**Sandbox scripts** (`prepare_model.py`, `generate_{task}.py`) must:
- Read `HELIX_LOCAL_MODEL_SERVICE_URL` / `HELIX_LOCAL_MODEL_SERVICE_TOKEN` from env.
- POST to `/models/prepare` or `/infer` with `Authorization: Bearer {TOKEN}`.
- Include `skill_name`, `model_spec`, `workspace_root` in every request; `/infer` also needs `task_type` and `inputs`.
- Print exactly one JSON object to stdout as the final result.

# Procedure

## Step 1: Read the current state of the skill

```json
{
  "job_name": "read-current-gen-skill",
  "code_type": "bash",
  "script": "ls skills/{path} && echo '--- SKILL.md ---' && cat skills/{path}/SKILL.md && echo '--- model_spec.json ---' && cat skills/{path}/model_spec.json && echo '--- host_adapter.py ---' && cat skills/{path}/host_adapter.py"
}
```

(Adjust the path for skill location; for built-in skills copy to `skills/{new-name}/` first — never edit under `skills/builtin_skills/` in place.)

## Step 2: Make the edit

Rewrite the target file(s) with `Path.write_text`:

```json
{
  "job_name": "edit-gen-skill-file",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('skills/{path}/{file}')\npath.write_text('''{updated_content}''', encoding='utf-8')\nprint(f'updated {path}')"
}
```

## Step 3a: Re-download weights (only if `model_spec.json` changed)

```json
{
  "job_name": "redownload-gen-skill-model",
  "code_type": "bash",
  "script": "helix model download --skill {skill-name}"
}
```

Skip this step for any other edit.

## Step 3b: Restart the coordinator (only if `model_spec.json` or `host_adapter.py` changed)

```json
{
  "job_name": "restart-local-model-service",
  "code_type": "bash",
  "script": "helix stop local-model-service && helix start local-model-service"
}
```

Skip this step if only scripts or SKILL.md changed — they don't run in the worker process.

## Step 4: Smoke-test the change

Run `scripts/prepare_model.py` followed by one inference call via `scripts/generate_{task}.py`. Both should print a JSON object with `status: "ok"`.

```json
{
  "job_name": "smoke-test-gen-skill",
  "code_type": "bash",
  "script": "python skills/{path}/scripts/prepare_model.py && python skills/{path}/scripts/generate_{task}.py --prompt 'smoke test' --output-path out/smoke.{ext}"
}
```

If smoke fails after an adapter edit, the most common causes are: forgot to restart the coordinator (Step 3b), new `_DEPENDENCIES` entry that doesn't install cleanly in the backend venv, or a `handle()` exception that wasn't caught and converted to `_error(...)`.

# Rules

- Always read the skill's existing state before editing.
- Never edit a file under `skills/builtin_skills/` directly — that tree is resynced from the package on every startup. To customize a built-in generative skill, copy its directory into `skills/{new-name}/` and edit the copy.
- Keep the frontmatter format: only `name` and `description`.
- Consult the Change → Follow-up Matrix before reporting "done" — a correctly edited skill that wasn't followed by the right download/restart looks broken on the next call.
- `host_adapter.py` must still subclass `_BaseBackend`, export `create_adapter(**kwargs)`, and return `_ok(...)` or `_error(...)` after every edit. Any other return shape is an interface violation.
- If you edit `model_spec.json`'s `backend` to a new value, you're building a fresh venv on the next download — expect an extra ~5GB of disk and slower first install.
