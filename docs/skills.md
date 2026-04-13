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
| `generate-image` | multi-script | Generate images (MLX backend) |
| `generate-audio` | multi-script | Generate audio (PyTorch backend) |
| `generate-video` | multi-script | Generate video (PyTorch backend) |

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

Generative skills use the local model service to run ML models on the host machine.

### Directory Structure

```
skills/my-gen-skill/
  SKILL.md                  Procedure
  model_spec.json           What to download
  host_adapter.py           How to load and run the model
  scripts/
    prepare_model.py        Calls /models/prepare
    generate.py             Calls /infer
```

### model_spec.json

Describes what `helix model download` should fetch:

```json
{
  "backend": "pytorch",
  "source": {"repo_id": "org/model-name"},
  "download_manifest": {
    "include": ["*.safetensors", "config.json"],
    "exclude": [],
    "required": ["config.json", "*.safetensors"]
  }
}
```

Optional `sources` field for runtime Python files that need downloading from GitHub.

### host_adapter.py

The host-side plugin. Runs on the host (not in Docker). Must export `create_adapter(**kwargs)`:

```python
from helix.runtime.local_model_service.adapters import _BaseBackend

class MyBackend(_BaseBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def _load(self):
        # Load model into self.model using self.model_root
        ...

    def handle(self, payload):
        if self.model is None:
            self._load()
        # Run inference, save output
        ...
        return self._ok(outputs={"output_path": rel_path}, message="done")

def create_adapter(**kwargs):
    return MyBackend(**kwargs)
```

Adapters are registered by **skill directory name** — no FAMILY or BACKEND exports needed.

### Skill Scripts

Scripts run in Docker and call the coordinator via HTTP. Key points:

- Read `HELIX_LOCAL_MODEL_SERVICE_URL` and `HELIX_LOCAL_MODEL_SERVICE_TOKEN` from env
- Include `"skill_name"` in every request payload
- Print one JSON object to stdout as the result

### Setup

```bash
helix model download --spec skills/my-gen-skill/model_spec.json
helix start local-model-service --workspace .
```

## Best Practices

- Default to no-script skills.
- User skills go under `skills/`, not `skills/builtin_skills/`.
- Use kebab-case directory names.
- Reference other skills by name instead of duplicating logic.
- Include concrete exec action JSON in procedures.
