---
name: Documentation Distillation
handler: scripts/documentation_distill.py
description: Distill useful task knowledge into standardized documents using runtime-managed persistence.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill to create or update reusable knowledge documents after meaningful task progress.

# Scope

This skill focuses only on document create and update.
Knowledge retrieval/search should be handled by a separate skill.

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. Use `stderr` only for unexpected failures.
3. Keep `stdout` concise so `runtime>` history stays readable.

# Storage

- Docs are persisted under runtime workspace `knowledge/docs/`.
- Index is persisted under `knowledge/index/catalog.json`.
- Persistence is handled directly by the skill script (runtime-local, no dev-package dependency).

# Actions

1. `create`
- Create a new knowledge doc.
- Requires at least `title` and one meaningful content field.

2. `update`
- Update an existing doc by `doc_id`.
- If `--body` is provided, it replaces doc body.
- Otherwise, script appends a structured update block.

# Preferred Action Input Template

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/documentation-distillation/scripts/documentation_distill.py",
  "script_args": [
    "--action", "create",
    "--title", "LM Studio connection troubleshooting",
    "--problem", "Connection refused from local endpoint",
    "--what-was-done", "Verified base URL and server state",
    "--reusable-pattern", "Check server up + endpoint + model mapping",
    "--caveats", "Local server must be running before request",
    "--tags", "lmstudio,network,debug"
  ]
}
```

Update example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/documentation-distillation/scripts/documentation_distill.py",
  "script_args": [
    "--action", "update",
    "--doc-id", "doc_abc123",
    "--what-was-done", "Added additional retry diagnostics",
    "--caveats", "Retry logic still depends on provider timeout"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "documentation-distillation",
  "status": "ok|error",
  "doc_path": "..."
}
```
