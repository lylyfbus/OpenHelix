---
name: Skill Authorization
handler: scripts/skill_authorization.py
description: Authorize complicated skill work with script-first execution and structured runtime evidence.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill when skill creation/update work is complex or uncertain and you want deterministic, script-first execution.

# Why Script-First

For uncertain skill tasks, run the helper script immediately so runtime history gets clean, structured evidence in `runtime> stdout`.

# Runtime Log Contract

1. `stdout` must contain one final JSON object.
2. Reserve `stderr` for unexpected runtime failures only.
3. Keep JSON concise so `workflow_hist` remains readable.

# Helper Script

- Path: `skills/all-agents/skill-authorization/scripts/skill_authorization.py`
- Supports two actions:
  1. `inspect`: inspect existing skill package status.
  2. `scaffold`: create/update a minimal skill skeleton.

# Preferred Action Input Template

Use `code_type=python` with `script_path` and `script_args` array:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-authorization/scripts/skill_authorization.py",
  "script_args": [
    "--action", "inspect",
    "--skill-id", "search-online-context",
    "--scope", "all-agents"
  ]
}
```

Scaffold example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-authorization/scripts/skill_authorization.py",
  "script_args": [
    "--action", "scaffold",
    "--skill-id", "new-skill-id",
    "--scope", "all-agents",
    "--description", "One-line purpose of this skill"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "skill-authorization",
  "status": "ok|error",
  "skill_created/updated": "target-skill-id-or-empty"
}
```

# Notes

- Use lowercase hyphenated `skill_id`.
- Use this skill before manually drafting large uncertain skill content.
