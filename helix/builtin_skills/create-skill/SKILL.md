---
name: Create Skill
description: Create a new skill with a SKILL.md and optional scripts.
---

# Purpose

Use this skill to create a new reusable skill under the workspace's skills/ directory.

# When To Use

- When a task pattern is worth reusing across sessions.
- When the user explicitly asks to create a skill.
- When you notice a repeated workflow that would benefit from a standardized procedure.

# Skill Structure

A skill is a directory containing at minimum a SKILL.md file:

```
skills/
  my-new-skill/
    SKILL.md                  <- Required: procedure and metadata
    scripts/                  <- Optional: pre-built scripts (if needed)
      my_script.py
```

# SKILL.md Specification

## Frontmatter

Every SKILL.md must begin with exactly these two fields:

```
---
name: Human-Readable Skill Name
description: One-line description of what the skill does.
---
```

## Required Body Sections

- `# Purpose` — what the skill does
- `# When To Use` — when to use it and when to skip it
- `# Procedure` — step-by-step instructions with exec action examples
- `# Rules` — constraints and guidelines

## Optional Body Sections

- `# Action Input Templates` — for skills with scripts
- `# Skill Dependencies` — when referencing other skills

# Script Modes

Choose based on **step complexity**, not step count:

- **No scripts**: Use when every step is simple file I/O or standard commands. The agent follows the SKILL.md procedure directly. Most skills should be this type.
- **Single script**: Use when one step is complex enough that writing the code fresh each time would be error-prone (e.g. API calls with auth/retry, binary format parsing).
- **Multiple scripts**: Use when multiple steps are independently complex (e.g. generate-image with prepare_model.py + generate_image.py).

# Procedure

## Step 1: Review Existing Skills as Examples

Before creating a new skill, read an existing skill for reference.

No-script example (simple procedure):
```json
{
  "job_name": "read-example-no-script",
  "code_type": "bash",
  "script": "cat skills/builtin_skills/retrieve-knowledge/SKILL.md"
}
```

Script-based example (with pre-built scripts):
```json
{
  "job_name": "read-example-with-scripts",
  "code_type": "bash",
  "script": "cat skills/builtin_skills/search-online-context/SKILL.md"
}
```

## Step 2: Create the Skill Directory

```json
{
  "job_name": "create-skill-dir",
  "code_type": "bash",
  "script": "mkdir -p skills/{skill-name}"
}
```

User-created skills go directly under `skills/`, not under `skills/builtin_skills/`.

## Step 3: Write the SKILL.md

```json
{
  "job_name": "write-skill-md",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('skills/{skill-name}/SKILL.md')\npath.write_text('''{skill_md_content}''', encoding='utf-8')\nprint(f'created {path}')"
}
```

## Step 4: Add Scripts (if needed)

Only if the skill requires pre-built scripts:

```json
{
  "job_name": "write-skill-script",
  "code_type": "python",
  "script": "from pathlib import Path\nscripts_dir = Path('skills/{skill-name}/scripts')\nscripts_dir.mkdir(parents=True, exist_ok=True)\nscript = scripts_dir / '{script_name}.py'\nscript.write_text('''{script_content}''', encoding='utf-8')\nprint(f'created {script}')"
}
```

## Step 5: Verify

```json
{
  "job_name": "verify-skill",
  "code_type": "bash",
  "script": "cat skills/{skill-name}/SKILL.md && echo '---' && find skills/{skill-name} -type f"
}
```

# Rules

- Use lowercase kebab-case for skill directory names.
- User-created skills go under `skills/`, not `skills/builtin_skills/`.
- Frontmatter must have exactly `name` and `description` — nothing else.
- Default to no-script skills. Only add scripts when step complexity justifies it.
- Reference existing skills by name in the procedure instead of duplicating their logic.
- For scripts: stdout should produce clear execution evidence; stderr only for failures.
