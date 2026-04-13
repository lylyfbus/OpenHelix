---
name: Update Skill
description: Update an existing skill's SKILL.md, scripts, or configuration.
---

# Purpose

Use this skill to improve or modify an existing skill — update its procedure, fix its scripts, or refine its description.

# When To Use

- When a skill's procedure needs improvement based on runtime experience.
- When a skill's scripts have bugs or need enhancements.
- When the user asks to update or improve an existing skill.

# Procedure

## Step 1: Read the Current Skill

```json
{
  "job_name": "read-existing-skill",
  "code_type": "bash",
  "script": "cat {workspace}/skills/{path}/SKILL.md"
}
```

## Step 2: List Skill Contents

```json
{
  "job_name": "list-skill-files",
  "code_type": "bash",
  "script": "find {workspace}/skills/{path} -type f"
}
```

## Step 3: Make Changes

Update the SKILL.md or scripts as needed:

```json
{
  "job_name": "update-skill-md",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('{workspace}/skills/{path}/SKILL.md')\npath.write_text('''{updated_content}''', encoding='utf-8')\nprint(f'updated {path}')"
}
```

## Step 4: Verify

Read the updated file to confirm the changes are correct:

```json
{
  "job_name": "verify-skill-update",
  "code_type": "bash",
  "script": "cat {workspace}/skills/{path}/SKILL.md"
}
```

# Rules

- Always read the current skill before modifying it.
- Keep the frontmatter format: only name and description fields.
- Preserve the existing procedure structure unless the update requires restructuring.
- If adding or modifying scripts, follow the script mode guidelines from the create-skill skill.
- Test any script changes by running them after the update.
