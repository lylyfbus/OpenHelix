---
name: Load Knowledge Docs
handler: scripts/load_knowledge_docs.py
description: Load related runtime knowledge documents into workflow history for reasoning.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill to load relevant knowledge documents from the runtime `knowledge/` folder before reasoning or answering.

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. Use `stderr` only for unexpected failures.
3. Keep output structured and concise.

# Script

- Path: `skills/all-agents/load-knowledge-docs/scripts/load_knowledge_docs.py`
- Executor: `python`
- Default workspace: current runtime working directory (`.`)

# Preferred Action Input Template

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/load-knowledge-docs/scripts/load_knowledge_docs.py",
  "script_args": [
    "--doc-id", "doc_abc123",
    "--doc-path", "knowledge/docs/doc_xyz789.md",
    "--max-docs", "4",
    "--max-chars-per-doc", "2200"
  ]
}
```

Path-only example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/load-knowledge-docs/scripts/load_knowledge_docs.py",
  "script_args": [
    "--doc-path", "knowledge/docs/doc_abc123.md",
    "--doc-path", "knowledge/docs/doc_def456.md",
    "--max-chars-per-doc", "3000"
  ]
}
```

Use `AVAILABLE KNOWLEDGE` metadata in system prompt to choose `doc-id` or `doc-path` first, then load selected docs with this skill.

# Output JSON Shape

```json
{
  "executed_skill": "load-knowledge-docs",
  "status": "ok|error",
  "knowledge_context": "..."
}
```
