---
name: Update Document
description: Update an existing knowledge document in the hierarchical knowledge library.
---

# Purpose

Use this skill to update an existing knowledge document with new findings, corrections, or additional content.

# When To Use

- When you have new information that supplements an existing document.
- When a previous document needs corrections or updates.
- When the user asks to update or revise documented knowledge.

# Procedure

## Step 1: Find the Document

Use the retrieve-knowledge skill to locate the document you want to update.

## Step 2: Read the Current Content

```json
{
  "job_name": "read-existing-doc",
  "code_type": "bash",
  "script": "cat {path_to_document}"
}
```

## Step 3: Update the Document

Write the updated content back to the same path:

```json
{
  "job_name": "update-knowledge-doc",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('{path_to_document}')\npath.write_text('''{updated_content}''', encoding='utf-8')\nprint(f'updated {path}')"
}
```

## Step 4: Update the Catalog

If the title or summary changed, update the corresponding catalog.json entry:

```json
{
  "job_name": "update-knowledge-catalog",
  "code_type": "python",
  "script": "import json\nfrom pathlib import Path\ncatalog_path = Path('knowledge/{category}/{subcategory}/catalog.json')\ncatalog = json.loads(catalog_path.read_text())\nfor entry in catalog:\n    if entry['path'] == '{path_to_document}':\n        entry['title'] = '{new_title}'\n        entry['summary'] = '{new_summary}'\n        break\ncatalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=True))\nprint(f'updated {catalog_path}')"
}
```

# Rules

- Always read the existing document before updating.
- Preserve existing content unless the update explicitly replaces it.
- Update the catalog entry if the title or summary changed.
- Do not move documents between categories — create a new one instead.
