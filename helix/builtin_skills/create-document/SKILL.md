---
name: Create Document
description: Create a new knowledge document in the hierarchical knowledge library.
---

# Purpose

Use this skill to persist useful task knowledge as structured documents in the workspace knowledge library. Documents are organized by category and subcategory.

# When To Use

- After meaningful task progress that produced reusable knowledge.
- When the user explicitly asks to document something.

# Knowledge Library Structure

```
knowledge/
  index.json                                    <- Global index (category/subcategory entries)
  {category}/
    {subcategory}/
      catalog.json                               <- Document metadata (title, summary, path)
      docs/
        {document-name}.md                       <- Actual documents
```

# Procedure: Create a New Document

## Step 1: Choose Category and Subcategory

Select an existing category/subcategory from the KNOWLEDGE INDEX, or define a new one if no existing entry fits.

## Step 2: Write the Document

Write a markdown file to the docs folder. Use a descriptive kebab-case filename.

```json
{
  "job_name": "write-knowledge-doc",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('knowledge/{category}/{subcategory}/docs/{name}.md')\npath.parent.mkdir(parents=True, exist_ok=True)\npath.write_text('''# {Title}\n\n## Problem\n\n{problem}\n\n## What Was Done\n\n{what_was_done}\n\n## Reusable Pattern\n\n{reusable_pattern}\n\n## Caveats\n\n{caveats}\n''', encoding='utf-8')\nprint(f'wrote {path}')"
}
```

## Step 3: Update the Catalog

Read the existing catalog, add the new entry, and write it back. Each entry has exactly three fields: title, summary, path.

```json
{
  "job_name": "update-knowledge-catalog",
  "code_type": "python",
  "script": "import json\nfrom pathlib import Path\ncatalog_path = Path('knowledge/{category}/{subcategory}/catalog.json')\ncatalog_path.parent.mkdir(parents=True, exist_ok=True)\ncatalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else []\ncatalog.append({\"title\": \"{Title}\", \"summary\": \"{summary}\", \"path\": \"knowledge/{category}/{subcategory}/docs/{name}.md\"})\ncatalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=True))\nprint(f'updated {catalog_path}')"
}
```

## Step 4: Update the Global Index

If this is a new category/subcategory, add it to index.json. Each entry has: category, subcategory, description, path.

```json
{
  "job_name": "update-knowledge-index",
  "code_type": "python",
  "script": "import json\nfrom pathlib import Path\nindex_path = Path('knowledge/index.json')\nindex_path.parent.mkdir(parents=True, exist_ok=True)\nindex = json.loads(index_path.read_text()) if index_path.exists() else []\nif not any(e.get('category') == '{category}' and e.get('subcategory') == '{subcategory}' for e in index):\n    index.append({\"category\": \"{category}\", \"subcategory\": \"{subcategory}\", \"description\": \"{description}\", \"path\": \"knowledge/{category}/{subcategory}\"})\n    index.sort(key=lambda e: (e['category'], e['subcategory']))\n    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=True))\n    print(f'added new entry to {index_path}')\nelse:\n    print('index entry already exists')"
}
```

# Mandatory Updates

Every document creation MUST update all three files:
1. The document itself (`docs/{name}.md`)
2. The subcategory catalog (`catalog.json`)
3. The global index (`index.json`) — add entry if subcategory is new

# Procedure: Update an Existing Document

1. Read the existing document.
2. Modify the content as needed.
3. Write the updated document back.
4. Update the catalog entry if the title or summary changed.

# Rules

- Use descriptive kebab-case filenames for documents (not UUIDs).
- Keep summaries concise (one sentence).
- Each catalog entry has exactly three fields: title, summary, path.
- Each index entry has exactly four fields: category, subcategory, description, path.
