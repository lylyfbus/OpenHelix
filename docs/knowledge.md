# Knowledge System

## Overview

The knowledge library is a hierarchical document store for reusable, long-term knowledge. Unlike conversation history (session-scoped), knowledge persists across all sessions in the workspace.

## Structure

```
knowledge/
  index.json                              Global classification index
  {category}/{subcategory}/
    catalog.json                          Document metadata
    docs/
      {document}.md                       Documents
```

### index.json

```json
[
  {
    "category": "computer-science",
    "subcategory": "programming",
    "description": "Programming languages, patterns, and practices",
    "path": "knowledge/computer-science/programming"
  }
]
```

### catalog.json

```json
[
  {
    "title": "Python Design Patterns",
    "summary": "Factory, observer, and strategy patterns for Python.",
    "path": "knowledge/computer-science/programming/docs/python-patterns.md"
  }
]
```

Three fields only: title, summary, path. No tags.

## Using Knowledge

Three built-in skills manage the knowledge lifecycle:

### Retrieve (`retrieve-knowledge`)

1. Read `knowledge/index.json` → pick a category
2. Read `knowledge/{category}/{subcategory}/catalog.json` → pick a document
3. Read the document

### Create (`create-document`)

Every creation updates three files:

1. Write the document to `knowledge/{category}/{subcategory}/docs/{name}.md`
2. Add entry to `knowledge/{category}/{subcategory}/catalog.json`
3. Add subcategory to `knowledge/index.json` (if new)

### Update (`update-document`)

1. Find the document via `retrieve-knowledge`
2. Read, modify, write back
3. Update catalog if title/summary changed

## Example

Agent learns Docker bridge networking needs specific DNS config:

**Create:**
```
knowledge/devops/docker/docs/bridge-networking.md     ← document
knowledge/devops/docker/catalog.json                  ← catalog entry added
knowledge/index.json                                  ← devops/docker entry added
```

**Later retrieval:**
```
cat knowledge/index.json                              → finds devops/docker
cat knowledge/devops/docker/catalog.json              → finds bridge-networking.md
cat knowledge/devops/docker/docs/bridge-networking.md  → reads content
```

## Best Practices

- Use descriptive kebab-case filenames.
- Keep summaries to one sentence — they're the search surface.
- Choose meaningful category/subcategory groupings.
- Document patterns and solutions, not raw data.
