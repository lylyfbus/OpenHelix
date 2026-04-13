---
name: Retrieve Knowledge
description: Search and load reusable long-term knowledge from the hierarchical knowledge library.
---

# Purpose

Use this skill to find and load relevant knowledge documents before reasoning or taking action. The knowledge library is organized by category and subcategory.

# When To Use

- When the task may benefit from previously documented knowledge.
- When the user asks about something that may have been researched before.
- Before starting a task that overlaps with past work.

# Knowledge Library Structure

```
knowledge/
  index.json                                  <- Global classification index
  {category}/
    {subcategory}/
      catalog.json                             <- Document metadata (title, summary, path)
      docs/
        {document}.md                          <- Actual documents
```

# Procedure

## Step 1: Read the Knowledge Index

```json
{
  "job_name": "read-knowledge-index",
  "code_type": "bash",
  "script": "cat knowledge/index.json"
}
```

The index is a JSON array. Each entry has: category, subcategory, description, path.
If the file doesn't exist, there is no knowledge library yet — skip retrieval.

## Step 2: Pick a Category

From the index, identify the category/subcategory most relevant to your current task.

## Step 3: Read the Catalog

```json
{
  "job_name": "read-knowledge-catalog",
  "code_type": "bash",
  "script": "cat knowledge/{category}/{subcategory}/catalog.json"
}
```

The catalog is a JSON array. Each entry has: title, summary, path.

## Step 4: Read the Document

```json
{
  "job_name": "read-knowledge-doc",
  "code_type": "bash",
  "script": "cat {path_from_catalog}"
}
```

# Rules

- Always start from the index. Do not guess document paths.
- Read the catalog before loading a document.
- Load only documents relevant to the current task.
- If no relevant category exists in the index, skip knowledge retrieval.
