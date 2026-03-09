"""Knowledge index loader — reads catalog.json for knowledge doc discovery.

Provides compact metadata rows that the agent can use to decide
which knowledge documents to load (via the load-knowledge-docs skill).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_tags(value: Any) -> list[str]:
    """Normalize tags field from list/string into list[str]."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def load_knowledge_catalog(
    knowledge_root: Path,
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Load knowledge doc metadata from ``knowledge/index/catalog.json``.

    Expected catalog entry format::

        {
          "doc_id": "llm-post-training",
          "title": "LLM Post-Training Overview",
          "path": "knowledge/docs/llm-post-training.md",
          "tags": ["llm", "rl", "training"],
          "quality_score": 0.9,
          "confidence": 0.85
        }

    Args:
        knowledge_root: Path to the ``knowledge/`` directory.
        limit: Maximum number of catalog entries to return.

    Returns:
        Sorted list of knowledge metadata dicts.
    """
    catalog_path = Path(knowledge_root) / "index" / "catalog.json"
    if not catalog_path.exists():
        return []

    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(raw, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        rows.append({
            "doc_id": doc_id,
            "title": str(item.get("title", "")).strip() or doc_id,
            "path": str(item.get("path", "")).strip() or f"knowledge/docs/{doc_id}.md",
            "tags": _normalize_tags(item.get("tags", [])),
            "quality_score": float(item.get("quality_score", 0.0) or 0.0),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
        })

    rows.sort(key=lambda r: r["doc_id"])
    return rows[:max(1, limit)]


def format_knowledge_for_prompt(catalog: list[dict[str, Any]]) -> str:
    """Format knowledge catalog rows into a prompt-injectable text block."""
    if not catalog:
        return "- (no knowledge docs found)"
    return "\n".join("- " + json.dumps(row, ensure_ascii=True) for row in catalog)
