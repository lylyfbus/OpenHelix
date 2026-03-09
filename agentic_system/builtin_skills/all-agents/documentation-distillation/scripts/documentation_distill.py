from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_EXECUTED_SKILL = "documentation-distillation"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _parse_tags(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _section(title: str, body: str) -> str:
    text = str(body).strip()
    return f"## {title}\n\n{text if text else '(empty)'}"


def _strip_h1(text: str) -> str:
    lines = str(text).splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def _build_body(args: argparse.Namespace) -> str:
    parts = [
        _section("Problem", args.problem),
        _section("What Was Done", args.what_was_done),
        _section("Reusable Pattern", args.reusable_pattern),
        _section("Caveats", args.caveats),
    ]
    refs = str(args.source_refs).strip()
    if refs:
        parts.append(_section("Source Refs", refs))
    tags = _parse_tags(args.tags)
    if tags:
        parts.append(_section("Tags", ", ".join(tags)))
    return "\n\n".join(parts)


def _ok(doc_path: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "doc_path": doc_path,
    }


def _err(doc_path: str = "") -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "doc_path": doc_path,
    }


def _knowledge_paths(workspace: Path) -> tuple[Path, Path, Path]:
    knowledge_root = workspace / "knowledge"
    docs_root = knowledge_root / "docs"
    index_root = knowledge_root / "index"
    docs_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)
    catalog_path = index_root / "catalog.json"
    return docs_root, index_root, catalog_path


def _load_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    if not catalog_path.exists():
        return []
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _save_catalog(catalog_path: Path, rows: list[dict[str, Any]]) -> None:
    catalog_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")


def _get_catalog_entry(catalog_rows: list[dict[str, Any]], doc_id: str) -> dict[str, Any]:
    for row in catalog_rows:
        if str(row.get("doc_id", "")).strip() == doc_id:
            return row
    return {}


def _upsert_catalog_entry(catalog_rows: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    doc_id = str(entry.get("doc_id", "")).strip()
    for idx, row in enumerate(catalog_rows):
        if str(row.get("doc_id", "")).strip() == doc_id:
            catalog_rows[idx] = entry
            return
    catalog_rows.append(entry)


def _doc_path(docs_root: Path, doc_id: str) -> Path:
    return docs_root / f"{doc_id}.md"


def _read_existing_doc(docs_root: Path, catalog_rows: list[dict[str, Any]], doc_id: str) -> dict[str, Any] | None:
    path = _doc_path(docs_root, doc_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    row = _get_catalog_entry(catalog_rows, doc_id)
    title = str(row.get("title", "")).strip()
    if not title:
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), doc_id)
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "quality_score": float(row.get("quality_score", 0.0) or 0.0),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "path": str(path),
    }


def run_create(workspace: Path, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    title = str(args.title).strip()
    if not title:
        return _err(doc_path="documentation_error: missing --title for create"), 1

    body = str(args.body).strip() if str(args.body).strip() else _build_body(args)
    if not body:
        return _err(doc_path="documentation_error: empty content for create"), 1

    tags = _parse_tags(args.tags)
    docs_root, _index_root, catalog_path = _knowledge_paths(workspace)
    catalog_rows = _load_catalog(catalog_path)

    doc_id = str(args.doc_id).strip() if str(args.doc_id).strip() else f"doc_{uuid4().hex[:12]}"
    doc_path = _doc_path(docs_root, doc_id)

    out = f"# {title}\n\n{body}\n"
    doc_path.write_text(out, encoding="utf-8")

    entry = {
        "doc_id": doc_id,
        "title": title,
        "path": str(doc_path.relative_to(workspace)),
        "tags": tags,
        "quality_score": float(args.quality_score),
        "confidence": float(args.confidence),
        "updated_at": _now_iso(),
    }
    _upsert_catalog_entry(catalog_rows, entry)
    _save_catalog(catalog_path, catalog_rows)

    return _ok(doc_path=str(doc_path.relative_to(workspace))), 0


def run_update(workspace: Path, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    doc_id = str(args.doc_id).strip()
    if not doc_id:
        return _err(doc_path="documentation_error: missing --doc-id for update"), 1

    docs_root, _index_root, catalog_path = _knowledge_paths(workspace)
    catalog_rows = _load_catalog(catalog_path)
    existing = _read_existing_doc(docs_root, catalog_rows, doc_id)
    if existing is None:
        return _err(doc_path=f"documentation_error: doc not found for doc_id={doc_id}"), 1

    current_title = str(existing.get("title", "")).strip() or "Untitled"
    current_text = str(existing.get("text", ""))
    current_body = _strip_h1(current_text)

    next_title = str(args.title).strip() or current_title
    if str(args.body).strip():
        next_body = str(args.body).strip()
    else:
        patch_parts = []
        if str(args.problem).strip():
            patch_parts.append(_section("Problem", args.problem))
        if str(args.what_was_done).strip():
            patch_parts.append(_section("What Was Done", args.what_was_done))
        if str(args.reusable_pattern).strip():
            patch_parts.append(_section("Reusable Pattern", args.reusable_pattern))
        if str(args.caveats).strip():
            patch_parts.append(_section("Caveats", args.caveats))
        if str(args.source_refs).strip():
            patch_parts.append(_section("Source Refs", args.source_refs))
        tags = _parse_tags(args.tags)
        if tags:
            patch_parts.append(_section("Tags", ", ".join(tags)))

        if patch_parts:
            next_body = (
                f"{current_body}\n\n## Update\n\n"
                + "\n\n".join(patch_parts)
            ).strip()
        else:
            return _err(doc_path="documentation_error: no update content provided"), 1

    doc_path = _doc_path(docs_root, doc_id)
    doc_path.write_text(f"# {next_title}\n\n{next_body}\n", encoding="utf-8")

    current_row = _get_catalog_entry(catalog_rows, doc_id)
    next_tags = _parse_tags(args.tags)
    if not next_tags:
        existing_tags = current_row.get("tags", [])
        if isinstance(existing_tags, list):
            next_tags = [str(tag).strip() for tag in existing_tags if str(tag).strip()]
        elif isinstance(existing_tags, str):
            next_tags = [item.strip() for item in existing_tags.split(",") if item.strip()]

    entry = {
        "doc_id": doc_id,
        "title": next_title,
        "path": str(doc_path.relative_to(workspace)),
        "tags": next_tags,
        "quality_score": float(args.quality_score),
        "confidence": float(args.confidence),
        "updated_at": _now_iso(),
    }
    _upsert_catalog_entry(catalog_rows, entry)
    _save_catalog(catalog_path, catalog_rows)

    return _ok(doc_path=str(doc_path.relative_to(workspace))), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update runtime knowledge docs.")
    parser.add_argument("--action", required=True, choices=["create", "update"])
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--doc-id", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--problem", default="")
    parser.add_argument("--what-was-done", default="")
    parser.add_argument("--reusable-pattern", default="")
    parser.add_argument("--caveats", default="")
    parser.add_argument("--source-refs", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--quality-score", default="0.0")
    parser.add_argument("--confidence", default="0.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action = str(args.action)
    workspace = Path(args.workspace).expanduser().resolve()

    try:
        if action == "create":
            out, code = run_create(workspace, args)
        else:
            out, code = run_update(workspace, args)
        print(json.dumps(out, ensure_ascii=True))
        return code
    except Exception as exc:
        out = _err(doc_path=f"documentation_error: unexpected exception: {exc}")
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
