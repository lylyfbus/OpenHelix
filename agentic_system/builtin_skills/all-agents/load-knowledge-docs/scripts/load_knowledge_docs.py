from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_EXECUTED_SKILL = "load-knowledge-docs"


def _ok(knowledge_context: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "knowledge_context": knowledge_context,
    }


def _err(knowledge_context: str = "") -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "knowledge_context": knowledge_context,
    }


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _catalog_by_doc_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        out[doc_id] = row
    return out


def _resolve_doc_from_id(workspace: Path, docs_root: Path, index_row: dict[str, Any], doc_id: str) -> tuple[str, Path]:
    title = str(index_row.get("title", "")).strip() or doc_id
    path_from_index = str(index_row.get("path", "")).strip()
    if path_from_index:
        candidate = (workspace / path_from_index).resolve()
    else:
        candidate = (docs_root / f"{doc_id}.md").resolve()
    return title, candidate


def _resolve_doc_from_path(workspace: Path, path_text: str) -> Path | None:
    raw = str(path_text).strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        resolved = candidate.resolve()
    except Exception:
        return None
    return resolved


def _safe_relpath(workspace: Path, path: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def _title_from_markdown(doc_id: str, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            if title:
                return title
    return doc_id


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _format_knowledge_context(blocks: list[dict[str, str]]) -> str:
    if not blocks:
        return ""
    out: list[str] = []
    for row in blocks:
        out.append(
            "\n".join(
                [
                    f"# {row.get('title', '')}".strip(),
                    f"path: {row.get('path', '')}",
                    "content:",
                    row.get("content", ""),
                ]
            )
        )
    return "\n\n---\n\n".join(out)


def run_load(
    workspace: Path,
    requested_doc_ids: list[str],
    requested_doc_paths: list[str],
    max_docs: int,
    max_chars_per_doc: int,
) -> dict[str, Any]:
    knowledge_root = workspace / "knowledge"
    docs_root = knowledge_root / "docs"
    catalog_path = knowledge_root / "index" / "catalog.json"
    catalog_rows = _load_catalog(catalog_path)
    index_by_id = _catalog_by_doc_id(catalog_rows)

    max_docs = max(1, int(max_docs))
    max_chars_per_doc = max(200, int(max_chars_per_doc))

    items: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()

    for doc_id in requested_doc_ids:
        normalized = str(doc_id).strip()
        if not normalized:
            continue
        row = index_by_id.get(normalized, {"doc_id": normalized})
        title, path = _resolve_doc_from_id(workspace, docs_root, row, normalized)
        key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        items.append((title, path))

    for path_text in requested_doc_paths:
        resolved = _resolve_doc_from_path(workspace, path_text)
        if resolved is None:
            continue
        key = str(resolved)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        items.append(("", resolved))

    if not items:
        return _err(
            knowledge_context=(
                "load_knowledge_error: no valid targets; "
                f"requested_doc_ids={requested_doc_ids}; requested_doc_paths={requested_doc_paths}"
            )
        )

    loaded_blocks: list[dict[str, str]] = []
    for title_hint, path in items:
        if len(loaded_blocks) >= max_docs:
            break
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not text.strip():
            continue
        doc_id = path.stem
        title = title_hint.strip() or _title_from_markdown(doc_id, text)
        loaded_blocks.append(
            {
                "title": title,
                "path": _safe_relpath(workspace, path),
                "content": _truncate(text, max_chars_per_doc),
            }
        )

    details_prefix = (
        "load_knowledge_ok: "
        f"requested={len(items)}; loaded={len(loaded_blocks)}; "
        f"max_docs={max_docs}; max_chars_per_doc={max_chars_per_doc}"
    )
    formatted_docs = _format_knowledge_context(loaded_blocks)
    knowledge_context = details_prefix if not formatted_docs else f"{details_prefix}\n\n{formatted_docs}"
    if not knowledge_context:
        return _err(
            knowledge_context=(
                "load_knowledge_error: no readable docs loaded; "
                f"requested_doc_ids={requested_doc_ids}; requested_doc_paths={requested_doc_paths}"
            )
        )
    return _ok(knowledge_context=knowledge_context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load selected knowledge docs from runtime workspace.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--doc-path", action="append", default=[])
    parser.add_argument("--max-docs", default="6")
    parser.add_argument("--max-chars-per-doc", default="2400")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    doc_ids = [str(item).strip() for item in args.doc_id if str(item).strip()]
    doc_paths = [str(item).strip() for item in args.doc_path if str(item).strip()]

    try:
        out = run_load(
            workspace=workspace,
            requested_doc_ids=doc_ids,
            requested_doc_paths=doc_paths,
            max_docs=int(args.max_docs),
            max_chars_per_doc=int(args.max_chars_per_doc),
        )
        print(json.dumps(out, ensure_ascii=True))
        return 0 if out.get("status") == "ok" else 1
    except ValueError as exc:
        out = _err(knowledge_context=f"load_knowledge_error: invalid numeric input: {exc}")
        print(json.dumps(out, ensure_ascii=True))
        return 1
    except Exception as exc:
        out = _err(knowledge_context=f"load_knowledge_error: unexpected exception: {exc}")
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
