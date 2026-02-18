from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4


class SkillEngine:
    def __init__(self, workspace: str | Path, packaged_root: str | Path | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.runtime_skills_root = self.workspace / "skills"
        default_packaged = Path(__file__).resolve().parents[2] / "skills"
        self.packaged_root = Path(packaged_root).resolve() if packaged_root else default_packaged
        self._bootstrap_runtime_skills()

    def load_skill_meta(self, scope: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        include_core = scope in {"core+all", "core"}
        include_all = scope in {"core+all", "all"}

        roots: list[tuple[Path, str]] = []
        if include_core:
            roots.append((self.runtime_skills_root / "core-agent", "core-agent"))
        if include_all:
            roots.append((self.runtime_skills_root / "all-agents", "all-agents"))

        for root, skill_scope in roots:
            if not root.exists():
                continue
            for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                skill_id = skill_dir.name
                skill_md_path = skill_dir / "SKILL.md"
                if not skill_md_path.exists():
                    continue
                text = ""
                try:
                    text = skill_md_path.read_text(encoding="utf-8")
                except OSError:
                    text = ""
                frontmatter = self._parse_frontmatter(text)
                rows.append(
                    {
                        "skill_id": skill_id,
                        "scope": skill_scope,
                        "name": str(frontmatter.get("name", skill_id)),
                        "handler": str(frontmatter.get("handler", "")),
                        "description": str(frontmatter.get("description", "")),
                        "required_tools": self._parse_csv_field(frontmatter.get("required_tools", "")),
                        "recommended_tools": self._parse_csv_field(frontmatter.get("recommended_tools", "")),
                        "forbidden_tools": self._parse_csv_field(frontmatter.get("forbidden_tools", "")),
                    }
                )

        rows = [row for row in rows if str(row.get("skill_id", "")).strip()]
        rows.sort(key=lambda item: (str(item.get("scope", "")), str(item.get("skill_id", ""))))
        return rows

    def load_skill(
        self,
        skill_ids: str | list[str],
        scope: str,
    ) -> list[dict[str, Any]]:
        if isinstance(skill_ids, str):
            requested = [skill_ids.strip()] if skill_ids.strip() else []
        else:
            requested = [str(item).strip() for item in skill_ids if str(item).strip()]
        if not requested:
            return []

        include_core = scope in {"core+all", "core"}
        include_all = scope in {"core+all", "all"}

        roots: list[Path] = []
        if include_core:
            roots.append(self.runtime_skills_root / "core-agent")
        if include_all:
            roots.append(self.runtime_skills_root / "all-agents")

        details: list[dict[str, Any]] = []

        for root in roots:
            if not root.exists():
                continue
            for skill_id in requested:
                skill_dir = root / skill_id
                if not skill_dir.exists() or not skill_dir.is_dir():
                    continue
                row = self._build_skill_payload(skill_dir, skill_id)
                details.append(row)

        return details

    def create_skill(self, proposal: dict[str, Any]) -> dict[str, Any]:
        payload = dict(proposal) if isinstance(proposal, dict) else {}
        action = str(payload.get("action", "create")).strip().lower()
        skill_id = str(payload.get("skill_id", "")).strip()
        scope = str(payload.get("scope", "all-agents")).strip()
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}

        proposals_dir = self.runtime_skills_root / "_proposals"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        proposal_id = f"proposal_{uuid4().hex[:12]}"
        proposal_path = proposals_dir / f"{proposal_id}.json"
        proposal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        applied = False
        reason = ""
        skill_path = ""

        if scope not in {"core-agent", "all-agents"}:
            reason = "invalid_scope"
        elif not skill_id:
            reason = "missing_skill_id"
        elif action not in {"create", "update"}:
            reason = "unsupported_action"
        else:
            skill_md = artifacts.get("skill_md")
            if isinstance(skill_md, str) and skill_md.strip():
                skill_dir = self.runtime_skills_root / scope / skill_id
                scripts_dir = skill_dir / "scripts"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
                for script in artifacts.get("scripts", []):
                    if not isinstance(script, dict):
                        continue
                    rel = str(script.get("path", "")).strip()
                    content = str(script.get("content", ""))
                    if not rel:
                        continue
                    rel_path = Path(rel)
                    if rel_path.is_absolute() or ".." in rel_path.parts:
                        continue
                    path = skill_dir / rel_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                applied = True
                skill_path = str(skill_dir / "SKILL.md")
            else:
                reason = "missing_skill_md"

        return {
            "proposal_path": str(proposal_path),
            "skill_id": skill_id,
            "scope": scope,
            "applied": applied,
            "skill_path": skill_path,
            "reason": reason,
        }

    def _bootstrap_runtime_skills(self) -> None:
        # Copy packaged skills into runtime workspace once; keep user-created skills intact.
        for scope in ("core-agent", "all-agents"):
            source_scope = self.packaged_root / scope
            target_scope = self.runtime_skills_root / scope
            target_scope.mkdir(parents=True, exist_ok=True)
            if not source_scope.exists():
                continue
            for skill_dir in sorted(p for p in source_scope.iterdir() if p.is_dir()):
                target_dir = target_scope / skill_dir.name
                if target_dir.exists():
                    continue
                shutil.copytree(skill_dir, target_dir)

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        end = -1
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end = idx
                break
        if end == -1:
            return {}
        payload: dict[str, str] = {}
        for raw in lines[1:end]:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip()
        return payload

    @staticmethod
    def _parse_csv_field(value: Any) -> list[str]:
        if value is None:
            return []
        raw = str(value).strip()
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]
    
    @staticmethod
    def _build_skill_payload(skill_dir: Path, skill_id: str) -> dict[str, Any]:
        skill_md_path = skill_dir / "SKILL.md"
        skill_md = ""
        if skill_md_path.exists():
            try:
                skill_md = skill_md_path.read_text(encoding="utf-8")
            except OSError:
                skill_md = ""

        payload: dict[str, Any] = {
            "skill_id": skill_id,
            "skill_md": skill_md,
            "scripts": [],
        }

        for file_path in sorted(p for p in skill_dir.rglob("*") if p.is_file()):
            rel_path = str(file_path.relative_to(skill_dir))
            if rel_path == "SKILL.md":
                continue
            parts = file_path.relative_to(skill_dir).parts
            if "__pycache__" in parts or file_path.suffix == ".pyc":
                continue
            top = parts[0] if parts else ""
            if top == "scripts":
                payload["scripts"].append({"path": rel_path})

        return payload
