from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class PromptEngine:
    _AGENT_ROLES_PLACEHOLDER = "{{AGENT_ROLES_FROM_JSON}}"
    _SKILLS_META_PLACEHOLDER = "{{SKILLS_META_FROM_JSON}}"
    WORKFLOW_SUMMARIZER_PROMPT = "\n".join(
        [
            "You are an objective observer of the full working history of an agentic system.",
            "Your task is to update workflow_summary using both the current workflow_summary and the full workflow_history.",
            "workflow_summary is a workflow tracker used by the agent brain to understand the full picture and current position.",
            "Input format:",
            "1) workflow_summary: current compact summary of workflow status.",
            "2) workflow_history: full workflow records in strict time order (earliest to latest).",
            "    - workflow_history line format: [UTC_ISO_TIMESTAMP] role> : content.",
            "Write concise factual summary text only. Do not invent facts and do not add speculation.",
            "Update policy:",
            "1) If workflow_history does not add important new information, keep workflow_summary unchanged.",
            "2) Only update when new facts materially change progress, blockers, decisions, or next focus.",
            "Focus requirements:",
            "1) Brief history from initial user intent to latest major changes.",
            "2) Current status: done, in progress, blocked.",
            "3) Current focus and immediate next useful direction.",
            "4) Prioritize recent workflow_history while preserving key earlier context.",
            "Return one JSON object wrapped in <output> and </output>.",
            "Do not output any text outside that block.",
            "Schema inside the block:",
            "{\"workflow_summary\":\"...\"}",
            "Example:",
            "<output>",
            "{\"workflow_summary\":\"...\"}",
            "</output>",
        ]
    )

    def __init__(self, workspace: str | Path, packaged_root: str | Path | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.runtime_prompts_root = self.workspace / "prompts"
        self.packaged_root = (
            Path(packaged_root).resolve()
            if packaged_root
            else Path(__file__).resolve().parents[1] / "prompts"
        )
        self.system_prompts_path = self.runtime_prompts_root / "agent_system_prompt.json"
        self.legacy_system_prompts_path = self.runtime_prompts_root / "agent_systemp_prompt.json"
        self.agent_role_descriptions_path = self.runtime_prompts_root / "agent_role_description.json"
        self._bootstrap_runtime_prompts()

    @staticmethod
    def _normalize_map(raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            role = str(key).strip()
            if not role:
                continue
            if isinstance(value, list):
                normalized[role] = "\n".join(str(item) for item in value)
            else:
                normalized[role] = str(value)
        return normalized

    @classmethod
    def _load_json_map(cls, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return cls._normalize_map(raw)

    def _bootstrap_runtime_prompts(self) -> None:
        self.runtime_prompts_root.mkdir(parents=True, exist_ok=True)

        if not self.system_prompts_path.exists():
            source = self.packaged_root / "agent_system_prompt.json"
            legacy_source = self.packaged_root / "agent_systemp_prompt.json"
            if source.exists():
                shutil.copy2(source, self.system_prompts_path)
            elif legacy_source.exists():
                shutil.copy2(legacy_source, self.system_prompts_path)
            else:
                self.system_prompts_path.write_text("{}", encoding="utf-8")

        if not self.agent_role_descriptions_path.exists():
            source = self.packaged_root / "agent_role_description.json"
            if source.exists():
                shutil.copy2(source, self.agent_role_descriptions_path)
            else:
                self.agent_role_descriptions_path.write_text("{}", encoding="utf-8")

    def _load_system_prompts(self) -> dict[str, str]:
        if not self.system_prompts_path.exists() and self.legacy_system_prompts_path.exists():
            return self._load_json_map(self.legacy_system_prompts_path)
        return self._load_json_map(self.system_prompts_path)

    def _build_skills_meta_section(self) -> str:
        skills_root = self.workspace / "skills"
        lines: list[str] = [f"Skills Metadata (Loaded from {skills_root}):"]
        rows: list[str] = []
        for scope in ("core-agent", "all-agents"):
            scope_root = skills_root / scope
            if not scope_root.exists():
                continue
            for skill_dir in sorted(path for path in scope_root.iterdir() if path.is_dir()):
                skill_md_path = skill_dir / "SKILL.md"
                if not skill_md_path.exists():
                    continue
                name = skill_dir.name
                description = ""
                try:
                    raw = skill_md_path.read_text(encoding="utf-8")
                    frontmatter = self._parse_frontmatter(raw)
                    if isinstance(frontmatter.get("name"), str) and frontmatter["name"].strip():
                        name = frontmatter["name"].strip()
                    if isinstance(frontmatter.get("description"), str):
                        description = frontmatter["description"].strip()
                except Exception:
                    pass
                summary = description if description else "No description."
                rows.append(f"- {skill_dir.name} [{scope}] - {name}: {summary}")
        if rows:
            lines.extend(rows)
        else:
            lines.append("- (no skills found)")
        return "\n".join(lines)

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

    def _get_system_prompt(self, agent_role: str) -> str:
        role = str(agent_role).strip()
        if role == "workflow_summarizer":
            return self.WORKFLOW_SUMMARIZER_PROMPT

        prompts = self._load_system_prompts()
        selected = prompts.get(role, "") if role else ""
        if not selected:
            return ""

        descriptions = self._load_json_map(self.agent_role_descriptions_path)
        lines = ["Agent Roles (Loaded from agent_role_description.json):"]
        if descriptions:
            for item_role in sorted(descriptions.keys()):
                desc = str(descriptions.get(item_role, "")).strip()
                lines.append(f"- {item_role}: {desc if desc else '(no description)'}")
        else:
            lines.append("- (no role descriptions found)")
        roles_section = "\n".join(lines)
        skills_section = self._build_skills_meta_section()

        text = selected.strip()
        if self._AGENT_ROLES_PLACEHOLDER in text:
            text = text.replace(self._AGENT_ROLES_PLACEHOLDER, roles_section)
        else:
            text = text + "\n\n" + roles_section
        if self._SKILLS_META_PLACEHOLDER in text:
            text = text.replace(self._SKILLS_META_PLACEHOLDER, skills_section)
        else:
            text = text + "\n\n" + skills_section
        return text

    def build_prompt(
        self,
        agent_role: str,
        input_payload: dict[str, Any],
    ) -> str:
        system_prompt = self._get_system_prompt(agent_role)
        sections: list[str] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            sections.append(str(system_prompt).strip())

        workflow_summary = input_payload.get("workflow_summary")
        workflow_history = input_payload.get("workflow_history")
        if workflow_summary is not None or workflow_history is not None:
            text_blocks: list[str] = []
            if workflow_summary is not None:
                summary_text = str(workflow_summary).strip()
                text_blocks.append("Workflow Summary:")
                text_blocks.append(summary_text if summary_text else "(empty)")
            if workflow_history is not None:
                if isinstance(workflow_history, list):
                    history_text = "\n".join(str(line) for line in workflow_history)
                else:
                    history_text = str(workflow_history)
                history_text = history_text.strip()
                text_blocks.append("Workflow History:")
                text_blocks.append(history_text if history_text else "(empty)")
            sections.append("\n".join(text_blocks))

        return "\n\n".join(sections)
