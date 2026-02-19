from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class PromptEngine:
    _AGENT_ROLES_PLACEHOLDER = "{{AGENT_ROLES_FROM_JSON}}"
    _SKILLS_META_PLACEHOLDER = "{{SKILLS_META_FROM_JSON}}"
    _RUNTIME_WORKSPACE_PLACEHOLDER = "{{RUNTIME_WORKSPACE}}"
    WORKFLOW_SUMMARIZER_PROMPT = "\n".join(
        [
            "You are an objective observer of the full working history of an agentic system.",
            "Your task is to update workflow_summary using both the current workflow_summary and the full workflow_history.",
            "Input format:",
            "1) workflow_summary: current compact summary of workflow status.",
            "    - workflow_summary is a workflow tracker used by the agent brain to understand the full picture and current position.",
            "2) workflow_history: full workflow records in strict time order (earliest to latest).",
            "    - workflow_history line format: [UTC_ISO_TIMESTAMP] role> : content.",
            "Write concise factual summary text only. Do not invent facts and do not add speculation.",
            "Update policy:",
            "1) If workflow_history does not add important new information, keep workflow_summary unchanged.",
            "2) Only update when new facts materially change progress, blockers, decisions, or next focus.",
            "3) If no update is needed, return an empty workflow_summary as sentinel:",
            "<output>",
            "{\"workflow_summary\": \"\"}",
            "</output>",
            "Focus requirements:",
            "1) Brief history from initial user intent to latest major changes.",
            "2) Current status: done, in progress, blocked.",
            "3) Current focus and immediate next useful direction.",
            "4) Prioritize recent workflow_history while preserving key earlier context.",
            "Output format:",
            "Return one JSON object wrapped in <output> and </output>.",
            "Do not output any text outside that block.",
            "{\"workflow_summary\":\"...\"}",
            "Example:",
            "<output>",
            "{\"workflow_summary\":\"...\"}",
            "</output>",
        ]
    )
    WORKFLOW_COMPACTOR_PROMPT = "\n".join(
        [
            "You are an objective observer compressing older workflow records for runtime memory control.",
            "Your task is to compact workflow_history into ONE concise chronological text block while preserving key facts.",
            "Use workflow_summary only as context support.",
            "Input format:",
            "1) workflow_summary: current compact summary of workflow status.",
            "    - workflow_summary is a workflow tracker used by the agent to understand the full picture and current position.",
            "2) workflow_history: older workflow records in strict time order (earliest to latest).",
            "    - workflow_history line format: [UTC_ISO_TIMESTAMP] role> : content.",
            "Compression target:",
            "- Output ONE paragraph string, no bullets, no newlines, no markdown, no JSON inside the string.",
            "- Keep it factual and chronological.",
            "- Keep major decisions, actions with outcomes, blockers, and unresolved loops.",
            "- Exclude advice, speculation, and future planning.",
            "- Keep it compact; target <= 1200 characters.",
            "Output format:",
            "Return one JSON object wrapped in <output> and </output>.",
            "Do not output any text outside that block.",
            "{\"workflow_hist_compact\":\"...\"}",
            "Example:",
            "<output>",
            "{\"workflow_hist_compact\":\"...\"}",
            "</output>",
        ]
    )

    def __init__(
        self,
        workspace: str | Path,
        packaged_root: str | Path | None = None,
        token_window_limit: int = int(256000*0.75),
        compact_keep_last_k: int = 10,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.runtime_prompts_root = self.workspace / "prompts"
        self.packaged_root = (
            Path(packaged_root).resolve()
            if packaged_root
            else Path(__file__).resolve().parents[1] / "prompts"
        )
        self.token_window_limit = int(token_window_limit)
        self.compact_keep_last_k = max(1, int(compact_keep_last_k))
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

    def _get_system_prompt(self, role: str, skill_engine: Any | None = None) -> str:
        role = str(role).strip()
        if role == "workflow_summarizer":
            return self.WORKFLOW_SUMMARIZER_PROMPT
        
        if role == "workflow_history_compactor":
            return self.WORKFLOW_COMPACTOR_PROMPT

        prompts = self._load_system_prompts()
        selected = prompts.get(role, "") if role else ""
        if not selected:
            return ""

        descriptions = self._load_json_map(self.agent_role_descriptions_path)
        lines = ["Below is the description of agent roles for your reference:"]
        if descriptions:
            for item_role in sorted(descriptions.keys()):
                desc = str(descriptions.get(item_role, "")).strip()
                lines.append(f"- {item_role}: {desc if desc else '(no description)'}")
        else:
            lines.append("- (no role descriptions found)")
        roles_section = "\n".join(lines)

        skill_scope = "core+all" if role == "core_agent" else "all"
        skills_lines = [f"Below is the available skill metadata:"]
        skills_text = ""
        try:
            skills_text = str(skill_engine.load_skill_meta(scope=skill_scope)).strip()
        except Exception:
            skills_text = ""
        if skills_text:
            skills_lines.append(skills_text)
        else:
            skills_lines.append("- (no skills found)")
        skills_section = "\n".join(skills_lines)

        text = selected.strip()
        if self._AGENT_ROLES_PLACEHOLDER in text:
            text = text.replace(self._AGENT_ROLES_PLACEHOLDER, roles_section)

        if self._SKILLS_META_PLACEHOLDER in text:
            text = text.replace(self._SKILLS_META_PLACEHOLDER, skills_section)
        if self._RUNTIME_WORKSPACE_PLACEHOLDER in text:
            text = text.replace(self._RUNTIME_WORKSPACE_PLACEHOLDER, str(self.workspace))
        return text

    def _compact_workflow_history(
        self,
        role: str,
        state: Any, 
        compact_keep_last_k: int,
        model_router: Any,
    ) -> None:
        head = state.workflow_hist[:-compact_keep_last_k] if compact_keep_last_k < len(state.workflow_hist) else []
        tail = state.workflow_hist[-compact_keep_last_k:] if compact_keep_last_k > 0 else []
        workflow_summary = state.workflow_summary if isinstance(state.workflow_summary, str) else ""
        system_prompt = self._get_system_prompt(role)
        sections: list[str] = []
        if system_prompt.strip():
            sections.append(system_prompt.strip())
        sections.append(
            "\n".join(
                [
                    "Workflow Summary:",
                    workflow_summary if workflow_summary else "(empty)",
                    "Workflow History:",
                    "\n".join(head) if head else "(empty)",
                ]
            )
        )
        final_prompt = "\n\n".join(sections)
        response = model_router.generate(
            role="workflow_history_compactor",
            final_prompt=final_prompt
        )
        workflow_hist_compact = str(response.get("workflow_hist_compact", ""))
        state.workflow_hist = [f"[{state.utc_now_iso()}] workflow_compactor> : {workflow_hist_compact}"] + tail

    def build_prompt(
        self,
        role: str,
        state: Any | None = None,
        model_router: Any | None = None,
        skill_engine: Any | None = None,
    ) -> str:
        system_prompt = self._get_system_prompt(role, skill_engine=skill_engine)
        workflow_summary = str(getattr(state, "workflow_summary", "")).strip()
        workflow_history: list[str] = getattr(state, "workflow_hist", [])
        workflow_history_lines = [line for line in workflow_history if line.strip()]
        sections: list[str] = []
        if system_prompt.strip():
            sections.append(system_prompt.strip())
        sections.append(
            "\n".join(
                [
                    "Workflow Summary:",
                    workflow_summary if workflow_summary else "(empty)",
                    "Workflow History:",
                    "\n".join(workflow_history_lines) if workflow_history_lines else "(empty)",
                ]
            )
        )
        final_prompt = "\n\n".join(sections)
        estimated_tokens = max(1, len(final_prompt) // 4)
        rounds = 0
        while rounds < 3 and estimated_tokens > self.token_window_limit:
            self._compact_workflow_history(
                role="workflow_history_compactor",
                state=state, 
                compact_keep_last_k=self.compact_keep_last_k, 
                model_router=model_router
            )
            workflow_history: list[str] = getattr(state, "workflow_hist", [])
            workflow_history_lines = [line for line in workflow_history if line.strip()]
            sections: list[str] = []
            if system_prompt.strip():
                sections.append(system_prompt.strip())
            sections.append(
                "\n".join(
                    [
                        "Workflow Summary:",
                        workflow_summary if workflow_summary else "(empty)",
                        "Workflow History:",
                        "\n".join(workflow_history_lines) if workflow_history_lines else "(empty)",
                    ]
                )
            )
            final_prompt = "\n\n".join(sections)
            estimated_tokens = max(1, len(final_prompt) // 4)
            rounds += 1

        return final_prompt
