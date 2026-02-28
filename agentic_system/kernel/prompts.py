from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PromptEngine:
    _AGENT_ROLES_PLACEHOLDER = "{{AGENT_ROLES_FROM_JSON}}"
    _SKILLS_META_PLACEHOLDER = "{{SKILLS_META_FROM_JSON}}"
    _KNOWLEDGE_META_PLACEHOLDER = "{{KNOWLEDGE_META_FROM_JSON}}"
    _BUILTIN_REFERENCE_LOADERS_PLACEHOLDER = "{{BUILTIN_REFERENCE_LOADERS}}"
    _RUNTIME_WORKSPACE_PLACEHOLDER = "{{RUNTIME_WORKSPACE}}"
    _LATEST_CONTEXT_PLACEHOLDER = "{{LATEST_CONTEXT}}"
    _DEFAULT_RETRY_LIMIT = 3
    _FORMAT_RETRY_LIMIT = 10
    WORKFLOW_SUMMARIZER_PROMPT = "\n".join(
        [
            "You are an objective observer and long-term memory maintainer for an agentic-system session.",
            "Your task is to maintain workflow_summary as canonical long-term memory of the whole session, keeping key facts from early and recent stages.",
            "Input format:",
            "1) workflow_summary: existing long-term summary memory from previous turns.",
            "2) workflow_history: chronological records (earliest to latest).",
            "    - workflow_history line format: [UTC_ISO_TIMESTAMP] role> content.",
            "Long-term memory update policy:",
            "1) Preserve durable key facts already stored in workflow_summary.",
            "2) Update facts when new evidence changes them; do not silently delete changed facts.",
            "3) Mark replaced facts as superseded when needed (for example: '(superseded: <new fact>)').",
            "4) Append newly confirmed key facts from workflow_history.",
            "5) Never forget initial user objective, major constraints/preferences, or key decisions unless explicitly invalidated.",
            "6) Keep full picture and key points only; remove low-level execution detail and repetition.",
            "Section policy:",
            "- Durable sections must keep long-term facts and evolve over time.",
            "- Current-state sections should reflect latest ground truth.",
            "Required workflow_summary structure (use exactly these section headers):",
            "Session Goal & Scope (durable)",
            "Constraints & Preferences (durable)",
            "Key Decisions (durable)",
            "Plan Evolution (durable)",
            "Progress Milestones (timeline)",
            "Current Status",
            "Open Loops & Next Actions",
            "Formatting requirements:",
            "- Use concise bullets under each section.",
            "- Keep content factual, non-speculative, and grounded in input.",
            "- Keep relationships clear: decision -> action -> outcome.",
            "- Keep summary compact but complete enough for long-horizon continuity.",
            "Material-change rule:",
            "- Only emit a new workflow_summary when there is material change in key facts, status, blockers, decisions, or next actions.",
            "- If no material change, return an empty workflow_summary string to preserve existing memory.",
            "Output format:",
            "Return one JSON object wrapped in <output> and </output>.",
            "Do not output any text outside that block.",
            "Example:",
            "<output>",
            "{\"workflow_summary\":\"\"}",
            "</output>",
        ]
    )
    WORKFLOW_COMPACTOR_PROMPT = "\n".join(
        [
            "You are an objective observer compressing older workflow records for runtime memory control.",
            "Your task is to compact older workflow_history context into ONE concise chronological text block.",
            "This output is for context-window control, not for progress tracking.",
            "Use workflow_summary only as supporting context to preserve important facts.",
            "Input format:",
            "1) workflow_summary: canonical full-progress summary.",
            "2) workflow_history: older workflow records in strict time order (earliest to latest).",
            "    - workflow_history line format: [UTC_ISO_TIMESTAMP] role> content.",
            "Compression target:",
            "- Output ONE paragraph string, no bullets, no newlines, no markdown, no JSON inside the string.",
            "- Keep it factual and chronological.",
            "- Keep major planning, decisions, actions with outcomes, blockers, and unresolved loops.",
            "- Exclude advice, speculation, and new planning.",
            "- Keep it compact; target <= 1200 characters.",
            "Output format:",
            "Return one JSON object wrapped in <output> and </output>.",
            "Do not output any text outside that block.",
            "Example:",
            "<output>",
            "{\"workflow_hist_compact\":\"...\"}",
            "</output>",
        ]
    )

    def __init__(
        self,
        workspace: str | Path,
        token_window_limit: int = int(256000*0.75),
        compact_keep_last_k: int = 10,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.runtime_prompts_root = self.workspace / "prompts"
        self.runtime_skills_root = self.workspace / "skills"
        self.runtime_knowledge_root = self.workspace / "knowledge"
        self.token_window_limit = int(token_window_limit)
        self.compact_keep_last_k = max(1, int(compact_keep_last_k))
        self.system_prompts_path = self.runtime_prompts_root / "agent_system_prompt.json"
        self.agent_role_descriptions_path = self.runtime_prompts_root / "agent_role_description.json"

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

    def _load_system_prompts(self) -> dict[str, str]:
        return self._load_json_map(self.system_prompts_path)

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

    def _load_skill_meta_text(self, role: str) -> str:
        role_name = str(role).strip()
        include_core = role_name == "core_agent"
        include_all = True
        builtin_loader_skill_ids = {"load-skill", "load-knowledge-docs"}

        roots: list[tuple[Path, str]] = []
        if include_core:
            roots.append((self.runtime_skills_root / "core-agent", "core-agent"))
        if include_all:
            roots.append((self.runtime_skills_root / "all-agents", "all-agents"))

        rows: list[dict[str, Any]] = []
        for root, scope in roots:
            if not root.exists():
                continue
            for skill_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                skill_id = skill_dir.name
                if skill_id in builtin_loader_skill_ids:
                    continue
                skill_md_path = skill_dir / "SKILL.md"
                if not skill_md_path.exists():
                    continue
                text = ""
                try:
                    text = skill_md_path.read_text(encoding="utf-8")
                except OSError:
                    text = ""
                frontmatter = self._parse_frontmatter(text)
                name = str(frontmatter.get("name", skill_id)).strip() or skill_id
                handler = str(frontmatter.get("handler", "")).strip()
                description = str(frontmatter.get("description", "")).strip()
                path = f"skills/{scope}/{skill_id}"
                handler_path = f"{path}/{handler}" if handler else ""
                rows.append(
                    {
                        "skill_id": skill_id,
                        "scope": scope,
                        "path": path,
                        "handler": handler_path,
                        "name": name,
                        "description": description,
                        "required_tools": self._parse_csv_field(frontmatter.get("required_tools", "")),
                        "recommended_tools": self._parse_csv_field(frontmatter.get("recommended_tools", "")),
                        "forbidden_tools": self._parse_csv_field(frontmatter.get("forbidden_tools", "")),
                    }
                )

        rows = [
            row
            for row in rows
            if str(row.get("skill_id", "")).strip() and str(row.get("scope", "")).strip()
        ]
        rows.sort(key=lambda item: (str(item.get("scope", "")), str(item.get("skill_id", ""))))
        if not rows:
            return "- (no skills found)"
        return "\n".join("- " + json.dumps(row, ensure_ascii=True) for row in rows)

    def _load_builtin_reference_loaders_text(self, role: str) -> str:
        role_name = str(role).strip()
        if role_name != "core_agent":
            return "- (built-in reference loaders not exposed for this role)"
        rows = [
            {
                "loader": "load-skill",
                "purpose": "Load full SKILL.md and scripts list for a target skill into workflow_history.",
                "script_path": "skills/all-agents/load-skill/scripts/load_skill.py",
                "code_type": "python",
                "required_args": ["--skill-id", "<skill_id>", "--scope", "all-agents|core-agent"],
                "example_action_input": {
                    "job_name": "load-skill-search-online-context",
                    "code_type": "python",
                    "script_path": "skills/all-agents/load-skill/scripts/load_skill.py",
                    "script_args": ["--skill-id", "search-online-context", "--scope", "all-agents"],
                },
            },
            {
                "loader": "load-knowledge-docs",
                "purpose": "Load selected knowledge docs into workflow_history by doc-id and/or doc-path.",
                "script_path": "skills/all-agents/load-knowledge-docs/scripts/load_knowledge_docs.py",
                "code_type": "python",
                "required_args": ["at least one of --doc-id or --doc-path"],
                "optional_args": ["--max-docs", "--max-chars-per-doc"],
                "example_action_input": {
                    "job_name": "load-knowledge-llm-post-training",
                    "code_type": "python",
                    "script_path": "skills/all-agents/load-knowledge-docs/scripts/load_knowledge_docs.py",
                    "script_args": ["--doc-id", "llm-post-training-rl-overview", "--max-docs", "4"],
                },
            },
        ]
        return "\n".join("- " + json.dumps(row, ensure_ascii=True) for row in rows)

    @staticmethod
    def _normalize_tags(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    def _load_knowledge_meta_text(self, role: str, limit: int = 80) -> str:
        role_name = str(role).strip()
        if role_name != "core_agent":
            return "- (knowledge meta not exposed for this role)"

        catalog_path = self.runtime_knowledge_root / "index" / "catalog.json"
        if not catalog_path.exists():
            return "- (no knowledge docs found)"
        try:
            raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            return "- (no knowledge docs found)"
        if not isinstance(raw, list):
            return "- (no knowledge docs found)"

        rows: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id:
                continue
            title = str(item.get("title", "")).strip() or doc_id
            path = str(item.get("path", "")).strip() or f"knowledge/docs/{doc_id}.md"
            tags = self._normalize_tags(item.get("tags", []))
            row = {
                "doc_id": doc_id,
                "title": title,
                "path": path,
                "tags": tags,
                "quality_score": float(item.get("quality_score", 0.0) or 0.0),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            }
            rows.append(row)

        rows.sort(key=lambda item: str(item.get("doc_id", "")))
        rows = rows[: max(1, int(limit))]
        if not rows:
            return "- (no knowledge docs found)"
        return "\n".join("- " + json.dumps(row, ensure_ascii=True) for row in rows)

    def _get_system_prompt(self, role: str) -> str:
        role = str(role).strip()
        if role == "workflow_summarizer":
            return self.WORKFLOW_SUMMARIZER_PROMPT
        
        if role == "workflow_compactor":
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

        skills_lines = [f"Below is the available skill metadata:"]
        skills_text = self._load_skill_meta_text(role)
        if skills_text:
            skills_lines.append(skills_text)
        else:
            skills_lines.append("- (no skills found)")
        skills_section = "\n".join(skills_lines)

        knowledge_lines = [f"Below is the available knowledge metadata:"]
        knowledge_text = self._load_knowledge_meta_text(role)
        if knowledge_text:
            knowledge_lines.append(knowledge_text)
        else:
            knowledge_lines.append("- (no knowledge docs found)")
        knowledge_section = "\n".join(knowledge_lines)

        loader_lines = [f"Below are built-in reference loaders:"]
        loader_text = self._load_builtin_reference_loaders_text(role)
        if loader_text:
            loader_lines.append(loader_text)
        else:
            loader_lines.append("- (no built-in reference loaders found)")
        loader_section = "\n".join(loader_lines)

        text = selected.strip()
        if self._AGENT_ROLES_PLACEHOLDER in text:
            text = text.replace(self._AGENT_ROLES_PLACEHOLDER, roles_section)
        if self._SKILLS_META_PLACEHOLDER in text:
            text = text.replace(self._SKILLS_META_PLACEHOLDER, skills_section)
        if self._KNOWLEDGE_META_PLACEHOLDER in text:
            text = text.replace(self._KNOWLEDGE_META_PLACEHOLDER, knowledge_section)
        if self._BUILTIN_REFERENCE_LOADERS_PLACEHOLDER in text:
            text = text.replace(self._BUILTIN_REFERENCE_LOADERS_PLACEHOLDER, loader_section)
        if self._RUNTIME_WORKSPACE_PLACEHOLDER in text:
            text = text.replace(self._RUNTIME_WORKSPACE_PLACEHOLDER, str(self.workspace))
        if self._LATEST_CONTEXT_PLACEHOLDER in text:
            text = text.replace(self._LATEST_CONTEXT_PLACEHOLDER, "{{LATEST_CONTEXT_VALUE}}")
        return text

    @staticmethod
    def _extract_latest_context(workflow_history_lines: list[str]) -> str:
        for line in reversed(workflow_history_lines):
            text = str(line).strip()
            if text:
                return text
        return ""

    def _build_core_prompt_text(
        self,
        *,
        system_prompt: str,
        workflow_summary: str,
        workflow_history_lines: list[str],
    ) -> str:
        latest_context = self._extract_latest_context(workflow_history_lines)
        sections: list[str] = []
        if system_prompt.strip():
            sections.append(system_prompt.strip())
        sections.append(
            "\n".join(
                [
                    "Latest Context",
                    "<latest_context>",
                    latest_context if latest_context else "(empty)",
                    "</latest_context>",
                    "Workflow Summary",
                    "<workflow_summary>",
                    workflow_summary if workflow_summary else "(empty)",
                    "</workflow_summary>",
                    "Workflow History",
                    "<workflow_history>",
                    "\n".join(workflow_history_lines) if workflow_history_lines else "(empty)",
                    "</workflow_history>",
                ]
            )
        )
        final_prompt = "\n\n".join(sections)
        return final_prompt.replace("{{LATEST_CONTEXT_VALUE}}", latest_context if latest_context else "(empty)")

    def _build_observer_prompt_text(
        self,
        *,
        system_prompt: str,
        workflow_summary: str,
        workflow_history_text: str,
    ) -> str:
        sections: list[str] = []
        if system_prompt.strip():
            sections.append(system_prompt.strip())
        sections.append(
            "\n".join(
                [
                    "<workflow_summary>",
                    workflow_summary if workflow_summary else "(empty)",
                    "</workflow_summary>",
                    "<workflow_history>",
                    workflow_history_text if workflow_history_text else "(empty)",
                    "</workflow_history>",
                ]
            )
        )
        return "\n\n".join(sections)

    def _generate_observer_value_with_retries(
        self,
        *,
        model_router: Any,
        role: str,
        final_prompt: str,
        value_key: str,
        require_non_empty: bool,
    ) -> tuple[str | None, str]:
        format_attempts = 0
        other_attempts = 0
        while True:
            parse_error = ""
            response: Any = {}
            try:
                response = model_router.generate(
                    role=role,
                    final_prompt=final_prompt,
                )
                if not isinstance(response, dict):
                    parse_error = f"{role} returned non-object response"
                elif not bool(response.get("_parse_ok", False)):
                    parse_error = str(response.get("_parse_error", "")).strip() or f"failed to parse {role} output"
            except Exception as exc:
                parse_error = f"{role} call failed: {exc}"

            if not parse_error:
                candidate = response.get(value_key)
                if isinstance(candidate, str):
                    value = candidate.strip()
                    if not require_non_empty or value:
                        return value, ""
                    parse_error = f"{value_key} must be a non-empty string"
                else:
                    parse_error = f"{value_key} must be a string"

            if self._is_missing_output_block_error(parse_error):
                format_attempts += 1
                if format_attempts < self._FORMAT_RETRY_LIMIT:
                    continue
            else:
                other_attempts += 1
                if other_attempts < self._DEFAULT_RETRY_LIMIT:
                    continue
            return None, parse_error

    def _compact_workflow_history(
        self,
        state: Any,
        model_router: Any,
        compact_keep_last_k: int
    ) -> None:
        head = state.workflow_hist[:-compact_keep_last_k] if compact_keep_last_k < len(state.workflow_hist) else []
        tail = state.workflow_hist[-compact_keep_last_k:] if compact_keep_last_k > 0 else []
        workflow_summary = state.workflow_summary if isinstance(state.workflow_summary, str) else ""
        system_prompt = self._get_system_prompt("workflow_compactor")
        final_prompt = self._build_observer_prompt_text(
            system_prompt=system_prompt,
            workflow_summary=workflow_summary,
            workflow_history_text="\n".join(head) if head else "",
        )
        workflow_hist_compact, parse_error = self._generate_observer_value_with_retries(
            model_router=model_router,
            role="workflow_history_compactor",
            final_prompt=final_prompt,
            value_key="workflow_hist_compact",
            require_non_empty=True,
        )
        if workflow_hist_compact is not None:
            state.workflow_hist = [f"[{state.utc_now_iso()}] workflow_compactor> {workflow_hist_compact}"] + tail
            return
        if parse_error:
            print()
            print(
                "runtime> workflow_history_compactor failed after retries; "
                "keeping workflow_history unchanged. "
                f"last_error={parse_error}"
            )
        return

    def _refresh_workflow_summary_if_needed(
        self,
        state: Any,
        model_router: Any,
    ) -> None:
        if state is None or model_router is None:
            return
        workflow_history: list[str] = getattr(state, "workflow_hist", [])
        workflow_history_lines = [line for line in workflow_history if line.strip()]
        workflow_summary = str(getattr(state, "workflow_summary", "")).strip()
        system_prompt = self._get_system_prompt("workflow_summarizer")
        final_prompt = self._build_observer_prompt_text(
            system_prompt=system_prompt,
            workflow_summary=workflow_summary,
            workflow_history_text="\n".join(workflow_history_lines) if workflow_history_lines else "",
        )
        candidate, parse_error = self._generate_observer_value_with_retries(
            model_router=model_router,
            role="workflow_summarizer",
            final_prompt=final_prompt,
            value_key="workflow_summary",
            require_non_empty=False,
        )
        if candidate is not None:
            if candidate:
                state.workflow_summary = candidate
            return
        if parse_error:
            print()
            print(
                "runtime> workflow_summarizer failed after retries; "
                "keeping workflow_summary unchanged. "
                f"last_error={parse_error}"
            )
        return

    @staticmethod
    def _is_missing_output_block_error(parse_error: str) -> bool:
        text = str(parse_error or "").strip().lower()
        return (
            "missing <output>...</output> block" in text
            or "empty <output> block" in text
        )

    def build_prompt(
        self,
        role: str,
        state: Any | None = None,
        model_router: Any | None = None,
    ) -> str:
        system_prompt = self._get_system_prompt(role)
        workflow_summary = str(getattr(state, "workflow_summary", "")).strip()
        workflow_history: list[str] = getattr(state, "workflow_hist", [])
        workflow_history_lines = [line for line in workflow_history if line.strip()]
        final_prompt = self._build_core_prompt_text(
            system_prompt=system_prompt,
            workflow_summary=workflow_summary,
            workflow_history_lines=workflow_history_lines,
        )
        estimated_tokens = max(1, len(final_prompt) // 4)
        if (
            role == "core_agent"
            and estimated_tokens > self.token_window_limit
            and state is not None
            and model_router is not None
        ):
            self._refresh_workflow_summary_if_needed(
                state=state,
                model_router=model_router,
            )
            self._compact_workflow_history(
                state=state,
                model_router=model_router,
                compact_keep_last_k=self.compact_keep_last_k
            )
            workflow_summary = str(getattr(state, "workflow_summary", "")).strip()
            workflow_history: list[str] = getattr(state, "workflow_hist", [])
            workflow_history_lines = [line for line in workflow_history if line.strip()]
            final_prompt = self._build_core_prompt_text(
                system_prompt=system_prompt,
                workflow_summary=workflow_summary,
                workflow_history_lines=workflow_history_lines,
            )

        return final_prompt
