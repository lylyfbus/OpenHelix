"""System Prompt Builder — assembles the agent's system prompt.

Reads prompt templates, injects skill/knowledge metadata, and
produces the final system prompt string for the Agent.

This replaces the relevant parts of the legacy ``kernel/prompts.py``
while being compatible with existing prompt JSON templates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .skill_loader import load_skills, format_skills_for_prompt
from .knowledge_loader import load_knowledge_catalog, format_knowledge_for_prompt


# --------------------------------------------------------------------------- #
# Placeholder tokens (matching the legacy prompt templates)
# --------------------------------------------------------------------------- #

_SKILLS_META = "{{SKILLS_META_FROM_JSON}}"
_KNOWLEDGE_META = "{{KNOWLEDGE_META_FROM_JSON}}"
_BUILTIN_LOADERS = "{{BUILTIN_REFERENCE_LOADERS}}"
_WORKSPACE = "{{RUNTIME_WORKSPACE}}"


# --------------------------------------------------------------------------- #
# Built-in reference loader metadata
# --------------------------------------------------------------------------- #

_BUILTIN_REFERENCE_LOADERS = [
    {
        "loader": "load-skill",
        "purpose": "Load full SKILL.md and scripts list for a target skill into workflow_history.",
        "script_path": "skills/all-agents/load-skill/scripts/load_skill.py",
        "code_type": "python",
        "required_args": ["--skill-id", "<skill_id>", "--scope", "all-agents|core-agent"],
    },
]


# --------------------------------------------------------------------------- #
# System prompt loading
# --------------------------------------------------------------------------- #


def _load_sys_prompt(path: Path) -> dict[str, str]:
    """Load a system prompt JSON file mapping role names to prompt text.

    When the value is a list of strings, they are joined with newlines.
    Returns an empty dict on any error.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        role = str(key).strip()
        if not role:
            continue
        if isinstance(value, list):
            result[role] = "\n".join(str(item) for item in value)
        else:
            result[role] = str(value)
    return result


# --------------------------------------------------------------------------- #
# PromptBuilder
# --------------------------------------------------------------------------- #


class PromptBuilder:
    """Build the system prompt for the Agent from templates + runtime metadata.

    Reads prompt templates from the package's ``agentic_system/prompts/``
    directory. Loads skill metadata from ``{workspace}/skills/`` and
    knowledge metadata from ``{workspace}/knowledge/``.

    Replaces template placeholders with the loaded metadata
    and returns the fully assembled system prompt string.

    Args:
        workspace: Root workspace directory (contains skills/, knowledge/).
    """

    # Package-level prompts directory
    _PACKAGE_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.skills_root = self.workspace / "skills"
        self.knowledge_root = self.workspace / "knowledge"
        self.prompts_root = self._PACKAGE_PROMPTS

    def build(self, role: str = "core_agent") -> str:
        """Build the complete system prompt for a given agent role.

        Steps:
            1. Load the role's prompt template from JSON.
            2. Load skill metadata and format for injection.
            3. Load knowledge catalog and format for injection.
            4. Replace all placeholders in the template.

        Args:
            role: Agent role (e.g. "core_agent").

        Returns:
            Fully assembled system prompt string. Empty string if
            no template is found for the role.
        """
        # 1. Load template
        templates = _load_sys_prompt(self.prompts_root / "agent_system_prompt.json")
        template = templates.get(role, "")
        if not template:
            return ""

        # 2. Load skills
        skills = load_skills(self.skills_root)
        skills_text = format_skills_for_prompt(skills)

        # 3. Load knowledge
        catalog = load_knowledge_catalog(self.knowledge_root)
        knowledge_text = format_knowledge_for_prompt(catalog)

        # 4. Built-in loaders
        loaders_text = "\n".join(
            "- " + json.dumps(row, ensure_ascii=True)
            for row in _BUILTIN_REFERENCE_LOADERS
        )

        # 5. Replace placeholders
        prompt = template
        if _SKILLS_META in prompt:
            prompt = prompt.replace(_SKILLS_META, skills_text)
        if _KNOWLEDGE_META in prompt:
            prompt = prompt.replace(_KNOWLEDGE_META, knowledge_text)
        if _BUILTIN_LOADERS in prompt:
            prompt = prompt.replace(_BUILTIN_LOADERS, loaders_text)
        if _WORKSPACE in prompt:
            prompt = prompt.replace(_WORKSPACE, str(self.workspace))

        return prompt
