"""Agent — the LLM brain.

A pure function conceptually: (system_prompt, state) → Action.
The agent itself is stateless; all state lives in the environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .action import Action, parse_action, ALLOWED_CORE_ACTIONS
from .state import State
from ..providers.openai_compat import LLMProvider


# --------------------------------------------------------------------------- #
# Prompt Building & Context Loading
# --------------------------------------------------------------------------- #

_SKILLS_META = "{{SKILLS_META_FROM_JSON}}"
_WORKSPACE_ROOT = "{{WORKSPACE_ROOT}}"
_SESSION_ROOT = "{{SESSION_ROOT}}"
_PROJECT_ROOT = "{{PROJECT_ROOT}}"
_DOCS_ROOT = "{{DOCS_ROOT}}"
_SUB_AGENT_ROLE = "{{SUB_AGENT_ROLE}}"

# Package-level prompts directory
_PACKAGE_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _load_sys_prompt(path: Path) -> dict[str, str]:
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
    result: dict[str, str] = {}
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _read_skill_row(skill_dir: Path, skills_root: Path) -> dict[str, Any] | None:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return None
    fm = _parse_frontmatter(text)
    name = fm.get("name", skill_dir.name).strip() or skill_dir.name
    description = fm.get("description", "").strip()
    path = f"skills/{skill_dir.relative_to(skills_root)}"
    return {
        "name": name,
        "description": description,
        "path": path,
    }


def _load_skills(skills_root: Path) -> list[dict[str, Any]]:
    """Load skills from the workspace skills directory.

    Handles two layouts:
    - ``skills/{skill}/SKILL.md``                  — user skill
    - ``skills/builtin_skills/{skill}/SKILL.md``   — built-in skill
    """
    skills_root = Path(skills_root)
    if not skills_root.exists():
        return []

    rows: list[dict[str, Any]] = []

    for entry in sorted(skills_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue

        if (entry / "SKILL.md").exists():
            # User skill directly under skills/
            row = _read_skill_row(entry, skills_root)
            if row:
                rows.append(row)
        elif entry.name == "builtin_skills":
            # Built-in skills under builtin_skills/
            for skill_dir in sorted(entry.iterdir()):
                if not skill_dir.is_dir() or skill_dir.name.startswith((".", "_")):
                    continue
                row = _read_skill_row(skill_dir, skills_root)
                if row:
                    rows.append(row)

    rows.sort(key=lambda r: r["path"])
    return rows


def _build_system_prompt(
    workspace_path: Path,
    role: str = "core_agent",
    *,
    session_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    docs_root: Optional[Path] = None,
    sub_agent_role: str = "",
) -> str:
    """Build the complete system prompt from templates + runtime metadata.

    The *role* selects which template to load from ``agent_system_prompt.json``
    (``core_agent`` or ``sub_agent``).
    """
    workspace = Path(workspace_path).expanduser().resolve()
    session_dir = Path(session_root).expanduser().resolve() if session_root is not None else workspace
    project_dir = Path(project_root).expanduser().resolve() if project_root is not None else session_dir / "project"
    docs_dir = Path(docs_root).expanduser().resolve() if docs_root is not None else session_dir / "docs"

    # 1. Load template
    templates = _load_sys_prompt(_PACKAGE_PROMPTS / "agent_system_prompt.json")
    template = templates.get(role, "")
    if not template:
        return ""

    # 2. Load skills
    skills = _load_skills(workspace / "skills")
    skills_text = "- (no skills found)" if not skills else "\n".join(
        "- " + json.dumps(row, ensure_ascii=True) for row in skills
    )

    # 3. Replace placeholders
    prompt = template
    if _SKILLS_META in prompt:
        prompt = prompt.replace(_SKILLS_META, skills_text)
    if _WORKSPACE_ROOT in prompt:
        prompt = prompt.replace(_WORKSPACE_ROOT, str(workspace))
    if _SESSION_ROOT in prompt:
        prompt = prompt.replace(_SESSION_ROOT, str(session_dir))
    if _PROJECT_ROOT in prompt:
        prompt = prompt.replace(_PROJECT_ROOT, str(project_dir))
    if _DOCS_ROOT in prompt:
        prompt = prompt.replace(_DOCS_ROOT, str(docs_dir))

    # Sub-agent-specific placeholders
    if _SUB_AGENT_ROLE in prompt:
        prompt = prompt.replace(_SUB_AGENT_ROLE, sub_agent_role)

    return prompt


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    """LLM-based agent that produces Actions from State.

    The agent is stateless — it reads the current State (built by the
    Environment) and produces an Action.  All persistent state lives
    in the Environment's history.

    The system prompt is built from the workspace's skills and prompt
    templates. The ``role`` selects which template to use
    (``core_agent`` or ``sub_agent``).

    Args:
        model: LLMProvider instance.
        workspace: Workspace directory (builds system prompt from skills).
        role: Agent role — selects the prompt template.
        allowed_actions: Which action types this agent can emit.
    """

    def __init__(
        self,
        model: LLMProvider,
        *,
        workspace: Path,
        role: str = "core_agent",
        session_root: Optional[Path] = None,
        project_root: Optional[Path] = None,
        docs_root: Optional[Path] = None,
        sub_agent_role: str = "",
        allowed_actions: frozenset[str] = ALLOWED_CORE_ACTIONS,
    ) -> None:
        self.model = model
        self.role = role
        self.allowed_actions = allowed_actions
        self.last_prompt = ""
        self._workspace_prompt_args = {
            "workspace_path": workspace,
            "role": role,
            "session_root": session_root,
            "project_root": project_root,
            "docs_root": docs_root,
            "sub_agent_role": sub_agent_role,
        }
        self.system_prompt = ""

    def act(
        self,
        state: State,
        *,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> Action:
        """Given current state, produce the next Action.

        Raises:
            ActionParseError: If the model output fails parsing/validation.
        """
        messages = self._build_messages(state)
        self.last_prompt = messages
        raw_output = self.model.generate(
            messages,
            chunk_callback=chunk_callback,
        )
        return parse_action(raw_output, allowed_actions=self.allowed_actions)

    # ----- prompt construction --------------------------------------------- #

    def _build_messages(self, state: State) -> list[dict[str, str]]:
        """Build the OpenAI chat messages array from system prompt + state.

        System message = system_prompt + workflow_summary (long-term memory).
        Adjacent turns that map to the same API role are merged so the
        array always alternates between ``user`` and ``assistant``.
        Each turn is prefixed with ``[role]`` so the LLM can still
        distinguish the original source.
        """
        self.system_prompt = _build_system_prompt(**self._workspace_prompt_args)
        system_content = self.system_prompt
        if state.workflow_summary:
            system_content += (
                "\n\n## Workflow Summary (long-term memory)\n"
                + state.workflow_summary
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        for turn in state.observation:
            api_role = "user" if turn.role in ("user", "runtime") else "assistant"
            content = f"[{turn.role}] {turn.content}"
            if len(messages) > 1 and messages[-1]["role"] == api_role:
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": api_role, "content": content})
        return messages
