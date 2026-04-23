"""Runtime Host — interactive REPL that wires all framework components.

Usage::

    host = RuntimeHost(
        workspace="/path/to/workspace",
        session_id="project-01",
        endpoint_url="http://localhost:11434/v1",
        model="llama3.1:8b",
    )
    host.start()
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from ..core.agent import Agent
from ..core.compactor import Compactor
from ..core.environment import Environment
from ..core.state import Turn
from .sandbox import HostSandboxExecutor
from ..providers.openai_compat import LLMProvider
from ..services.searxng import discover as discover_searxng
from ..services.local_model_service import discover as discover_lms
from .loop import run_loop
from . import sub_agent_meta
from .approval import ApprovalPolicy
from .display import StreamingDisplay, write_runtime
from .debug import render_session_view_html, open_file_in_viewer


class RuntimeHost:
    """Interactive REPL host for the agentic framework.

    On startup, the host:
    1. Bootstraps built-in skills into the workspace
    2. Discovers running services (SearXNG, local model service)
    3. Prepares the host-shell sandbox
    4. Resumes previous session if available

    Args:
        workspace: Global workspace root for shared skills, knowledge, and sessions.
        session_id: Session identifier used to resume/persist project state.
        endpoint_url: LLM API endpoint URL.
        model: Model name.
        api_key: LLM API key (empty for unauthenticated endpoints).
        mode: Execution mode ("auto" or "controlled").
    """

    _SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        candidate = session_id.strip()
        if not candidate or not RuntimeHost._SESSION_ID_RE.fullmatch(candidate):
            raise ValueError(
                "session_id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$"
            )
        return candidate

    @staticmethod
    def _read_session_payload(session_path: Path) -> dict[str, Any] | None:
        try:
            raw = json.loads(session_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return raw if isinstance(raw, dict) else None

    _VIEW_FIELDS = ("full_history", "observation", "workflow_summary", "last_prompt")

    HELP_TEXT = "\n".join([
        "Commands:",
        "  /help                        Show this help.",
        "  /status                      Show session status.",
        "  /view <field>                Inspect the main session (fields: " + ", ".join(_VIEW_FIELDS) + ").",
        "  /view <field> <role>         Inspect a sub-agent's field by role.",
        "  /view sub_agents             List all sub-agents created in this session.",
        "  /exit                        Quit.",
    ])

    def __init__(
        self,
        workspace: Path,
        *,
        session_id: str,
        endpoint_url: str,
        model: str,
        api_key: str = "",
        mode: str = "controlled",
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.knowledge_root = self.workspace / "knowledge"
        self.knowledge_root.mkdir(parents=True, exist_ok=True)
        knowledge_index = self.knowledge_root / "index.json"
        if not knowledge_index.exists():
            knowledge_index.write_text("[]\n", encoding="utf-8")
        self.session_id = self._normalize_session_id(session_id)
        self.session_root = (self.workspace / "sessions" / self.session_id).resolve()
        self.project_root = self.session_root / "project"
        self.docs_root = self.session_root / "docs"
        self.state_root = self.session_root / ".state"
        for path in (self.session_root, self.project_root, self.docs_root, self.state_root):
            path.mkdir(parents=True, exist_ok=True)
        self.session_state_path = (self.state_root / "session_state.json").resolve()
        self._session_loaded = False
        self.mode = mode
        self._sandbox_status_fields: dict[str, str] = {}
        self._user_input_session = self._build_user_input_session()

        # 1. Bootstrap built-in skills into workspace
        self._bootstrap_skills()

        # 2. LLM provider
        self._model = LLMProvider(endpoint_url=endpoint_url, model=model, api_key=api_key)

        # 3. Agent
        self._agent = Agent(
            self._model,
            workspace=self.workspace,
            session_root=self.session_root,
            project_root=self.project_root,
            docs_root=self.docs_root,
            sub_agents_meta=sub_agent_meta.format_for_prompt(
                sub_agent_meta.load(self.state_root)
            ),
        )

        # 4. Discover services (started via `helix start`)
        searxng = discover_searxng()
        lms = discover_lms()
        local_model_env: dict[str, str] = {}
        if lms:
            local_model_env = {
                "HELIX_LOCAL_MODEL_SERVICE_URL": f"http://127.0.0.1:{lms['port']}",
                "HELIX_LOCAL_MODEL_SERVICE_TOKEN": lms["token"],
            }

        # 5. Host-shell sandbox
        self._sandbox_executor = HostSandboxExecutor(
            self.workspace,
            session_id=self.session_id,
            searxng_base_url=searxng["base_url"] if searxng else "",
            local_model_service_env=local_model_env,
        )
        self._sandbox_executor.prepare_runtime()
        self._sandbox_status_fields = self._sandbox_executor.status_fields()
        for key, value in self._sandbox_executor.tool_environment().items():
            os.environ[key] = value

        # 6. Compactor (LLM-based context summarization)
        self._compactor = Compactor(self._model)

        # 7. Environment (sandbox + compactor + history)
        self._env = Environment(
            workspace=self.workspace,
            executor=self._sandbox_executor,
            mode=mode,
            compactor=self._compactor,
            state_root=self.state_root,
        )
        self._env.approval_profile = self._sandbox_executor.approval_profile

        # 8. Resume previous session if available
        raw_session = None
        if self._env.load_session(self.session_state_path):
            raw_session = self._read_session_payload(self.session_state_path) or {}
        if raw_session is not None:
            saved_prompt = raw_session.get("last_prompt", "")
            self._agent.last_prompt = saved_prompt if isinstance(saved_prompt, list) else str(saved_prompt or "")
            self._session_loaded = True

        # 9. Approval policy (execution gate for controlled mode)
        self._approval = ApprovalPolicy(
            mode=mode,
            prompt=self._prompt_approval_choice,
        )
        self._env.on_before_execute(self._approval)

    # ----- Input ------------------------------------------------------------- #

    @staticmethod
    def _build_user_input_session() -> object:
        """Create multiline prompt session (Ctrl+D submits)."""
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _submit(event: Any) -> None:
            event.app.exit(result=event.app.current_buffer.text)

        style = Style.from_dict({
            "badge": "bold bg:#585858 #ffffff",
        })
        return PromptSession(key_bindings=bindings, style=style)

    def _prompt_approval_choice(self, prompt_text: str) -> str:
        """Read approval input using the same Ctrl+D/Ctrl+C semantics as user input."""
        return str(
            self._user_input_session.prompt(
                prompt_text,
                multiline=True,
                prompt_continuation=lambda _w, _n, _s: "... ",
            )
        )

    # ----- Bootstrap -------------------------------------------------------- #

    def _bootstrap_skills(self) -> None:
        """Sync built-in skills from the package into the workspace.

        Copies ``helix/builtin_skills/`` into ``{workspace}/skills/builtin_skills/``.
        Each built-in skill directory is replaced on startup so updates to
        the package propagate automatically. User-created skills elsewhere
        under ``{workspace}/skills/`` are never touched.
        """
        builtin_skills_root = Path(__file__).resolve().parent.parent / "builtin_skills"
        if not builtin_skills_root.exists():
            return

        ws_builtin = self.workspace / "skills" / "builtin_skills"
        ws_builtin.mkdir(parents=True, exist_ok=True)
        manifest_path = self.workspace / ".runtime" / "builtin_skills_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = []
        previously_managed = {
            str(item).strip()
            for item in previous_manifest
            if isinstance(item, str) and str(item).strip()
        }
        currently_managed: set[str] = set()

        for skill_dir in sorted(p for p in builtin_skills_root.iterdir() if p.is_dir()):
            if skill_dir.name.startswith((".", "_")):
                continue
            if not (skill_dir / "SKILL.md").exists():
                continue
            target_skill = ws_builtin / skill_dir.name
            currently_managed.add(skill_dir.name)
            # Replace entire skill directory to pick up updates
            if target_skill.exists():
                if target_skill.is_dir():
                    shutil.rmtree(target_skill)
                else:
                    target_skill.unlink()
            shutil.copytree(skill_dir, target_skill)

        for skill_name in sorted(previously_managed - currently_managed):
            target = ws_builtin / skill_name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        manifest_path.write_text(
            json.dumps(sorted(currently_managed), indent=2),
            encoding="utf-8",
        )

    # ----- REPL ------------------------------------------------------------- #

    def start(self) -> int:
        """Run the interactive REPL until the user exits.

        Returns:
            Exit code (0 for normal exit).
        """
        print(f"Agentic System — model={self._model.model}, mode={self.mode}")
        print(f"Workspace: {self.workspace}")
        state = "resumed" if self._session_loaded else "new"
        print(f"Session: {self.session_id} ({state})")
        print("Sandbox: docker")
        print("Type /help for commands. Type /exit to quit.")
        print("Multiline: Enter adds lines, Ctrl+D submits, Ctrl+C cancels.\n")

        try:
            while True:
                # Read user input
                try:
                    user_input = self._user_input_session.prompt(
                        [("class:badge", " user> "), ("", " ")],
                        multiline=True,
                        prompt_continuation=lambda _w, _n, _s: "... ",
                    )
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print("\nInterrupted. Use /exit to quit.")
                    continue

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    result = self._handle_command(user_input)
                    if result is None:  # exit signal
                        break
                    if result:
                        print(result)
                    continue

                # Process user message through the agent loop
                self._process_message(user_input)

            return 0
        finally:
            self._shutdown()

    def _process_message(self, user_text: str) -> None:
        """Record user message and run the agent loop to completion."""
        print()  # blank line after user input
        self._env.record(Turn(role="user", content=user_text))
        display = StreamingDisplay()

        try:
            run_loop(
                self._agent,
                self._env,
                model=self._model,
                on_turn_start=display.reset,
                on_turn_end=display.commit,
                on_turn_error=display.discard,
                on_token_chunk=display,
            )
        except RuntimeError as exc:
            message = f"Agent error: {exc}"
            write_runtime(f"runtime> {message}", sys.stdout)
            self._env.record(Turn(role="runtime", content=message))
        finally:
            # The loop already refreshes sub_agents_meta on the agent
            # immediately after each delegation, so we only need to persist
            # session state here.
            self._persist_session()

    def _handle_command(self, command_line: str) -> Optional[str]:
        """Process slash commands. Returns None for exit, string for output."""
        parts = command_line.strip().split()
        cmd = parts[0].lower()

        if cmd == "/exit":
            return None
        if cmd == "/help":
            return self.HELP_TEXT
        if cmd == "/status":
            return self._status_text()
        if cmd == "/view":
            if len(parts) < 2:
                return (
                    "Usage:\n"
                    "  /view <field>               Inspect main session\n"
                    "  /view <field> <role>        Inspect a sub-agent's field by role\n"
                    "  /view sub_agents            List sub-agents in this session\n"
                    f"Fields: {', '.join(self._VIEW_FIELDS)}"
                )
            field = parts[1].lower()
            if field == "sub_agents":
                return self._list_sub_agents()
            if field not in self._VIEW_FIELDS:
                return (
                    f"Unknown field: {field!r}. "
                    f"Fields: {', '.join(self._VIEW_FIELDS)}. "
                    "Use '/view sub_agents' to list sub-agents."
                )
            if len(parts) >= 3:
                return self._open_sub_agent_field_view(parts[2], field)
            return self._open_session_field_view(field)
        return f"Unknown command: {cmd}. Use /help."

    def _status_text(self) -> str:
        """Build session status overview."""
        lines = [
            f"llm_endpoint_url={self._model.endpoint_url}",
            f"llm_model={self._model.model}",
            f"mode={self.mode}",
            f"sandbox_backend=docker",
            f"workspace={self.workspace}",
            f"session_id={self.session_id}",
            f"session_state={self._session_state()}",
            f"session_root={self.session_root}",
            f"project_root={self.project_root}",
            f"docs_root={self.docs_root}",
            f"state_root={self.state_root}",
            f"searxng={os.environ.get('SEARXNG_BASE_URL', 'not set')}",
            f"full_history_turns={len(self._env.full_history)}",
            f"observation_turns={len(self._env.observation)}",
            f"system_prompt_length={len(self._agent.system_prompt)} chars",
        ]
        for key, value in self._sandbox_status_fields.items():
            if key == "sandbox_backend":
                continue
            lines.append(f"{key}={value}")
        return "\n".join(lines)

    def _shutdown(self) -> None:
        """Persist state before exit."""
        try:
            self._persist_session()
        except Exception:
            pass
        try:
            self._sandbox_executor.shutdown()
        except Exception:
            pass

    def _open_field_view(
        self,
        *,
        field: str,
        session_path: Path,
        view_id: str,
        view_filename: str,
        description: str,
    ) -> str:
        """Read a JSON session file, render a field-specific HTML view, and open it.

        ``description`` is a short lowercase noun ("session view",
        "sub-agent view") used in the success message.
        """
        raw_session = self._read_session_payload(session_path)
        if raw_session is None:
            return f"Unable to read state file: {session_path}"

        value = raw_session.get(field, "(missing)")
        if field == "last_prompt" and not str(value):
            value = "(none yet)"
        html = render_session_view_html(
            session_id=view_id,
            field=field,
            session_path=session_path,
            value=value,
        )

        view_dir = self.state_root / "views"
        view_dir.mkdir(parents=True, exist_ok=True)
        view_path = (view_dir / view_filename).resolve()
        view_path.write_text(html, encoding="utf-8")

        if open_file_in_viewer(view_path):
            return f"Opened {description}: {view_path}"
        return f"{description.capitalize()} written: {view_path}"

    def _open_session_field_view(self, field: str) -> str:
        """Persist and open a field-specific HTML view for the current session."""
        self._persist_session()
        return self._open_field_view(
            field=field,
            session_path=self.session_state_path,
            view_id=self.session_id,
            view_filename=f"{self.session_id}.{field}.html",
            description="session view",
        )

    def _list_sub_agents(self) -> str:
        """List sub-agents persisted under this session's state_root."""
        meta = sub_agent_meta.load(self.state_root)
        if not meta:
            return "No sub-agents have been created in this session yet."
        lines = ["Sub-agents (use `/view <field> <role>` to inspect):"]
        for entry in meta:
            role = entry.get("role", "?")
            desc = entry.get("description", "") or "(no description)"
            state_file = self.state_root / "sub_agents" / f"{role}.json"
            marker = "persisted" if state_file.exists() else "no state yet"
            lines.append(f"  {role}: {desc} [{marker}]")
        return "\n".join(lines)

    def _open_sub_agent_field_view(self, role: str, field: str) -> str:
        """Open an HTML view for a specific sub-agent's field."""
        sub_state_path = (self.state_root / "sub_agents" / f"{role}.json").resolve()
        if not sub_state_path.exists():
            return (
                f"No state file for sub-agent {role!r} at {sub_state_path}. "
                "Use '/view sub_agents' to list available sub-agents."
            )
        return self._open_field_view(
            field=field,
            session_path=sub_state_path,
            view_id=f"{self.session_id}.sub_agent.{role}",
            view_filename=f"{self.session_id}.sub_agent.{role}.{field}.html",
            description="sub-agent view",
        )

    def _persist_session(self) -> None:
        """Persist session state when a named session is active."""
        self._env.save_session(
            self.session_state_path,
            extra_fields={"last_prompt": getattr(self._agent, "last_prompt", "")}
        )

    def _session_state(self) -> str:
        """Return a short user-visible description of current session mode."""
        return "loaded" if self._session_loaded else "new"
