"""Runtime Host — interactive REPL that wires all framework components.

Replaces the legacy ``runtime.py`` + ``FlowEngine`` + ``StorageEngine``
with a clean host built on the new core abstractions.

Usage::

    host = RuntimeHost(workspace="/path/to/workspace", session_id="project-01", provider="ollama")
    host.start()
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from ..core.agent import Agent
from ..core.compactor import Compactor
from ..core.environment import Environment
from ..core.state import Turn
from .sandbox import DockerSandboxExecutor, docker_is_available
from ..providers import create_provider
from .local_model_service import LocalModelServiceManager, local_model_service_supported
from .loop import run_loop
from .approval import ApprovalPolicy
from .display import StreamingDisplay, write_framed_text
from .debug import render_session_view_html, open_file_in_viewer


# --------------------------------------------------------------------------- #
# Package-level paths
# --------------------------------------------------------------------------- #

_BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "builtin_skills"
_BUILTIN_SKILLS_MANIFEST_REL = Path(".runtime") / "builtin_skills_manifest.json"


# --------------------------------------------------------------------------- #
# Default tool configuration
# --------------------------------------------------------------------------- #

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _normalize_session_id(session_id: str) -> str:
    """Validate and normalize a session identifier."""
    candidate = session_id.strip()
    if not candidate or not _SESSION_ID_RE.fullmatch(candidate):
        raise ValueError(
            "session_id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$"
        )
    return candidate


def _read_session_payload(session_path: Path) -> dict[str, Any] | None:
    """Read a persisted session JSON payload."""
    try:
        raw = json.loads(session_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


# --------------------------------------------------------------------------- #
# RuntimeHost
# --------------------------------------------------------------------------- #


class RuntimeHost:
    """Interactive REPL host for the agentic framework.

    Wires together:
    - ``PromptBuilder`` for system prompt assembly
    - ``Agent`` for LLM-based reasoning
    - ``Environment`` for sandbox execution + history management
    - ``run_loop()`` for orchestration
    - ``ApprovalPolicy`` for controlled execution gating

    On startup, the host:
    1. Bootstraps built-in skills into the workspace
    2. Prepares runtime-managed tool environment (SearXNG, local services, etc.)
    3. Builds the system prompt from the workspace content

    Args:
        workspace: Global workspace root for shared skills, knowledge, and sessions.
        session_id: Session identifier used to resume/persist project state.
        base_url: LLM API base URL (default from LLM_BASE_URL env or http://localhost:11434/v1).
        api_key: LLM API key (default from LLM_API_KEY env).
        model: Model name (default from LLM_MODEL env or llama3.1:8b).
        mode: Execution mode ("auto" or "controlled").
        sandbox_backend: Internal/testing hook. Only ``docker`` is supported.
    """

    HELP_TEXT = "\n".join([
        "Commands:",
        "  /help            Show this help.",
        "  /status          Show session status.",
        "  /full_history    Open the session full_history view.",
        "  /observation     Open the session observation view.",
        "  /workflow_summary  Open the session workflow_summary view.",
        "  /last_prompt     Open the session last_prompt view.",
        "  /exit            Quit.",
    ])

    def __init__(
        self,
        workspace: Path,
        *,
        session_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mode: str = "controlled",
        sandbox_backend: str = "docker",
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if session_id is None:
            raise ValueError("session_id is required.")
        self.session_id = _normalize_session_id(session_id)
        self.session_root = (self.workspace / "sessions" / self.session_id).resolve()
        self.project_root = self.session_root / "project"
        self.docs_root = self.session_root / "docs"
        self.state_root = self.session_root / ".state"
        for path in (self.session_root, self.project_root, self.docs_root, self.state_root):
            path.mkdir(parents=True, exist_ok=True)
        self.session_path = (self.state_root / "session.json").resolve()
        self._session_loaded = False
        self.mode = mode
        self.requested_sandbox_backend = str(sandbox_backend).strip().lower() or "docker"
        self.resolved_sandbox_backend = "docker"
        self._sandbox_status_fields: dict[str, str] = {}
        self._local_model_service: LocalModelServiceManager | None = None
        self._prompt_session = self._build_prompt_session()

        # 1. Bootstrap built-in skills into workspace
        self._bootstrap_skills()

        # 2. Build components
        self._model = create_provider(base_url=base_url, api_key=api_key, model=model)

        self._stream_display = StreamingDisplay()
        self._sandbox_executor: object | None = None

        # Create the core agent — Agent owns prompt building from workspace
        self._agent = Agent(
            self._model,
            workspace=self.workspace,
            session_id=self.session_id,
            session_root=self.session_root,
            project_root=self.project_root,
            docs_root=self.docs_root,
            state_root=self.state_root,
        )

        sandbox_executor = self._resolve_sandbox_executor(
            sandbox_backend=self.requested_sandbox_backend,
        )
        self._sandbox_executor = sandbox_executor

        # Create environment with sandbox executor and compactor
        self._compactor = Compactor(self._model)
        self._env = Environment(
            workspace=self.workspace,
            executor=sandbox_executor,
            mode=mode,
            compactor=self._compactor,
        )
        self._env.approval_profile = getattr(
            sandbox_executor,
            "approval_profile",
            "docker-online-rw-workspace-v1",
        )
        raw_session = None
        if self._env.load_session(self.session_path):
            raw_session = _read_session_payload(self.session_path) or {}
        if raw_session is not None:
            self._agent.last_prompt = str(raw_session.get("last_prompt", "") or "")
            self._session_loaded = True

        # Register approval policy as execution gate
        self._approval = ApprovalPolicy(
            mode=mode,
            prompt=self._prompt_approval_choice,
        )
        self._env.on_before_execute(self._approval)

    # ----- Input ------------------------------------------------------------- #

    @staticmethod
    def _build_prompt_session() -> object:
        """Create multiline prompt session (Ctrl+D submits)."""
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _submit(event: Any) -> None:
            event.app.exit(result=event.app.current_buffer.text)

        return PromptSession(key_bindings=bindings)

    def _prompt_approval_choice(self, prompt_text: str) -> str:
        """Read approval input using the same Ctrl+D/Ctrl+C semantics as user input."""
        return str(
            self._prompt_session.prompt(
                prompt_text,
                multiline=True,
                prompt_continuation=lambda _w, _n, _s: "... ",
            )
        )

    # ----- Bootstrap -------------------------------------------------------- #

    def _bootstrap_skills(self) -> None:
        """Sync built-in skills from the package into the workspace.

        Copies ``helix/builtin_skills/`` into ``{workspace}/skills/``.
        Each built-in skill directory is replaced on startup so updates to
        the package propagate automatically. User-created skills in the
        workspace are preserved.
        """
        if not _BUILTIN_SKILLS_ROOT.exists():
            return

        ws_skills = self.workspace / "skills"
        ws_skills.mkdir(parents=True, exist_ok=True)
        manifest_path = self.workspace / _BUILTIN_SKILLS_MANIFEST_REL
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

        for scope_dir in sorted(p for p in _BUILTIN_SKILLS_ROOT.iterdir() if p.is_dir()):
            if scope_dir.name.startswith((".", "_")):
                continue
            target_scope = ws_skills / scope_dir.name
            target_scope.mkdir(parents=True, exist_ok=True)

            for skill_dir in sorted(p for p in scope_dir.iterdir() if p.is_dir()):
                if skill_dir.name.startswith((".", "_")):
                    continue
                if not (skill_dir / "SKILL.md").exists():
                    continue
                target_skill = target_scope / skill_dir.name
                managed_rel = f"{scope_dir.name}/{skill_dir.name}"
                currently_managed.add(managed_rel)
                # Replace entire skill directory to pick up updates
                if target_skill.exists():
                    if target_skill.is_dir():
                        shutil.rmtree(target_skill)
                    else:
                        target_skill.unlink()
                shutil.copytree(skill_dir, target_skill)

        for managed_rel in sorted(previously_managed - currently_managed):
            target = ws_skills / managed_rel
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        manifest_path.write_text(
            json.dumps(sorted(currently_managed), indent=2),
            encoding="utf-8",
        )

    # ----- Agent & streaming ------------------------------------------------ #

    @property
    def stream_display(self) -> StreamingDisplay:
        """Access the streaming display for loop integration."""
        return self._stream_display

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
        print(f"Sandbox: {self.resolved_sandbox_backend}")
        print("Type /help for commands. Type /exit to quit.")
        print("Multiline: Enter adds lines, Ctrl+D submits, Ctrl+C cancels.\n")

        try:
            while True:
                # Read user input
                try:
                    user_input = self._prompt_session.prompt(
                        "user> ",
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
        # Record user turn
        self._env.record(Turn(role="user", content=user_text))

        try:
            # Run agent loop (prints agent responses as they happen)
            run_loop(
                self._agent,
                self._env,
                model=self._model,
                on_turn_start=self._stream_display.reset,
                on_turn_end=self._stream_display.commit,
                on_turn_error=self._stream_display.discard,
                on_token_chunk=self._stream_display,
            )
        except RuntimeError as exc:
            message = f"Agent error: {exc}"
            write_framed_text(f"runtime> {message}", sys.stdout)
            self._env.record(Turn(role="runtime", content=message))
        finally:
            # Persist state after each interaction
            self._persist_session()

    def _handle_command(self, command_line: str) -> Optional[str]:
        """Process slash commands. Returns None for exit, string for output."""
        cmd = command_line.strip().split()[0].lower()

        if cmd == "/exit":
            return None
        if cmd == "/help":
            return self.HELP_TEXT
        if cmd == "/status":
            return self._status_text()
        if cmd == "/full_history":
            return self._open_session_field_view("full_history")
        if cmd == "/observation":
            return self._open_session_field_view("observation")
        if cmd == "/workflow_summary":
            return self._open_session_field_view("workflow_summary")
        if cmd == "/last_prompt":
            return self._open_session_field_view("last_prompt")
        return f"Unknown command: {cmd}. Use /help."

    def _status_text(self) -> str:
        """Build session status overview."""
        lines = [
            f"llm_base_url={self._model.base_url}",
            f"llm_model={self._model.model}",
            f"mode={self.mode}",
            f"sandbox_backend={self.resolved_sandbox_backend}",
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

    def _resolve_sandbox_executor(
        self,
        *,
        sandbox_backend: str,
    ) -> object:
        """Resolve the configured sandbox backend into a callable executor."""
        backend = str(sandbox_backend).strip().lower() or "docker"
        if backend != "docker":
            raise ValueError(f"Unsupported sandbox_backend: {sandbox_backend}")

        available, reason = docker_is_available()
        if not available:
            raise ValueError(f"Docker sandbox unavailable: {reason}")

        executor = DockerSandboxExecutor(
            self.workspace,
            session_id=self.session_id,
        )
        if local_model_service_supported() and hasattr(executor, "attach_local_model_service"):
            service = LocalModelServiceManager(
                self.workspace,
                session_id=self.session_id,
            )
            service.start()
            executor.attach_local_model_service(service.tool_environment())
            self._local_model_service = service
        prepare_runtime = getattr(executor, "prepare_runtime", None)
        try:
            if callable(prepare_runtime):
                prepare_runtime()
        except Exception:
            if self._local_model_service is not None:
                with contextlib.suppress(Exception):
                    self._local_model_service.stop()
                self._local_model_service = None
            raise
        self.resolved_sandbox_backend = "docker"
        self._sandbox_status_fields = executor.status_fields()
        for key, value in executor.tool_environment().items():
            os.environ[key] = value
        return executor

    def _shutdown(self) -> None:
        """Persist state before exit."""
        try:
            self._persist_session()
        except Exception:
            pass
        sandbox_shutdown = getattr(self._sandbox_executor, "shutdown", None)
        if callable(sandbox_shutdown):
            try:
                sandbox_shutdown()
            except Exception:
                pass
        if self._local_model_service is not None:
            try:
                self._local_model_service.stop()
            except Exception:
                pass
            self._local_model_service = None

    def _open_session_field_view(self, field: str) -> str:
        """Persist and open a field-specific HTML view for the current session."""
        self._persist_session()
        raw_session = _read_session_payload(self.session_path)
        if raw_session is None:
            return f"Unable to read session file: {self.session_path}"

        value = raw_session.get(field, "(missing)")
        if field == "last_prompt" and not str(value):
            value = "(none yet)"
        html = render_session_view_html(
            session_id=self.session_id,
            field=field,
            session_path=self.session_path,
            value=value,
        )

        view_dir = self.state_root / "views"
        view_dir.mkdir(parents=True, exist_ok=True)
        view_path = (view_dir / f"{self.session_id}.{field}.html").resolve()
        view_path.write_text(html, encoding="utf-8")

        if open_file_in_viewer(view_path):
            return f"Opened session view: {view_path}"
        return f"Session view written: {view_path}"

    def _persist_session(self) -> None:
        """Persist session state when a named session is active."""
        self._env.save_session(
            self.session_path,
            extra_fields={"last_prompt": getattr(self._agent, "last_prompt", "")}
        )

    def _session_state(self) -> str:
        """Return a short user-visible description of current session mode."""
        return "loaded" if self._session_loaded else "new"
