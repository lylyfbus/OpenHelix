from __future__ import annotations

from pathlib import Path

from .kernel import (
    FlowEngine,
    KnowledgeEngine,
    PolicyEngine,
    PromptEngine,
    StorageEngine,
    SkillEngine,
)
from .kernel.model_router import ModelRouter


class AgentRuntime:
    def __init__(
        self,
        workspace: str | Path,
        provider: str = "ollama",
        mode: str = "controlled",
        session_id: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.provider = str(provider).strip().lower() or "ollama"
        self.mode = mode
        self.state = StorageEngine(workspace=self.workspace, session_id=session_id)
        if session_id is not None:
            self.state.load_state()
        self.model_router = ModelRouter(provider=self.provider, model_name=model_name)
        self.skill_engine = SkillEngine(workspace=self.workspace)
        self.prompt_engine = PromptEngine(workspace=self.workspace, token_window_limit=9000)
        self.knowledge = KnowledgeEngine(workspace=self.workspace)
        self.policy = PolicyEngine()
        self.engine = FlowEngine(
            workspace=self.workspace,
            mode=self.mode,
            model_router=self.model_router,
            prompt_engine=self.prompt_engine,
            skill_engine=self.skill_engine,
            knowledge_engine=self.knowledge,
            policy_engine=self.policy,
            approval_handler=self._default_approval_prompt,
        )

        self._persist()

    @staticmethod
    def _default_approval_prompt(signature: str) -> tuple[bool, str]:
        print()
        print("Runtime confirmation required for exec action.")
        print(signature)
        print("Approve this execution? [y/N]")
        choice = input("> ").strip().lower()
        if choice in {"y", "yes", "once"}:
            return True, "once"
        return False, "deny"

    def _help_text(self) -> str:
        return "\n".join(
            [
                "Commands:",
                "  /help            Show help.",
                "  /status          Show runtime status overview.",
                "  /status workflow_summary   Show workflow_summary.",
                "  /status workflow_hist      Show workflow_hist lines.",
                "  /status full_proc_hist     Show full_proc_hist lines.",
                "  /status action_hist        Show LLM selected action history.",
                "  /status core_agent_prompt  Show the last full prompt sent to core_agent.",
                "  /refresh         Start a new session in current workspace.",
                "  /exit            Quit.",
            ]
        )

    def _status_overview_text(self) -> str:
        return "\n".join(
            [
                f"session_id={self.state.session_id}",
                f"provider={self.provider}",
                f"mode={self.mode}",
                f"full_proc_hist_lines={len(self.state.full_proc_hist)}",
                f"workflow_hist_lines={len(self.state.workflow_hist)}",
                f"action_hist_lines={len(getattr(self.state, 'action_hist', []))}",
            ]
        )

    def _status_workflow_summary_text(self) -> str:
        summary = getattr(self.state, "workflow_summary", "")
        text = summary if isinstance(summary, str) else ""
        return text if text.strip() else "(empty)"

    def _status_workflow_hist_text(self) -> str:
        rows = getattr(self.state, "workflow_hist", [])
        if not isinstance(rows, list) or not rows:
            return "(empty)"
        return "\n".join(str(line) for line in rows)

    def _status_full_proc_hist_text(self) -> str:
        rows = getattr(self.state, "full_proc_hist", [])
        if not isinstance(rows, list) or not rows:
            return "(empty)"
        return "\n".join(str(line) for line in rows)

    def _status_action_hist_text(self) -> str:
        rows = getattr(self.state, "action_hist", [])
        if not isinstance(rows, list) or not rows:
            return "(empty)"
        return "\n".join(str(line) for line in rows)

    def _status_core_agent_prompt_text(self) -> str:
        text = getattr(self.engine, "last_core_agent_prompt", "")
        value = text if isinstance(text, str) else ""
        return value if value.strip() else "(empty)"

    def _handle_command(self, command_line: str) -> str:
        parts = command_line.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd in {"/help"}:
            return self._help_text()
        if cmd in {"/refresh"}:
            return "__REFRESH__"
        if cmd in {"/exit"}:
            return "__EXIT__"
        if cmd == "/status":
            target = str(parts[1]).strip().lower() if len(parts) > 1 else ""
            if not target:
                return self._status_overview_text()
            if target == "workflow_summary":
                return self._status_workflow_summary_text()
            if target == "workflow_hist":
                return self._status_workflow_hist_text()
            if target == "full_proc_hist":
                return self._status_full_proc_hist_text()
            if target == "action_hist":
                return self._status_action_hist_text()
            if target == "core_agent_prompt":
                return self._status_core_agent_prompt_text()
            return "Unknown /status target. Use: workflow_summary | workflow_hist | full_proc_hist | action_hist | core_agent_prompt"
        return f"Unknown command: {cmd}. Use /help."

    def start(
        self,
        show_banner: bool = True,
    ) -> int:
        if show_banner:
            print(f"Session {self.state.session_id} started in provider={self.provider}, mode={self.mode}")
            print("Type /help for commands. Type /exit to quit.")

        try:
            while True:
                status = self.engine.run_session(
                    state=self.state,
                    command_handler=self._handle_command,
                )
                if status == "__REFRESH__":
                    self._persist()
                    self.state = StorageEngine(workspace=self.workspace, session_id=None)
                    print(f"Session refreshed. New session_id={self.state.session_id}")
                    continue
                break
            return 0
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._persist()

    def _persist(self) -> None:
        self.state.save_state()
