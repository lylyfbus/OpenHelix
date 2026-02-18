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
        self.prompt_engine = PromptEngine(workspace=self.workspace)
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
                "  /status          Show runtime status.",
                "  /refresh         Start a new session in current workspace.",
                "  /exit            Quit.",
            ]
        )

    def _status_text(self) -> str:
        workflow_summary = getattr(self.state, "workflow_summary", "")
        return "\n".join(
            [
                f"session_id={self.state.session_id}",
                f"provider={self.provider}",
                f"mode={self.mode}",
                f"full_proc_hist_lines={len(self.state.full_proc_hist)}",
                f"workflow_hist_lines={len(self.state.workflow_hist)}",
                f"workflow_summary={workflow_summary if isinstance(workflow_summary, str) else ''}",
            ]
        )

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
            return self._status_text()
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
