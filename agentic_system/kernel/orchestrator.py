from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .constants import DEFAULT_LIMITS, TERMINAL_TOKENS
from .executors import execute
from .model_router import ModelRouter
from .prompts import PromptEngine
from .storage import StorageEngine


class FlowEngine:
    def __init__(
        self,
        workspace: str | Path,
        mode: str,
        approval_handler: Callable[[str], tuple[bool, str]] | None = None,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.mode = mode
        self.approval_handler = approval_handler
        self.limits = deepcopy(DEFAULT_LIMITS)
        if limits:
            self.limits.update(limits)

    def _ensure_runtime_fields(self, state: StorageEngine) -> None:
        if not isinstance(getattr(state, "full_proc_hist", None), list):
            state.full_proc_hist = []
        if not isinstance(getattr(state, "workflow_hist", None), list):
            state.workflow_hist = []
        if not isinstance(getattr(state, "workflow_summary", None), str):
            state.workflow_summary = ""

    def _run_sub_agent_loop(
        self,
        state: StorageEngine,
        *,
        agent_role: str,
        model_router: ModelRouter,
        prompt_engine: PromptEngine,
    ) -> None:
        role = str(agent_role).strip()
        if not role:
            text = "chat_with_sub_agent requires action_input.agent_role"
        else:
            text = f"sub-agent loop placeholder for role '{role}'"
        state.update_state(
            role="runtime",
            text=text,
            prompt_engine=prompt_engine,
            model_router=model_router,
        )
        state.save_state()

    def _run_core_agent_loop(
        self,
        state: StorageEngine,
        *,
        model_router: ModelRouter,
        prompt_engine: PromptEngine,
    ) -> None:
        max_turns = int(self.limits.get("max_inner_turns", 60))
        turns = 0

        self._ensure_runtime_fields(state)
        response = model_router.generate(
            role="core_agent",
            state=state,
            prompt_engine=prompt_engine,
        )
        state.update_state(
            role="core_agent",
            text=str(response.get("raw_response", "")),
            prompt_engine=prompt_engine,
            model_router=model_router,
        )
        state.save_state()
        action = str(response.get("action", "none")).strip().lower()
        action_input = dict(response.get("action_input", {}))

        while action not in TERMINAL_TOKENS and turns < max_turns:
            turns += 1
            if action == "chat_with_requester":
                break
            elif action == "chat_with_sub_agent":
                agent_role = str(action_input.get("agent_role", "")).strip()
                self._run_sub_agent_loop(
                    state=state,
                    agent_role=agent_role,
                    model_router=model_router,
                    prompt_engine=prompt_engine,
                )
            elif action == "exec":
                if not isinstance(action_input, dict):
                    state.update_state(
                        role="runtime",
                        text="exec action requires object action_input",
                        prompt_engine=prompt_engine,
                        model_router=model_router,
                    )
                    state.save_state()
                else:
                    code_type = str(action_input.get("code_type", "bash")).strip().lower()
                    script_path = str(action_input.get("script_path", "")).strip()
                    script = str(action_input.get("script", "")).strip()
                    try:
                        exec_result = execute(
                            code_type=code_type,
                            script_path=script_path,
                            script=script,
                            workspace=self.workspace,
                        )
                        state.update_state(
                            role="runtime",
                            text=json.dumps(exec_result, ensure_ascii=True),
                            prompt_engine=prompt_engine,
                            model_router=model_router,
                        )
                    except Exception as exc:
                        state.update_state(
                            role="runtime",
                            text=f"exec error: {exc}",
                            prompt_engine=prompt_engine,
                            model_router=model_router,
                        )
                    state.save_state()
            else:
                state.update_state(
                    role="runtime",
                    text=f"unsupported action: {action}",
                    prompt_engine=prompt_engine,
                    model_router=model_router,
                )
                state.save_state()

            self._ensure_runtime_fields(state)
            response = model_router.generate(
                role="core_agent",
                state=state,
                prompt_engine=prompt_engine,
            )
            state.update_state(
                role="core_agent",
                text=str(response.get("raw_response", "")),
                prompt_engine=prompt_engine,
                model_router=model_router,
            )
            state.save_state()
            action = str(response.get("action", "none")).strip().lower()
            action_input = dict(response.get("action_input", {}))

        if turns >= max_turns:
            state.update_state(
                role="runtime",
                text=f"max turns reached ({max_turns}); ending current loop",
                prompt_engine=prompt_engine,
                model_router=model_router,
            )
            state.save_state()

    def run_session(
        self,
        state: StorageEngine,
        command_handler: Callable[[str], str] | None = None,
        *,
        model_router: ModelRouter,
        prompt_engine: PromptEngine,
        skill_engine: Any | None = None,
        knowledge_engine: Any | None = None,
        policy_engine: Any | None = None,
    ) -> None:
        while True:
            try:
                line = input("user> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nInterrupted. Use /exit to quit.")
                continue

            stripped = line.strip()
            if not stripped:
                print("No input provided.")
                continue

            if command_handler is not None and stripped.startswith("/"):
                command_out = command_handler(stripped)
                if command_out == "__EXIT__":
                    break
                if command_out:
                    print(command_out)
                continue

            state.update_state(
                role="user",
                text=stripped,
                prompt_engine=prompt_engine,
                model_router=model_router,
            )
            state.save_state()
            self._run_core_agent_loop(
                state,
                model_router=model_router,
                prompt_engine=prompt_engine,
            )
