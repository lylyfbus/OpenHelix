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
        model_router: ModelRouter | None = None,
        prompt_engine: PromptEngine | None = None,
        skill_engine: Any | None = None,
        knowledge_engine: Any | None = None,
        policy_engine: Any | None = None,
        approval_handler: Callable[[str], tuple[bool, str]] | None = None,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.mode = mode
        self.model_router = model_router
        self.prompt_engine = prompt_engine
        self.skill_engine = skill_engine
        self.knowledge_engine = knowledge_engine
        self.policy_engine = policy_engine
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

    def _confirm_exec(self, action_input: dict[str, Any]) -> bool:
        if str(self.mode).strip().lower() == "auto":
            return True
        if self.approval_handler is None:
            return True
        signature = json.dumps(
            {
                "action": "exec",
                "code_type": str(action_input.get("code_type", "bash")).strip().lower(),
                "script_path": str(action_input.get("script_path", "")).strip(),
                "script_preview": str(action_input.get("script", "")).strip()[:240],
            },
            ensure_ascii=True,
        )
        try:
            allowed, _scope = self.approval_handler(signature)
            return bool(allowed)
        except Exception:
            return False

    @staticmethod
    def _normalize_llm_response(response: Any) -> tuple[str, str, dict[str, Any]]:
        if not isinstance(response, dict):
            return "", "none", {}
        raw_response = str(response.get("raw_response", ""))
        action = str(response.get("action", "none")).strip().lower() or "none"
        action_input_raw = response.get("action_input", {})
        action_input = dict(action_input_raw) if isinstance(action_input_raw, dict) else {}
        return raw_response, action, action_input

    @staticmethod
    def _stream_to_stdout(token: str) -> None:
        if token:
            print(token, end="", flush=True)

    def _run_core_agent_loop(
        self,
        state: StorageEngine,
    ) -> None:
        model_router = self.model_router
        prompt_engine = self.prompt_engine
        if model_router is None or prompt_engine is None:
            raise RuntimeError("FlowEngine requires model_router and prompt_engine to run")

        max_turns = int(self.limits.get("max_inner_turns", 60))
        turns = 0

        self._ensure_runtime_fields(state)
        final_prompt = prompt_engine.build_prompt(
            role="core_agent",
            state=state,
            model_router=model_router,
        )

        response = model_router.generate(
            role="core_agent",
            final_prompt=final_prompt,
            raw_response_callback=self._stream_to_stdout,
        )
        print()
        raw_response, action, action_input = self._normalize_llm_response(response)
        state.update_state(
            role="core_agent",
            text=raw_response,
            prompt_engine=prompt_engine,
            model_router=model_router,
        )
        state.save_state()

        while action not in TERMINAL_TOKENS and turns < max_turns:
            turns += 1
            if action == "chat_with_requester":
                break
            elif action == "chat_with_sub_agent":
                state.update_state(
                    role="runtime",
                    text="chat_with_sub_agent is disabled in current runtime",
                    prompt_engine=prompt_engine,
                    model_router=model_router,
                )
                state.save_state()
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
                    if not self._confirm_exec(action_input):
                        state.update_state(
                            role="runtime",
                            text="exec denied by requester",
                            prompt_engine=prompt_engine,
                            model_router=model_router,
                        )
                        state.save_state()
                        continue
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
                        state.save_state()

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
            final_prompt = prompt_engine.build_prompt(
                role="core_agent",
                state=state,
                model_router=model_router,
            )

            response = model_router.generate(
                role="core_agent",
                final_prompt=final_prompt,
                raw_response_callback=self._stream_to_stdout,
            )
            print()
            raw_response, action, action_input = self._normalize_llm_response(response)
            state.update_state(
                role="core_agent",
                text=raw_response,
                prompt_engine=prompt_engine,
                model_router=model_router,
            )
            state.save_state()

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
    ) -> str:
        model_router = self.model_router
        prompt_engine = self.prompt_engine
        if model_router is None or prompt_engine is None:
            raise RuntimeError("FlowEngine requires model_router and prompt_engine to run")

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
                    return "__EXIT__"
                if command_out == "__REFRESH__":
                    return "__REFRESH__"
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
            )
        return "__EXIT__"
