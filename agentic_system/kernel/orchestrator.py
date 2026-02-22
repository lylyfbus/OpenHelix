from __future__ import annotations

import json
import select
import shlex
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .executors import ExecJob, collect_exec_job_result, start_exec_job, terminate_exec_job
from .model_router import ModelRouter
from .prompts import PromptEngine
from .storage import StorageEngine

DEFAULT_LIMITS = {
    "max_inner_turns": 60,
    "max_invalid_action_retries": 3,
}


class FlowEngine:
    def __init__(
        self,
        workspace: str | Path,
        mode: str,
        model_router: ModelRouter | None = None,
        prompt_engine: PromptEngine | None = None,
        approval_handler: Callable[[str], tuple[bool, str]] | None = None,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.mode = mode
        self.model_router = model_router
        self.prompt_engine = prompt_engine
        self.approval_handler = approval_handler
        self.last_core_agent_prompt: str = ""
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
        if not isinstance(getattr(state, "action_hist", None), list):
            state.action_hist = []
        if not isinstance(getattr(state, "exec_approval_exact", None), list):
            state.exec_approval_exact = []
        if not isinstance(getattr(state, "exec_approval_pattern", None), list):
            state.exec_approval_pattern = []

    @staticmethod
    def _normalize_script_args(raw_script_args: Any) -> list[str]:
        if isinstance(raw_script_args, (list, tuple)):
            return [str(arg).strip() for arg in raw_script_args if str(arg).strip()]
        if isinstance(raw_script_args, str):
            text = raw_script_args.strip()
            if not text:
                return []
            try:
                return [arg for arg in shlex.split(text) if arg.strip()]
            except ValueError:
                return [text]
        return []

    def _build_exec_exact_signature(self, action_input: dict[str, Any]) -> str:
        code_type = str(action_input.get("code_type", "bash")).strip().lower() or "bash"
        script_path = str(action_input.get("script_path", "")).strip()
        script = str(action_input.get("script", "")).strip()
        script_args = self._normalize_script_args(action_input.get("script_args", []))
        normalized = {
            "action": "exec",
            "code_type": code_type,
            "script_path": script_path,
            "script": script,
            "script_args": script_args,
        }
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True)

    def _build_exec_pattern_signature(self, action_input: dict[str, Any]) -> str:
        code_type = str(action_input.get("code_type", "bash")).strip().lower() or "bash"
        script_path = str(action_input.get("script_path", "")).strip()
        script = str(action_input.get("script", "")).strip()
        if script_path:
            return f"exec|{code_type}|script_path|{script_path}"
        compact_inline = " ".join(script.split())[:240]
        return f"exec|{code_type}|inline|{compact_inline}"

    @staticmethod
    def _format_exec_value_lines(label: str, value: Any) -> list[str]:
        lines = [f"- {label}:"]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
        if not text:
            return lines
        for row in str(text).splitlines():
            lines.append(f"  {row}")
        return lines

    def _format_exec_result_text(self, exec_result: Any) -> str:
        if not isinstance(exec_result, dict):
            lines: list[str] = []
            lines.extend(self._format_exec_value_lines("stdout", ""))
            lines.extend(self._format_exec_value_lines("stderr", str(exec_result)))
            return "\n".join(lines)
        lines = []
        job_id = str(exec_result.get("job_id", "")).strip()
        status = str(exec_result.get("status", "")).strip()
        if job_id:
            lines.append(f"- job_id: {job_id}")
        if status:
            lines.append(f"- status: {status}")
        lines.extend(self._format_exec_value_lines("stdout", exec_result.get("stdout", "")))
        lines.extend(self._format_exec_value_lines("stderr", exec_result.get("stderr", "")))
        return "\n".join(lines)

    @staticmethod
    def _format_core_agent_record(
        state: StorageEngine,
        raw_response: str,
        action: str,
        action_input: Any,
    ) -> str:
        action_name = str(action or "").strip().lower() or "unknown"
        payload = dict(action_input) if isinstance(action_input, dict) else {}
        prefix = f"[{state.utc_now_iso()}] core_agent> : "
        indent = " " * len(prefix)
        lines: list[str] = [
            str(raw_response or ""),
            f"{indent}> next_action: {action_name}",
        ]
        if payload:
            lines.append(f"{indent}> action_input:")
            for key, value in payload.items():
                if isinstance(value, str) and "\n" in value:
                    lines.append(f"{indent}>    - {key}:")
                    for sub_line in value.splitlines():
                        lines.append(f"{indent}>        {sub_line}")
                    continue
                lines.append(f"{indent}>    - {key}: {json.dumps(value, ensure_ascii=True)}")
        else:
            lines.append(f"{indent}> action_input: {{}}")
        return "\n".join(lines)

    def _build_exec_approval_prompt(
        self,
        action_input: dict[str, Any],
    ) -> str:
        code_type = str(action_input.get("code_type", "bash")).strip().lower() or "bash"
        script_path = str(action_input.get("script_path", "")).strip()
        script_args = self._normalize_script_args(action_input.get("script_args", []))
        script_preview_full = str(action_input.get("script", "")).strip()
        script_preview = script_preview_full[:500]

        lines = [
            "action: exec",
            f"- code_type: {code_type}",
            f"- script_path: {script_path if script_path else '(none)'}",
            f"- script_args: {json.dumps(script_args, ensure_ascii=True)}",
            "- script_preview:",
        ]
        if script_preview:
            for row in script_preview.splitlines():
                lines.append(f"    {row}")
            if len(script_preview_full) > 500:
                lines.append("    ... (truncated)")
        return "\n".join(lines)

    def _confirm_exec(self, state: StorageEngine, action_input: dict[str, Any]) -> bool:
        if str(self.mode).strip().lower() == "auto":
            return True
        exact_signature = self._build_exec_exact_signature(action_input)
        pattern_signature = self._build_exec_pattern_signature(action_input)
        if exact_signature in state.exec_approval_exact:
            return True
        if pattern_signature in state.exec_approval_pattern:
            return True
        if self.approval_handler is None:
            return True
        signature = self._build_exec_approval_prompt(
            action_input=action_input,
        )
        try:
            allowed, scope = self.approval_handler(signature)
            if not bool(allowed):
                return False
            scope_name = str(scope).strip().lower()
            if scope_name in {"session", "exact", "allow-session", "allow-exact", "s"}:
                if exact_signature not in state.exec_approval_exact:
                    state.exec_approval_exact.append(exact_signature)
            elif scope_name in {"pattern", "allow-pattern", "p"}:
                if pattern_signature not in state.exec_approval_pattern:
                    state.exec_approval_pattern.append(pattern_signature)
            return True
        except Exception:
            return False

    @staticmethod
    def _build_exec_cancel_note(job_id: str, reason: str, signals: list[str]) -> str:
        signal_text = " -> ".join(signals) if signals else "none"
        return "\n".join(
            [
                f"[runtime] exec terminated by requester ({reason})",
                f"[runtime] job_id={job_id}",
                f"[runtime] signals={signal_text}",
            ]
        )

    def _cancel_running_jobs(
        self,
        *,
        running_jobs: dict[str, ExecJob],
        status_by_job: dict[str, str],
        stderr_notes_by_job: dict[str, str],
        reason: str,
    ) -> None:
        for job_id in list(running_jobs.keys()):
            job = running_jobs.get(job_id)
            if job is None:
                continue
            try:
                cancel_meta = terminate_exec_job(job, reason=reason)
                signals = [str(item) for item in cancel_meta.get("signals", [])]
            except Exception as exc:
                signals = []
                stderr_notes_by_job[job_id] = (
                    self._build_exec_cancel_note(job_id=job_id, reason=reason, signals=signals)
                    + f"\n[runtime] cancellation error: {exc}"
                )
            else:
                stderr_notes_by_job[job_id] = self._build_exec_cancel_note(
                    job_id=job_id,
                    reason=reason,
                    signals=signals,
                )
            status_by_job[job_id] = "cancelled"

    def _wait_for_exec_jobs(self, jobs: list[ExecJob]) -> list[dict[str, str]]:
        if not jobs:
            return []

        jobs_by_id = {job.job_id: job for job in jobs}
        running_jobs: dict[str, ExecJob] = dict(jobs_by_id)
        status_by_job: dict[str, str] = {job.job_id: "running" for job in jobs}
        stderr_notes_by_job: dict[str, str] = {}
        spinner = ["|", "/", "-", "\\"]
        spinner_index = 0
        has_status_line = False
        last_status_at = 0.0

        while running_jobs:
            now = time.time()
            if now - last_status_at >= 0.25:
                marker = spinner[spinner_index % len(spinner)]
                spinner_index += 1
                ids = ", ".join(running_jobs.keys())
                print(
                    f"\rruntime> [{marker}] exec running job_ids={ids} (Ctrl+C to cancel all, /cancel <job_id>)",
                    end="",
                    flush=True,
                )
                has_status_line = True
                last_status_at = now

            done_ids = [job_id for job_id, job in running_jobs.items() if job.process.poll() is not None]
            for job_id in done_ids:
                job = running_jobs.pop(job_id)
                if status_by_job.get(job_id) != "cancelled":
                    return_code = int(job.process.returncode or 0)
                    status_by_job[job_id] = "completed" if return_code == 0 else "failed"

            if not running_jobs:
                break

            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.25)
            except KeyboardInterrupt:
                if has_status_line:
                    print()
                    has_status_line = False
                self._cancel_running_jobs(
                    running_jobs=running_jobs,
                    status_by_job=status_by_job,
                    stderr_notes_by_job=stderr_notes_by_job,
                    reason="Ctrl+C",
                )
                continue
            except (OSError, ValueError):
                time.sleep(0.25)
                continue

            if not readable:
                continue

            line = sys.stdin.readline()
            command = str(line or "").strip()
            if not command:
                time.sleep(0.05)
                continue

            if command.startswith("/cancel"):
                if has_status_line:
                    print()
                    has_status_line = False
                parts = command.split(maxsplit=1)
                if len(parts) == 1:
                    self._cancel_running_jobs(
                        running_jobs=running_jobs,
                        status_by_job=status_by_job,
                        stderr_notes_by_job=stderr_notes_by_job,
                        reason="/cancel",
                    )
                    continue

                target_job_id = parts[1].strip()
                if target_job_id.lower() == "all":
                    self._cancel_running_jobs(
                        running_jobs=running_jobs,
                        status_by_job=status_by_job,
                        stderr_notes_by_job=stderr_notes_by_job,
                        reason="/cancel",
                    )
                    continue

                target_job = running_jobs.get(target_job_id)
                if target_job is None:
                    print(f"runtime> unknown job_id for /cancel: {target_job_id}")
                    continue

                try:
                    cancel_meta = terminate_exec_job(target_job, reason="/cancel")
                    signals = [str(item) for item in cancel_meta.get("signals", [])]
                    stderr_notes_by_job[target_job_id] = self._build_exec_cancel_note(
                        job_id=target_job_id,
                        reason="/cancel",
                        signals=signals,
                    )
                except Exception as exc:
                    stderr_notes_by_job[target_job_id] = (
                        self._build_exec_cancel_note(
                            job_id=target_job_id,
                            reason="/cancel",
                            signals=[],
                        )
                        + f"\n[runtime] cancellation error: {exc}"
                    )
                status_by_job[target_job_id] = "cancelled"
                print(f"runtime> cancellation requested for job_id={target_job_id}")
                continue

            if has_status_line:
                print()
                has_status_line = False
            print("runtime> exec jobs are running; use /cancel <job_id> or Ctrl+C.")

        if has_status_line:
            print()

        out: list[dict[str, str]] = []
        for job in jobs:
            job_id = job.job_id
            stderr_append = stderr_notes_by_job.get(job_id, "")
            try:
                result = collect_exec_job_result(job, stderr_append=stderr_append)
            except Exception as exc:
                result = {
                    "stdout": "",
                    "stderr": f"[runtime] failed to collect exec output: {exc}",
                    "return_code": 1,
                }
            status = status_by_job.get(job_id, "running")
            if status == "running":
                return_code = int(result.get("return_code", 0) or 0)
                status = "completed" if return_code == 0 else "failed"
            out.append(
                {
                    "job_id": job_id,
                    "status": status,
                    "stdout": str(result.get("stdout", "")),
                    "stderr": str(result.get("stderr", "")),
                }
            )
        return out

    @staticmethod
    def _normalize_llm_response(response: Any) -> tuple[str, str, dict[str, Any]]:
        if not isinstance(response, dict):
            return "", "none", {}
        raw_response = str(response.get("raw_response", ""))
        action = str(response.get("action", "none")).strip().lower() or "none"
        action_input_raw = response.get("action_input", {})
        action_input = dict(action_input_raw) if isinstance(action_input_raw, dict) else {}
        return raw_response, action, action_input

    def _build_stream_printer(self, role: str) -> tuple[Callable[[str], None], Callable[[str], None]]:
        role_name = str(role).strip() or "assistant"
        started = {"value": False}

        def on_token(token: str) -> None:
            if not token:
                return
            if not started["value"]:
                print()
                print(f"{role_name}> ", end="", flush=True)
                started["value"] = True
            print(token, end="", flush=True)

        def finish(raw_response: str) -> None:
            if started["value"]:
                print()
                return
            text = str(raw_response or "").strip()
            if text:
                print()
                print(f"{role_name}> {text}")

        return on_token, finish

    def _run_core_agent_loop(
        self,
        state: StorageEngine,
    ) -> None:
        model_router = self.model_router
        prompt_engine = self.prompt_engine
        if model_router is None or prompt_engine is None:
            raise RuntimeError("FlowEngine requires model_router and prompt_engine to run")

        max_turns = int(self.limits.get("max_inner_turns", 999))
        max_invalid_action_retries = int(self.limits.get("max_invalid_action_retries", 3))
        turns = 0
        invalid_action_retries = 0

        self._ensure_runtime_fields(state)
        final_prompt = prompt_engine.build_prompt(
            role="core_agent",
            state=state,
            model_router=model_router,
        )
        self.last_core_agent_prompt = final_prompt
        on_chunk, finish_stream = self._build_stream_printer("core_agent")

        response = model_router.generate(
            role="core_agent",
            final_prompt=final_prompt,
            raw_response_callback=on_chunk,
        )
        raw_response, action, action_input = self._normalize_llm_response(response)
        state.append_action(role="core_agent", action=action, action_input=action_input)
        finish_stream(raw_response)
        state.update_state(
            role="core_agent",
            text=self._format_core_agent_record(
                state=state,
                raw_response=raw_response,
                action=action,
                action_input=action_input,
            ),
        )
        state.save_state()

        while turns < max_turns:
            turns += 1
            if action == "chat_with_requester":
                break
            elif action == "chat_with_sub_agent":
                invalid_action_retries = 0
                state.update_state(
                    role="runtime",
                    text="chat_with_sub_agent is disabled in current runtime",
                )
                print()
                print(f"runtime> chat_with_sub_agent is disabled in current runtime")
                state.save_state()
            elif action == "exec":
                invalid_action_retries = 0
                if not isinstance(action_input, dict):
                    state.update_state(
                        role="runtime",
                        text="exec action requires object action_input",
                    )
                    print()
                    print(f"runtime> exec action requires object action_input")
                    state.save_state()
                else:
                    if not self._confirm_exec(state, action_input):
                        state.update_state(
                            role="runtime",
                            text="exec denied by requester",
                        )
                        print()
                        print(f"runtime> exec denied by requester")
                        state.save_state()
                        break
                    try:
                        job_id = f"job_{uuid4().hex[:8]}"
                        job = start_exec_job(
                            action_input=action_input,
                            workspace=self.workspace,
                            job_id=job_id,
                        )
                        print()
                        print(
                            "runtime> [exec started] "
                            f"job_id={job_id} (Ctrl+C to cancel all, /cancel {job_id} to cancel this job)"
                        )

                        exec_results = self._wait_for_exec_jobs([job])
                        for exec_result in exec_results:
                            exec_text = self._format_exec_result_text(exec_result)
                            state.update_state(
                                role="runtime",
                                text=exec_text,
                            )
                            print()
                            print(f"runtime> {exec_text}")
                            state.save_state()

                    except Exception as exc:
                        state.update_state(
                            role="runtime",
                            text=f"exec error: {exc}",
                        )
                        print()
                        print(f"runtime> exec error: {exc}")
                        state.save_state()
            elif action == "keep_reasoning":
                invalid_action_retries = 0
                pass
            else:
                invalid_action_retries += 1
                correction = (
                    f"You chose invalid next action '{action}'. Please double check your last statement "
                    "and select one allowed action from chat_with_requester, keep_reasoning, and exec."
                )
                state.update_state(
                    role="runtime",
                    text=correction,
                )
                print()
                print(f"runtime> {correction}")
                state.save_state()
                if invalid_action_retries >= max_invalid_action_retries:
                    stop_reason = (
                        f"max invalid action retries reached ({max_invalid_action_retries}); "
                        "ending current loop"
                    )
                    state.update_state(
                        role="runtime",
                        text=stop_reason,
                    )
                    print()
                    print(f"runtime> {stop_reason}")
                    state.save_state()
                    break

            self._ensure_runtime_fields(state)
            final_prompt = prompt_engine.build_prompt(
                role="core_agent",
                state=state,
                model_router=model_router,
            )
            self.last_core_agent_prompt = final_prompt
            on_chunk, finish_stream = self._build_stream_printer("core_agent")

            response = model_router.generate(
                role="core_agent",
                final_prompt=final_prompt,
                raw_response_callback=on_chunk,
            )
            raw_response, action, action_input = self._normalize_llm_response(response)
            state.append_action(role="core_agent", action=action, action_input=action_input)
            finish_stream(raw_response)
            state.update_state(
                role="core_agent",
                text=self._format_core_agent_record(
                    state=state,
                    raw_response=raw_response,
                    action=action,
                    action_input=action_input,
                ),
            )
            state.save_state()

        if turns >= max_turns:
            state.update_state(
                role="runtime",
                text=f"max turns reached ({max_turns}); ending current loop",
            )
            print()
            print(f"runtime> max turns reached ({max_turns}); ending current loop")
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
            )
            state.save_state()
            self._run_core_agent_loop(
                state,
            )
        return "__EXIT__"
