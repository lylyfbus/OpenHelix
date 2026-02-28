from __future__ import annotations

import json
import re
import select
import shlex
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .executors import ExecJob, collect_exec_job_result, start_exec_job, terminate_exec_job
from .history_utils import (
    build_exec_exact_signature,
    build_exec_path_signature,
    build_exec_pattern_signature,
    build_exec_result_lines,
    format_core_agent_record,
    format_history_block,
    format_history_record,
    format_ui_block,
    normalize_script_args,
)
from .model_router import ModelRouter
from .prompts import PromptEngine
from .storage import StorageEngine

DEFAULT_LIMITS = {
    "max_inner_turns": 60,
    "max_invalid_action_retries": 3,
    "max_invalid_output_retries": 3,
}


class FlowEngine:
    def __init__(
        self,
        workspace: str | Path,
        mode: str,
        model_router: ModelRouter | None = None,
        prompt_engine: PromptEngine | None = None,
        approval_handler: Callable[[str], tuple[bool, str]] | None = None,
        write_policy_handler: Callable[[str, list[str]], str | None] | None = None,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.mode = mode
        self.model_router = model_router
        self.prompt_engine = prompt_engine
        self.approval_handler = approval_handler
        self.write_policy_handler = write_policy_handler
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
        if not isinstance(getattr(state, "exec_approval_path", None), list):
            state.exec_approval_path = []
        if not isinstance(getattr(state, "exec_auto_write_allowlist", None), list):
            state.exec_auto_write_allowlist = []

    def _is_auto_mode(self) -> bool:
        return str(self.mode).strip().lower() == "auto"

    def _normalize_allow_path(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return str(candidate.resolve())

    def _collect_external_path_suggestions(
        self,
        action_input: dict[str, Any],
        exec_result: dict[str, Any] | None = None,
    ) -> list[str]:
        suggestions: list[str] = []
        seen: set[str] = set()

        def _push(text: str) -> None:
            token = str(text or "").strip()
            if not token:
                return
            if "=" in token and not token.startswith("="):
                _, rhs = token.split("=", 1)
                token = rhs.strip()
            token = token.strip("'\"")
            if token.startswith("~"):
                normalized = self._normalize_allow_path(token)
            elif token.startswith("/"):
                normalized = self._normalize_allow_path(token)
            else:
                return
            if not normalized:
                return
            candidate = Path(normalized)
            if candidate == self.workspace or self.workspace in candidate.parents:
                return
            if normalized in seen:
                return
            seen.add(normalized)
            suggestions.append(normalized)

        def _scan_text_for_paths(text: str) -> None:
            payload = str(text or "")
            if not payload:
                return
            matches = re.findall(r"(~\/[^\s\"'`<>|;:,]+|\/[^\s\"'`<>|;:,]+)", payload)
            for token in matches:
                _push(token)

        _push(str(action_input.get("script_path", "")))
        for arg in normalize_script_args(action_input.get("script_args", [])):
            _push(arg)
        script_value = str(action_input.get("script", "")).strip()
        if script_value:
            try:
                script_tokens = shlex.split(script_value)
            except ValueError:
                script_tokens = script_value.split()
            for token in script_tokens:
                _push(token)
        if isinstance(exec_result, dict):
            _scan_text_for_paths(str(exec_result.get("stderr", "")))
            _scan_text_for_paths(str(exec_result.get("stdout", "")))
        return suggestions

    @staticmethod
    def _is_write_policy_violation_result(exec_result: dict[str, Any]) -> bool:
        if not isinstance(exec_result, dict):
            return False
        if not bool(exec_result.get("write_policy_enabled", False)):
            return False
        if str(exec_result.get("status", "")).strip().lower() != "failed":
            return False
        stderr_text = str(exec_result.get("stderr", ""))
        stdout_text = str(exec_result.get("stdout", ""))
        combined_text = f"{stderr_text}\n{stdout_text}".lower()
        markers = (
            "operation not permitted",
            "read-only file system",
            "sandbox",
            "file-write",
            "/dev/null",
        )
        return any(marker in combined_text for marker in markers)

    def _record_exec_result(self, state: StorageEngine, exec_result: dict[str, Any]) -> None:
        exec_lines = build_exec_result_lines(exec_result)
        if not exec_lines:
            return
        first_line = exec_lines[0]
        continuation_lines = exec_lines[1:]
        history_text = format_history_block(
            state=state,
            role="runtime",
            first_line=first_line,
            continuation_lines=continuation_lines,
        )
        ui_text = format_ui_block(
            role="runtime",
            first_line=first_line,
            continuation_lines=continuation_lines,
        )
        state.update_state(
            text=history_text,
        )
        print()
        print(ui_text)
        state.save_state()

    def _emit_runtime_note(self, state: StorageEngine, text: str) -> None:
        state.update_state(
            text=format_history_record(
                state=state,
                role="runtime",
                text=text,
            ),
        )
        print()
        print(f"runtime> {text}")
        state.save_state()

    def _build_exec_approval_prompt(
        self,
        action_input: dict[str, Any],
    ) -> str:
        code_type = str(action_input.get("code_type", "bash")).strip().lower() or "bash"
        job_name = str(action_input.get("job_name", "none")).strip() or "none"
        script_path = str(action_input.get("script_path", "")).strip()
        script_args = normalize_script_args(action_input.get("script_args", []))
        script_preview_full = str(action_input.get("script", "")).strip()
        script_preview = script_preview_full[:500]

        lines = [
            "action: exec",
            f"- job_name: {job_name}",
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
        exact_signature = build_exec_exact_signature(action_input)
        pattern_signature = build_exec_pattern_signature(action_input)
        path_signature = build_exec_path_signature(action_input)
        if exact_signature in state.exec_approval_exact:
            return True
        if pattern_signature in state.exec_approval_pattern:
            return True
        if path_signature and path_signature in state.exec_approval_path:
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
            elif scope_name in {"path", "allow-path", "skill", "k"}:
                if path_signature and path_signature not in state.exec_approval_path:
                    state.exec_approval_path.append(path_signature)
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

    def _wait_for_exec_jobs(self, jobs: list[ExecJob]) -> list[dict[str, Any]]:
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
        prev_status_len = 0

        def _format_running_ids(ids: list[str]) -> str:
            if not ids:
                return "(none)"
            preview = ids[:3]
            rendered = ", ".join(preview)
            remaining = len(ids) - len(preview)
            if remaining > 0:
                rendered += f", +{remaining} more"
            return rendered

        while running_jobs:
            now = time.time()
            if now - last_status_at >= 0.25:
                marker = spinner[spinner_index % len(spinner)]
                spinner_index += 1
                ids = list(running_jobs.keys())
                status_line = (
                    f"runtime> [{marker}] exec running "
                    f"jobs={len(ids)} ids={_format_running_ids(ids)}"
                )
                pad = ""
                if prev_status_len > len(status_line):
                    pad = " " * (prev_status_len - len(status_line))
                print(f"\r{status_line}{pad}", end="", flush=True)
                prev_status_len = len(status_line)
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
                    if prev_status_len > 0:
                        print(f"\r{' ' * prev_status_len}\r", end="", flush=True)
                    print()
                    has_status_line = False
                    prev_status_len = 0
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
                    if prev_status_len > 0:
                        print(f"\r{' ' * prev_status_len}\r", end="", flush=True)
                    print()
                    has_status_line = False
                    prev_status_len = 0
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
                if prev_status_len > 0:
                    print(f"\r{' ' * prev_status_len}\r", end="", flush=True)
                print()
                has_status_line = False
                prev_status_len = 0
            print("runtime> exec jobs are running; use /cancel <job_id> or Ctrl+C.")

        if has_status_line:
            if prev_status_len > 0:
                print(f"\r{' ' * prev_status_len}\r", end="", flush=True)
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
                    "job_name": str(job.job_name),
                    "job_id": job_id,
                    "status": status,
                    "stdout": str(result.get("stdout", "")),
                    "stderr": str(result.get("stderr", "")),
                    "write_policy_enabled": bool(result.get("write_policy_enabled", False)),
                    "write_policy_mode": str(result.get("write_policy_mode", "")),
                    "write_policy_backend": str(result.get("write_policy_backend", "")),
                    "write_policy_workspace": str(result.get("write_policy_workspace", "")),
                    "write_policy_external_roots": list(result.get("write_policy_external_roots", []))
                    if isinstance(result.get("write_policy_external_roots", []), list)
                    else [],
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

    @staticmethod
    def _is_transient_model_error(text: str) -> bool:
        payload = str(text or "").strip().lower()
        if not payload:
            return False
        transient_markers = (
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "service unavailable",
            "temporarily overloaded",
            "network error",
            "timed out",
            "timeout",
        )
        return any(marker in payload for marker in transient_markers)

    @staticmethod
    def _retry_delay_seconds_for_attempt(attempt_index: int) -> int:
        return min(8, max(1, 2**max(0, attempt_index)))

    def _build_stream_printer(self) -> tuple[Callable[[str], None], Callable[[], None]]:
        def on_token(token: str) -> None:
            if not token:
                return
            print(token, end="", flush=True)

        def finish() -> None:
            print()

        return on_token, finish

    def _call_core_agent(
        self,
        *,
        state: StorageEngine,
        model_router: ModelRouter,
        prompt_engine: PromptEngine,
    ) -> tuple[str, dict[str, Any], bool]:
        max_invalid_output_retries = int(self.limits.get("max_invalid_output_retries", 3))
        last_failure_kind = "invalid_output"
        for attempt_idx in range(max_invalid_output_retries):
            self._ensure_runtime_fields(state)
            final_prompt = prompt_engine.build_prompt(
                role="core_agent",
                state=state,
                model_router=model_router,
            )
            self.last_core_agent_prompt = final_prompt
            print()
            print("core_agent> ", end="", flush=True)
            on_chunk, finish_stream = self._build_stream_printer()
            response: Any = {}
            call_error: Exception | None = None
            try:
                response = model_router.generate(
                    role="core_agent",
                    final_prompt=final_prompt,
                    raw_response_callback=on_chunk,
                )
            except Exception as exc:
                call_error = exc
            finish_stream()
            if call_error is not None:
                last_failure_kind = "model_call"
                error_text = str(call_error).strip() or repr(call_error)
                retryable = self._is_transient_model_error(error_text)
                should_retry = attempt_idx < (max_invalid_output_retries - 1)
                if should_retry and retryable:
                    delay_seconds = self._retry_delay_seconds_for_attempt(attempt_idx)
                    runtime_note = (
                        f"core_agent model call failed: {error_text}. "
                        f"Retrying in {delay_seconds}s "
                        f"({attempt_idx + 1}/{max_invalid_output_retries})."
                    )
                elif should_retry:
                    delay_seconds = 0
                    runtime_note = (
                        f"core_agent model call failed: {error_text}. "
                        f"Retrying ({attempt_idx + 1}/{max_invalid_output_retries})."
                    )
                else:
                    delay_seconds = 0
                    runtime_note = f"core_agent model call failed: {error_text}."
                self._emit_runtime_note(state=state, text=runtime_note)
                if should_retry:
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                break
            parse_ok = bool(response.get("_parse_ok", False)) if isinstance(response, dict) else False
            if not parse_ok:
                last_failure_kind = "invalid_output"
                parse_error = str(response.get("_parse_error", "")).strip() if isinstance(response, dict) else ""
                state.update_state(
                    text=format_history_record(
                        state=state,
                        role="core_agent",
                        text="[invalid_output_rejected]",
                    ),
                )
                runtime_note = (
                    "invalid core_agent output contract: "
                    + (parse_error if parse_error else "failed to parse model output")
                    + '. Regenerate with <output>{"raw_response":"...","action":"chat_with_requester|keep_reasoning|exec","action_input":{}}</output>.'
                )
                self._emit_runtime_note(state=state, text=runtime_note)
                continue
            raw_response, action, action_input = self._normalize_llm_response(response)
            state.append_action(role="core_agent", action=action, action_input=action_input)
            state.update_state(
                text=format_core_agent_record(
                    state=state,
                    raw_response=raw_response,
                    action=action,
                    action_input=action_input,
                ),
            )
            state.save_state()
            return action, action_input, True

        if last_failure_kind == "model_call":
            stop_reason = (
                f"max core_agent model call retries reached ({max_invalid_output_retries}); ending current loop"
            )
        else:
            stop_reason = (
                f"max invalid output retries reached ({max_invalid_output_retries}); ending current loop"
            )
        self._emit_runtime_note(state=state, text=stop_reason)
        return "chat_with_requester", {}, False

    def _handle_exec_action(self, state: StorageEngine, action_input: Any) -> bool:
        if not isinstance(action_input, dict):
            self._emit_runtime_note(state=state, text="exec action requires object action_input")
            return True
        if not self._confirm_exec(state, action_input):
            self._emit_runtime_note(state=state, text="exec denied by requester")
            return False

        try:
            job_name = str(action_input.get("job_name", "none")).strip() or "none"
            write_policy_mode = "workspace_write_only" if self._is_auto_mode() else "none"
            write_override_used = False
            while True:
                job_id = f"job_{uuid4().hex[:8]}"
                job = start_exec_job(
                    action_input=action_input,
                    workspace=self.workspace,
                    job_id=job_id,
                    job_name=job_name,
                    write_policy_mode=write_policy_mode,
                    external_write_roots=list(getattr(state, "exec_auto_write_allowlist", [])),
                )
                print()
                print(
                    "runtime> [exec started] "
                    f"job_name={job_name} job_id={job_id} "
                    f"(Ctrl+C to cancel all, /cancel {job_id} to cancel this job)"
                )

                exec_results = self._wait_for_exec_jobs([job])
                if not exec_results:
                    break
                exec_result = exec_results[0]
                self._record_exec_result(state=state, exec_result=exec_result)
                if (
                    not self._is_auto_mode()
                    or write_override_used
                    or not self._is_write_policy_violation_result(exec_result)
                ):
                    break

                stderr_text = str(exec_result.get("stderr", "")).strip()
                stdout_text = str(exec_result.get("stdout", "")).strip()
                stderr_lines = [line for line in stderr_text.splitlines() if line.strip()]
                stdout_lines = [line for line in stdout_text.splitlines() if line.strip()]
                stderr_preview = "\n".join(stderr_lines[-6:]) if stderr_lines else "(empty)"
                stdout_preview = "\n".join(stdout_lines[-6:]) if stdout_lines else "(empty)"
                suggested_paths = self._collect_external_path_suggestions(action_input, exec_result)
                details = "\n".join(
                    [
                        f"job_name={job_name}",
                        f"job_id={str(exec_result.get('job_id', '')).strip() or 'unknown'}",
                        "Failure appears to be blocked by workspace write policy.",
                        "stdout tail:",
                        stdout_preview,
                        "stderr tail:",
                        stderr_preview,
                    ]
                )
                allow_path_raw = ""
                if self.write_policy_handler is not None:
                    try:
                        allow_path_raw = str(self.write_policy_handler(details, suggested_paths) or "").strip()
                    except Exception:
                        allow_path_raw = ""
                allow_path = self._normalize_allow_path(allow_path_raw)
                if not allow_path:
                    break
                allowlist = list(getattr(state, "exec_auto_write_allowlist", []))
                if allow_path not in allowlist:
                    allowlist.append(allow_path)
                    state.exec_auto_write_allowlist = allowlist
                override_note = (
                    "auto-mode write override approved: "
                    f"added writable path {allow_path}; retrying current exec once"
                )
                self._emit_runtime_note(state=state, text=override_note)
                write_override_used = True
        except Exception as exc:
            self._emit_runtime_note(state=state, text=f"exec error: {exc}")

        return True

    def _handle_invalid_action(
        self,
        *,
        state: StorageEngine,
        action: str,
        invalid_action_retries: int,
        max_invalid_action_retries: int,
    ) -> tuple[int, bool]:
        next_retries = invalid_action_retries + 1
        correction = (
            f"You chose invalid next action '{action}'. Please double check your last statement "
            "and select one allowed action from chat_with_requester, keep_reasoning, and exec."
        )
        self._emit_runtime_note(state=state, text=correction)
        if next_retries < max_invalid_action_retries:
            return next_retries, False

        stop_reason = (
            f"max invalid action retries reached ({max_invalid_action_retries}); "
            "ending current loop"
        )
        self._emit_runtime_note(state=state, text=stop_reason)
        return next_retries, True

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

        action, action_input, ok = self._call_core_agent(
            state=state,
            model_router=model_router,
            prompt_engine=prompt_engine,
        )
        if not ok:
            return

        while turns < max_turns:
            turns += 1
            if action == "chat_with_requester":
                break
            elif action == "chat_with_sub_agent":
                invalid_action_retries = 0
                self._emit_runtime_note(
                    state=state,
                    text="chat_with_sub_agent is disabled in current runtime",
                )
            elif action == "exec":
                invalid_action_retries = 0
                should_continue = self._handle_exec_action(state=state, action_input=action_input)
                if not should_continue:
                    break
            elif action == "keep_reasoning":
                invalid_action_retries = 0
                pass
            else:
                invalid_action_retries, should_stop = self._handle_invalid_action(
                    state=state,
                    action=action,
                    invalid_action_retries=invalid_action_retries,
                    max_invalid_action_retries=max_invalid_action_retries,
                )
                if should_stop:
                    break

            action, action_input, ok = self._call_core_agent(
                state=state,
                model_router=model_router,
                prompt_engine=prompt_engine,
            )
            if not ok:
                return

        if turns >= max_turns:
            self._emit_runtime_note(
                state=state,
                text=f"max turns reached ({max_turns}); ending current loop",
            )

    def process_user_message(
        self,
        *,
        state: StorageEngine,
        user_text: str,
    ) -> None:
        model_router = self.model_router
        prompt_engine = self.prompt_engine
        if model_router is None or prompt_engine is None:
            raise RuntimeError("FlowEngine requires model_router and prompt_engine to run")

        stripped = str(user_text).strip()
        if not stripped:
            raise ValueError("user_text must be non-empty")

        state.update_state(
            text=format_history_record(
                state=state,
                role="user",
                text=stripped,
            ),
        )
        state.save_state()
        self._run_core_agent_loop(state)
