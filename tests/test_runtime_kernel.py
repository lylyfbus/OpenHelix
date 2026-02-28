from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from agentic_system.cli import build_parser
from agentic_system.kernel.model_router import ModelRouter
from agentic_system.kernel.agent_loop import FlowEngine
from agentic_system.kernel.prompts import PromptEngine
from agentic_system.kernel.storage import StorageEngine
from agentic_system.runtime import AgentRuntime


class _DummyPromptEngine:
    def __init__(self, prompt: str = "prompt") -> None:
        self.prompt = prompt
        self.calls: list[tuple[str, str]] = []

    def build_prompt(self, role: str, state: StorageEngine | None = None, model_router: object | None = None) -> str:
        self.calls.append((role, str(getattr(state, "session_id", ""))))
        return self.prompt


class _DummyModelRouter:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(
        self,
        role: str = "core_agent",
        final_prompt: str | None = None,
        raw_response_callback: object | None = None,
    ) -> dict[str, object]:
        self.calls.append(role)
        if not self.responses:
            raise RuntimeError("no dummy responses left")
        response = dict(self.responses.pop(0))
        callback = raw_response_callback if callable(raw_response_callback) else None
        if callback is not None:
            token = str(response.get("raw_response", ""))
            if token:
                callback(token)
        return response


class _ObserverRouter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self,
        role: str = "core_agent",
        final_prompt: str | None = None,
        raw_response_callback: object | None = None,
    ) -> dict[str, object]:
        self.calls.append(role)
        if role == "workflow_summarizer":
            return {
                "_parse_ok": True,
                "_parse_error": "",
                "workflow_summary": "updated summary",
            }
        if role == "workflow_history_compactor":
            return {
                "_parse_ok": True,
                "_parse_error": "",
                "workflow_hist_compact": "older context compacted",
            }
        return {
            "_parse_ok": True,
            "_parse_error": "",
            "raw_response": "done",
            "action": "chat_with_requester",
            "action_input": {},
        }


class _FailingModelRouter:
    def __init__(self, error_text: str) -> None:
        self.error_text = error_text
        self.calls: list[str] = []

    def generate(
        self,
        role: str = "core_agent",
        final_prompt: str | None = None,
        raw_response_callback: object | None = None,
    ) -> dict[str, object]:
        self.calls.append(role)
        raise RuntimeError(self.error_text)


class RuntimeKernelTests(unittest.TestCase):
    def test_cli_parser_accepts_canonical_image_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--workspace",
                "tmp_workspace",
                "--image-analysis-provider",
                "ollama",
                "--image-analysis-model",
                "llava:latest",
                "--image-generation-provider",
                "ollama",
                "--image-generation-model",
                "x/z-image-turbo",
            ]
        )
        self.assertEqual(args.image_analysis_provider, "ollama")
        self.assertEqual(args.image_analysis_model, "llava:latest")
        self.assertEqual(args.image_generation_provider, "ollama")
        self.assertEqual(args.image_generation_model, "x/z-image-turbo")

    def test_cli_parser_rejects_removed_legacy_image_flags(self) -> None:
        parser = build_parser()
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--workspace",
                        "tmp_workspace",
                        "--vision-provider",
                        "ollama",
                    ]
                )

    def test_runtime_injects_only_canonical_image_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            runtime = AgentRuntime(
                workspace=tmp,
                provider="ollama",
                image_analysis_provider="ollama",
                image_analysis_model="llava:latest",
                image_generation_provider="ollama",
                image_generation_model="x/z-image-turbo",
            )
            try:
                self.assertEqual(os.getenv("IMAGE_ANALYSIS_PROVIDER"), "ollama")
                self.assertEqual(os.getenv("IMAGE_ANALYSIS_MODEL"), "llava:latest")
                self.assertEqual(os.getenv("IMAGE_GENERATION_PROVIDER"), "ollama")
                self.assertEqual(os.getenv("IMAGE_GENERATION_MODEL"), "x/z-image-turbo")
                self.assertIsNone(os.getenv("VISION_PROVIDER"))
                self.assertIsNone(os.getenv("VISION_MODEL"))
                self.assertIsNone(os.getenv("IMAGE_GEN_PROVIDER"))
                self.assertIsNone(os.getenv("IMAGE_GEN_MODEL"))
            finally:
                runtime.shutdown()

    def test_flow_engine_process_user_message_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = StorageEngine(workspace=tmp)
            model_router = _DummyModelRouter(
                [
                    {
                        "_parse_ok": True,
                        "_parse_error": "",
                        "raw_response": "acknowledged",
                        "action": "chat_with_requester",
                        "action_input": {},
                    }
                ]
            )
            prompt_engine = _DummyPromptEngine(prompt="core prompt")
            engine = FlowEngine(
                workspace=tmp,
                mode="controlled",
                model_router=model_router,  # type: ignore[arg-type]
                prompt_engine=prompt_engine,  # type: ignore[arg-type]
            )
            with patch("builtins.print"):
                engine.process_user_message(state=state, user_text="hello runtime")

            joined = "\n".join(state.workflow_hist)
            self.assertIn("user> hello runtime", joined)
            self.assertIn("core_agent> acknowledged", joined)
            self.assertEqual(len(state.action_hist), 1)
            self.assertIn("action=chat_with_requester", state.action_hist[0])
            self.assertEqual(engine.last_core_agent_prompt, "core prompt")

    def test_flow_engine_invalid_output_retries_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = StorageEngine(workspace=tmp)
            model_router = _DummyModelRouter(
                [
                    {"_parse_ok": False, "_parse_error": "missing <output>...</output> block"},
                    {"_parse_ok": False, "_parse_error": "missing <output>...</output> block"},
                ]
            )
            prompt_engine = _DummyPromptEngine(prompt="core prompt")
            engine = FlowEngine(
                workspace=tmp,
                mode="controlled",
                model_router=model_router,  # type: ignore[arg-type]
                prompt_engine=prompt_engine,  # type: ignore[arg-type]
                limits={"max_invalid_output_retries": 2},
            )
            with patch("builtins.print"):
                engine.process_user_message(state=state, user_text="trigger invalid contract path")

            joined = "\n".join(state.workflow_hist)
            self.assertIn("core_agent> [invalid_output_rejected]", joined)
            self.assertIn("invalid core_agent output contract:", joined)
            self.assertIn("max invalid output retries reached (2); ending current loop", joined)

    def test_flow_engine_model_call_error_retries_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = StorageEngine(workspace=tmp)
            model_router = _FailingModelRouter("HTTP 429: service temporarily overloaded")
            prompt_engine = _DummyPromptEngine(prompt="core prompt")
            engine = FlowEngine(
                workspace=tmp,
                mode="controlled",
                model_router=model_router,  # type: ignore[arg-type]
                prompt_engine=prompt_engine,  # type: ignore[arg-type]
                limits={"max_invalid_output_retries": 2},
            )
            with patch("builtins.print"), patch("time.sleep"):
                engine.process_user_message(state=state, user_text="trigger model error retries")

            joined = "\n".join(state.workflow_hist)
            self.assertIn("core_agent model call failed:", joined)
            self.assertIn("max core_agent model call retries reached (2); ending current loop", joined)

    def test_prompt_engine_triggers_summary_and_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state = StorageEngine(workspace=workspace)
            state.workflow_summary = "previous summary"
            state.workflow_hist = [
                f"[{state.utc_now_iso()}] user> one {'x' * 120}",
                f"[{state.utc_now_iso()}] runtime> two {'x' * 120}",
                f"[{state.utc_now_iso()}] core_agent> three {'x' * 120}",
                f"[{state.utc_now_iso()}] runtime> four {'x' * 120}",
            ]
            prompt_engine = PromptEngine(
                workspace=workspace,
                token_window_limit=30,
                compact_keep_last_k=2,
            )
            router = _ObserverRouter()

            final_prompt = prompt_engine.build_prompt(
                role="core_agent",
                state=state,
                model_router=router,  # type: ignore[arg-type]
            )

            self.assertIn("workflow_summarizer", router.calls)
            self.assertIn("workflow_history_compactor", router.calls)
            self.assertEqual(state.workflow_summary, "updated summary")
            self.assertEqual(len(state.workflow_hist), 3)
            self.assertIn("workflow_compactor> older context compacted", state.workflow_hist[0])
            self.assertIn("Latest Context", final_prompt)
            self.assertIn("Workflow Summary", final_prompt)
            self.assertIn("Workflow History", final_prompt)

    def test_model_router_parse_reports_contract_error(self) -> None:
        payload, parse_error = ModelRouter._parse_json_payload_with_error(
            '<output>{"raw_response":"ok","action":"none","action_input":{}}</output>',
            role="core_agent",
        )
        self.assertIsNone(payload)
        self.assertIn("action must be one of chat_with_requester, keep_reasoning, exec", parse_error)


if __name__ == "__main__":
    unittest.main()
