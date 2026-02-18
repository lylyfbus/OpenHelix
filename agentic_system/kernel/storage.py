from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class StorageEngine:
    def __init__(self, workspace: str | Path, session_id: str | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sessions_root = self.workspace / "sessions"
        self.knowledge_root = self.workspace / "knowledge"
        self.skills_root = self.workspace / "skills"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        (self.knowledge_root / "docs").mkdir(parents=True, exist_ok=True)
        (self.knowledge_root / "index").mkdir(parents=True, exist_ok=True)
        (self.skills_root / "core-agent").mkdir(parents=True, exist_ok=True)
        (self.skills_root / "all-agents").mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{uuid4().hex[:12]}"
        self.session_dir = self.sessions_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.session_dir / "state.json"
        # Session state fields.
        self.full_proc_hist: list[str] = []
        self.workflow_hist: list[str] = []
        self.workflow_summary: str = ""

    def load_state(self) -> bool:
        if not self.state_path.exists():
            return False
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return False
        self._deserialize_state(raw)
        return True

    def save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        payload = self._serialize_state()
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def update_state(
        self,
        *,
        role: str | None = None,
        text: str | None = None,
        to_workflow_hist: bool = True,
        prompt_engine: Any | None = None,
        model_router: Any | None = None,
    ) -> None:
        if role is not None:
            line = self.format_line(role, text or "")
            self.full_proc_hist.append(line)
            if to_workflow_hist:
                self.workflow_hist.append(line)

        if not self.workflow_hist:
            return
        
        if prompt_engine is None or model_router is None:
            return

        try:
            out = model_router.generate(
                role="workflow_summarizer",
                state=self,
                prompt_engine=prompt_engine,
            )
            if not isinstance(out, dict):
                return
            candidate = out.get("workflow_summary")
            if isinstance(candidate, str) and candidate.strip():
                self.workflow_summary = candidate.strip()
        except Exception:
            return

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @classmethod
    def format_line(cls, role: str, text: str) -> str:
        return f"[{cls.utc_now_iso()}] {role}> : {text}"

    def _serialize_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "full_proc_hist": self.full_proc_hist,
            "workflow_hist": self.workflow_hist,
            "workflow_summary": self.workflow_summary,
        }

    def _deserialize_state(self, raw: dict[str, Any]) -> None:
        loaded_id = str(raw.get("session_id", "")).strip()
        if loaded_id:
            self.session_id = loaded_id
            self.session_dir = self.sessions_root / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.state_path = self.session_dir / "state.json"
        full_proc_hist = raw.get("full_proc_hist", [])
        workflow_hist = raw.get("workflow_hist", raw.get("llm_hist", []))
        workflow_summary = raw.get("workflow_summary", raw.get("runtime_summary", ""))
        self.full_proc_hist = list(full_proc_hist if isinstance(full_proc_hist, list) else [])
        self.workflow_hist = list(workflow_hist if isinstance(workflow_hist, list) else [])
        self.workflow_summary = str(workflow_summary if isinstance(workflow_summary, str) else "")
