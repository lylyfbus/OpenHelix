"""Runtime Host — interactive REPL that wires all framework components.

Replaces the legacy ``runtime.py`` + ``FlowEngine`` + ``StorageEngine``
with a clean host built on the new core abstractions.

Usage::

    host = RuntimeHost(workspace="/path/to/workspace", provider="ollama")
    host.start()
"""

from __future__ import annotations

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

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:  # pragma: no cover
    PromptSession = None  # type: ignore[assignment]
    KeyBindings = None  # type: ignore[assignment]

from ..core.agent import Agent
from ..core.environment import Environment
from ..core.loop import run_loop
from ..core.state import Turn
from .sandbox import sandbox_executor
from .approval import ApprovalPolicy


# --------------------------------------------------------------------------- #
# Streaming display
# --------------------------------------------------------------------------- #

_JSON_ESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def extract_streaming_response(partial_text: str) -> "str | None":
    """Extract the 'response' value from a partial JSON stream.

    Used to stream tokens to the UI during generation. Returns the
    extracted response text so far, or None if the key hasn't appeared yet.
    """
    # Look for "response": "... pattern
    marker = '"response"'
    idx = partial_text.find(marker)
    if idx == -1:
        return None

    # Skip past the key and colon
    after_key = partial_text[idx + len(marker) :]
    colon_idx = after_key.find(":")
    if colon_idx == -1:
        return None

    after_colon = after_key[colon_idx + 1 :].lstrip()
    if not after_colon or after_colon[0] != '"':
        return None

    # Extract and decode the partial JSON string value.
    result_chars: list[str] = []
    i = 1  # skip opening quote
    while i < len(after_colon):
        ch = after_colon[i]
        if ch == "\\":
            if i + 1 >= len(after_colon):
                break
            esc = after_colon[i + 1]
            if esc in _JSON_ESCAPE_MAP:
                result_chars.append(_JSON_ESCAPE_MAP[esc])
                i += 2
                continue
            if esc == "u":
                if i + 6 > len(after_colon):
                    break
                hex_value = after_colon[i + 2 : i + 6]
                if not re.fullmatch(r"[0-9a-fA-F]{4}", hex_value):
                    break
                result_chars.append(chr(int(hex_value, 16)))
                i += 6
                continue
            # Unknown or malformed escape; stop until more text arrives.
            break
        if ch == '"':
            break
        result_chars.append(ch)
        i += 1
    return "".join(result_chars) if result_chars else None

def _render_session_view_html(
    *,
    session_id: str,
    field: str,
    session_path: Path,
    value: object,
) -> str:
    """Render a field-specific session JSON view as a readable HTML page."""
    def _render_text_view(*, eyebrow: str, body_text: str) -> str:
        title = escape(f"Session View - {session_id} - {field}")
        escaped_body = escape(body_text)
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f3efe6;
        --panel: rgba(255, 252, 245, 0.92);
        --border: rgba(46, 58, 89, 0.14);
        --text: #1f2430;
        --muted: #6a7280;
        --accent: #1e6aa8;
        --shadow: 0 20px 50px rgba(31, 36, 48, 0.10);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Menlo, Monaco, "SFMono-Regular", Consolas, monospace;
        background:
          radial-gradient(circle at top left, rgba(30, 106, 168, 0.12), transparent 34%),
          linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
        color: var(--text);
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 24px 48px;
      }}
      .header {{
        margin-bottom: 20px;
      }}
      .eyebrow {{
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 28px;
        line-height: 1.2;
      }}
      .sub {{
        margin: 0;
        color: var(--muted);
        font-size: 14px;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .panel-head {{
        padding: 16px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.72);
        font-size: 13px;
        color: var(--muted);
      }}
      pre {{
        margin: 0;
        padding: 22px 24px 28px;
        font-size: 13px;
        line-height: 1.55;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }}
    </style>
  </head>
  <body>
    <main>
      <div class="header">
        <div class="eyebrow">{escape(eyebrow)}</div>
        <h1>{escape(field)}</h1>
        <p class="sub">Session <strong>{escape(session_id)}</strong></p>
      </div>
      <section class="panel">
        <div class="panel-head">{escape(str(session_path))}</div>
        <pre>{escaped_body}</pre>
      </section>
    </main>
  </body>
</html>
"""

    if field == "last_prompt":
        prompt_text = str(value) if str(value) else "(none yet)"
        return _render_text_view(
            eyebrow="Agentic System Prompt View",
            body_text=prompt_text,
        )

    if field in {"full_history", "observation"}:
        turns = value if isinstance(value, list) else []
        lines: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            timestamp = str(turn.get("timestamp", "") or "")
            role = str(turn.get("role", "unknown") or "unknown")
            content = str(turn.get("content", "") or "")
            prefix = f"[{timestamp}] {role}> "
            content_lines = content.split("\n")
            continuation = " " * len(prefix)
            rendered = "\n".join(
                [f"{prefix}{content_lines[0]}"]
                + [f"{continuation}{cl}" for cl in content_lines[1:]]
            )
            lines.append(rendered)
        body_text = "\n".join(lines) if lines else "(empty)"
        return _render_text_view(
            eyebrow="Agentic System Timeline View",
            body_text=body_text,
        )

    payload = {
        "session_id": session_id,
        "field": field,
        "source_session_file": str(session_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "value": value,
    }
    pretty_json = json.dumps(payload, indent=2, ensure_ascii=False)
    escaped_json = escape(pretty_json)
    title = escape(f"Session View - {session_id} - {field}")

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f3efe6;
        --panel: rgba(255, 252, 245, 0.92);
        --border: rgba(46, 58, 89, 0.14);
        --text: #1f2430;
        --muted: #6a7280;
        --accent: #1e6aa8;
        --shadow: 0 20px 50px rgba(31, 36, 48, 0.10);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Menlo, Monaco, "SFMono-Regular", Consolas, monospace;
        background:
          radial-gradient(circle at top left, rgba(30, 106, 168, 0.12), transparent 34%),
          linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
        color: var(--text);
      }}
      main {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 24px 48px;
      }}
      .header {{
        margin-bottom: 20px;
      }}
      .eyebrow {{
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 28px;
        line-height: 1.2;
      }}
      .sub {{
        margin: 0;
        color: var(--muted);
        font-size: 14px;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .panel-head {{
        padding: 16px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.72);
        font-size: 13px;
        color: var(--muted);
      }}
      pre {{
        margin: 0;
        padding: 22px 24px 28px;
        font-size: 13px;
        line-height: 1.55;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }}
    </style>
  </head>
  <body>
    <main>
      <div class="header">
        <div class="eyebrow">Agentic System Session View</div>
        <h1>{escape(field)}</h1>
        <p class="sub">Session <strong>{escape(session_id)}</strong></p>
      </div>
      <section class="panel">
        <div class="panel-head">{escape(str(session_path))}</div>
        <pre>{escaped_json}</pre>
      </section>
    </main>
  </body>
</html>
"""


def _open_file_in_viewer(path: Path) -> bool:
    """Best-effort open of a file in the platform's default viewer."""
    try:
        if sys.platform == "darwin":
            subprocess.run(
                ["open", str(path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        if sys.platform.startswith("linux"):
            subprocess.run(
                ["xdg-open", str(path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
    except Exception:
        return False
    return False


def _read_session_payload(path: Path) -> dict[str, object] | None:
    """Read a session JSON object from disk."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_session_payload(path: Path, payload: dict[str, object]) -> None:
    """Write a session JSON object to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


class StreamingDisplay:
    """Stateful streaming callback that buffers only the parsed response.

    Accumulates raw LLM tokens and uses extract_streaming_response() to
    track the latest response text. The text is only printed if the turn
    later passes parsing/validation. Raw JSON structure remains hidden.
    """

    def __init__(self, output: "TextIO" = None) -> None:
        import sys as _sys
        self._output = output or _sys.stdout
        self._accumulated = ""
        self._response_text = ""
        self._current_name = "agent"

    def __call__(self, token: str) -> None:
        """Called per token during model.generate(stream=True)."""
        self._accumulated += token
        response = extract_streaming_response(self._accumulated)
        if response is None:
            return
        self._response_text = response

    def reset(self, name: str = "agent") -> None:
        """Reset state for a new turn."""
        self._accumulated = ""
        self._response_text = ""
        self._current_name = name

    def commit(self) -> None:
        """Print the buffered response after successful parsing."""
        if not self._response_text:
            return
        self._output.write(f"\n{self._current_name}> {self._response_text}\n")
        self._output.flush()

    def discard(self) -> None:
        """Drop any buffered response from a failed parse attempt."""
        self._accumulated = ""
        self._response_text = ""


# --------------------------------------------------------------------------- #
# Package-level paths
# --------------------------------------------------------------------------- #

_BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "builtin_skills"


# --------------------------------------------------------------------------- #
# Default tool configuration
# --------------------------------------------------------------------------- #

_TOOL_DEFAULTS = {
    "IMAGE_ANALYSIS_PROVIDER": "ollama",
    "IMAGE_ANALYSIS_MODEL": "glm-ocr",
    "IMAGE_GENERATION_PROVIDER": "ollama",
    "IMAGE_GENERATION_MODEL": "x/z-image-turbo",
    "SEARXNG_BASE_URL": "http://127.0.0.1:8888",
}

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #


def _create_provider(
    provider_name: str,
    *,
    model: Optional[str] = None,
) -> "Agent":
    """Create the appropriate ModelProvider from a provider name string.

    Returns an object satisfying the ``ModelProvider`` protocol.
    """
    name = provider_name.strip().lower() or "ollama"

    if name == "ollama":
        from ..providers.ollama import OllamaProvider
        return OllamaProvider(model=model)

    # All other providers go through OpenAI-compatible adapter
    from ..providers.openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(provider=name, model=model)


def _normalize_session_id(session_id: str) -> str:
    """Validate and normalize a session identifier."""
    candidate = session_id.strip()
    if not candidate or not _SESSION_ID_RE.fullmatch(candidate):
        raise ValueError(
            "session_id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$"
        )
    return candidate


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
    2. Configures tool environment variables (image models, SearXNG, etc.)
    3. Builds the system prompt from the workspace content

    Args:
        workspace: Working directory for the agent session.
        session_id: Optional session identifier used to resume/persist state.
        provider: LLM provider name ("ollama", "deepseek", "lmstudio", "zai", etc.).
        mode: Execution mode ("auto" or "controlled").
        model: Model name override (uses provider defaults if not specified).
        image_analysis_provider: Override for IMAGE_ANALYSIS_PROVIDER.
        image_analysis_model: Override for IMAGE_ANALYSIS_MODEL.
        image_generation_provider: Override for IMAGE_GENERATION_PROVIDER.
        image_generation_model: Override for IMAGE_GENERATION_MODEL.
        searxng_base_url: Override for SEARXNG_BASE_URL.
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
        provider: str = "ollama",
        mode: str = "controlled",
        model: Optional[str] = None,
        image_analysis_provider: Optional[str] = None,
        image_analysis_model: Optional[str] = None,
        image_generation_provider: Optional[str] = None,
        image_generation_model: Optional[str] = None,
        searxng_base_url: Optional[str] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.session_id = _normalize_session_id(session_id) if session_id else None
        self.session_path = (
            self.workspace / ".sessions" / f"{self.session_id}.json"
            if self.session_id
            else None
        )
        self._session_loaded = False
        self.provider_name = provider
        self.mode = mode
        self._prompt_session = self._build_prompt_session()

        # 1. Bootstrap built-in skills into workspace
        self._bootstrap_skills()

        # 2. Configure tool environment variables
        self._configure_tool_environment(
            image_analysis_provider=image_analysis_provider,
            image_analysis_model=image_analysis_model,
            image_generation_provider=image_generation_provider,
            image_generation_model=image_generation_model,
            searxng_base_url=searxng_base_url,
        )

        # 3. Build components
        self._model = _create_provider(provider, model=model)

        self._stream_display = StreamingDisplay(sys.stdout)

        # Create the core agent — Agent owns prompt building from workspace
        self._agent = Agent(
            self._model,
            workspace=self.workspace,
        )

        # Create environment with sandbox executor
        self._env = Environment(
            workspace=self.workspace,
            executor=sandbox_executor,
            mode=mode,
        )
        if self.session_path is not None:
            if self._env.load_session(self.session_path):
                raw_session = _read_session_payload(self.session_path) or {}
                self._agent.last_prompt = str(raw_session.get("last_prompt", "") or "")
                self._session_loaded = True

        # Wire model into environment for sub-agent delegation
        self._env.set_model_ref(self._model)

        # Register approval policy as execution gate
        self._approval = ApprovalPolicy(mode=mode)
        self._env.on_before_execute(self._approval)

    # ----- Input ------------------------------------------------------------- #

    @staticmethod
    def _build_prompt_session() -> object:
        """Create multiline prompt session (Ctrl+D submits)."""
        if PromptSession is None or KeyBindings is None:
            return None
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _submit(event: Any) -> None:
            event.app.exit(result=event.app.current_buffer.text)

        return PromptSession(key_bindings=bindings)

    # ----- Bootstrap -------------------------------------------------------- #

    def _bootstrap_skills(self) -> None:
        """Sync built-in skills from the package into the workspace.

        Copies ``agentic_system/builtin_skills/`` into ``{workspace}/skills/``.
        Each built-in skill directory is replaced on startup so updates to
        the package propagate automatically. User-created skills in the
        workspace are preserved.
        """
        if not _BUILTIN_SKILLS_ROOT.exists():
            return

        ws_skills = self.workspace / "skills"
        ws_skills.mkdir(parents=True, exist_ok=True)

        for scope_dir in sorted(p for p in _BUILTIN_SKILLS_ROOT.iterdir() if p.is_dir()):
            if scope_dir.name.startswith((".", "_")):
                continue
            target_scope = ws_skills / scope_dir.name
            target_scope.mkdir(parents=True, exist_ok=True)

            for skill_dir in sorted(p for p in scope_dir.iterdir() if p.is_dir()):
                if skill_dir.name.startswith((".", "_")):
                    continue
                target_skill = target_scope / skill_dir.name
                # Replace entire skill directory to pick up updates
                if target_skill.exists():
                    if target_skill.is_dir():
                        shutil.rmtree(target_skill)
                    else:
                        target_skill.unlink()
                shutil.copytree(skill_dir, target_skill)

    # ----- Tool environment ------------------------------------------------- #

    def _configure_tool_environment(
        self,
        *,
        image_analysis_provider: Optional[str] = None,
        image_analysis_model: Optional[str] = None,
        image_generation_provider: Optional[str] = None,
        image_generation_model: Optional[str] = None,
        searxng_base_url: Optional[str] = None,
    ) -> None:
        """Set environment variables for skill scripts to read.

        Priority: CLI flag > existing env var > built-in default.
        Uses ``os.environ.setdefault()`` so existing env vars are preserved,
        then applies explicit CLI overrides on top.
        """
        # Apply defaults (only if not already set in environment)
        for key, default in _TOOL_DEFAULTS.items():
            os.environ.setdefault(key, default)

        # Apply explicit CLI overrides (these win over everything)
        overrides = {
            "IMAGE_ANALYSIS_PROVIDER": image_analysis_provider,
            "IMAGE_ANALYSIS_MODEL": image_analysis_model,
            "IMAGE_GENERATION_PROVIDER": image_generation_provider,
            "IMAGE_GENERATION_MODEL": image_generation_model,
            "SEARXNG_BASE_URL": searxng_base_url,
        }
        for key, value in overrides.items():
            if value is not None and value.strip():
                os.environ[key] = value.strip()

    # ----- Agent & streaming ------------------------------------------------ #

    @property
    def stream_display(self) -> StreamingDisplay:
        """Access the streaming display for loop integration."""
        return self._stream_display

    # ----- REPL ------------------------------------------------------------- #

    def start(self, show_banner: bool = True) -> int:
        """Run the interactive REPL until the user exits.

        Returns:
            Exit code (0 for normal exit).
        """
        if show_banner:
            print(f"Agentic System — provider={self.provider_name}, mode={self.mode}")
            print(f"Workspace: {self.workspace}")
            if self.session_id:
                state = "resumed" if self._session_loaded else "new"
                print(f"Session: {self.session_id} ({state})")
            else:
                print("Session: ephemeral")
            print("Type /help for commands. Type /exit to quit.")
            print("Multiline: Enter adds lines, Ctrl+D submits, Ctrl+C cancels.\n")

        try:
            while True:
                # Read user input
                try:
                    if self._prompt_session is not None:
                        user_input = self._prompt_session.prompt(
                            "user> ",
                            multiline=True,
                            prompt_continuation=lambda _w, _n, _s: "... ",
                        )
                    else:
                        user_input = input("user> ")
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
        self._stream_display._output = sys.stdout

        try:
            # Run agent loop (prints agent responses as they happen)
            run_loop(
                self._agent,
                self._env,
                on_turn_start=self._stream_display.reset,
                on_turn_end=self._stream_display.commit,
                on_turn_error=self._stream_display.discard,
                on_token_chunk=self._stream_display,
            )
        except RuntimeError as exc:
            message = f"Agent error: {exc}"
            print(f"\nruntime> {message}")
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
        return "\n".join([
            f"provider={self.provider_name}",
            f"mode={self.mode}",
            f"workspace={self.workspace}",
            f"session_id={self.session_id or 'none'}",
            f"session_state={self._session_state()}",
            f"image_analysis={os.environ.get('IMAGE_ANALYSIS_PROVIDER', 'none')}"
            f"/{os.environ.get('IMAGE_ANALYSIS_MODEL', 'none')}",
            f"image_generation={os.environ.get('IMAGE_GENERATION_PROVIDER', 'none')}"
            f"/{os.environ.get('IMAGE_GENERATION_MODEL', 'none')}",
            f"searxng={os.environ.get('SEARXNG_BASE_URL', 'not set')}",
            f"full_history_turns={len(self._env.full_history)}",
            f"observation_turns={len(self._env.observation)}",
            f"system_prompt_length={len(self._agent.system_prompt)} chars",
        ])

    def _shutdown(self) -> None:
        """Persist state before exit."""
        try:
            self._persist_session()
        except Exception:
            pass

    def _open_session_field_view(self, field: str) -> str:
        """Persist and open a field-specific HTML view for the current session."""
        if self.session_path is None or self.session_id is None:
            return "Inspection commands require --session-id because no session file exists."

        self._persist_session()
        raw_session = _read_session_payload(self.session_path)
        if raw_session is None:
            return f"Unable to read session file: {self.session_path}"

        value = raw_session.get(field, "(missing)")
        if field == "last_prompt" and not str(value):
            value = "(none yet)"
        html = _render_session_view_html(
            session_id=self.session_id,
            field=field,
            session_path=self.session_path,
            value=value,
        )

        view_dir = self.session_path.parent / "views"
        view_dir.mkdir(parents=True, exist_ok=True)
        view_path = (view_dir / f"{self.session_id}.{field}.html").resolve()
        view_path.write_text(html, encoding="utf-8")

        if _open_file_in_viewer(view_path):
            return f"Opened session view: {view_path}"
        return f"Session view written: {view_path}"

    def _persist_session(self) -> None:
        """Persist session state when a named session is active."""
        if self.session_path is None:
            return
        self._env.save_session(self.session_path)
        payload = _read_session_payload(self.session_path) or {}
        payload["last_prompt"] = self._agent.last_prompt
        _write_session_payload(self.session_path, payload)

    def _session_state(self) -> str:
        """Return a short user-visible description of current session mode."""
        if self.session_id is None:
            return "ephemeral"
        return "loaded" if self._session_loaded else "new"
