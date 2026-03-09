"""Runtime Host — interactive REPL that wires all framework components.

Replaces the legacy ``runtime.py`` + ``FlowEngine`` + ``StorageEngine``
with a clean host built on the new core abstractions.

Usage::

    host = RuntimeHost(workspace="/path/to/workspace", provider="ollama")
    host.start()
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from ..core.agent import Agent
from ..core.environment import Environment
from ..core.loop import run_loop
from ..core.state import Turn
from .sandbox import sandbox_executor
from .approval import ApprovalPolicy


# --------------------------------------------------------------------------- #
# Streaming display
# --------------------------------------------------------------------------- #


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

    # Extract string value (handle escaped quotes)
    result_chars: list[str] = []
    i = 1  # skip opening quote
    while i < len(after_colon):
        ch = after_colon[i]
        if ch == "\\" and i + 1 < len(after_colon):
            result_chars.append(after_colon[i + 1])
            i += 2
            continue
        if ch == '"':
            break
        result_chars.append(ch)
        i += 1

    return "".join(result_chars) if result_chars else None


class StreamingDisplay:
    """Stateful streaming callback that shows only the parsed response.

    Accumulates raw LLM tokens and uses extract_streaming_response() to
    progressively display the response text with an 'agent>' prefix.
    Raw JSON structure (tags, keys, action, action_input) is hidden.
    """

    def __init__(self, output: "TextIO" = None) -> None:
        import sys as _sys
        self._output = output or _sys.stdout
        self._accumulated = ""
        self._displayed_len = 0
        self._prefix_printed = False
        self._current_name = "agent"

    def __call__(self, token: str) -> None:
        """Called per token during model.generate(stream=True)."""
        self._accumulated += token
        response = extract_streaming_response(self._accumulated)
        if response is None:
            return

        if not self._prefix_printed:
            self._output.write(f"\n{self._current_name}> ")
            self._prefix_printed = True

        # Print only the NEW characters since last display
        new_text = response[self._displayed_len:]
        if new_text:
            self._output.write(new_text)
            self._output.flush()
            self._displayed_len = len(response)

    def reset(self, name: str = "agent") -> None:
        """Reset state for a new turn."""
        self._accumulated = ""
        self._displayed_len = 0
        self._prefix_printed = False
        self._current_name = name

    def finalize(self) -> None:
        """Print trailing newline after generation completes."""
        if self._prefix_printed:
            self._output.write("\n")
            self._output.flush()


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
        "  /exit            Quit.",
    ])

    def __init__(
        self,
        workspace: Path,
        *,
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
        self.provider_name = provider
        self.mode = mode

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

        # Wire model into environment for sub-agent delegation
        self._env.set_model_ref(self._model)

        # Register approval policy as execution gate
        self._approval = ApprovalPolicy(mode=mode)
        self._env.on_before_execute(self._approval)

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
            print("Type /help for commands. Type /exit to quit.\n")

        try:
            while True:
                # Read user input
                try:
                    user_input = input("user> ").strip()
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

        # Run agent loop (prints agent responses as they happen)
        run_loop(
            self._agent,
            self._env,
            on_turn_start=self._stream_display.reset,
            on_turn_end=self._stream_display.finalize,
            on_token_chunk=self._stream_display,
        )

        # Persist state after each interaction
        self._env.save_session(self.workspace / ".session")

    def _handle_command(self, command_line: str) -> Optional[str]:
        """Process slash commands. Returns None for exit, string for output."""
        cmd = command_line.split()[0].lower()

        if cmd == "/exit":
            return None
        if cmd == "/help":
            return self.HELP_TEXT
        if cmd == "/status":
            return self._status_text()
        return f"Unknown command: {cmd}. Use /help."

    def _status_text(self) -> str:
        """Build session status overview."""
        return "\n".join([
            f"provider={self.provider_name}",
            f"mode={self.mode}",
            f"workspace={self.workspace}",
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
            self._env.save_session(self.workspace / ".session")
        except Exception:
            pass
