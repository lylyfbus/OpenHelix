"""LLM provider and protocol.

Exports:
    ModelProvider: Protocol that all providers must satisfy.
    LLMProvider: Universal OpenAI-compatible provider.
    create_provider: Factory to create an LLMProvider from CLI/env args.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol


class ModelProvider(Protocol):
    """Minimal contract for LLM providers."""

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text from the given prompt."""
        ...


def create_provider(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> "LLMProvider":
    """Create an LLMProvider from explicit args or environment variables."""
    from .openai_compat import LLMProvider

    return LLMProvider(base_url=base_url, api_key=api_key, model=model)
