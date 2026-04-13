"""LLM provider.

Exports:
    LLMProvider: Universal OpenAI-compatible provider.
"""

from .openai_compat import LLMProvider

__all__ = ["LLMProvider"]
