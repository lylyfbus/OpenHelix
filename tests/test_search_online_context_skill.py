"""Verification tests for the built-in search-online-context multi-script skill."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = (
    ROOT
    / "helix"
    / "builtin_skills"
    / "search-online-context"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search_phase = _load_module(SKILL_DIR / "scripts" / "search_searxng.py", "search_phase_script")
fetch_phase = _load_module(SKILL_DIR / "scripts" / "fetch_pages.py", "fetch_phase_script")
combined_phase = _load_module(SKILL_DIR / "scripts" / "search_and_fetch.py", "combined_phase_script")


def test_skill_md_declares_multi_mode():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert "script_mode: multi" in text
    assert "handler: scripts/search_searxng.py" in text
    assert "scripts/fetch_pages.py" in text
    print("  search-online-context SKILL.md multi mode OK")


def test_search_phase_ok():
    original = search_phase.search_searxng
    try:
        search_phase.search_searxng = lambda **kwargs: [
            {
                "rank": 1,
                "title": "Forecast",
                "url": "https://example.com/forecast",
                "snippet": "Forecast details",
                "engines": ["searxng"],
            }
        ]
        out = search_phase.run(
            query="chicago weather",
            limit=5,
            timeout=20,
            searxng_base_url="http://127.0.0.1:8888",
            language="en-US",
            categories="general",
            safesearch=1,
        )
        assert out["status"] == "ok"
        assert out["phase"] == "search"
        assert "https://example.com/forecast" in str(out["search_results"])
        print("  search phase OK")
    finally:
        search_phase.search_searxng = original


def test_fetch_phase_ok():
    original = fetch_phase.fetch_urls
    try:
        fetch_phase.fetch_urls = lambda **kwargs: [
            {
                "title": "https://example.com/forecast",
                "url": "https://example.com/forecast",
                "status": "ok",
                "context": "Forecast body text",
                "error": "",
            }
        ]
        out = fetch_phase.run(
            urls=["https://example.com/forecast"],
            context_chars=2500,
            max_total_context_chars=12000,
            timeout=20,
        )
        assert out["status"] == "ok"
        assert out["phase"] == "fetch"
        assert "Forecast body text" in str(out["fetched_context"])
        print("  fetch phase OK")
    finally:
        fetch_phase.fetch_urls = original


def test_combined_phase_ok():
    original_search = combined_phase.search_searxng
    original_fetch = combined_phase.fetch_urls
    try:
        combined_phase.search_searxng = lambda **kwargs: [
            {
                "rank": 1,
                "title": "Forecast",
                "url": "https://example.com/forecast",
                "snippet": "Forecast details",
                "engines": ["searxng"],
            }
        ]
        combined_phase.fetch_urls = lambda **kwargs: [
            {
                "title": "https://example.com/forecast",
                "url": "https://example.com/forecast",
                "status": "ok",
                "context": "Forecast body text",
                "error": "",
            }
        ]
        out = combined_phase.run(
            query="chicago weather",
            limit=5,
            fetch_count=2,
            context_chars=2500,
            max_total_context_chars=12000,
            timeout=20,
            searxng_base_url="http://127.0.0.1:8888",
            language="en-US",
            categories="general",
            safesearch=1,
        )
        assert out["status"] == "ok"
        assert out["phase"] == "search-and-fetch"
        assert "Forecast body text" in str(out["fetched_context"])
        print("  combined phase OK")
    finally:
        combined_phase.search_searxng = original_search
        combined_phase.fetch_urls = original_fetch


if __name__ == "__main__":
    print("=== Search Online Context Skill ===")
    test_skill_md_declares_multi_mode()
    test_search_phase_ok()
    test_fetch_phase_ok()
    test_combined_phase_ok()
    print("\n✅ All search-online-context skill tests passed!")
