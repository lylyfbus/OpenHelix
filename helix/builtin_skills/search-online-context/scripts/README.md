# Multi-Script Phase Map

Use one script per bounded research phase and let the core agent reason between executions.

Default scaffold:
- phase: search -> script: scripts/search_searxng.py
- phase: fetch -> script: scripts/fetch_pages.py
- phase: quick-single-round -> script: scripts/search_and_fetch.py

Preferred flow:
1. Run `search_searxng.py` with a broad query.
2. Read the returned ranked URLs/snippets.
3. Run `fetch_pages.py` for selected URLs.
4. Refine the query and repeat as needed.

Use `search_and_fetch.py` only as a shortcut for a quick one-round bootstrap when the extra control is not needed.
