---
name: Search Online And Fetch Context
description: Search current online sources with SearXNG and fetch selected page text through explicit research phases.
---

# Purpose

Use this skill when you need current online information and want evidence-driven web research with explicit search and fetch phases.

# When To Use

Use when:
- the task depends on current or external online information
- the agent needs direct source context before answering
- iterative search and cross-checking will improve answer quality

Skip when:
- workspace knowledge already contains the needed information
- a local file or existing runtime evidence answers the question
- the task is simple enough that external search is unnecessary

# Skill Mode

- `script_mode: multi`
- Preferred for real research because the agent should reason between phases.
- Default phase scripts:
  - `scripts/search_searxng.py`: run one search round and return ranked results only
  - `scripts/fetch_pages.py`: fetch and clean text from selected URLs
  - `scripts/search_and_fetch.py`: compatibility shortcut for one combined round; use only when the extra phase control is not needed

# Procedure

1. Gather context:
   - derive a focused initial query from the user request and current workflow state
   - choose search parameters only as specific as needed
2. Search:
   - run `scripts/search_searxng.py` first
   - inspect ranked results, snippets, domains, and entities
3. Select targets:
   - choose the most promising URLs to inspect more deeply
   - prefer authoritative or directly relevant sources
4. Fetch:
   - run `scripts/fetch_pages.py` for selected URLs
   - read `fetched_context` and note useful facts, contradictions, or missing details
5. Iterate:
   - refine the query from fetched evidence
   - repeat search then fetch until answer quality is sufficient
6. Verify and report:
   - cross-check important claims across fetched sources when the task warrants it
   - stop when the evidence is enough for a confident answer

Recommended search rounds: 2-5 unless the requester asks for deeper research.

# Runtime Contract

All scripts in this skill must:
1. print one final JSON object to stdout
2. use stderr only for unexpected runtime failures
3. keep stdout concise but informative so workflow history remains readable

The next reasoning step should inspect runtime stdout/stderr before deciding the next phase.

# Action Input Templates

## Phase 1: Search

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/search-online-context/scripts/search_searxng.py",
  "script_args": [
    "--query", "site:forecast.weather.gov chicago tomorrow weather",
    "--limit", "8",
    "--language", "en-US",
    "--categories", "general",
    "--safesearch", "1"
  ]
}
```

## Phase 2: Fetch Selected Pages

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/search-online-context/scripts/fetch_pages.py",
  "script_args": [
    "--url", "https://forecast.weather.gov/MapClick.php?lat=41.88&lon=-87.63",
    "--url", "https://www.weather.gov/lot/",
    "--context-chars", "2500",
    "--max-total-context-chars", "12000"
  ]
}
```

## Optional Shortcut: One Combined Round

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/search-online-context/scripts/search_and_fetch.py",
  "script_args": [
    "--query", "site:forecast.weather.gov chicago tomorrow weather",
    "--limit", "8",
    "--fetch", "4",
    "--context-chars", "2500",
    "--max-total-context-chars", "15000"
  ]
}
```

# Output JSON Shape

## Search Phase

```json
{
  "executed_skill": "search-online-context",
  "phase": "search",
  "status": "ok|error",
  "query": "...",
  "search_results": "..."
}
```

## Fetch Phase

```json
{
  "executed_skill": "search-online-context",
  "phase": "fetch",
  "status": "ok|error",
  "fetched_context": "..."
}
```

## Combined Shortcut

```json
{
  "executed_skill": "search-online-context",
  "phase": "search-and-fetch",
  "status": "ok|error",
  "query": "...",
  "fetched_context": "..."
}
```

# Error Handling Rule

1. If the search phase returns no useful results, reformulate the query instead of repeating the same one.
2. If fetch results are weak or contradictory, fetch a more authoritative subset of URLs or run a narrower follow-up query.
3. If the backend is unavailable or repeatedly failing, stop internal retries and return to the requester with a concise blocker summary.
4. Do not keep appending more fetched pages if the existing evidence is already enough to answer.

# Skill Dependencies

- `load-skill`: load this skill before first use so the phase structure is visible in workflow history.
- `file-based-planning`: use for large research tasks that need persistent notes across many rounds.
- `documentation-distillation`: use after meaningful research when the findings should become reusable knowledge.

# Notes

- Ensure SearXNG is running and reachable through `SEARXNG_BASE_URL`.
- Prefer `search_searxng.py` plus `fetch_pages.py` for non-trivial research because the agent gets a clearer decision point between phases.
- Keep fetch targets selective; fetching too many pages reduces signal and bloats workflow history.
- Defaults: `--limit 8`, `--context-chars 2500`, `--max-total-context-chars 15000`.
