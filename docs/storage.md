# Storage Layout

## Workspace (per-project)

```
{workspace}/
  skills/                                   Reusable skill procedures
    builtin_skills/                         Managed by OpenHelix (synced on startup)
      search-online-context/
      generate-image/
      create-skill/
      ...
    my-custom-skill/                        User-created (never touched by OpenHelix)

  knowledge/                                Knowledge library
    index.json                              Global classification index
    {category}/{subcategory}/
      catalog.json                          Document metadata
      docs/*.md                             Documents

  sessions/{session-id}/                    Per-session data
    project/                                Project repos, apps, code
    docs/                                   Plans, research, notes
    .state/
      session_state.json                    Conversation history + workflow summary

  .runtime/                                 Runtime data (not user-facing)
    docker/cache/                           Persistent sandbox cache (pip, npm, venv)
    logs/                                   Exec stdout/stderr log files
    tmp/                                    Temporary files
    builtin_skills_manifest.json            Tracks managed skill names
```

## Global (`~/.helix`)

```
~/.helix/
  services/
    searxng/                                SearXNG service data
      state.json                            Service state (container, network, URL)
      config/settings.yml                   SearXNG configuration
      data/                                 SearXNG cache

    local-model-service/                    Local model service data
      state.json                            Service state (PID, port, token)
      models/                               Downloaded model weights
        {repo-id}/                          One directory per model
      venvs/                                Per-backend Python environments
        mlx/
        pytorch/
      sources/                              Downloaded runtime source files
        {skill-name}/{commit}/
```

## Key Files

| File | Purpose |
|---|---|
| `session_state.json` | Full conversation history, observation window, workflow summary |
| `builtin_skills_manifest.json` | List of built-in skill names (for cleanup on updates) |
| `services/*/state.json` | Service endpoint info (read by RuntimeHost for discovery) |
| `knowledge/index.json` | Global knowledge classification index |
| `*/catalog.json` | Per-subcategory document metadata |
| `*/SKILL.md` | Skill procedure and metadata |
| `*/model_spec.json` | Model download specification |
| `*/host_adapter.py` | Host-side ML model plugin |
