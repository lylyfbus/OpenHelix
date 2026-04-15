# Storage Layout

OpenHelix keeps state in two clearly separated places:

- **Local (per-workspace)** — everything tied to a specific project lives inside your `--workspace` directory: the agent's skills, its knowledge library, session state, sandbox cache. Pick up a workspace directory and you pick up the whole project.
- **Global (`~/.helix`)** — machine-wide state that doesn't belong to any one workspace: running services, downloaded ML model weights, per-backend Python venvs. Delete this whole tree to reclaim GB; it can always be re-created.

Neither location holds any secrets. Everything is plain files you can read, edit, and version-control.

## Local — your workspace

```
{workspace}/
  skills/                                   Reusable skill procedures
    builtin_skills/                         Managed by OpenHelix — synced on startup, do not edit
      search-online-context/
      generate-image/
      ...
    my-custom-skill/                        User-created — yours, never touched by OpenHelix

  knowledge/                                Knowledge library (see knowledge.md)
    index.json                              Global classification index
    {category}/{subcategory}/
      catalog.json                          Document metadata
      docs/*.md                             Actual knowledge documents

  sessions/{session-id}/                    Per-session data, keyed by --session-id
    project/                                Project repos, apps, code the agent produces
    docs/                                   Plans, research notes, session artifacts
    .state/
      session_state.json                    Core-agent conversation history + workflow summary
      sub_agents_meta.json                  Registry of sub-agents created this session
      sub_agents/
        {role}.json                         Per-sub-agent persisted state (history, observation, last_prompt)
      views/
        {session-id}.{field}.html           HTML snapshots written by /view <field>
        {session-id}.sub_agent.{role}.{field}.html    HTML snapshots written by /view <field> <role>

  .runtime/                                 Runtime scratch space (never committed to git)
    docker/cache/                           Persistent sandbox cache (pip, npm, venv)
    logs/                                   Per-exec stdout/stderr log files
    tmp/                                    Temporary files
    builtin_skills_manifest.json            Tracks which built-in skills OpenHelix manages
```

**Rules of thumb:**

- **Files you write freely** — `skills/my-*/` (your own skills), `knowledge/**/*.md` (knowledge docs), `sessions/{id}/project/` (whatever the agent is building).
- **Files you read but shouldn't hand-edit** — `sessions/{id}/.state/` (the agent's memory, including core-agent and sub-agent state), `.runtime/*` (runtime-managed).
- **Files OpenHelix overwrites on every startup** — `skills/builtin_skills/*`. Put customizations in a copy under `skills/` at the top level instead.
- **Safe to version-control** — `skills/my-*/`, `knowledge/`, `sessions/{id}/project/`, `sessions/{id}/docs/`.
- **Should be gitignored** — `.runtime/`, `skills/builtin_skills/` (synced from the package), `sessions/{id}/.state/`.

## Global — `~/.helix`

```
~/.helix/
  services/
    searxng/                                SearXNG search service
      state.json                            Service state (container, network, URL)
      config/settings.yml                   SearXNG configuration
      data/                                 SearXNG cache

    local-model-service/                    Local ML model service
      state.json                            Service state (PID, port, auth token)
      models/                               Downloaded model weights from HuggingFace Hub.
        uqer1244--MLX-z-image/              One directory per HF repo_id; slashes
        notapalindrome--ltx23-mlx-av-q4/    become `--` in the directory name.
        ...
      venvs/                                Per-backend Python environments
        mlx/                                Shared by mlx-backed skills
        pytorch/                            Shared by pytorch-backed skills
        {custom-backend}/                   One per custom backend name
```

**Rules of thumb:**

- **Shared across every workspace on the machine** — model weights, venvs, running services. Start OpenHelix from two different workspaces and both see the same downloaded models.
- **All model weights come from [HuggingFace Hub](https://huggingface.co)** — this is currently the only download source. If a model is gated or private, set `HF_TOKEN` in your shell before running `helix model download`.
- **Safe to delete** — any directory under `~/.helix/services/local-model-service/models/` if you no longer need that model. Next `helix model download --skill X` will re-fetch it from HuggingFace.
- **Do not hand-edit** — `state.json` files describe live services; touch them while a service is running and things break. Use `helix stop SERVICE` first.
- **Each model directory's layout comes from the upstream** — OpenHelix doesn't rearrange what the HuggingFace CLI produces. An adapter may add a co-located `_runner/` subdirectory with fetched upstream Python files; that's normal.

## Key Files Reference

| File | Lives in | Purpose |
|---|---|---|
| `sessions/{id}/.state/session_state.json` | workspace | Core-agent history, observation window, compacted workflow summary |
| `sessions/{id}/.state/sub_agents_meta.json` | workspace | Registry of sub-agent personas (role + description) created this session |
| `sessions/{id}/.state/sub_agents/{role}.json` | workspace | Per-sub-agent persisted state (history, observation, last_prompt) — restored on re-delegation to the same role |
| `.runtime/builtin_skills_manifest.json` | workspace | List of built-in skill names (tracked for clean upgrades) |
| `~/.helix/services/*/state.json` | global | Service endpoint info (read by `helix status` and auto-discovered on startup) |
| `knowledge/index.json` | workspace | Global knowledge classification index |
| `knowledge/{cat}/{sub}/catalog.json` | workspace | Per-subcategory document metadata |
| `skills/**/SKILL.md` | workspace | Skill procedure + frontmatter metadata |
| `skills/**/model_spec.json` | workspace | Model download specification for a generative skill |
| `skills/**/host_adapter.py` | workspace | Host-side ML model plugin for a generative skill |
