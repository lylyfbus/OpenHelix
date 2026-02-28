# Agentic System Design

Date: 2026-02-23
Status: Implemented Baseline (code-aligned)
Owner: Yang Liu

## 1) Purpose

This document describes the current implemented design of the Agentic System runtime in this repository.

It is written for readers who want to understand:
1. What the system does today.
2. How the runtime loop works.
3. Where safety and control boundaries are enforced.
4. How memory, skills, and knowledge are represented.

## 2) Design Principles

1. Keep runtime deterministic even when model output is non-deterministic.
2. Treat runtime as the safety and execution boundary.
3. Keep side effects explicit and auditable through `exec` actions.
4. Persist state so sessions can be resumed.
5. Grow capabilities through skills and knowledge assets, not runtime complexity.

## 3) Architecture Overview

### 3.1 Core Components

1. `AgentRuntime` (`/Users/yangliu/Projects/Business/AgenticSystem/agentic_system/runtime.py`)
- Initializes workspace, state, model router, prompt engine, and flow engine.
- Bootstraps runtime assets (`prompts/`, `skills/`) into the workspace.
- Owns interactive command handling (`/help`, `/status`, `/refresh`, `/exit`).

2. `FlowEngine` (`/Users/yangliu/Projects/Business/AgenticSystem/agentic_system/kernel/agent_loop.py`)
- Runs the core agent loop.
- Streams model `raw_response` tokens to UI.
- Validates output contract and retries invalid generations.
- Executes approved jobs and writes runtime evidence to history.

3. `PromptEngine` (`/Users/yangliu/Projects/Business/AgenticSystem/agentic_system/kernel/prompts.py`)
- Loads role system prompts from runtime workspace.
- Injects role descriptions, skill metadata, knowledge metadata, workspace path.
- Builds final prompt from system prompt + workflow summary + workflow history.
- Triggers one-pass history compaction only when token estimate exceeds limit.

4. `ModelRouter` (`/Users/yangliu/Projects/Business/AgenticSystem/agentic_system/kernel/model_router.py`)
- Selects provider adapter and role-specific model.
- Streams model output.
- Parses `<output>...</output>` JSON payload with parse diagnostics.
- Extracts `raw_response` stream from generated JSON text.

5. `Executors` (`/Users/yangliu/Projects/Business/AgenticSystem/agentic_system/kernel/executors.py`)
- Normalizes `exec` input shape.
- Starts isolated background jobs (`start_new_session=True`).
- Captures `stdout` and `stderr` via temp files.
- Supports cancellation escalation (`SIGINT -> SIGTERM -> SIGKILL`).

6. `StorageEngine` (`/Users/yangliu/Projects/Business/AgenticSystem/agentic_system/kernel/storage.py`)
- Persists session state atomically to `state.json`.
- Stores workflow history, full process history, summary, action history, approval caches.

### 3.2 Workspace Layout

For a given runtime workspace (`--workspace`), the runtime uses:

1. `sessions/<session_id>/state.json`
2. `prompts/agent_system_prompt.json`
3. `prompts/agent_role_description.json`
4. `skills/core-agent/...`
5. `skills/all-agents/...`
6. `knowledge/docs/...`
7. `knowledge/index/catalog.json`

Bootstrap rule:
1. Packaged prompts and skills are copied only if missing in workspace.
2. Existing workspace files are preserved.

## 4) Session and CLI Behavior

CLI entry (`/Users/yangliu/Projects/Business/AgenticSystem/agentic_system/cli.py`) supports:

1. `--workspace`
2. `--provider`: `ollama | lmstudio | zai | deepseek | openai_compatible`
3. `--model-name`
4. `--mode`: `auto | controlled`
5. `--session-id` (resume existing session)

Runtime commands:

1. `/help`
2. `/status`
3. `/status workflow_summary`
4. `/status workflow_hist`
5. `/status full_proc_hist`
6. `/status action_hist`
7. `/status core_agent_prompt`
8. `/refresh` (new session in same workspace)
9. `/exit`

## 5) Core Agent Loop (Implemented)

### 5.1 High-Level Flow

1. User input is appended to histories.
2. Runtime generates a core-agent turn with strict output validation.
3. Runtime executes chosen action (`chat_with_requester`, `keep_reasoning`, or `exec`).
4. Runtime loops until control is returned to requester or loop limits are reached.

### 5.2 Generation Cycle

For each generation attempt:

1. Build final prompt from current state.
2. Print `core_agent> ` and stream tokens from `raw_response`.
3. Parse model output from `<output>...</output>`.
4. Validate output contract.

If invalid:

1. Append runtime validation error line to histories.
2. Regenerate immediately.
3. Stop safely if `max_invalid_output_retries` is reached.

If valid:

1. Append action record to `action_hist`.
2. Append core-agent readable record to histories.
3. Continue to action handling.

### 5.3 Allowed Actions

1. `chat_with_requester`
- Ends internal loop and returns control to user.

2. `keep_reasoning`
- Continues internal loop without user handoff.

3. `exec`
- Runs through approval gate (except in `auto` mode).
- Starts background job.
- Waits for completion/cancellation.
- Appends structured runtime result (job summary + stdout/stderr).

Invalid action handling:

1. Runtime appends correction message.
2. Regenerates.
3. Stops when `max_invalid_action_retries` is reached.

## 6) Output Contract and Validation

The runtime expects one JSON object inside `<output>...</output>` with top-level keys:

1. `raw_response` (non-empty string)
2. `action` (`chat_with_requester | keep_reasoning | exec`)
3. `action_input` (object)

Action-specific checks:

1. For `chat_with_requester` or `keep_reasoning`, `action_input` must be `{}`.
2. For `exec`, `action_input` must satisfy executor schema.

Validation and streaming are intentionally separated:

1. Streaming occurs during generation.
2. Contract validation occurs after generation completes.

## 7) Exec Model, Approval, and Cancellation

### 7.1 Exec Input Schema

`exec` requires:

1. `code_type`: `bash` or `python`
2. Exactly one of:
- `script` (inline)
- `script_path` (path to script)
3. Optional `script_args` only when `script_path` is used
4. `job_name` is expected by prompt contract (runtime fallback is `none` if missing)

### 7.2 Execution Semantics

1. Jobs run in workspace cwd.
2. Each job has `job_id` and `job_name`.
3. Output is captured as `stdout` and `stderr`.
4. Runtime writes an evidence block into histories.

### 7.3 Approval Modes

1. `auto`: no prompt, execute directly.
2. `controlled`: prompt with choices:
- `y`: allow once
- `s`: allow same exact exec for session
- `p`: allow same pattern for session
- default/no: deny

Persisted approval caches in state:

1. `exec_approval_exact`
2. `exec_approval_pattern`

### 7.4 Cancellation Controls

While jobs are running:

1. `Ctrl+C`: cancel all running jobs.
2. `/cancel`: cancel all running jobs.
3. `/cancel <job_id>`: cancel specific job.

Cancellation metadata is appended to `stderr` for audit trail.

## 8) Prompt Construction and Context Management

### 8.1 Prompt Composition

Final prompt sections:

1. Role system prompt with placeholders resolved.
2. `Workflow Summary:`
3. `Workflow History:`

Injected placeholders include:

1. Role descriptions (`agent_role_description.json`)
2. Skill metadata snapshot (from workspace skills)
3. Knowledge metadata snapshot (from `knowledge/index/catalog.json`, core-agent only)
4. Runtime workspace path

### 8.2 History Compaction Strategy

Compaction runs only for `core_agent` when estimated prompt tokens exceed limit.

Current behavior:

1. Refresh `workflow_summary` once using `workflow_summarizer`.
2. Compact old head of `workflow_hist` using `workflow_history_compactor`.
3. Keep recent `K` lines unchanged (`compact_keep_last_k`).
4. Replace head with one `workflow_compactor>` line.

`full_proc_hist` remains full and un-compacted.

## 9) Persistence and Recovery

State is persisted in `sessions/<session_id>/state.json` with fields:

1. `session_id`
2. `full_proc_hist`
3. `workflow_hist`
4. `workflow_summary`
5. `action_hist`
6. `exec_approval_exact`
7. `exec_approval_pattern`

Recovery:

1. Start runtime with `--session-id <existing_id>`.
2. Runtime loads previous state and continues with current workspace assets.

## 10) Skills and Knowledge Baseline

Current packaged all-agent skills:

1. `load-skill`
2. `search-online-context`
3. `load-knowledge-docs`
4. `documentation-distillation`
5. `skill-creation`

Design intent:

1. Runtime remains minimal and executor-first.
2. Domain behavior is implemented by skill scripts + skill docs.
3. Knowledge docs are runtime workspace assets and are discoverable by metadata.

## 11) Provider Abstraction

Supported providers:

1. `ollama`
2. `lmstudio`
3. `zai`
4. `deepseek`
5. `openai_compatible`

Role model mapping currently covers:

1. `core_agent`
2. `workflow_summarizer`
3. `workflow_history_compactor`

All providers share the same runtime contract and loop behavior through `ModelRouter`.

## 12) Current Constraints

1. Sub-agent execution path is intentionally disabled.
2. Output parser requires `<output>...</output>` with valid JSON object.
3. Raw-response streaming extractor is JSON-key based and depends on `"raw_response"` emission.
4. `docs/specs/*` are not fully aligned with current implementation and should be treated as legacy drafts until rewritten.

## 13) Example Run

```bash
python -m agentic_system \
  --workspace ./runtime_test \
  --provider ollama \
  --model-name "glm-5:cloud" \
  --mode controlled
```
