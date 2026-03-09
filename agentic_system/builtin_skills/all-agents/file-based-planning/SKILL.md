---
name: File Based Planning

handler: scripts/init_planning.py
script_mode: multi
description: Manus-style file-based planning with persistent markdown files for complex multi-step tasks
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Implement Manus-style file-based planning for complex multi-step tasks. Use persistent markdown files as "working memory on disk" to overcome AI agent limitations like volatile memory, goal drift, and hidden errors.

# When To Use

**Use this pattern for:**
- Multi-step tasks (3+ steps)
- Research tasks
- Building/creating projects
- Tasks spanning many tool calls
- Anything requiring organization across context resets

**Skip for:**
- Simple questions
- Single-file edits
- Quick lookups

# Skill Mode

This skill uses **multi** script mode with three helper scripts:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `init_planning.py` | Initialize planning files from templates | Start of new task |
| `check_complete.py` | Verify all phases are complete | Before reporting done |
| `session_catchup.py` | Recover context from previous session | After context reset |

The LLM reasons between script executions according to the Procedure section.

# The Core Principle

```
Context Window = RAM (volatile, limited)
Filesystem = Disk (persistent, unlimited)

→ Anything important gets written to disk.
```

# The 3-File Pattern

| File | Purpose | When to Update |
|------|---------|----------------|
| `task_plan.md` | Phases, progress, decisions | After each phase |
| `findings.md` | Research, discoveries | After ANY discovery |
| `progress.md` | Session log, test results | Throughout session |

# Procedure

## Step 1: Check for Existing Session

Before starting work, check if planning files already exist:

```bash
ls task_plan.md findings.md progress.md 2>/dev/null
```

If files exist:
1. Read all three files to recover context
2. Check current phase status in task_plan.md
3. Resume from where you left off

## Step 2: Initialize Planning Files (if needed)

If no planning files exist, run the initialization script.

## Step 3: Fill In the Plan

Immediately after initialization:
1. Edit `task_plan.md` with the specific goal and phases
2. List key questions to answer
3. Set Phase 1 to `in_progress`

## Step 4: Execute with Attention Manipulation

- **Before major decisions:** Re-read `task_plan.md`
- **After every 2 view/browser/search operations:** Save findings to `findings.md`
- **After completing each phase:** Update status and log errors

## Step 5: Verify Completion

Before reporting task complete, run check_complete.py.

# Runtime Contract

All scripts output a JSON object to stdout as the final line:

- `stdout`: Contains execution evidence and final JSON result
- `stderr`: Reserved for unexpected runtime failures only
- Exit code: 0 for success, non-zero for failures

Scripts are idempotent and safe to re-run.

# Action Input Templates

## Initialize Planning Files

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/file-based-planning/scripts/init_planning.py",
  "script_args": ["--project-name", "<brief-task-name>"]
}
```

## Check Completion Status

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/file-based-planning/scripts/check_complete.py"
}
```

## Session Catchup

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/file-based-planning/scripts/session_catchup.py",
  "script_args": []
}
```

# Output JSON Shape

## init_planning.py Output

```json
{
  "executed_skill": "file-based-planning",
  "action": "init",
  "status": "ok|partial|error|dry_run",
  "project_name": "string",
  "created": ["task_plan.md", "findings.md", "progress.md"],
  "skipped": [],
  "errors": [],
  "message": "string"
}
```

## check_complete.py Output

```json
{
  "executed_skill": "file-based-planning",
  "action": "check_complete",
  "status": "complete|in_progress|no_plan|no_phases|error",
  "message": "string",
  "total": 0,
  "complete": 0,
  "in_progress": 0,
  "pending": 0,
  "error_code": ""
}
```

## session_catchup.py Output

```json
{
  "executed_skill": "file-based-planning",
  "action": "session_catchup",
  "status": "active_session|no_session|partial_session|error",
  "message": "string",
  "files": {},
  "recommendation": "resume|init|recover",
  "file_errors": []
}
```

# Error Handling Rule

1. Scripts always return one final JSON object on stdout, including error scenarios.
2. **Missing templates** in `init_planning.py` return `status = "partial"` (or `"error"` if nothing could be initialized) with details in JSON `errors`.
3. **File permission/read issues** return structured JSON status (`error` or `partial_session`) instead of tracebacks.
4. **Invalid/no phase plan** in `check_complete.py` returns `status = "no_phases"` with a helpful message.
5. Re-run safety: helper scripts are idempotent and safe to re-run.

# Critical Rules

## Rule 1: Create Plan First
Never start a complex task without `task_plan.md`. Non-negotiable.

## Rule 2: The 2-Action Rule
After every 2 view/browser/search operations, IMMEDIATELY save key findings to text files.

## Rule 3: Read Before Decide
Before major decisions, read the plan file to keep goals in attention window.

## Rule 4: Update After Act
After completing any phase, mark status and log errors.

## Rule 5: Log ALL Errors
Every error goes in the plan file to build knowledge and prevent repetition.

## Rule 6: Never Repeat Failures
If action failed, next_action must be different. Track attempts, mutate approach.

# The 3-Strike Error Protocol

```
ATTEMPT 1: Diagnose & Fix
  → Read error carefully
  → Identify root cause
  → Apply targeted fix

ATTEMPT 2: Alternative Approach
  → Same error? Try different method
  → NEVER repeat exact same failing action

ATTEMPT 3: Broader Rethink
  → Question assumptions
  → Search for solutions
  → Consider updating the plan

AFTER 3 FAILURES: Escalate to User
  → Explain what you tried
  → Share the specific error
  → Ask for guidance
```

# The 5-Question Reboot Test

| Question | Answer Source |
|----------|---------------|
| Where am I? | Current phase in task_plan.md |
| Where am I going? | Remaining phases |
| What's the goal? | Goal statement in plan |
| What have I learned? | findings.md |
| What have I done? | progress.md |

# Read vs Write Decision Matrix

| Situation | Action | Reason |
|-----------|--------|--------|
| Just wrote a file | DON'T read | Content still in context |
| Viewed image/PDF | Write findings NOW | Multimodal → text before lost |
| Browser returned data | Write to file | Screenshots don't persist |
| Starting new phase | Read plan/findings | Re-orient if context stale |
| Error occurred | Read relevant file | Need current state to fix |
| Resuming after gap | Read all planning files | Recover state |

# Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Stuff everything in context | Store large content in files |
| Start executing immediately | Create plan file FIRST |
| Repeat failed actions | Track attempts, mutate approach |
| Hide errors and retry silently | Log errors to plan file |
| State goals once and forget | Re-read plan before decisions |

# Skill Dependencies

This skill references:
- **search-online-context** - For research phases
- **documentation-distillation** - For capturing learnings

# Templates

Templates stored in `skills/all-agents/file-based-planning/templates/`:

- `task_plan.md` - Phase tracking template
- `findings.md` - Research storage template
- `progress.md` - Session log template

# Notes

- Planning files go in project working directory, not skill folder
- Use `exec` action to run helper scripts
- Update phase status: pending → in_progress → complete
- Re-read plan before major decisions (attention manipulation)
- Log ALL errors - they help avoid repetition
- Template path is resolved relative to the script location (bootstrap-safe, no machine-specific absolute paths)
