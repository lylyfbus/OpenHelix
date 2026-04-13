---
name: Brainstorming
description: Turn rough ideas into validated design direction through structured dialogue before implementation.
---

# Purpose

Use this skill to convert an idea into a clear, reviewable design before building code or running major implementation actions.

# When To Use

Use when:
- requirements are ambiguous or incomplete
- user asks to design, plan, explore options, or "think first"
- task may cause rework if implementation starts too early

Do not use as a long discovery loop when user asked for immediate execution and requirements are already explicit.

# Skill Mode

- script_mode: `none`
- This is a process skill (reasoning + dialogue), not a dedicated runtime script.
- Core Agent should choose among:
  - `chat` for questions/validation
  - `think` for internal synthesis
  - `exec` only when reading/writing concrete workspace artifacts is necessary

# Procedure

1. Gather context:
   - inspect relevant project files/docs only as needed
   - identify current state, constraints, and obvious unknowns
2. Clarify with one focused question at a time:
   - prefer multiple-choice questions when possible
   - do not ask multiple unrelated questions in one turn
3. Lock objective:
   - confirm purpose, constraints, success criteria, and non-goals
4. Explore alternatives:
   - propose 2-3 approaches with trade-offs
   - lead with recommended option and brief rationale
5. Draft design incrementally:
   - present in small sections (about 200-300 words each)
   - ask requester to validate each section before continuing
6. Verify readiness:
   - confirm open risks, dependencies, and test/verification strategy
   - ask explicit go/no-go before implementation handoff
7. Optional persistence:
   - if requested, save approved design under workspace (for example `docs/plans/<date>-<topic>-design.md`)
8. Final reporting:
   - report the selected direction, key trade-offs, and immediate next step to requester.

# Runtime Contract

No dedicated skill script is required.

If `exec` is used during brainstorming:
1. keep operations scoped to runtime workspace
2. make `stdout` concise and informative (what was checked/created)
3. use `stderr` only for actionable failures
4. prefer deterministic, reviewable artifacts (notes/design docs) over hidden assumptions

# Action Input Templates

No required script template for this skill.

Typical action patterns:

1. Ask a focused clarifying question:
```json
{
  "action": "chat",
  "action_input": {}
}
```

2. Synthesize constraints/options internally:
```json
{
  "action": "think",
  "action_input": {}
}
```

3. Optional: persist approved design to workspace:
```json
{
  "action": "exec",
  "action_input": {
    "job_name": "save-design-doc",
    "code_type": "bash",
    "script": "mkdir -p docs/plans && cat > docs/plans/2026-02-25-example-design.md <<'EOF'\n# Example Design\n...\nEOF\necho \"saved docs/plans/2026-02-25-example-design.md\""
  }
}
```

# Output JSON Shape

No dedicated script output JSON is required for this skill.

If optional `exec` is used for persistence, stdout should clearly expose the resulting artifact path(s).

# Error Handling Rule

1. If requester intent is still unclear after several clarification turns, present a short assumption set and ask for explicit confirmation.
2. If context lookup fails (missing files, permission issues), state the blocker and continue brainstorming with explicit assumptions.
3. If design persistence fails, return to requester with failure summary and next minimal fix option.
4. Stop internal looping once design is validated or requester chooses to proceed directly.

# Skill Dependencies

Use dependency skills only when needed:
- `search-online-context`: gather external facts when design depends on current/public info.
- `file-based-planning`: structure complex execution phases after design approval.
- `documentation-distillation`: convert validated design decisions into reusable knowledge.

# Notes

- "Too simple to design" is an anti-pattern; even small tasks should have a minimal design statement.
- Keep brainstorming concise and decision-oriented; avoid endless ideation loops.
- Do not start major implementation until requester confirms design direction.
