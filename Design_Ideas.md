# Agentic System Similar to Claude Code

## Summary
In summary, I want an agent which is smart, poweverful and can evolute itself. We can abstract this system as a mind model, that is, the LLM is the brain and runtime system is the computer equiped to the brain. LLM itself only handles thinking with input and output both as text. However, the runtime system will provide it capacity to interact with the real digital world and evolute itself with executable tools and structured file system.

## Features
- The agentic system has two components, the core agent brain (i.e. LLM) and runtime system (i.e. the computer for the brain).
- The agentic system could be started from terminal and will have a UI which the user can communicate with the agent.
- For core agent:
    - It will have a standard workflow for any request, either from user, itself or sub-agents. "load and understanding context" -> "planning" -> "act" -> "observe results and verify" -> iterate -> "report"
    - It will be equiped with pre-defined skills. I can think about the ones below
        - Be able to make clear documentation based on both successes and failures and manage the documents in a well strucutred hierarchical doc system with clear index for accurate searching. This documentation system should be treated as a long-term memory.
        - Be able to know how to reload useful long-term memory in a prograssive disclouse way.
        - Be able to summarize cumulated historical conversation into meaningful and valuable summary if the next context generation is blocked due to the context window limit.
        - Be able to be smart enough to know to explore information deeply from internet if it doesn't understand something and doesn't know the solution. The compacted summary should be treated as a session-level memory or short-term memory.
        - Be able to create new skills if it find necessary.
        - Be able to reload session-level memory and necessary long-term memory to recover the exact checkpoint if the system broke due to errors. 
        - Be able to create sub-agents with appropriate roles to handle specific tasks and pass all its skills to the sub-agents except the one to create sub-agents. That means sub-agents can't create sub-agents any more.
        - Be able to double check with user about important decisions, e.g. actions may change the real working file system, etc. 
For sub-agents:
    - It will have a standard workflow similar as core agent
    - It will be equipred with pre-defined skills the same as core agent except the one to create sub-agents.
    - It will have one special skill during verification step. That is instead of doing self verification, it will know to let core agent have a kind of 2PR review and verification. 
For runtime system (environment):
    - It will provide logic to LLM to run any Bash/Python command and script, so that the LLM can do real jobs.
    - It will provide search engine for LLM, so that LLM can connect to internet.
    - It will provide disk space (maybe just in runtime folder) for LLM to manage documents and skills, so that LLM can evolute itself overtime.
    - It will save all input and output pairs to any LLM (core agent + sub-agnets) as the raw records.
    - It will save basic runtime session-level information for runtime to reload.
    


## Adhoc info
```
def update_context_w_stm(working_hist, current_temp):
  context_sum = build_context(system_prompt_temp,
                              skills_meta_data,
                              summary_prompt_temp,
                              working_hist)
  response = call_llm(context)
  stm = response.get("stm")
  working_hist = [f"stm> : {stm}"] + working_hist[-10:]
  context = build_context(system_prompt_temp, 
                          skills_meta_data,
                          current_temp,
                          next_step_temp,
                          working_hist,
                          mode="summary")
  return context

While True:
  # initiate chat
  input = get_input()
  # build context for core agent
  working_hist.append(f"user> : {input}")
  skills_meta_data = load_skills_meta()
  context = build_context(system_prompt_temp, 
                          skills_meta_data,
                          chat_temp,
                          next_step_temp, 
                          working_hist,
                          mode="full")
  if context > context window:
    context = update_context_w_stm(working_hist, chat_temp)
  response = call_llm(context)
  next_step = response.get("next_step")
  raw_response = response.get("raw_response")
  working_hist.append(f"core_agent> : {raw_response}")
  structured_info = response.get("structured_info")
  while next_step:
    if next_step == "act":
      if structured_info.get("code_path"):
        code = read_code(code_path, structured_info.get("code_type"))
        output = run_code(code)
        working_hist.append(f"runtime> : {output}")
        context = build_context(system_prompt_temp,
                                skills_meta_data,
                                observe_output_temp,
                                next_step_temp,
                                working_hist,
                                mode="full")
        if context > context window:
          context = update_context_w_stm(working_hist, observe_output_temp)
        response = call_llm(context)
        next_step = response.get("next_step")
        raw_response = response.get("raw_response")
        working_hist.append(f"core_agent> : {raw_response}")
        structured_info = response.get("structured_info")
      elif structured_info.get("code"):
        output = run_code(code, structured_info.get("code_type"))
        working_hist.append(f"runtime> : {output}")
        context = build_context(system_prompt_temp,
                                skills_meta_data,
                                observe_output_temp,
                                next_step_temp,
                                working_hist,
                                mode="full")
        if context > context window:
          context = update_context_w_stm(working_hist, observe_output_temp)
        response = call_llm(context)
        next_step = response.get("next_step")
        raw_response = response.get("raw_response")
        working_hist.append(f"core_agent> : {raw_response}")
        structured_info = response.get("structured_info")
    elif next_step == "verify":
      context = build_context(system_prompt_temp,
                              skills_meta_data,
                              verify_temp,
                              next_step_temp,
                              working_hist,
                              mode="full")
      if context > context window:
          context = update_context_w_stm(working_hist, verify_temp)
      response = call_llm(context)
      next_step = response.get("next_step")
      raw_response = response.get("raw_response")
      working_hist.append(f"core_agent> : {raw_response}")
      structured_info = response.get("structured_info")
    elif next_step == "retrive_ltm":
      context = build_context(system_prompt_temp,
                              skills_meta_data,
                              ltm_retrieve_temp,
                              next_step_temp,
                              working_hist,
                              mode="full")
      if context > context window:
        context = update_context_w_stm(working_hist, ltm_retrieve_temp)
      response = call_llm(context)
      next_step = response.get("next_step")
      raw_response = response.get("raw_response")
      working_hist.append(f"core_agent> : {raw_response}")
      structured_info = response.get("structured_info")
      for ltm_meta in structured_info.get("ltms"):
        ltm = get_ltm(ltm_meta)
        working_hist.append(f"retrieved memory> {ltm}")
    elif next_step == "plan":
      context = build_context(system_prompt_temp,
                              skills_meta_data,
                              plan_temp,
                              next_step_temp,
                              working_hist,
                              mode="full")
      if context > context window:
        context = update_context_w_stm(working_hist, plan_temp)
      response = call_llm(context)
      next_step = response.get("next_step")
      raw_response = response.get("raw_response")
      working_hist.append(f"core_agent> : {raw_response}")
      structured_info = response.get("structured_info")
    elif next_step == "do_tasks":
      for task in structured_info.get("tasks"):
        ...
    elif next_step == "document":
      ...
    elif next_step == "create_skill":
      ...
    elif next_step == "create sub_agent":
      ...
    elif next_step == "assign task":
      ...
    elif next_step == "report":
      ...
  
  context = build_context(system_prompt_temp,
                          skills_meta_data,
                          loop_summary_temp,
                          working_hist,
                          mode="full")
  if context > context window:
    context = update_context_w_stm(working_hist, loop_summary_temp)
  response = call_llm(context)
  strem(response.get("raw_response"))
```



System Prompt = """
You are the Core Agent in a runtime-controlled agentic system.

ROLE
- You own user interaction, global planning, delegation decisions, verification quality, and final reporting.
- You are responsible for deciding when reusable skills/knowledge should be proposed.
- You must not bypass runtime policy gates, approval rules, or executor constraints.

RUNTIME AUTHORITY
- Runtime controls execution, policy gating, persistence, indexing, and recovery.
- Runtime appends a live capability section each turn.
- Treat runtime-injected capabilities as the authoritative can-do list for this turn.
- Treat runtime-injected skills metadata as authoritative procedures for this turn.
- Prefer skill-guided procedures before ad-hoc commands.
- If a needed capability or skill is missing, state the gap and propose creation via the proper loop step.
- Treat `input_context` as authoritative runtime-selected context.
- Runtime will provide either full chat history or compacted summary plus recent turns.
- Runtime may also provide long-term memory (LTM).

OPERATING LOOP (STRICT ORDER)
1) Context
2) Plan
3) Act
4) Verify
5) Iterate
6) Document
7) Promotion Check
8) Report

STEP RULES
- Context:
  - Understand the current input and memory state.
  - Select relevant memory documents from provided LTM index snapshot.
  - No external web research and no side-effect execution in this step.
- Plan:
  - Produce minimal, ordered, verifiable actions.
  - Planning is read-only.
  - Bind each action to applicable skills when possible.
- Act:
  - Execute only necessary actions, only through allowed executors.
  - Respect policy gates and approval outcomes.
  - Follow skill procedures referenced in the plan.
- Verify:
  - Validate outcomes against explicit checks with concrete evidence.
- Iterate:
  - Decide to continue, replan, ask user, or finish.
- Document:
  - Produce structured memory update proposals (STM + optional LTM candidates) with provenance.
- Promotion Check:
  - Decide whether repeated successful workflows should be proposed as permanent skills.
- Report:
  - Return concise final answer with evidence, risks, and open questions.

OUTPUT DISCIPLINE
- For each loop step, return only the schema required by runtime.
- Do not mix schemas across steps.
- Do not emit hidden chain-of-thought; provide only required structured outputs and concise rationale fields.

TRUTHFULNESS AND EVIDENCE
- Never fabricate computed, stochastic, or environment-derived results.
- When execution is required, use executors and cite resulting evidence.
- Separate verified facts from assumptions.

SAFETY
- Prefer minimal-risk actions first.
- Escalate uncertainty or ambiguity early.
- Request user clarification when acceptance criteria are unclear.

PROMOTION POLICY
- Promotion threshold is advisory, not hardcoded.
- Use repeated successful evidence as signal.
- Recommend promotion; runtime and policy determine final application.

=== RUNTIME_INJECTED_CAPABILITIES_START ===
{{CAPABILITY_SNAPSHOT}}
=== RUNTIME_INJECTED_CAPABILITIES_END ===
"""

Standard Working Loop (Core Agent)

1. Context
0). Runtime pre-check (token guard + compaction policy)
  a). Build candidate conversational input:
    - Preferred: [all previous input/output pairs] + [current input].
  b). Estimate prompt tokens for:
    - System Prompt + Context Step Prompt + candidate input + ltm_index_snapshot.
  c). If within budget:
    - Set `input_mode = "full_history"`.
    - Use full history.
  d). If exceeds budget:
    - Run compaction:
      1) Summarize older history into STM update (`summary`, `open_loops`, `active_entities`).
      2) Keep only recent exact turns (small sliding window) + current input.
    - Set `input_mode = "stm_compacted"`.
  e). Persist compaction event:
    - mode, reason, dropped window size, retained window size.

1). Call LLM
- LLM input: `System Prompt + Context Step Prompt`.
Context Step Prompt:
"""
[STEP: CONTEXT]

Goal:
Select the most relevant long-term memory documents for this turn.
This step is retrieval-only.

Hard rules:
1) Do NOT execute tools.
2) Do NOT perform external web research.
3) Select only from `ltm_index_snapshot`.
4) Return JSON only, matching the required schema.

Selection guidance:
- Prefer high relevance to current input.
- Prefer higher confidence/quality.
- Prefer diversity (avoid near-duplicate docs).
- Select a minimal sufficient set.
- If nothing is relevant, return `{"selected_doc_ids": [], "selected_reasons": {}}`.

Additional schema rules:
- selected_doc_ids length must be <= max_doc_ids.
- selected_reasons keys must exactly match selected_doc_ids.
- selected_doc_ids must be unique (no duplicates).
- selected_doc_ids must be ordered by relevance (highest first).

Input:
{
  "input_context": {
    "mode": "full_history | stm_compacted",
    "current_input": "...",
    "compression_reason": "normal | token_limit_guard",
    "full_history": [...],              // present only when mode=full_history
    "stm": "...",                       // present only when mode=stm_compacted
    "recent_exact_turns": [...]         // present only when mode=stm_compacted
  },
  "ltm_index_snapshot": [
    {
      "doc_id": "doc_123",
      "title": "...",
      "summary_short": "...",
      "tags": ["..."],
      "quality_score": 0.82,
      "confidence": 0.79
    }
  ],
  "selection_limits": {"max_doc_ids": 10}
}

Required output schema:
{
  "selected_doc_ids": ["doc_123", "doc_456"],
  "selected_reasons": {
    "doc_123": "...",
    "doc_456": "..."
  }
}
"""

2). Runtime-controlled actions
  a). Validate schema and scope of selected docs.
  b). Read selected LTM docs/chunks by `doc_id`.
  c). Build `LTM_pack`:
{
  "ltm": [
    {
      "type": "decision|howto|failure|fact|policy",
      "title": "Meaningful doc title",
      "text": "Distilled knowledge text..."
    }
  ]
}
  d). Persist retrieval trace event.
  e). Do not trigger external web research in this step.

2. Plan
1). Call LLM
- LLM input: `System Prompt + Plan Step Prompt`.
Plan Step Prompt:
"""
[STEP: PLAN]

Goal:
Produce the minimal executable plan for the current input.

Hard rules:
1) Planning is read-only: do NOT execute tools.
2) Use only executors in `constraints.allowed_executors`.
3) If no execution is needed, return `actions: []`.
4) Include explicit, objective verification checks.
5) Return JSON only, matching the required schema.
6) Each action must include `skills_to_apply`.
7) If no skill applies, use `skills_to_apply: []` and explain in `skill_reason`.
8) If a required skill is missing, note it in `missing_skills`.

Action design policy:
- Actions may include information-gathering to resolve unknowns.
- With allowed executors limited to `bash` and `pythonexec`, represent web research as executable bash/python actions.
- Do not execute here; only propose action args.

Input:
{
  "input_context": {
    "mode": "full_history | stm_compacted",
    "current_input": "...",
    "compression_reason": "normal | token_limit_guard",
    "full_history": [...],              // present only when mode=full_history
    "stm": "...",                       // present only when mode=stm_compacted
    "recent_exact_turns": [...]         // present only when mode=stm_compacted
  },
  "LTM_pack": {
    "ltm": [
      {
        "type": "decision|howto|failure|fact|policy",
        "title": "...",
        "text": "..."
      }
    ]
  },
  "constraints": {
    "allowed_executors": ["Bash", "PythonExec"],
    "planning_is_read_only": true,
    "allow_delegate": false
  }
}

Required output schema:
{
  "objective": "...",
  "assumptions": ["..."],
  "steps": ["..."],
  "actions": [
    {
      "id": "a1",
      "type": "bash|pythonexec",
      "purpose": "...",
      "skills_to_apply": ["web-research", "execution-protocol"],
      "skill_reason": "...",
      "params": {
        "query": "...",
        "target_path": "...",
        "timeout_sec": 30
      },
      "risk": "low|medium|high"
    }
  ],
  "verification_checks": [
    {
      "id": "v1",
      "check": "...",
      "type": "exact_match|contains|artifact_exists|numeric_range|manual_review"
    }
  ],
  "risk": "low|medium|high",
  "needs_user_confirmation": false,
  "missing_skills": []
}
"""

2). Runtime-controlled actions
  a). Validate and normalize plan schema.
  b). Enforce allowed action types and parameter schema (intent-level `params`, not executable args).
  c). Enforce read-only planning phase (no execution here).
  d). Validate `skills_to_apply` against runtime-injected skill metadata.
  e). For each action, load referenced `SKILL.md`/scripts and resolve an executable action draft (`command` or `code`) from `params`.
  f). If resolution fails, mark skill/param gaps and route to replan/repair.
  g). Reject or repair invalid plan shape.
  h). Persist planning trace event (including resolved-action draft metadata, no execution yet).

3. Act
1). Runtime-controlled actions
  a). Creates action queue from approved plan.
  b). For each action:
    - resolves full skill bundle for that action
    - builds Act Step Prompt with action + resolved skill details
Act Step Prompt:
"""
[STEP: ACT]

Goal:
Refine the current action for safe, correct execution using the resolved skill bundle.
This step may refine execution arguments but must not execute directly.

Hard rules:
1) Do NOT execute tools directly in this response.
2) Keep executor type unchanged (`bash` or `pythonexec`).
3) Respect runtime constraints and policy requirements.
4) Keep `skills_to_apply` unchanged unless runtime explicitly allows skill substitution.
5) Return JSON only, matching required schema.
6) If current action is already appropriate, set `adjust_action=false`.

Refinement policy:
- Prefer minimal edits.
- Improve correctness, safety, and determinism.
- Preserve intent, objective, and verification compatibility.
- If required inputs are missing, request stop/replan via `blockers`.

Input:
{
  "objective": "...",
  "plan_id": "plan_001",
  "action": {
    "id": "a1",
    "type": "bash|pythonexec",
    "purpose": "...",
    "skills_to_apply": ["web-research", "execution-protocol"],
    "params": {
      "query": "...",
      "target_path": "...",
      "timeout_sec": 30
    },
    "executor_draft": {
      "command": "...",
      "code": "..."
    },
    "risk": "low|medium|high"
  },
  "resolved_skill_bundle": {
    "skills": [
      {
        "skill_id": "web-research",
        "procedure": ["...", "..."],
        "script_entrypoints": [{"name": "search", "path": "..."}],
        "safety_notes": ["..."]
      },
      {
        "skill_id": "execution-protocol",
        "procedure": ["...", "..."],
        "script_entrypoints": [],
        "safety_notes": ["..."]
      }
    ]
  },
  "constraints": {
    "allowed_executors": ["Bash", "PythonExec"],
    "allow_delegate": false,
    "side_effect_policy": "ask|allow|deny"
  },
  "prior_step_evidence": [
    {"event_id": "evt_123", "summary": "..."}
  ]
}

Required output schema:
{
  "adjust_action": false,
  "replacement_action": null,
  "reason": "...",
  "safety_notes": ["..."],
  "blockers": []
}

If adjust_action=true:
{
  "adjust_action": true,
  "replacement_action": {
    "id": "a1",
    "type": "bash|pythonexec",
    "purpose": "...",
    "skills_to_apply": ["web-research", "execution-protocol"],
    "params": {
      "query": "...",
      "target_path": "...",
      "timeout_sec": 30
    },
    "executor_draft": {
      "command": "...",
      "code": "..."
    },
    "risk": "low|medium|high"
  },
  "reason": "...",
  "safety_notes": ["..."],
  "blockers": []
}
"""
    - call LLM for optional refinement
    - runtime validates refinement (cannot break constraints)
    - runtime policy-gates
    - runtime executes
    - runtime records evidence/audit
  c). Continue until queue done or stop condition.

Verify
LLM input: verification_checks + execution_evidence
LLM output: VerifyReport

{
  "checks": [{"name": "...", "passed": true, "evidence_refs": ["evt_..."]}],
  "overall_passed": true,
  "gaps": []
}
Runtime-controlled actions:

Run deterministic sanity checks (exit codes, missing outputs).

Reconcile LLM verification with deterministic checks.

Persist authoritative pass/fail state.

Iterate
LLM input: VerifyReport + remaining objective + limits
LLM output: IterateDecision

{
  "decision": "continue|replan|ask_user|done",
  "reason": "...",
  "next_focus": "..."
}
Runtime-controlled actions:

Enforce loop/time/cost limits.

Trigger replan path if needed.

Create checkpoint before risky transitions.

Document
LLM input: turn evidence + current STM + LTM index summary
LLM output: MemoryPatch

{
  "stm_update": {
    "summary_delta": "...",
    "open_loops_add": ["..."],
    "open_loops_resolve": ["..."],
    "next_actions": ["..."]
  },
  "ltm_candidates": [
    {
      "title": "...",
      "summary_short": "...",
      "tags": ["..."],
      "keywords": ["..."],
      "confidence": 0.82,
      "quality_score": 0.78,
      "source_event_ids": ["evt_..."]
    }
  ]
}
Runtime-controlled actions:

Validate schema and normalize metadata.

Commit STM update (default path).

Gate LTM promotion by policy/rules.

Dedupe/chunk/hash LTM docs.

Update index incrementally.

Promotion Check
LLM input: repeated-pattern history + success evidence
LLM output: PromotionProposal

{
  "propose_skill": true,
  "scope": "core-agent|all-agents",
  "name": "...",
  "justification": "...",
  "source_runs": ["session_x/..."]
}
Runtime-controlled actions:

Enforce approval path.

Apply or reject promotion.

Refresh skill registry if applied.

Report
LLM input: final verified state + citations
LLM output: user-facing FinalReport

{
  "answer": "...",
  "evidence": [{"source": "evt_...", "note": "..."}],
  "risks": ["..."],
  "next_options": ["..."]
}
Runtime-controlled actions:

Persist final assistant output.
Persist structured run summary.
Emit UI-safe final response.
If you want, I can now insert this exact contract into system_design.md as a dedicated “Core Agent I/O Contract” section.

```
# Reference runtime logic: LLM-directed step routing + runtime safety kernel
# (pseudocode, intentionally framework-agnostic)

from collections import deque
from copy import deepcopy
from datetime import datetime

# ----------------------------
# Constants / policies
# ----------------------------

CORE_NEXT_STEPS = {
    "context",
    "retrieve_ltm",
    "plan",
    "do_tasks",
    "act",
    "verify",
    "iterate",
    "create_sub_agent",
    "assign_task",
    "document",
    "create_skill",
    "promotion_check",
    "report",
    None,
}
SUB_NEXT_STEPS = {
    "context",
    "retrieve_ltm",
    "plan",
    "do_tasks",
    "act",
    "verify",
    "iterate",
    "document",
    "create_skill",
    "promotion_check",
    "report",
    None,
}
TERMINAL_TOKENS = {None, "none", "no_next_step", "null"}

STEP_PROMPTS = {
    "context": CONTEXT_STEP_PROMPT,
    "retrieve_ltm": RETRIEVE_LTM_STEP_PROMPT,
    "plan": PLAN_STEP_PROMPT,
    "do_tasks": DO_TASKS_STEP_PROMPT,
    "act": ACT_STEP_PROMPT,
    "verify": VERIFY_STEP_PROMPT,
    "iterate": ITERATE_STEP_PROMPT,
    "create_sub_agent": CREATE_SUB_AGENT_STEP_PROMPT,
    "assign_task": ASSIGN_TASK_STEP_PROMPT,
    "document": DOCUMENT_STEP_PROMPT,
    "create_skill": CREATE_SKILL_STEP_PROMPT,
    "promotion_check": PROMOTION_CHECK_STEP_PROMPT,
    "report": REPORT_STEP_PROMPT,
    "invalid_step_repair": INVALID_STEP_REPAIR_PROMPT,
    "stm_compaction": STM_COMPACTION_PROMPT,
}

DEFAULT_LIMITS = {
    "max_inner_turns": 60,
    "max_exec_actions": 40,
    "max_subagent_depth": 2,
    "token_budget": 12000,
}

# ----------------------------
# Data model
# ----------------------------

def new_event(kind, payload):
    return {
        "event_id": f"evt_{datetime.utcnow().timestamp()}",
        "ts": datetime.utcnow().isoformat(),
        "kind": kind,
        "payload": payload,
    }

def init_agent_state(agent_kind, role, objective, working_hist_seed=None):
    return {
        "agent_kind": agent_kind,            # core|sub
        "role": role,                        # e.g. core-orchestrator / researcher
        "objective": objective,
        "working_hist": list(working_hist_seed or []),  # LLM-facing compact context
        "runtime_events": [],                # authoritative audit log
        "memory_pack": {"stm": None, "ltm": []},
        "ltm_cache": {},
        "plan": None,
        "task_queue": deque(),               # tasks from plan
        "active_task": None,
        "active_action": None,
        "execution_evidence": [],
        "subagents": {},                     # id -> profile
        "final_report": None,
        "terminated": False,
    }

# ----------------------------
# Runtime utility functions
# ----------------------------

def allowed_steps(agent_kind):
    return CORE_NEXT_STEPS if agent_kind == "core" else SUB_NEXT_STEPS

def normalize_next_step(token):
    if token in TERMINAL_TOKENS:
        return None
    return token

def append_hist(state, role, text):
    state["working_hist"].append(f"{role}> : {text}")

def append_event(state, kind, payload):
    state["runtime_events"].append(new_event(kind, payload))

def load_capability_snapshot(agent_kind):
    # Includes: skills metadata, executor constraints, policy mode, approvals profile, etc.
    return {
        "skills_meta": load_skills_meta(scope="core+all" if agent_kind == "core" else "all"),
        "executors": ["Bash", "PythonExec"],
        "policy_mode": get_policy_mode(),
        "approval_profile": get_approval_profile(),
    }

def build_envelope_for_step(step, state, caps):
    base = {
        "agent_kind": state["agent_kind"],
        "role": state["role"],
        "objective": state["objective"],
        "input_context": build_input_context(state["working_hist"]),
        "memory_pack": state["memory_pack"],
        "capability_snapshot": caps,
        "available_next_steps": sorted([x for x in allowed_steps(state["agent_kind"]) if x is not None]) + [None],
        "constraints": {
            "allowed_executors": caps["executors"],
            "planning_is_read_only": step == "plan",
            "allow_delegate": state["agent_kind"] == "core",
        },
    }
    # step-specific additions
    if step == "do_tasks":
        base["task_queue_preview"] = list(state["task_queue"])
        base["active_task"] = state["active_task"]
    if step == "act":
        base["active_task"] = state["active_task"]
        base["active_action"] = state["active_action"]
        base["recent_observation"] = state["execution_evidence"][-1] if state["execution_evidence"] else None
    if step == "verify":
        base["execution_evidence"] = state["execution_evidence"][-10:]
        base["plan"] = state["plan"]
    return base

def estimate_prompt_tokens(prompt_text):
    # replace with model-specific tokenizer
    return len(prompt_text) // 4

def compact_history_if_needed(state, caps, step_prompt, token_budget):
    prompt_preview = build_prompt(SYSTEM_PROMPT, step_prompt, build_envelope_for_step("context", state, caps))
    if estimate_prompt_tokens(prompt_preview) <= token_budget:
        return

    # isolated STM utility call (runtime-triggered)
    comp_prompt = build_prompt(
        SYSTEM_PROMPT,
        STEP_PROMPTS["stm_compaction"],
        {"working_hist": state["working_hist"], "current_stm": state["memory_pack"]["stm"]},
    )
    comp_out = call_llm(comp_prompt)  # schema: {stm:{summary,open_loops,active_entities,resolved_loops}}
    stm = comp_out.get("stm", {})
    state["memory_pack"]["stm"] = stm

    # Keep compact STM line + recent turns
    recent = state["working_hist"][-10:]
    state["working_hist"] = [f"stm> : {stm}"] + recent

    append_event(
        state,
        "stm_compaction",
        {
            "reason": "token_limit_guard",
            "retained_turns": len(recent),
            "stm_keys": list(stm.keys()) if isinstance(stm, dict) else [],
        },
    )

def validate_llm_step_output(step, out):
    # minimum shared contract
    assert isinstance(out, dict), "step output must be object"
    assert "next_step" in out, "missing next_step"
    assert "raw_response" in out, "missing raw_response"
    assert "structured_info" in out, "missing structured_info"
    # per-step schema validation should be strict in real code
    return True

def validate_next_step_or_repair(state, proposed_step):
    nxt = normalize_next_step(proposed_step)
    if nxt in allowed_steps(state["agent_kind"]):
        return nxt

    append_event(state, "invalid_next_step", {"proposed": proposed_step})
    # repair call
    caps = load_capability_snapshot(state["agent_kind"])
    env = build_envelope_for_step("context", state, caps)
    repair_prompt = build_prompt(SYSTEM_PROMPT, STEP_PROMPTS["invalid_step_repair"], env)
    repair_out = call_llm(repair_prompt)
    validate_llm_step_output("invalid_step_repair", repair_out)
    repaired = normalize_next_step(repair_out.get("next_step"))
    if repaired in allowed_steps(state["agent_kind"]):
        append_hist(state, "core_agent", repair_out.get("raw_response", ""))
        return repaired

    # hard fallback
    return "report"

# ----------------------------
# Step handlers
# ----------------------------

def handle_context(state, structured):
    selected = structured.get("selected_doc_ids", [])
    reasons = structured.get("selected_reasons", {})
    # runtime validation
    assert len(selected) == len(set(selected)), "duplicate doc ids"
    assert set(selected) == set(reasons.keys()), "reasons keys mismatch"

    docs = []
    for doc_id in selected:
        doc = read_ltm_doc_by_id(doc_id)   # runtime storage read
        if doc:
            docs.append(doc)

    state["memory_pack"]["ltm"] = docs
    append_event(state, "context_selected", {"doc_ids": selected})

def handle_retrieve_ltm(state, structured):
    ltms = []
    for meta in structured.get("ltms", []):
        doc = get_ltm(meta)
        if doc:
            ltms.append(doc)
            append_hist(state, "retrieved_memory", str(doc))
    state["memory_pack"]["ltm"].extend(ltms)
    append_event(state, "retrieve_ltm", {"count": len(ltms)})

def handle_plan(state, structured, caps):
    # PlanSpec expected to include tasks with route=act|assign_task
    plan = validate_plan_schema(structured, caps)  # strict validation
    state["plan"] = plan
    state["task_queue"] = deque(plan.get("tasks", []))
    append_event(state, "plan_approved", {"task_count": len(state["task_queue"])})

def handle_do_tasks(state, structured):
    # LLM may choose next task and route
    if not state["task_queue"]:
        return {"force_next_step": "verify"}

    requested_task_id = structured.get("task_id")
    requested_route = structured.get("route")  # act|assign_task|done

    if requested_route == "done":
        return {"force_next_step": "verify"}

    # pick task
    task = None
    if requested_task_id:
        for t in list(state["task_queue"]):
            if t.get("task_id") == requested_task_id:
                task = t
                break
    if task is None:
        task = state["task_queue"][0]

    state["active_task"] = task
    state["active_action"] = None

    if requested_route == "assign_task" and state["agent_kind"] == "core":
        return {"force_next_step": "assign_task"}
    return {"force_next_step": "act"}

def resolve_action_from_skills(task, caps):
    # deterministic resolver from task intent + skill metadata
    # returns executable draft for Bash/PythonExec
    return compile_executable_draft(task, caps["skills_meta"])

def handle_act(state, structured, caps):
    task = state["active_task"]
    if not task:
        return {"status": "error", "force_next_step": "do_tasks"}

    # Optionally allow LLM-provided refinement params (validated)
    refined_params = structured.get("refine_params")
    if refined_params:
        task = apply_refinement_params(task, refined_params)

    executable = resolve_action_from_skills(task, caps)
    gate = policy_gate(executable)
    if not gate.get("allowed"):
        append_event(state, "policy_block", {"task_id": task.get("task_id"), "reason": gate.get("reason")})
        obs = {"status": "blocked", "reason": gate.get("reason")}
    else:
        result = execute(executable)  # ONLY Bash/PythonExec
        obs = compact_observation(result)
        state["execution_evidence"].append(result)
        append_event(state, "tool_result", result)

    append_hist(state, "runtime", str(obs))
    return {"status": obs.get("status", "ok"), "force_next_step": "verify"}

def handle_verify(state, structured):
    rep = validate_verify_schema(structured)
    append_event(state, "verify", rep)
    if rep.get("overall_passed", False):
        # task-level completion
        if state["active_task"] is not None:
            try:
                state["task_queue"].remove(state["active_task"])
            except ValueError:
                pass
            state["active_task"] = None
        if state["task_queue"]:
            return {"force_next_step": "do_tasks"}
        return {"force_next_step": "iterate"}
    return {"force_next_step": "iterate"}

def handle_iterate(state, structured):
    decision = structured.get("decision", "continue")
    append_event(state, "iterate", {"decision": decision})
    if decision in {"done", "finish"}:
        return {"force_next_step": "document"}
    if decision in {"replan"}:
        return {"force_next_step": "plan"}
    if decision in {"ask_user"}:
        return {"force_next_step": "report"}
    return {"force_next_step": "do_tasks"}

def handle_create_sub_agent(state, structured):
    if state["agent_kind"] != "core":
        return {"force_next_step": "iterate"}

    spec = validate_subagent_spec(structured)
    sub_id = create_subagent(spec)  # runtime registry
    state["subagents"][sub_id] = spec
    append_event(state, "subagent_created", {"subagent_id": sub_id, "role": spec.get("role")})
    return {"force_next_step": "assign_task"}

def handle_assign_task(state, structured):
    if state["agent_kind"] != "core":
        return {"force_next_step": "iterate"}

    assignment = validate_assignment(structured, state["subagents"], state["active_task"])
    sub_ctx = init_agent_state(
        agent_kind="sub",
        role=assignment["role"],
        objective=assignment["objective"],
        working_hist_seed=[
            f"core_assignment> : {assignment}",
            f"task_context> : {state['active_task']}",
        ],
    )
    # core-provided isolated system prompt should be injected in call path
    sub_result = run_agent_loop(sub_ctx, depth=assignment.get("depth", 1))
    append_event(state, "subagent_result", {"summary": sub_result.get("final_report")})
    append_hist(state, "runtime", f"subagent_result> : {sub_result.get('final_report')}")
    return {"force_next_step": "verify"}

def handle_document(state, structured):
    patch = validate_memory_patch(structured)
    apply_memory_patch(patch)  # runtime commit/gating
    append_event(state, "document_patch_applied", {"ok": True})
    return {"force_next_step": "promotion_check"}

def handle_create_skill(state, structured):
    proposal = validate_skill_proposal(structured)
    apply_or_queue_skill_proposal(proposal)  # policy-gated
    append_event(state, "skill_proposal", proposal)
    return {"force_next_step": "iterate"}

def handle_promotion_check(state, structured):
    proposal = validate_promotion_proposal(structured)
    apply_promotion_if_approved(proposal)
    append_event(state, "promotion_check", proposal)
    return {"force_next_step": "report"}

def handle_report(state, raw_response, structured):
    state["final_report"] = raw_response
    append_event(state, "assistant_report", {"text": raw_response, "structured": structured})
    state["terminated"] = True
    return {"force_next_step": None}

# ----------------------------
# Core loop engine
# ----------------------------

def run_agent_loop(state, depth=0):
    limits = deepcopy(DEFAULT_LIMITS)
    if depth > limits["max_subagent_depth"]:
        state["terminated"] = True
        state["final_report"] = "Subagent depth limit reached."
        return state

    current_step = "context"
    turns = 0
    actions_executed = 0

    while not state["terminated"] and turns < limits["max_inner_turns"]:
        turns += 1
        caps = load_capability_snapshot(state["agent_kind"])

        # pre-call compaction
        compact_history_if_needed(state, caps, STEP_PROMPTS[current_step], limits["token_budget"])

        envelope = build_envelope_for_step(current_step, state, caps)
        prompt = build_prompt(SYSTEM_PROMPT, STEP_PROMPTS[current_step], envelope)

        llm_out = call_llm(prompt)  # expected: {next_step, raw_response, structured_info}
        validate_llm_step_output(current_step, llm_out)

        append_hist(state, "core_agent" if state["agent_kind"] == "core" else "sub_agent", llm_out["raw_response"])
        append_event(state, "llm_step_output", {"step": current_step, "next_step": llm_out.get("next_step")})

        # execute current step
        structured = llm_out.get("structured_info", {})
        forced = None

        if current_step == "context":
            handle_context(state, structured)
        elif current_step == "retrieve_ltm":
            handle_retrieve_ltm(state, structured)
        elif current_step == "plan":
            handle_plan(state, structured, caps)
        elif current_step == "do_tasks":
            forced = handle_do_tasks(state, structured).get("force_next_step")
        elif current_step == "act":
            out = handle_act(state, structured, caps)
            forced = out.get("force_next_step")
            if out.get("status") == "ok":
                actions_executed += 1
        elif current_step == "verify":
            forced = handle_verify(state, structured).get("force_next_step")
        elif current_step == "iterate":
            forced = handle_iterate(state, structured).get("force_next_step")
        elif current_step == "create_sub_agent":
            forced = handle_create_sub_agent(state, structured).get("force_next_step")
        elif current_step == "assign_task":
            forced = handle_assign_task(state, structured).get("force_next_step")
        elif current_step == "document":
            forced = handle_document(state, structured).get("force_next_step")
        elif current_step == "create_skill":
            forced = handle_create_skill(state, structured).get("force_next_step")
        elif current_step == "promotion_check":
            forced = handle_promotion_check(state, structured).get("force_next_step")
        elif current_step == "report":
            forced = handle_report(state, llm_out["raw_response"], structured).get("force_next_step")

        if actions_executed >= limits["max_exec_actions"]:
            append_event(state, "limit_reached", {"kind": "max_exec_actions"})
            current_step = "report"
            continue

        # choose next step: runtime-forced override > llm proposal
        proposed_next = forced if forced is not None else llm_out.get("next_step")
        next_step = validate_next_step_or_repair(state, proposed_next)

        if next_step in TERMINAL_TOKENS:
            state["terminated"] = True
            if not state["final_report"]:
                state["final_report"] = llm_out["raw_response"]
            break

        current_step = next_step

    if not state["final_report"]:
        state["final_report"] = "Loop ended by runtime limits."
    return state

# ----------------------------
# Top-level chat session
# ----------------------------

def run_core_session():
    core = init_agent_state(
        agent_kind="core",
        role="core_orchestrator",
        objective="Handle user requests end-to-end safely.",
    )

    while True:
        user_text = get_input()
        if user_text.strip().lower() in {"/exit", "exit", "quit"}:
            break

        append_hist(core, "user", user_text)
        append_event(core, "user_input", {"text": user_text})

        # each user turn launches a fresh working loop from context
        core["terminated"] = False
        core["final_report"] = None
        result = run_agent_loop(core, depth=0)

        stream(result["final_report"])
        append_hist(core, "core_agent", result["final_report"])
```

Do you know SKILL which is introduced by Anthropic for agent design?

python -m agentic_system --workspace ./runtime_test --provider lmstudio --model-name "zai-org/glm-4.7-flash" --mode controlled