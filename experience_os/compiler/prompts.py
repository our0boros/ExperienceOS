"""LLM prompt 模板常量。

包含 Phase 1 轨迹分段、Phase 5 harness 合成、ArtifactJudge 评估
三个 prompt 模板。
"""

SEGMENT_PROMPT = """\
You are analysing agent execution trajectories. Segment the following trajectory
into semantic sub-tasks by identifying action boundaries. Return JSON:
{"segments": [{"steps": [0,1], "label": "lookup customer"}, ...]}

Trajectory (task: {task}):
{steps_json}
"""

SYNTHESIS_PROMPT = """\
You are a code-synthesis agent. Given the following structured experience,
write a single self-contained Python function called `run()` that performs
the task by calling `call_tool(name, **kwargs)`.

Available globals inside run():
  - call_tool(name, **kwargs): call a tool by name with keyword arguments.
        Use the EXACT tool name as shown in the trajectories below.
        Returns a dict/list (auto-parsed from JSON) when the tool returns
        structured data.  Some tools return a plain string on success
        (e.g. a user ID) — that is NOT an error.  Only strings starting
        with "Error" indicate failure.  Recommended pattern:
            r = call_tool("get_order_details", order_id=oid)
            if isinstance(r, str) and r.startswith("Error"):
                return r
            # r is now a dict or a legitimate string value
        Skip any canonical step whose action is "reasoning" (it is a
        thought, not a real tool call).
  - params: a dict of task-specific parameters extracted from the reference
        solution.  These are the CORRECT values for THIS task instance —
        use them directly instead of hardcoding values from the example
        trajectories.  For example:
            order_id = params["order_id"]
            item_ids = params["item_ids"]
        If a key might be absent, use params.get("key", None) and handle None.
  - env.snapshot(): returns the current environment state

Rules:
1. Define ONLY the `run()` function (and any helpers it needs).
2. Use call_tool() for every external action, with the EXACT tool name and
   parameter names shown in the canonical_steps.
3. **Never hardcode task-specific values** (user_id, order_id, item_ids,
   product_id, etc.) from the example trajectories.  Always read them from
   `params`.  The example traces are for understanding the FLOW, not for
   copying values.
4. If params does not contain a needed value, derive it from a tool call
   result (e.g. user_id from order_details["user_id"]) rather than
   hardcoding.
5. Return the final result string from run().
6. Keep the code minimal and robust.  Handle missing keys gracefully.
7. Do NOT use any external libraries.

Example harness for the action pattern below:

{example_harness}

Experience record:
  task_type: {task_type}
  preconditions: {preconditions}
  canonical_steps: {steps_json}
  invariants: {invariants}
  terminal_verifier: {terminal_verifier}

Task context:
  goal: {cot_goal}
  constraints: {cot_constraints}
  risk: {cot_risk}
  milestones: {cot_milestones}

Example trajectories (observation -> action -> result):
{example_traces}
{repair_section}
Write the Python code now (output only the code, no markdown fences):
"""


SUBSTEP_SYNTHESIS_PROMPT = """\
You are an expert Python developer. Write a harness for a SINGLE tool call.

Task: {capability}
Tool: {tool_name}
Description: {description}

Your harness MUST:
1. Define ONLY a ``run()`` function with NO arguments
2. Make exactly ONE call to ``call_tool("{tool_name}", ...)``
3. Access parameters via the ``params`` dict (already available)
4. Return the tool result directly — do NOT add extra steps
5. Handle errors: if result is a string starting with "Error", return it

Preconditions:
{preconditions}

Invariants:
{invariants}

Parameters:
{params_list}

Example calls (observation → tool_name(arguments) → result):
{example_traces}

CRITICAL: This is a SINGLE-STEP harness. Do NOT add extra tool calls beyond ``{tool_name}``.
Write ONLY the Python code (no markdown fences):
"""

JUDGE_PROMPT = """\
You are evaluating whether a recurring sub-step pattern from agent execution traces
is worth compiling into a reusable artifact (executable harness, text skill, or verifier).

Sub-step pattern:
  intent: {intent}
  action: {action_name}  ({action_type})
  observed {support_count} times across different task instances
  success rate: {success_rate:.0%}
  example contexts (the conditions when this sub-step runs):
{example_contexts}
  example parameter variations:
{example_params}

Evaluate on four criteria:
1. **Generalisability** — can this sub-step be parameterised and applied to unseen instances?
   (yes = the action and its parameters follow a stable pattern; no = highly dependent on specific values)
2. **Stability** — does the sub-step have fixed preconditions and predictable outcomes?
   (yes = rarely fails when context matches; no = frequently fails or context is unpredictable)
3. **Value** — would compiling this save significant LLM reasoning cost?
   (high = repetitive, deterministic, frequent; low = one-off, requires judgment)
4. **Granularity** — is the sub-step at the right level for an artifact?
   (too fine = a single tool call hardly needs compilation;
    good = a sequence of 2-5 steps with clear boundaries;
    too coarse = spans multiple unrelated concerns)

Return JSON:
{{
  "verdict": "harness" | "skill" | "verifier" | "skip",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of the decision",
  "estimated_steps": <int>,  // how many steps the compiled artifact would contain
  "suggested_capability": "short capability label (e.g. user_lookup)"
}}
"""
