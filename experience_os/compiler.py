"""Harness Inductor — the six-phase compile + sandbox validation pipeline.

Implements §3.3 of the research proposal:

    Phase 1 — trajectory segmentation
    Phase 2 — precondition / postcondition extraction
    Phase 3 — invariant mining
    Phase 4 — step abstraction & parameterisation
    Phase 5 — harness synthesis (LLM generates executable code)
    Phase 6 — sandbox replay validation

The Bayesian *induction trigger* (§2.2, §3.5) decides *when* this pipeline runs:

    H* = argmax P(H | T_c) ∝ P(T_c | H) · P(H)

    P(H) ∝ exp(-λ · MDL(H))      # simplicity prior
    P(T_c | H) = replay success  # likelihood = coverage

Induction fires when ``support_count >= MIN_SUPPORT`` (new harness) or
``f2_failure_count >= 2`` (patch).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from experience_os.config import Config
from experience_os.environment import BaseEnvironment, TaskRequest
from experience_os.llm import LLMClient
from experience_os.models import (
    ArtifactType,
    ExperienceRecord,
    FailureType,
    Harness,
    HarnessStatus,
    ParamStep,
    Step,
    SubStepOutcome,
    SubStepPattern,
    SubStepPlan,
    Trajectory,
)
from experience_os.repository import Repository

log = logging.getLogger(__name__)


class ValidationResult(Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


@dataclass
class HarnessCandidate:
    """Output of the synthesis phase, pre-validation."""

    harness: Harness
    source_trajectories: list[Trajectory]


# ======================================================================
# Prompts
# ======================================================================
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
        Returns a string, or an auto-parsed dict/list if the result is JSON.
        For JSON results, you can access fields directly: result["key"]
  - params: a dict of task parameters (e.g. params["user_id"])
  - env.snapshot(): returns the current environment state

Rules:
1. Define ONLY the `run()` function (and any helpers it needs).
2. Use call_tool() for every external action, with the EXACT tool name and
   parameter names shown in the canonical_steps.
3. Return the final result string from run().
4. Keep the code minimal and robust.  Handle missing keys gracefully.
5. Do NOT use any external libraries.

Example harness for the action pattern below:

{example_harness}

Experience record:
  task_type: {task_type}
  preconditions: {preconditions}
  canonical_steps: {steps_json}
  invariants: {invariants}
  terminal_verifier: {terminal_verifier}

Example trajectories (observation -> action -> result):
{example_traces}

Write the Python code now (output only the code, no markdown fences):
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


class HarnessInductor:
    """Compiles trajectories into validated harnesses."""

    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        repo: Repository,
    ) -> None:
        self.config = config
        self.llm = llm
        self.repo = repo
        # Phase 1 segmentation results kept across induce() calls
        self._segments: list[list[int]] = []

    # ==================================================================
    # induction triggers (full-task level + sub-step pattern level)
    # ==================================================================
    def check_triggers(self, task_type: str) -> str | None:
        """Return the trigger type if induction should fire, else ``None``.

        Checks two levels:
        1. **Full-task level**: ``support_count >= MIN_SUPPORT`` → ``"new_harness"``
        2. **Sub-step pattern level**: any pattern reached MIN_SUPPORT → ``"substep_pattern"``

        Returns one of: ``"new_harness"``, ``"substep_pattern"``, ``"patch"``.
        """
        stats = self.repo.get_stats(task_type)
        # new harness: enough support, no active harness yet
        if stats.current_harness_id is None:
            if self.repo.support_count(task_type) >= self.config.induction.min_support:
                return "new_harness"
        # patch: consecutive F2 failures
        else:
            harness = self.repo.get_harness(stats.current_harness_id)
            if harness:
                f2 = harness.failure_counts.get(FailureType.F2_IMPLEMENTATION_ERROR.value, 0)
                if f2 >= self.config.induction.f2_patch_trigger:
                    return "patch"
        # sub-step pattern trigger: any pattern across ALL recent trajectories
        # that has reached min_support is a candidate (judged by ArtifactJudge
        # inside induce() to prevent false positives from noisy patterns)
        all_trajs = list(self.repo._trajectories.values())
        patterns = self._discover_substep_patterns(all_trajs)
        for key, p in patterns.items():
            if p.support_count >= self.config.induction.min_support:
                return "substep_pattern"
        return None

    # ==================================================================
    # Phase 0 — sub-step pattern discovery (cross-trajectory)
    # ==================================================================
    def _discover_substep_patterns(self, trajectories: list[Trajectory]) -> dict[str, SubStepPattern]:
        """Group steps by (action_name, intent) across all trajectories.

        Each **step** is treated as a sub-step observation.  Patterns are keyed by
        ``action_name:intent``.  A pattern's ``support_count`` increments when at least
        one trajectory contained that step; ``success_count`` increments when the step's
        result was an error-free tool response (not when the full task succeeded).

        Patterns with ``support_count >= MIN_SUPPORT`` are candidates for
        :meth:`_judge_artifact_value`.
        """
        patterns: dict[str, SubStepPattern] = {}

        for traj in trajectories:
            seen_intents = set()
            for step in traj.steps:
                action = step.action.split("(")[0].strip()
                intent = step.sub_step_intent or action
                key = f"{action}:{intent}"
                if key in seen_intents:
                    continue  # count once per trajectory
                seen_intents.add(key)

                if key not in patterns:
                    patterns[key] = SubStepPattern(
                        intent=intent,
                        action_name=action,
                        action_type=step.action_type,
                    )
                p = patterns[key]
                p.support_count += 1
                # Sub-step success: tool call returned without error
                step_success = bool(step.result) and "Error" not in step.result and "error" not in step.result.lower()
                if step_success:
                    p.success_count += 1
                if len(p.example_contexts) < 5:
                    ctx = step.observation[:200] if step.observation else traj.task_description[:200]
                    p.example_contexts.append(ctx or f"step: {step.action[:80]}")
                if step.metadata.get("params") and len(p.example_params) < 3:
                    p.example_params.append(step.metadata["params"])

        return patterns

    # ==================================================================
    # ArtifactJudge — LLM evaluation of sub-step pattern value
    # ==================================================================
    def _judge_artifact_value(self, pattern: SubStepPattern) -> ArtifactType:
        """Ask the LLM whether *pattern* is worth compiling into an artifact.

        Updates ``pattern.artifact_value_score``, ``pattern.artifact_type``,
        and ``pattern.skip_reason`` in place.
        """
        ctx_lines = "\n".join(f"  - {c[:150]}" for c in pattern.example_contexts)
        param_lines = "\n".join(f"  - {p}" for p in pattern.example_params[:3]) or "  (none)"

        prompt = JUDGE_PROMPT.format(
            intent=pattern.intent,
            action_name=pattern.action_name,
            action_type=pattern.action_type,
            support_count=pattern.support_count,
            success_rate=pattern.success_rate,
            example_contexts=ctx_lines,
            example_params=param_lines,
        )
        try:
            data = self.llm.chat_json([
                {"role": "system", "content": "You evaluate sub-step patterns for artifact compilation value."},
                {"role": "user", "content": prompt},
            ])
            verdict = data.get("verdict", "skip")
            confidence = float(data.get("confidence", 0.0))
            pattern.artifact_value_score = confidence
            pattern.skip_reason = data.get("reasoning", "")

            if verdict in ("harness", "skill", "verifier"):
                pattern.artifact_type = verdict
                log.info("ArtifactJudge: %s → %s (confidence=%.2f) — %s",
                         pattern.intent, verdict, confidence, pattern.skip_reason[:80])
                return ArtifactType(verdict)
            else:
                log.info("ArtifactJudge: %s → SKIP (confidence=%.2f) — %s",
                         pattern.intent, confidence, pattern.skip_reason[:80])
                return ArtifactType.SKIP
        except Exception as exc:
            log.warning("ArtifactJudge failed for %s: %s", pattern.intent, exc)
            pattern.skip_reason = f"judge_error: {exc}"
            return ArtifactType.SKIP

    # ==================================================================
    # Phase 1 — trajectory segmentation
    # ==================================================================
    def _segment(self, trajectories: list[Trajectory]) -> list[list[int]]:
        """Return step-index segments for the first trajectory (representative).

        For the minimal implementation we treat the full trajectory as one
        segment if there are few steps; otherwise we use the LLM to find
        boundaries.

        Results are stored in ``self._segments`` for use downstream.
        """
        if not trajectories:
            self._segments = []
            return []
        rep = trajectories[0]
        if len(rep.steps) <= 3:
            self._segments = [list(range(len(rep.steps)))]
            return self._segments
        steps_json = json.dumps(
            [{"i": i, "action": s.action, "result": s.result[:80]} for i, s in enumerate(rep.steps)],
            ensure_ascii=False,
        )
        try:
            data = self.llm.chat_json([
                {"role": "system", "content": "You segment agent trajectories into semantic sub-tasks."},
                {"role": "user", "content": SEGMENT_PROMPT.format(task=rep.task_description, steps_json=steps_json)},
            ])
            segs = data.get("segments", [])
            self._segments = [s.get("steps", []) for s in segs] or [list(range(len(rep.steps)))]
            return self._segments
        except Exception as exc:
            log.warning("Segmentation failed (%s), using whole trajectory", exc)
            self._segments = [list(range(len(rep.steps)))]
            return self._segments

    # ==================================================================
    # Phase 2 — precondition extraction (intersection across trajectories)
    # ==================================================================
    @staticmethod
    def _intersect_preconditions(trajectories: list[Trajectory]) -> dict:
        if not trajectories:
            return {}
        env_dicts = [t.env_snapshot.attributes for t in trajectories]
        common: dict = {}
        for key in env_dicts[0]:
            # collect values, handling unhashable types (lists, dicts)
            vals = []
            for d in env_dicts:
                if key in d:
                    v = d[key]
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v, sort_keys=True)
                    if v not in vals:
                        vals.append(v)
            if len(vals) == 1:
                val = vals[0]
                # only keep scalar preconditions; skip complex/mutable ones
                if not isinstance(val, str) or val.startswith("[") or val.startswith("{"):
                    continue  # skip list/dict-derived values
                common[key] = val
            else:
                # record the set as a list (any-of constraint)
                common[key] = sorted(str(v) for v in vals)
        return common

    # ==================================================================
    # Phase 3 — invariant mining (predicates true across all trajectories)
    # ==================================================================
    @staticmethod
    def _mine_invariants(trajectories: list[Trajectory]) -> list[str]:
        invariants: list[str] = []
        # simple heuristic: if all trajectories share a common action prefix,
        # that prefix is an invariant.
        if not trajectories:
            return invariants
        first_actions = [t.steps[0].action for t in trajectories if t.steps]
        if len(set(first_actions)) == 1:
            invariants.append(f"first action is always: {first_actions[0]}")
        # all outcomes success
        if all(t.outcome == "success" for t in trajectories):
            invariants.append("outcome must be success")
        return invariants

    # ==================================================================
    # Phase 4 — step abstraction & parameterisation
    # ==================================================================
    def _abstract_steps(self, trajectories: list[Trajectory]) -> list[ParamStep]:
        """Find the longest common action sequence and parameterise concrete values."""
        if not trajectories:
            return []
        # use the shortest trajectory as the template
        rep = min(trajectories, key=lambda t: len(t.steps))
        actions = [s.action for s in rep.steps]

        param_steps: list[ParamStep] = []
        for i, action in enumerate(actions):
            # Parse the action: tool_name({"key": "value", ...}) → tool_name(key={param}, ...)
            import re
            m = re.match(r'(\w+)\((.*)\)', action)
            if not m:
                param_steps.append(ParamStep(template=action, params=[], action_type=rep.steps[i].action_type))
                continue
            tool_name = m.group(1)
            args_str = m.group(2).strip()
            params: list[str] = []

            try:
                # Try to parse as JSON dict → extract parameter names
                args_dict = json.loads(args_str)
                if isinstance(args_dict, dict):
                    param_parts = []
                    for k, v in args_dict.items():
                        pname = str(k)
                        params.append(pname)
                        param_parts.append(f"{pname}={{{pname}}}")
                    template = f"{tool_name}({', '.join(param_parts)})"
                else:
                    # Non-dict JSON — treat as single value
                    pname = "value"
                    params.append(pname)
                    template = f"{tool_name}({{{pname}}})"
            except (json.JSONDecodeError, ValueError):
                # Fallback: extract key=value pairs from the string
                for pair in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                    pname = pair.group(1)
                    if pname not in params:
                        params.append(pname)
                if params:
                    param_parts = [f"{p}={{{p}}}" for p in params]
                    template = f"{tool_name}({', '.join(param_parts)})"
                else:
                    template = action  # keep original

            param_steps.append(ParamStep(template=template, params=params, action_type=rep.steps[i].action_type))
        return param_steps

    # ==================================================================
    # Phase 5 — harness synthesis (LLM generates code)
    # ==================================================================
    def _synthesize(self, record: ExperienceRecord, trajectories: list[Trajectory]) -> str:
        steps_json = json.dumps(
            [{"template": s.template, "params": s.params} for s in record.param_steps],
            ensure_ascii=False,
        )
        example_traces = "\n---\n".join(
            "\n".join(f"  {s.action} -> {s.result[:60]}" for s in t.steps[:6])
            for t in trajectories[:2]
        )

        # Build a dynamic example harness from the first trajectory's actual action names
        example_action = trajectories[0].steps[0].action.split("(")[0].strip() if trajectories and trajectories[0].steps else "tool_name"
        example_params_list = record.param_steps[0].params if record.param_steps else []
        example_params_call = ", ".join(f'{p}={p}' for p in example_params_list[:3])
        param_get = example_params_list[0] if example_params_list else "value"
        example_harness = f"""\
def run():
    # Example using the EXACT tool name from the trajectory
    {param_get} = params.get("{param_get}", "default_{param_get}")
    result = call_tool("{example_action}", {example_params_call})
    return result"""

        prompt = SYNTHESIS_PROMPT.format(
            task_type=record.task_type,
            preconditions=json.dumps(record.candidate_preconditions, ensure_ascii=False),
            steps_json=steps_json,
            invariants=record.invariants,
            terminal_verifier=record.terminal_verifier,
            example_traces=example_traces,
            example_harness=example_harness,
        )
        code = self.llm.chat(
            [{"role": "system", "content": "You are an expert Python code generator."},
             {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        log.debug("Raw LLM synthesis output:\n%s", code)
        # strip markdown fences if present
        if "```" in code:
            parts = code.split("```")
            # take the longest code block
            code = max(parts, key=len).strip()
            if code.startswith("python"):
                code = code[6:].strip()
        return code

    # ==================================================================
    # Phase 6 — sandbox replay validation
    # ==================================================================
    def _validate(
        self,
        candidate: Harness,
        source_trajectories: list[Trajectory],
        env: BaseEnvironment,
    ) -> tuple[ValidationResult, float]:
        """Replay the harness against each source task; require success_rate ≥ threshold."""
        successes = 0
        for traj in source_trajectories:
            # extract params from trajectory step metadata or structured CoT
            params = {}
            for s in traj.steps:
                if s.metadata.get("params"):
                    params.update(s.metadata["params"])
            request = TaskRequest(
                task_id=traj.task_id,
                task_description=traj.task_description,
                task_type=traj.task_type,
                params=params,
                expected_output=traj.structured_cot.goal or "",
            )
            result = env.execute_harness(candidate, request)
            if result.success:
                successes += 1
        rate = successes / len(source_trajectories) if source_trajectories else 0.0
        if rate >= self.config.induction.validation_threshold:
            return ValidationResult.APPROVED, rate
        if rate > 0.0:
            return ValidationResult.NEEDS_REVISION, rate
        return ValidationResult.REJECTED, rate

    # ==================================================================
    # full induction pipeline (dual-level)
    # ==================================================================
    def induce(
        self,
        task_type: str,
        env: BaseEnvironment,
        trigger: str | None = None,
        substep_pattern: SubStepPattern | None = None,
    ) -> Optional[Harness]:
        """Run the full six-phase induction.

        Supports two entry points:
        * **Full-task induction** (trigger = ``"new_harness"``): compiles all
          successful trajectories for *task_type*.
        * **Sub-step induction** (trigger = ``"substep_pattern"``): compiles
          only the steps matching the discovered *substep_pattern* across all
          trajectories, regardless of full-task success.

        Returns the validated :class:`Harness` or ``None`` if rejected.
        """
        all_trajs = list(self.repo._trajectories.values()) if substep_pattern else []

        if substep_pattern:
            # --- Sub-step level induction ---
            # Collect steps matching this pattern across ALL trajectories
            matching_trajs = []
            for traj in all_trajs:
                matched_steps = [
                    s for s in traj.steps
                    if s.sub_step_intent == substep_pattern.intent
                    or s.action.startswith(substep_pattern.action_name)
                ]
                if matched_steps:
                    # Build a synthetic trajectory containing only the matched steps
                    synth = Trajectory(
                        task_id=traj.task_id,
                        task_description=traj.task_description,
                        task_type=substep_pattern.intent,
                        steps=matched_steps,
                        structured_cot=traj.structured_cot,
                        env_snapshot=traj.env_snapshot,
                        outcome="success" if all(s.result and "Error" not in s.result for s in matched_steps) else "failure",
                        tokens_used=traj.tokens_used,
                        latency_seconds=traj.latency_seconds / max(1, len(traj.steps)) * len(matched_steps),
                    )
                    matching_trajs.append(synth)

            if not matching_trajs:
                log.warning("Sub-step induction: no matching trajectories for %s", substep_pattern.intent)
                return None

            trajectories = matching_trajs
            induced_type = substep_pattern.intent
            capability = substep_pattern.intent
            log.info("Sub-step induction triggered for '%s' (%d matched steps across %d trajs)",
                     induced_type, sum(len(t.steps) for t in trajectories), len(trajectories))
        else:
            # --- Full-task level induction (original logic) ---
            trajectories = self.repo.trajectories_for_type(task_type, success_only=True)
            if len(trajectories) < self.config.induction.min_support:
                log.info("Not enough support for %s (%d < %d)",
                         task_type, len(trajectories), self.config.induction.min_support)
                return None
            induced_type = task_type
            capability = task_type

        if not trigger:
            trigger = self.check_triggers(task_type)
        if not trigger:
            return None
        log.info("Induction triggered (%s) for type=%s", trigger, induced_type)

        # --- Phase 1 (segmentation) — store results in self._segments ---
        # For sub-step induction the steps are already pre-segmented
        if not substep_pattern:
            self._segment(trajectories)
        else:
            self._segments = [list(range(len(trajectories[0].steps)))] if trajectories else []

        # --- Phase 2 ---
        preconditions = self._intersect_preconditions(trajectories)

        # --- Phase 3 ---
        invariants = self._mine_invariants(trajectories)

        # --- Phase 4 ---
        param_steps = self._abstract_steps(trajectories)

        # terminal verifier
        terminal = trajectories[0].structured_cot.goal or f"{induced_type} completed"

        # build experience record (Layer 1)
        record = ExperienceRecord(
            task_type=induced_type,
            candidate_preconditions=preconditions,
            param_steps=param_steps,
            invariants=invariants,
            terminal_verifier=terminal,
            source_trajectory_ids=[t.id for t in trajectories],
            support_count=max(t.support_count for t in trajectories) if hasattr(trajectories[0], 'support_count') and substep_pattern else len(trajectories),
        )
        self.repo.add_record(record)

        # --- Phase 5: synthesis ---
        code = self._synthesize(record, trajectories)

        # determine parent (for patches)
        stats = self.repo.get_stats(task_type)
        parent_id = stats.current_harness_id if trigger == "patch" else None
        existing = self.repo.get_harness(parent_id) if parent_id else None
        version = (existing.version + 1) if existing else 1

        harness = Harness(
            name=induced_type.replace(" ", "_"),
            version=version,
            parent_id=parent_id,
            task_type=induced_type,
            capability=capability,
            description=trajectories[0].task_description,
            preconditions=preconditions,
            procedure_code=code,
            invariants=invariants,
            terminal_verifier=terminal,
            params=[p for ps in param_steps for p in ps.params],
            source_record_ids=[record.id],
        )

        # --- Phase 6: validation ---
        result, rate = self._validate(harness, trajectories, env)
        harness.verification = type(harness.verification)(
            success_rate=rate, test_count=len(trajectories)
        )

        if result == ValidationResult.APPROVED:
            if parent_id:
                self.repo.deprecate(parent_id)
            self.repo.add_harness(harness)
            log.info("Harness %s APPROVED (replay rate=%.2f)", harness.full_name, rate)
            return harness
        elif result == ValidationResult.NEEDS_REVISION:
            log.warning("Harness %s needs revision (replay rate=%.2f)", harness.full_name, rate)
            harness.status = HarnessStatus.DRAFT
            self.repo.add_harness(harness)
            return None
        else:
            log.warning("Harness for %s REJECTED (replay rate=%.2f)", induced_type, rate)
            return None
