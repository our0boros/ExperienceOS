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
    ExperienceRecord,
    FailureType,
    Harness,
    HarnessStatus,
    ParamStep,
    Step,
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
        Examples: call_tool("lookup", key="customer")
                  call_tool("submit", payload="result string")
        Returns a string, or an auto-parsed dict/list if the result is JSON.
        For JSON results, you can access fields directly: result["key"]
  - params: a dict of task parameters (e.g. params["key"])
  - env.snapshot(): returns the current environment state

Rules:
1. Define ONLY the `run()` function (and any helpers it needs).
2. Use call_tool() for every external action, passing arguments as kwargs.
3. Return the final result string from run().
4. Keep the code minimal and robust.  Handle missing keys gracefully.
5. Do NOT use any external libraries.

Example harness for a lookup-and-submit task:

def run():
    key = params.get("key", "")
    value = call_tool("lookup", key=key)
    call_tool("submit", payload=value)
    return value

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

    # ==================================================================
    # induction triggers
    # ==================================================================
    def check_triggers(self, task_type: str) -> str | None:
        """Return the trigger type if induction should fire, else ``None``.

        Returns one of: ``"new_harness"``, ``"patch"``.
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
        return None

    # ==================================================================
    # Phase 1 — trajectory segmentation
    # ==================================================================
    def _segment(self, trajectories: list[Trajectory]) -> list[list[int]]:
        """Return step-index segments for the first trajectory (representative).

        For the minimal implementation we treat the full trajectory as one
        segment if there are few steps; otherwise we use the LLM to find
        boundaries.
        """
        if not trajectories:
            return []
        rep = trajectories[0]
        if len(rep.steps) <= 3:
            return [list(range(len(rep.steps)))]
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
            return [s.get("steps", []) for s in segs] or [list(range(len(rep.steps)))]
        except Exception as exc:
            log.warning("Segmentation failed (%s), using whole trajectory", exc)
            return [list(range(len(rep.steps)))]

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

        # detect concrete values (strings in quotes or numbers) → replace with {param}
        param_steps: list[ParamStep] = []
        for i, action in enumerate(actions):
            template = action
            params: list[str] = []
            # crude parametrisation: replace quoted strings and numbers
            import re
            for m in re.finditer(r"['\"]([^'\"]+)['\"]", template):
                pname = f"arg_{len(params)}"
                template = template.replace(m.group(0), f"{{{pname}}}", 1)
                params.append(pname)
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
        prompt = SYNTHESIS_PROMPT.format(
            task_type=record.task_type,
            preconditions=json.dumps(record.candidate_preconditions, ensure_ascii=False),
            steps_json=steps_json,
            invariants=record.invariants,
            terminal_verifier=record.terminal_verifier,
            example_traces=example_traces,
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
    # full induction pipeline
    # ==================================================================
    def induce(
        self,
        task_type: str,
        env: BaseEnvironment,
    ) -> Optional[Harness]:
        """Run the full six-phase induction for *task_type*.

        Returns the validated :class:`Harness` or ``None`` if rejected.
        """
        trajectories = self.repo.trajectories_for_type(task_type, success_only=True)
        if len(trajectories) < self.config.induction.min_support:
            log.info("Not enough support for %s (%d < %d)",
                     task_type, len(trajectories), self.config.induction.min_support)
            return None

        trigger = self.check_triggers(task_type)
        if not trigger:
            return None
        log.info("Induction triggered (%s) for task_type=%s", trigger, task_type)

        # --- Phase 1 ---
        self._segment(trajectories)

        # --- Phase 2 ---
        preconditions = self._intersect_preconditions(trajectories)

        # --- Phase 3 ---
        invariants = self._mine_invariants(trajectories)

        # --- Phase 4 ---
        param_steps = self._abstract_steps(trajectories)

        # terminal verifier: the goal of the CoT
        terminal = trajectories[0].structured_cot.goal or "task completed"

        # build experience record (Layer 1)
        record = ExperienceRecord(
            task_type=task_type,
            candidate_preconditions=preconditions,
            param_steps=param_steps,
            invariants=invariants,
            terminal_verifier=terminal,
            source_trajectory_ids=[t.id for t in trajectories],
            support_count=len(trajectories),
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
            name=task_type.replace(" ", "_"),
            version=version,
            parent_id=parent_id,
            task_type=task_type,
            capability=task_type,
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
            # deprecate parent if patching
            if parent_id:
                self.repo.deprecate(parent_id)
            self.repo.add_harness(harness)
            log.info("Harness %s APPROVED (replay rate=%.2f)", harness.full_name, rate)
            return harness
        elif result == ValidationResult.NEEDS_REVISION:
            log.warning("Harness %s needs revision (replay rate=%.2f)", harness.full_name, rate)
            # store as draft for inspection
            harness.status = HarnessStatus.DRAFT
            self.repo.add_harness(harness)
            return None
        else:
            log.warning("Harness for %s REJECTED (replay rate=%.2f)", task_type, rate)
            return None
