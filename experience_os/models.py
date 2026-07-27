"""Core data models for ExperienceOS.

These map directly to the formalisation in the research proposal:

    * :class:`Step`         — a single (observation, action) pair in a trajectory.
    * :class:`Trajectory`  — Layer-0 raw experience log.
    * :class:`ExperienceRecord` — Layer-1 semantic summary induced from a cluster
      of trajectories.
    * :class:`Harness`     — Layer-2 compiled, parameterised executable scaffold.
    * :class:`TaskTypeStats` — accumulation statistics per task type.
    * :class:`EnvironmentSnapshot` — env state used for precondition matching.

The :class:`Harness` schema implements the extended Hoare Triple
``H = <P, steps, I, Q, R>`` from §2.1 of the proposal.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# =====================================================================
# Environment
# =====================================================================
@dataclass
class EnvironmentSnapshot:
    """A flat key-value view of the execution environment.

    Used for precondition matching.  Examples::

        {"os": "linux", "app": "gmail", "version": "3.2", "has_write_perm": True}
    """

    attributes: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def satisfies(self, key: str, expected: Any) -> bool:
        actual = self.attributes.get(key)
        if isinstance(expected, list) and not isinstance(expected, str):
            return actual in expected
        return actual == expected


# =====================================================================
# Trajectory (Layer 0)
# =====================================================================
@dataclass
class Step:
    """A single observation-action pair."""

    observation: str
    action: str  # serialised action description or tool call
    action_type: str = "generic"  # read / write / think / generic
    result: str = ""  # post-action observation or tool output
    metadata: dict[str, Any] = field(default_factory=dict)
    sub_step_intent: str = ""  # intent label from sub-step decomposition (§3.5 sub-step tracking)


@dataclass
class StructuredCoT:
    """Structured reasoning trace accompanying a trajectory.

    Acts as the *observation window* onto latent task variables (see Discuss §5).
    """

    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    risk: str = ""
    milestones: list[str] = field(default_factory=list)
    reflection: str = ""


@dataclass
class Trajectory:
    """Layer-0 raw experience: a complete agent execution trace."""

    task_id: str
    task_description: str
    task_type: str = ""  # semantic cluster label
    steps: list[Step] = field(default_factory=list)
    structured_cot: StructuredCoT = field(default_factory=StructuredCoT)
    env_snapshot: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    outcome: str = "success"  # "success" | "failure"
    tokens_used: int = 0
    latency_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: _uid("traj_"))
    sub_step_plan: list[SubStepPlan] = field(default_factory=list)  # Phase 0 decomposition
    phase: str = ""  # "warmup" | "eval" | "" — 实验阶段标记

    def fingerprint(self) -> str:
        """A stable hash for dedup / replay comparison."""
        raw = f"{self.task_type}|{self.task_description}|{len(self.steps)}"
        for s in self.steps:
            raw += f"|{s.action}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# =====================================================================
# Sub-step tracking (§3.5 sub-step pattern discovery)
# =====================================================================
@dataclass
class SubStepPlan:
    """A planned sub-step within a larger task.

    Produced by Phase 0 (sub-step decomposition) before execution.
    Each sub-step records its intent/goal and the expected action type,
    enabling cross-trajectory pattern matching independent of full-task success.
    """

    intent: str          # e.g. "find_user_id_by_email"
    goal: str            # what this sub-step aims to achieve
    context: str = ""    # preconditions / environment context at this step
    expected_action: str = ""  # the action expected (e.g. tool name)
    action_type: str = "generic"


@dataclass
class SubStepPattern:
    """A recurring sub-step pattern discovered across multiple trajectories.

    When the same intent (+ similar context) appears in >= MIN_SUPPORT
    trajectories, the pattern is a candidate for artifact induction.
    """

    intent: str                      # semantic label (e.g. "user_lookup_by_email")
    action_name: str                 # canonical action name (e.g. "find_user_id_by_email")
    action_type: str = "generic"     # read / write / think / generic
    description: str = ""            # human-readable description for retrieval
    support_count: int = 0           # number of distinct trajectories this pattern appeared in
    success_count: int = 0           # number of successful single-step executions
    # 贝叶斯权重字段（从 substeps 表聚合得出）
    success_in_full_tasks: int = 0   # 此子步骤在成功全任务中的出现次数
    total_appearances: int = 0       # 此子步骤在任何全任务中的出现次数
    bayesian_score: float = 0.0      # Beta-Binomial 可信度 (α=1,β=1)
    example_contexts: list[str] = field(default_factory=list)  # context samples for LLM judgment
    example_params: list[dict] = field(default_factory=list)   # parameter variations
    artifact_value_score: float = 0.0  # 0-1, set by ArtifactJudge
    artifact_type: str = ""            # "harness" | "skill" | "verifier" | "" (not yet judged)
    skip_reason: str = ""              # if judged not worth compiling, why
    id: str = field(default_factory=lambda: _uid("ssp_"))
    # 三要素检索签名
    input_schema: str = ""           # JSON: {"requires":[...],"from":"..."}
    output_schema: str = ""          # JSON: {"produces":[...],"type":"..."}
    effect: str = ""                 # "read_only" | "write" | "mixed"
    intent_embedding: Optional[list[float]] = None  # cached retrieval vector

    @property
    def success_rate(self) -> float:
        return self.success_count / self.support_count if self.support_count else 0.0


class ArtifactType(str, Enum):
    """The kind of artifact a sub-step pattern should be compiled into."""

    HARNESS = "harness"     # executable Python code (deterministic, bypasses LLM)
    SKILL = "skill"         # text skill document (guides LLM, doesn't bypass)
    VERIFIER = "verifier"   # post-condition checker (run after agent execution)
    SKIP = "skip"           # not worth compiling


@dataclass
class SubStepOutcome:
    """The result of executing one sub-step within a task.

    Accumulated per task execution and used to update SubStepPattern stats.
    """

    intent: str
    action_name: str
    action_type: str = "generic"
    context: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str = ""
    # 预测契约验证（Phase A，来自 flow.md 融合）
    prediction_verification: Optional[PredictionVerification] = None


# =====================================================================
# Prediction Contract (flow.md fusion — Phase A)
# =====================================================================
@dataclass
class PredictionContract:
    """Agent's structured prediction BEFORE executing a sub-step.

    Maps to the Hoare Triple: expected_input → P, expected_output/effect → Q.
    Derived from the agent's reasoning text immediately preceding a tool call.

    Design reference: docs/ExperienceOS.md §5.1.1, flow.md §1.
    """

    step_id: str = ""
    intent: str = ""                 # what sub-step this prediction is for
    expected_input: str = ""         # natural-language expected input / precondition
    expected_output: str = ""        # natural-language expected output / postcondition
    expected_effect: str = ""        # expected side-effect (e.g. "order status changed")
    confidence: float = 0.5          # agent's self-assessed confidence (0.0–1.0)
    agent_reasoning: str = ""        # raw reasoning snippet this contract was extracted from


@dataclass
class PredictionVerification:
    """Post-execution comparison: predicted (contract) vs actual (tool result).

    Produces a quality label that feeds into Bayesian experience gating.
    Design reference: docs/ExperienceOS.md §5.1.2.
    """

    contract: Optional[PredictionContract] = None
    actual_output: str = ""          # actual tool result (truncated)
    actual_effect: str = ""          # observed side-effect
    prediction_accurate: bool = False
    divergence_reason: str = ""      # why prediction and actual diverged (if they did)
    quality_label: str = ""          # high_quality | lucky_success | implementation_defect | negative_sample

    @classmethod
    def from_outcome(
        cls,
        contract: Optional[PredictionContract],
        outcome: SubStepOutcome,
        parent_task_success: bool = False,
    ) -> PredictionVerification:
        """Factory: compare prediction contract against actual step outcome.

        Quality labels (ExperienceOS.md §5.1.2):

        - prediction OK + step OK → ``"high_quality"`` (weight ×1.0)
        - prediction BAD + step OK → ``"lucky_success"`` (weight ×0.3)
        - prediction OK + step FAIL → ``"implementation_defect"`` (→ F2)
        - prediction BAD + step FAIL → ``"negative_sample"`` (boundary record)

        When the step fails, we distinguish infrastructure errors (timeout,
        connection, rate-limit — agent's plan was correct) from domain errors
        (not found, invalid input — agent's prediction was wrong about what
        the system would return).
        """
        prediction_accurate = False
        divergence_reason = ""

        # Infrastructure error markers — agent's prediction was likely correct
        # but the system failed to execute.
        _infra_errors = {
            "timeout", "connection", "rate limit", "unauthorized",
            "server error", "internal error", "unavailable",
            "try again", "capacity", "overloaded",
        }

        if contract is not None:
            # Check prediction accuracy regardless of step success/failure
            expected_keywords = _extract_keywords(contract.expected_output)
            actual_lower = (outcome.error or "").lower()
            result_text = str(outcome.params.get("_result_summary", actual_lower))

            if not outcome.success and any(
                ie in actual_lower for ie in _infra_errors
            ):
                # Infrastructure failure → agent's prediction was correct,
                # the system just couldn't execute.
                prediction_accurate = True
                divergence_reason = f"Infrastructure failure: {outcome.error}"
            elif expected_keywords:
                match_count = sum(
                    1 for kw in expected_keywords if kw.lower() in result_text
                )
                prediction_accurate = match_count >= len(expected_keywords) * 0.5
                if not prediction_accurate:
                    divergence_reason = (
                        f"Expected keywords {expected_keywords} not found in result"
                    )
            elif outcome.success:
                # No explicit expected output + step succeeded → assume accurate
                prediction_accurate = True
            else:
                # No expected output + step failed → can't verify, default accurate
                prediction_accurate = True

            if prediction_accurate and not outcome.success and not divergence_reason:
                divergence_reason = outcome.error or "step failed"

        # Quality label
        if prediction_accurate and outcome.success:
            quality_label = "high_quality"
        elif not prediction_accurate and outcome.success:
            quality_label = "lucky_success"
        elif prediction_accurate and not outcome.success:
            quality_label = "implementation_defect"
        else:
            quality_label = "negative_sample"

        return cls(
            contract=contract,
            actual_output=(outcome.error or "")[:500],
            actual_effect="",
            prediction_accurate=prediction_accurate,
            divergence_reason=divergence_reason,
            quality_label=quality_label,
        )


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a prediction text for heuristic matching."""
    if not text:
        return []
    # Split on common delimiters, filter noise words
    noise = {"the", "a", "an", "is", "are", "be", "to", "of", "in", "for",
             "on", "with", "and", "or", "will", "should", "would", "could",
             "this", "that", "it", "its", "i", "we", "you", "they"}
    words = text.replace(",", " ").replace(".", " ").replace(":", " ").split()
    return [w for w in words if len(w) > 2 and w.lower() not in noise][:6]


# =====================================================================
# Experience Record (Layer 1)
# =====================================================================
@dataclass
class ParamStep:
    """A parameterised step in a canonical action sequence."""

    template: str  # e.g. "call_api(endpoint={endpoint}, params={params})"
    params: list[str] = field(default_factory=list)
    action_type: str = "generic"


@dataclass
class ExperienceRecord:
    """Layer-1 semantic summary induced from a cluster of trajectories."""

    task_type: str
    candidate_preconditions: dict[str, Any] = field(default_factory=dict)
    param_steps: list[ParamStep] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    terminal_verifier: str = ""  # description of success end-state
    observed_variations: list[str] = field(default_factory=list)
    source_trajectory_ids: list[str] = field(default_factory=list)
    support_count: int = 0
    # 从源轨迹提取的示例任务描述，供 compiler.py 在归纳时填充到 Harness.example_tasks
    example_task_descriptions: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _uid("rec_"))


# =====================================================================
# Harness / Artifact (Layer 2)
# =====================================================================
class FailureType(str, Enum):
    """Harness execution failure classification (§3.4)."""

    F1_PRECONDITION_GAP = "F1"  # constraint gap
    F2_IMPLEMENTATION_ERROR = "F2"  # selector/timing bug
    F3_ENVIRONMENT_DRIFT = "F3"  # UI/API change
    F4_OUT_OF_SCOPE = "F4"  # task outside harness capability


class HarnessStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class VerificationMeta:
    """Sandbox replay validation metadata."""

    method: str = "sandbox_replay"
    success_rate: float = 0.0
    test_count: int = 0
    last_validated: float = field(default_factory=time.time)


@dataclass
class Harness:
    """A compiled executable scaffold (the ``Artifact``).

    Implements ``H = <P, steps, I, Q, R>``:
        P = preconditions, steps = procedure, I = invariants,
        Q = postconditions/terminal_verifier, R = rollback.
    """

    # identity
    id: str = field(default_factory=lambda: _uid("harn_"))
    name: str = ""
    version: int = 1
    parent_id: Optional[str] = None  # version DAG link
    status: HarnessStatus = HarnessStatus.ACTIVE

    # semantic
    task_type: str = ""
    description: str = ""  # natural-language description for retrieval
    capability: str = ""  # e.g. "document_validation"
    # 示例任务描述（用于检索增强，§5.5.1 检索向量维度）
    example_tasks: list[str] = field(default_factory=list)

    # version DAG 边类型（§P1-4）："patch" | "specialization" | "composition" | ""
    edge_type: str = ""
    split_reason: str = ""  # 特化分裂原因（如 "variation: 3_steps_alt"）
    merge_source_ids: list[str] = field(default_factory=list)  # composition 合并来源

    # Hoare triple components
    preconditions: dict[str, Any] = field(default_factory=dict)
    soft_preconditions: dict[str, Any] = field(default_factory=dict)
    procedure_code: str = ""  # the actual executable Python code
    invariants: list[str] = field(default_factory=list)
    terminal_verifier: str = ""
    rollback: str = ""

    # parameters
    params: list[str] = field(default_factory=list)

    # provenance & validation
    source_record_ids: list[str] = field(default_factory=list)
    verification: VerificationMeta = field(default_factory=VerificationMeta)
    failure_counts: dict[str, int] = field(default_factory=dict)
    embedding: Optional[list[float]] = None  # cached retrieval vector

    # timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    @property
    def full_name(self) -> str:
        return f"{self.name}-v{self.version}" if self.name else self.id

    def retrieval_text(self) -> str:
        """The text used to compute the harness embedding vector."""
        parts = [
            f"task_type: {self.task_type}",
            f"capability: {self.capability}",
            f"description: {self.description}",
            f"preconditions: {self.preconditions}",
        ]
        # 示例任务描述作为检索向量维度（§5.5.1），取前 3 条避免噪声
        if self.example_tasks:
            parts.append(f"example_tasks: {'; '.join(self.example_tasks[:3])}")
        return "\n".join(parts)

    def mdl(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3) -> float:
        """Minimum Description Length prior score (§2.2).

        Lower = simpler.  Used in the Bayesian induction criterion.
        """
        n_steps = self.procedure_code.count("\n") + 1 if self.procedure_code else 0
        n_params = len(self.params)
        n_invariants = len(self.invariants)
        return alpha * n_steps + beta * n_params + gamma * n_invariants


# =====================================================================
# Task-type statistics (Layer 3 / Meta-experience)
# =====================================================================
@dataclass
class TaskTypeStats:
    """Accumulation statistics for one task type (§4 of Discuss)."""

    task_type: str = ""

    # counts
    total_executions: int = 0
    harness_executions: int = 0
    agent_executions: int = 0

    # quality
    harness_successes: int = 0
    agent_successes: int = 0

    # failures
    failure_counts: dict[str, int] = field(default_factory=dict)

    # env coverage
    observed_envs: list[str] = field(default_factory=list)

    # induction state
    current_harness_id: Optional[str] = None
    last_induction_time: Optional[float] = None

    # estimated token savings
    estimated_token_savings: int = 0

    @property
    def harness_success_rate(self) -> float:
        return self.harness_successes / self.harness_executions if self.harness_executions else 0.0

    @property
    def agent_success_rate(self) -> float:
        return self.agent_successes / self.agent_executions if self.agent_executions else 0.0

    @property
    def support_count(self) -> int:
        """Trajectories accumulated for this task type."""
        return self.agent_executions  # during accumulation, agent path


# =====================================================================
# Execution result
# =====================================================================
@dataclass
class ExecutionResult:
    """Outcome of running a harness or agent on a task."""

    success: bool
    path: str  # "harness" | "harness_with_fallback" | "agent_fallback"
    harness_id: Optional[str] = None
    tokens_used: int = 0
    latency_seconds: float = 0.0
    failure_type: Optional[str] = None
    trajectory: Optional[Trajectory] = None
    output: str = ""
