"""Harness Inductor — 六阶段编译 + 沙箱验证管线。

实现了研究提案 §3.3：

    Phase 1 — trajectory segmentation
    Phase 2 — precondition / postcondition extraction
    Phase 3 — invariant mining
    Phase 4 — step abstraction & parameterisation
    Phase 5 — harness synthesis (LLM generates executable code)
    Phase 6 — sandbox replay validation

Bayesian *induction trigger* (§2.2, §3.5) 决定 *何时* 运行该管线：

    H* = argmax P(H | T_c) ∝ P(T_c | H) · P(H)

    P(H) ∝ exp(-λ · MDL(H))      # simplicity prior
    P(T_c | H) = replay success  # likelihood = coverage

当 ``support_count >= MIN_SUPPORT``（新 harness）或
``f2_failure_count >= 2``（补丁）时触发归纳。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from experience_os.config import Config
from experience_os.environment import BaseEnvironment, TaskRequest
from experience_os.models import (
    ArtifactType,
    ExperienceRecord,
    FailureType,
    Harness,
    HarnessStatus,
    ParamStep,
    Step,
    StructuredCoT,
    SubStepOutcome,
    SubStepPattern,
    SubStepPlan,
    Trajectory,
)
from experience_os.repository import Repository
from experience_os.services import Services
from experience_os.stores import ExperienceStore
from experience_os.compiler import algorithms as algo
from experience_os.compiler.prompts import (
    JUDGE_PROMPT,
    SEGMENT_PROMPT,
    SYNTHESIS_PROMPT,
)

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


class HarnessInductor:
    """Compiles trajectories into validated harnesses."""

    def __init__(
        self,
        config: Config,
        services: Services,
        repo: Repository,
        *,
        experience_store: Optional[ExperienceStore] = None,
    ) -> None:
        self.config = config
        self.services = services
        self.chat = services.chat
        self.embedding = services.embedding
        self.repo = repo
        self.experience_store = experience_store
        # Phase 1 segmentation results kept across induce() calls
        self._segments: list[list[int]] = []

        if experience_store is None:
            log.info(
                "HarnessInductor created without ExperienceStore — "
                "substep discovery will fall back to legacy repo._trajectories path. "
                "Inject via HarnessInductor(..., experience_store=store)."
            )

    # ==================================================================
    # induction triggers (full-task level + sub-step pattern level)
    # ==================================================================
    def check_triggers(
        self, task_type: str = "",
        min_support: int | None = None,
        min_success_rate: float = 0.5,
        min_bayesian_score: float = 0.3,
    ) -> list[tuple[str, "SubStepPattern | None"]]:
        """返回所有满足触发条件的 ``(trigger_type, pattern)`` 对。

        检查三个级别：
        1. **Full-task level**: ``support_count >= MIN_SUPPORT`` → ``("new_harness", None)``
        2. **Patch level**: ``F2 >= f2_patch_trigger`` → ``("patch", None)``
        3. **Sub-step pattern level**: 从 substeps 表（优先）或轨迹 steps 聚合，
           满足 ``support >= min_support AND success_rate >= min_success_rate
           AND bayesian_score >= min_bayesian_score`` → ``("substep_pattern", pattern)``

        贝叶斯权重（§1.2）确保「只在失败全任务中出现的子步骤」也有非零分数。
        """
        if min_support is None:
            min_support = self.config.induction.min_support

        candidates: list[tuple[str, SubStepPattern | None]] = []

        # 1. Full-task trigger
        if task_type:
            stats = self.repo.get_stats(task_type)
            if stats.current_harness_id is None:
                if self.repo.support_count(task_type) >= min_support:
                    candidates.append(("new_harness", None))
            else:
                harness = self.repo.get_harness(stats.current_harness_id)
                if harness:
                    f2 = harness.failure_counts.get(FailureType.F2_IMPLEMENTATION_ERROR.value, 0)
                    if f2 >= self.config.induction.f2_patch_trigger:
                        candidates.append(("patch", None))

        # 2. Sub-step pattern trigger — try substeps table first (current experiment only)
        patterns = self._discover_substep_patterns_from_store(
            experiment_id=getattr(self, '_current_experiment_id', '')
        )
        if not patterns:
            # LEGACY FALLBACK: 直接读 repo._trajectories 做子步骤发现
            # 新代码应确保注入 ExperienceStore，避免走此路径
            log.warning(
                "Substep pattern discovery falling back to legacy path "
                "(repo._trajectories). Inject an ExperienceStore into "
                "HarnessInductor to use the substeps table instead."
            )
            all_trajs = list(self.repo._trajectories.values())
            patterns = self._discover_substep_patterns(all_trajs)

        # P2.1 意图聚类：合并 embedding 相似的 pattern
        patterns = self._cluster_patterns(patterns)

        for key, p in patterns.items():
            if p.support_count < min_support:
                continue
            if p.success_rate < min_success_rate:
                continue
            # 贝叶斯权重（Phase A: 已含预测准确率调整）
            bs = getattr(p, 'bayesian_score', None)
            if bs is not None and bs < min_bayesian_score:
                log.debug(
                    "Pattern %s filtered by bayesian score: %.3f < %.3f",
                    p.intent, bs, min_bayesian_score,
                )
                continue
            # pre-filter: 已有 ACTIVE harness 的 pattern 跳过（避免重复归纳）
            if self.repo.active_harnesses_for_type(p.intent):
                continue
            # Phase A: 记录预测质量
            pred_info = ""
            if p.example_contexts:
                first_ctx = p.example_contexts[0] if p.example_contexts else ""
                if "[pred_acc=" in first_ctx:
                    pred_info = f" {first_ctx}"
            log.info("Pattern candidate: %s (bs=%.3f, support=%d)%s",
                     p.intent, bs, p.support_count, pred_info)
            candidates.append(("substep_pattern", p))

        return candidates

    def _discover_substep_patterns_from_store(
        self, experiment_id: str = ""
    ) -> dict[str, SubStepPattern]:
        """从 ``substeps`` 表（ExperienceLibrary）聚合子步骤模式。

        优先使用此数据源（§2 新架构），因为子步骤是独立一等实体，
        不受全任务成功率限制。

        Phase A（预测契约）：使用 prediction-adjusted Bayesian score 替代
        原始 bayesian_score，区分高质量经验与侥幸成功。

        Args:
            experiment_id: 可选，限定实验范围（避免跨实验污染）。
        """
        try:
            store = self.experience_store
            if store is not None:
                rows = store.aggregate_substep_patterns(
                    experiment_id=experiment_id,
                    min_support=self.config.induction.min_support,
                )
            else:
                log.debug(
                    "No ExperienceStore injected — substep discovery skipped. "
                    "Inject an ExperienceStore via HarnessInductor(experience_store=...)."
                )
                return {}
            patterns: dict[str, SubStepPattern] = {}
            for row in rows:
                intent = row["intent"]
                tool = row["tool_name"]
                key = f"{tool}:{intent}"

                # Phase A: 优先使用预测调整后的贝叶斯评分
                score = row.get("adjusted_bayesian") or row.get("bayesian_score") or 0.0
                pred_acc = row.get("prediction_accuracy", 1.0)
                quality_dist = row.get("quality_label_dist", {})

                p = SubStepPattern(
                    intent=intent,
                    action_name=tool,
                    support_count=row["support_count"] or 0,
                    success_count=row["success_count"] or 0,
                    success_in_full_tasks=row["success_in_full_tasks"] or 0,
                    total_appearances=row["total_appearances"] or 0,
                    bayesian_score=score,
                )
                # 附加预测质量信息到 example_contexts 用于日志
                if pred_acc < 1.0 or quality_dist:
                    qual_summary = (
                        f"[pred_acc={pred_acc:.2f}, quality={quality_dist}]"
                    )
                    if p.example_contexts:
                        p.example_contexts[0] = qual_summary + " " + p.example_contexts[0]
                    else:
                        p.example_contexts.append(qual_summary)
                patterns[key] = p

            # 日志：报告预测契约对候选模式的影响
            if patterns and any(
                row.get("prediction_accuracy", 1.0) < 1.0 for row in rows
            ):
                n_adjusted = sum(
                    1 for row in rows
                    if row.get("adjusted_bayesian", 0) != row.get("bayesian_score", 0)
                )
                log.info(
                    "Phase A prediction contracts: %d/%d patterns had adjusted scores",
                    n_adjusted, len(rows),
                )
            return patterns
        except Exception:
            return {}

    # ==================================================================
    # P2.1 意图聚类 — embedding 相似的工具合并为同一 capability
    # ==================================================================
    def _cluster_patterns(
        self, patterns: dict[str, SubStepPattern]
    ) -> dict[str, SubStepPattern]:
        """将 embedding 相似（cosine ≥ 0.85）的 pattern 合并。

        例如 ``find_user_id_by_name_zip`` 和 ``find_user_id_by_email``
        都映射到 ``user_lookup`` capability。
        """
        if len(patterns) <= 1:
            return patterns

        try:
            _embed = self.embedding
        except Exception:
            return patterns

        seen: set[str] = set()
        clustered: dict[str, SubStepPattern] = {}

        for key, p in patterns.items():
            if key in seen:
                continue
            # 找相似 pattern
            similar: list[SubStepPattern] = []
            for key2, p2 in patterns.items():
                if key2 == key or key2 in seen:
                    continue
                try:
                    sim = _embed._cosine(
                        _embed.embed(p.description or p.intent),
                        _embed.embed(p2.description or p2.intent),
                    )
                    if sim >= 0.85:
                        similar.append(p2)
                        seen.add(key2)
                except Exception:
                    pass

            if similar:
                # 合并：用更短的 action_name 作为 capability
                all_names = [p.action_name] + [s.action_name for s in similar]
                capability = min(all_names, key=len) if all_names else p.action_name
                for s in similar:
                    p.support_count += s.support_count
                    p.success_count += s.success_count
                    p.example_contexts.extend(s.example_contexts[:2])
                log.info("Clustered %s + %d similar → capability=%s",
                         p.intent, len(similar), capability)
            else:
                capability = ""

            seen.add(key)
            cluster_key = capability or p.intent
            p.description = cluster_key  # 用聚类后的 key 作为 description 供 retrieval 使用
            clustered[cluster_key] = p

        return clustered

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
            data = self.chat.complete_json([
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

        委托给 :func:`experience_os.compiler.algorithms._segment`，
        并将结果保存到 ``self._segments`` 供下游阶段使用。
        """
        self._segments = algo._segment(trajectories, self.chat)
        return self._segments

    # ==================================================================
    # Phase 5 — harness synthesis (LLM generates code)
    # ==================================================================
    def _synthesize(self, record: ExperienceRecord, trajectories: list[Trajectory],
                    repair_context: str = "", *,
                    is_substep: bool = False, tool_name: str = "") -> str:
        steps_json = json.dumps(
            [{"template": s.template, "params": s.params} for s in record.param_steps],
            ensure_ascii=False,
        )
        example_traces = "\n---\n".join(
            "\n".join(f"  {s.action} -> {s.result[:60]}" for s in t.steps[:6])
            for t in trajectories[:2]
        )

        # StructuredCoT 完整字段（goal/constraints/risk/milestones）传入 prompt
        cot = trajectories[0].structured_cot if trajectories else StructuredCoT(goal="")

        # 如果有 repair_context，构造修复提示段，引导 LLM 修正先前失败
        if repair_context:
            repair_section = (
                "\n\nPrevious version failed validation with these errors:\n"
                f"{repair_context}\n"
                "Please fix the code to handle these cases."
            )
        else:
            repair_section = ""

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

        # 选择 prompt：子步骤用简化版（单工具调用），全任务用完整版
        if is_substep and tool_name:
            from experience_os.compiler.prompts import SUBSTEP_SYNTHESIS_PROMPT
            params_list = ", ".join(record.param_steps[0].params) if record.param_steps else "none"
            prompt = SUBSTEP_SYNTHESIS_PROMPT.format(
                capability=record.task_type,
                tool_name=tool_name,
                description=record.terminal_verifier or record.task_type,
                preconditions=json.dumps(record.candidate_preconditions, ensure_ascii=False),
                invariants=record.invariants,
                params_list=params_list,
                example_traces=example_traces,
            )
        else:
            prompt = SYNTHESIS_PROMPT.format(
                task_type=record.task_type,
                preconditions=json.dumps(record.candidate_preconditions, ensure_ascii=False),
                steps_json=steps_json,
                invariants=record.invariants,
                terminal_verifier=record.terminal_verifier,
                example_traces=example_traces,
                example_harness=example_harness,
                cot_goal=cot.goal,
                cot_constraints="; ".join(cot.constraints) if cot.constraints else "none",
                cot_risk=cot.risk or "none",
                cot_milestones="; ".join(cot.milestones) if cot.milestones else "none",
                repair_section=repair_section,
            )
        code = self.chat.chat(
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
        *,
        env_builder: Optional["Callable[[Trajectory], BaseEnvironment]"] = None,
        threshold: float | None = None,
    ) -> tuple[ValidationResult, float]:
        """Replay the harness against each source task; require success_rate ≥ threshold.

        When ``env_builder`` is provided, a fresh environment is constructed for
        each trajectory (correct for stateful envs like τ-bench where each task
        has its own initial DB state).  Otherwise the single ``env`` is reused
        (suitable for stateless / mock envs).
        """
        if threshold is None:
            threshold = self.config.induction.validation_threshold
        successes = 0
        for traj in source_trajectories:
            # extract params from trajectory step metadata or structured CoT
            params: dict = {}
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
            replay_env = env_builder(traj) if env_builder else env
            result = replay_env.execute_harness(candidate, request)
            if result.success:
                successes += 1
        rate = successes / len(source_trajectories) if source_trajectories else 0.0
        if rate >= threshold:
            return ValidationResult.APPROVED, rate
        if rate > 0.0:
            return ValidationResult.NEEDS_REVISION, rate
        return ValidationResult.REJECTED, rate

    # ==================================================================
    # Phase 6 辅助 — 收集验证失败的错误信息（§5.3.5 修复重试）
    # ==================================================================
    def _collect_repair_context(self, harness: Harness, trajectories: list[Trajectory], env: BaseEnvironment, *, env_builder: Optional["Callable[[Trajectory], BaseEnvironment]"] = None) -> str:
        """收集验证失败的错误信息，供修复重试使用。

        对每条源轨迹重放 harness，提取失败任务的输出（异常 traceback 等），
        最多保留 3 条错误，每条截断到 300 字符，避免 prompt 过长。
        """
        errors: list[str] = []
        for traj in trajectories:
            params: dict = {}
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
            replay_env = env_builder(traj) if env_builder else env
            result = replay_env.execute_harness(harness, request)
            if not result.success and result.output:
                errors.append(f"Task {traj.task_id}: {result.output[:300]}")
        # 最多 3 条错误
        return "\n---\n".join(errors[:3])

    # ==================================================================
    # Phase 7 辅助 — 变体检测（§5.3.4 特化分裂）
    # ==================================================================
    def _detect_variations(self, harness: Harness, trajectories: list[Trajectory]) -> list[tuple[list[Trajectory], str]]:
        """检测轨迹中的变体模式（不同的工具序列长度或不同工具组合）。

        返回 ``[(variant_trajectories, signature), ...]``，
        ``signature`` 是变体的标识（如 "toolA_toolB_toolC"）。

        当存在多个不同的工具序列，且某序列占比 < 80% 且至少 2 条轨迹时，
        触发特化分裂。
        """
        if len(trajectories) < 4:
            return []

        # 按工具序列签名分组
        signatures: dict[tuple[str, ...], list[Trajectory]] = {}
        for t in trajectories:
            tools = tuple(s.action.split("(")[0].strip() for s in t.steps)
            signatures.setdefault(tools, []).append(t)

        # 如果有多个不同的工具序列，且主序列占比 < 80%，则触发分裂
        total = len(trajectories)
        variations: list[tuple[list[Trajectory], str]] = []
        # 按组大小降序，跳过最大的主序列
        for sig, group in sorted(signatures.items(), key=lambda x: -len(x[1])):
            ratio = len(group) / total
            if ratio < 0.8 and len(group) >= 2:
                # 这是一个变体
                short_sig = "_".join(sig[:3]) if sig else "empty"
                variations.append((group, short_sig))
        return variations

    # ==================================================================
    # full induction pipeline (dual-level)
    # ==================================================================
    def induce(
        self,
        task_type: str,
        env: BaseEnvironment,
        trigger: str | None = None,
        substep_pattern: SubStepPattern | None = None,
        env_builder: Optional["Callable[[Trajectory], BaseEnvironment]"] = None,
        *,
        trajectories: Optional[list[Trajectory]] = None,
        parent_harness_id: Optional[str] = None,
    ) -> Optional[Harness]:
        """Run the full seven-phase induction.

        Supports three entry points:
        * **Full-task induction** (trigger = ``"new_harness"``): compiles all
          successful trajectories for *task_type*.
        * **Sub-step induction** (trigger = ``"substep_pattern"``): compiles
          only the steps matching the discovered *substep_pattern* across all
          trajectories, regardless of full-task success.
        * **Specialization induction** (trigger = ``"specialization"``):
          compiles a variant harness from a subset of trajectories, linked to
          a parent via *parent_harness_id*.

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
            if trajectories is None:
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

        # --- Phase 2-4: 按分段结果处理（修复 1） ---
        # 当存在多个分段时，每段独立提取 preconditions/invariants/param_steps，
        # 然后合并：preconditions 取交集，invariants 合并去重，param_steps 按段拼接。
        # 单段（含整条轨迹）时行为与原有一致（向后兼容）。
        if self._segments and len(self._segments) > 1:
            all_preconditions: list[dict] = []
            all_invariants: list[str] = []
            all_param_steps: list[ParamStep] = []
            for seg_indices in self._segments:
                if not seg_indices:
                    continue
                seg_trajs = [algo._extract_segment_steps(t, seg_indices) for t in trajectories]
                all_preconditions.append(algo._intersect_preconditions(seg_trajs))
                all_invariants.extend(algo._mine_invariants(seg_trajs))
                all_param_steps.extend(algo._abstract_steps(seg_trajs))
            preconditions = algo._merge_preconditions(all_preconditions)
            # 合并去重（保序）
            seen_inv: set[str] = set()
            invariants = [x for x in all_invariants if not (x in seen_inv or seen_inv.add(x))]
            param_steps = all_param_steps
        else:
            # --- Phase 2 ---
            preconditions = algo._intersect_preconditions(trajectories)
            # --- Phase 3 ---
            invariants = algo._mine_invariants(trajectories)
            # --- Phase 4 ---
            param_steps = algo._abstract_steps(trajectories)

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

        # --- Phase 5 + 6: LLM 合成 + 沙箱验证（含 NEEDS_REVISION 修复重试循环, §5.3.5）---
        # determine parent and edge type for version DAG
        if parent_harness_id:
            parent_id = parent_harness_id
        elif trigger == "patch":
            stats = self.repo.get_stats(task_type)
            parent_id = stats.current_harness_id
        else:
            parent_id = None
        existing = self.repo.get_harness(parent_id) if parent_id else None
        version = (existing.version + 1) if existing else 1

        max_repair_attempts = 2
        repair_context = ""
        code = ""
        harness: Optional[Harness] = None
        validation_result = ValidationResult.REJECTED
        validation_rate = 0.0

        for attempt in range(max_repair_attempts + 1):
            if attempt > 0:
                log.info("Repair attempt %d/%d for %s", attempt, max_repair_attempts, induced_type)

            # --- Phase 5: synthesis ---
            code = self._synthesize(
                record, trajectories, repair_context=repair_context,
                is_substep=(trigger in ("substep_pattern", "specialization")),
                tool_name=substep_pattern.action_name if substep_pattern else "",
            )

            # 分离 soft preconditions（版本、分辨率等允许降级匹配的字段）
            _soft_keys = {"version", "browser", "screen_resolution", "latency"}
            _hard = {k: v for k, v in preconditions.items() if k not in _soft_keys}
            _soft = {k: v for k, v in preconditions.items() if k in _soft_keys}

            # 收集示例任务描述（用于检索增强，取前 3 条避免噪声）
            _examples = list(dict.fromkeys(  # 去重保序
                t.task_description for t in trajectories[:5]
                if t.task_description
            ))[:3]

            # 确定 version DAG 边类型
            _edge_type = ""
            if trigger == "patch":
                _edge_type = "patch"
            elif trigger in ("specialization", "substep_pattern"):
                _edge_type = "specialization" if parent_id else ""

            harness = Harness(
                name=induced_type.replace(" ", "_"),
                version=version,
                parent_id=parent_id,
                edge_type=_edge_type,
                task_type=induced_type,
                capability=capability,
                description=trajectories[0].task_description,
                preconditions=_hard,
                soft_preconditions=_soft,
                example_tasks=_examples,
                procedure_code=code,
                invariants=invariants,
                terminal_verifier=terminal,
                params=list(dict.fromkeys(p for ps in param_steps for p in ps.params)),
                source_record_ids=[record.id],
            )

            # --- Phase 6: validation ---
            # P1.2 读工具跳过验证：effect=read_only 的单步 harness 直接 APPROVE
            _is_readonly = (
                trigger in ("substep_pattern", "specialization")
                and substep_pattern is not None
                and getattr(substep_pattern, 'effect', '') == 'read_only'
            )
            if _is_readonly:
                validation_result = ValidationResult.APPROVED
                validation_rate = 1.0
                log.info("Read-only harness %s: skipping validation (direct APPROVE)",
                         harness.name)
            else:
                # 子步骤模式用更低阈值（单工具调用验证受 env 状态影响大）
                _threshold = self.config.induction.validation_threshold
                if trigger in ("substep_pattern", "specialization"):
                    _threshold = 0.0  # 任何成功即通过
                validation_result, validation_rate = self._validate(
                    harness, trajectories, env, env_builder=env_builder,
                    threshold=_threshold,
                )
            harness.verification = type(harness.verification)(
                success_rate=validation_rate, test_count=len(trajectories)
            )

            if validation_result == ValidationResult.APPROVED:
                break
            elif validation_result == ValidationResult.NEEDS_REVISION:
                # 收集失败信息供修复重试使用
                repair_context = self._collect_repair_context(
                    harness, trajectories, env, env_builder=env_builder,
                )
                log.warning("Harness %s needs revision (rate=%.2f), collecting repair context",
                            harness.full_name, validation_rate)
            else:
                # REJECTED，重试无意义
                break

        # 处理最终验证结果
        if validation_result == ValidationResult.APPROVED:
            # P1.1 去重：如果已有同 task_type 的 ACTIVE harness，deprecate 旧的
            for old in self.repo.active_harnesses_for_type(harness.task_type):
                if old.id != harness.id:
                    old.status = HarnessStatus.DEPRECATED
                    self.repo.add_harness(old)
                    log.info("Deprecated old harness %s (replaced by %s)",
                             old.full_name, harness.full_name)
            if parent_id:
                self.repo.deprecate(parent_id)
            assert harness is not None
            harness.status = HarnessStatus.ACTIVE
            self.repo.add_harness(harness)
            log.info("Harness %s APPROVED (replay rate=%.2f)", harness.full_name, validation_rate)

            # --- Phase 7: 变体检测与特化分裂（§5.3.4）---
            if len(trajectories) > 3:
                variations = self._detect_variations(harness, trajectories)
                if variations:
                    log.info("Detected %d variation(s) for %s, triggering specialization",
                             len(variations), induced_type)
                    # 记录变体签名到父 harness 的 split_reason
                    var_sigs = [sig for _, sig in variations]
                    harness.split_reason = "variations: " + ", ".join(var_sigs)
                    self.repo.add_harness(harness)  # 持久化 split_reason
                    for var_tasks, var_signature in variations:
                        var_type = f"{induced_type}__{var_signature}"
                        if not self.repo.records_for_type(var_type):
                            # 为变体任务组触发特化归纳
                            log.info("Specializing %s → %s (%d tasks)",
                                     induced_type, var_type, len(var_tasks))
                            self.induce(
                                var_type, env,
                                trigger="specialization",
                                parent_harness_id=harness.id,
                                env_builder=env_builder,
                                trajectories=var_tasks,
                            )
            return harness
        elif validation_result == ValidationResult.NEEDS_REVISION:
            assert harness is not None
            log.warning("Harness %s still needs revision after %d repairs (rate=%.2f)",
                        harness.full_name, max_repair_attempts, validation_rate)
            harness.status = HarnessStatus.DRAFT
            self.repo.add_harness(harness)
            return None
        else:
            log.warning("Harness for %s REJECTED (replay rate=%.2f)", induced_type, validation_rate)
            return None
