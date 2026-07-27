"""Runtime Router: two-stage harness retrieval + precondition matching.

Stage 1 — *semantic retrieval* (coarse):  cosine similarity between the task
description embedding and each harness's cached embedding vector.

Stage 2 — *precondition matching* (fine):  verify every hard precondition holds
in the current :class:`~experience_os.models.EnvironmentSnapshot`;  soft
precondition mismatches allow degraded execution.

Decision logic (§3.2):
    hard conditions satisfied    → use harness (high confidence)
    only soft conditions miss     → use harness (degraded, medium confidence)
    no match                     → fallback to agent
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from experience_os.services import EmbeddingService
    from experience_os.models import SubStepPattern

from experience_os.models import EnvironmentSnapshot, Harness
from experience_os.repository import Repository
from experience_os.services import Services

log = logging.getLogger(__name__)


class MatchLevel(str, Enum):
    FULL = "full"
    SOFT = "soft"
    NONE = "none"


@dataclass
class RetrievalResult:
    harness: Optional[Harness]
    level: MatchLevel
    confidence: float
    reason: str


class RuntimeRouter:
    """Selects a harness for a given task, or signals agent fallback."""

    SOFT_KEYS = {"version", "browser", "screen_resolution", "latency"}

    # 四层降级的 embedding 阈值
    HIGH_CONF_THRESHOLD = 0.85   # cosine ≥ 此值 → 直接使用，不消耗 LLM
    LOW_CONF_THRESHOLD = 0.65    # cosine < 此值 → 直接回退 ReAct，不浪费 LLM

    def __init__(
        self, repo: Repository, services: Services, top_k: int = 5,
    ) -> None:
        self.repo = repo
        self._services = services
        self.top_k = top_k

    # ------------------------------------------------------------------
    # embedding cache
    # ------------------------------------------------------------------
    def _ensure_embedding(self, harness: Harness) -> list[float]:
        if harness.embedding is not None:
            return harness.embedding
        if self._services is None:
            raise RuntimeError("RuntimeRouter: services required for embedding retrieval")
        vec = self._services.embedding.embed(harness.retrieval_text())
        harness.embedding = vec
        self.repo.add_harness(harness)  # persist
        return vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    # ------------------------------------------------------------------
    # stage 1 — semantic retrieval
    # ------------------------------------------------------------------
    def _semantic_search(self, task_description: str) -> list[tuple[Harness, float]]:
        if self._services is None:
            raise RuntimeError("RuntimeRouter: services required for embedding retrieval")
        query_vec = self._services.embedding.embed(task_description)
        scored: list[tuple[Harness, float]] = []
        for h in self.repo.active_harnesses():
            vec = self._ensure_embedding(h)
            sim = self._cosine(query_vec, vec)
            scored.append((h, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]

    # ------------------------------------------------------------------
    # stage 2 — precondition matching
    # ------------------------------------------------------------------
    def _check_preconditions(
        self, harness: Harness, env: EnvironmentSnapshot
    ) -> MatchLevel:
        hard_fail = False
        soft_fail = False
        for key, expected in harness.preconditions.items():
            if not env.satisfies(key, expected):
                if key in self.SOFT_KEYS:
                    soft_fail = True
                else:
                    hard_fail = True
        if hard_fail:
            return MatchLevel.NONE
        if soft_fail:
            return MatchLevel.SOFT
        return MatchLevel.FULL

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def select(
        self,
        task_description: str,
        env: EnvironmentSnapshot,
        task_type: str = "",
    ) -> RetrievalResult:
        """Return the best harness match, or ``NONE`` for agent fallback."""
        candidates = self._semantic_search(task_description)

        # if task_type is known, prefer harnesses of the same type first
        if task_type:
            typed = [c for c in candidates if c[0].task_type == task_type]
            others = [c for c in candidates if c[0].task_type != task_type]
            candidates = typed + others

        for harness, sim in candidates:
            level = self._check_preconditions(harness, env)
            if level == MatchLevel.FULL:
                return RetrievalResult(harness, MatchLevel.FULL, sim, "full precondition match")
            if level == MatchLevel.SOFT:
                return RetrievalResult(
                    harness, MatchLevel.SOFT, sim * 0.8, "soft precondition mismatch (degraded)"
                )

        # fallback: if task_type matches but semantic/precondition didn't work,
        # check if there's an exact task_type active harness with empty preconditions
        if task_type:
            for h in self.repo.active_harnesses_for_type(task_type):
                if not h.preconditions or all(
                    env.satisfies(k, v) for k, v in h.preconditions.items()
                ):
                    return RetrievalResult(
                        h, MatchLevel.FULL, 0.5, "task_type fallback match"
                    )

        if candidates:
            return RetrievalResult(
                None, MatchLevel.NONE, 0.0, "precondition mismatch on all candidates"
            )
        return RetrievalResult(None, MatchLevel.NONE, 0.0, "no harnesses in registry")

    # ------------------------------------------------------------------
    # 四层子步骤意图检索（§2.1）
    # ------------------------------------------------------------------
    def retrieve_substep_harness(
        self,
        intent: str,
        available_inputs: set[str] | None = None,
        needed_outputs: set[str] | None = None,
        effect_constraint: str | None = None,
    ) -> RetrievalResult:
        """按四层降级策略检索匹配的子步骤 harness。

        返回 ``RetrievalResult``，其中 ``level`` 表示匹配置信度：
        - FULL:   exact match 或 embedding ≥ 0.85
        - SOFT:   embedding 模糊匹配 (0.65~0.85)，需 LLM 评估
        - NONE:   无匹配，应回退 ReAct
        """
        from experience_os.models import SubStepPattern

        # 收集已知的子步骤模式（从 harness registry + substeps 聚合）
        patterns = self._gather_substep_patterns()

        # Layer 1: exact match
        exact = next((p for p in patterns if p.intent == intent), None)
        if exact is not None:
            h = self.repo.get_harness_for_capability(exact.intent)
            if h:
                return RetrievalResult(h, MatchLevel.FULL, 1.0,
                                       f"exact intent match: {exact.intent}")

        # Layer 2a/2b/2c: embedding matching
        try:
            high, fuzzy, rejected = self._services.embedding.match_intent(
                intent, patterns,
                high_threshold=self.HIGH_CONF_THRESHOLD,
                low_threshold=self.LOW_CONF_THRESHOLD,
            )
        except RuntimeError:
            # No embedding service available — fall back to exact only
            return RetrievalResult(None, MatchLevel.NONE, 0.0,
                                   "no embedding service, exact match only")

        # Layer 2a: high confidence → use directly
        if high:
            best_pattern, sim = high[0]
            # Also check input/output signature constraints
            if self._check_io_signature(best_pattern, available_inputs, needed_outputs):
                h = self.repo.get_harness_for_capability(best_pattern.intent)
                if h:
                    return RetrievalResult(h, MatchLevel.FULL, sim,
                                           f"embedding match (cos={sim:.2f}): {best_pattern.intent}")
                # Pattern exists but no harness yet → SOFT (need induction)
                return RetrievalResult(None, MatchLevel.SOFT, sim * 0.8,
                                       f"candidate pattern, no harness yet: {best_pattern.intent}")

        # Layer 2b: fuzzy → flag for LLM evaluation (caller decides)
        if fuzzy:
            best_pattern, sim = fuzzy[0]
            if self._check_io_signature(best_pattern, available_inputs, needed_outputs):
                return RetrievalResult(None, MatchLevel.SOFT, sim,
                                       f"fuzzy match, needs LLM eval: {best_pattern.intent}")

        # Layer 2c: low confidence → reject
        if rejected:
            return RetrievalResult(None, MatchLevel.NONE, rejected[0].support_count / 10.0 if hasattr(rejected[0], 'support_count') else 0.0,
                                   f"no match (best cos < {self.LOW_CONF_THRESHOLD}), fallback to ReAct")

        return RetrievalResult(None, MatchLevel.NONE, 0.0, "no sub-step pattern match")

    def _gather_substep_patterns(self) -> list:
        """收集所有已知子步骤模式（从 harness registry 聚合）。"""
        from experience_os.models import SubStepPattern
        patterns: list[SubStepPattern] = []

        # 从活跃 harness 中提取子步骤模式
        for h in self.repo.active_harnesses():
            if h.capability and h.capability not in [p.intent for p in patterns]:
                p = SubStepPattern(
                    intent=h.capability,
                    action_name=h.name,
                    description=h.description,
                )
                patterns.append(p)

        return patterns

    @staticmethod
    def _check_io_signature(
        pattern, available_inputs: set[str] | None, needed_outputs: set[str] | None,
    ) -> bool:
        """检查子步骤模式的输入/输出签名与需求的匹配度。"""
        if available_inputs is None and needed_outputs is None:
            return True  # no constraints → always match
        # TODO: parse pattern's input_schema / output_schema for precise matching
        return True
