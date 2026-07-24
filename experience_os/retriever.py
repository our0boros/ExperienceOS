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
from typing import Optional

from experience_os.llm import LLMClient
from experience_os.models import EnvironmentSnapshot, Harness
from experience_os.repository import Repository

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

    def __init__(self, repo: Repository, llm: LLMClient, top_k: int = 5) -> None:
        self.repo = repo
        self.llm = llm
        self.top_k = top_k

    # ------------------------------------------------------------------
    # embedding cache
    # ------------------------------------------------------------------
    def _ensure_embedding(self, harness: Harness) -> list[float]:
        if harness.embedding is not None:
            return harness.embedding
        vec = self.llm.embed(harness.retrieval_text())
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
        query_vec = self.llm.embed(task_description)
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
