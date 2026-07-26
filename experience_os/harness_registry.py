"""HarnessRegistry: O(1) intent→Harness lookup with embedding fallback.

Maps Agent tool-call intents (e.g. ``get_user_details``) directly to compiled
harnesses, eliminating the need for per-request LLM inference on stable,
deterministic sub-steps.
"""
from __future__ import annotations

import logging
from typing import Optional

from experience_os.models import Harness
from experience_os.repository import Repository

log = logging.getLogger(__name__)


class HarnessRegistry:
    """Intent → Harness registry.

    Usage::

        registry = HarnessRegistry(repo)
        registry.load_all()                    # load from repository

        harness = registry.lookup("get_user_details")
        if harness:
            result = harness.execute({"user_id": "..."})
    """

    def __init__(self, repo: Optional[Repository] = None) -> None:
        self.repo = repo
        self._by_intent: dict[str, Harness] = {}      # "get_user_details" → latest version
        self._by_name: dict[str, Harness] = {}         # "get_user_details-v1" → exact version
        self._stats: dict[str, dict] = {}              # intent → {calls, successes, failures}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, harness: Harness) -> None:
        """Register (or update) a harness by its capability/intent."""
        intent = harness.capability or harness.task_type
        # Version management: keep the latest version by default
        if intent in self._by_intent:
            existing = self._by_intent[intent]
            if harness.version > existing.version:
                log.info("Registry: %s → %s (updated v%d → v%d)",
                         intent, harness.full_name, existing.version, harness.version)
                self._by_intent[intent] = harness
        else:
            log.info("Registry: %s → %s (registered)", intent, harness.full_name)
            self._by_intent[intent] = harness

        self._by_name[harness.full_name] = harness
        if intent not in self._stats:
            self._stats[intent] = {"calls": 0, "successes": 0, "failures": 0}

    def register_all(self, harnesses: list[Harness]) -> None:
        for h in harnesses:
            self.register(h)

    # ------------------------------------------------------------------
    # load from repository
    # ------------------------------------------------------------------
    def load_all(self) -> int:
        """Load all ACTIVE harnesses from the repository."""
        if not self.repo:
            return 0
        count = 0
        for h in self.repo.active_harnesses():
            self.register(h)
            count += 1
        log.info("HarnessRegistry: loaded %d harnesses from repository", count)
        return count

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def lookup(self, intent: str) -> Optional[Harness]:
        """O(1) intent → latest harness lookup.

        Args:
            intent: Tool/action name, e.g. ``"get_user_details"``.

        Returns:
            The latest harness for this intent, or ``None``.
        """
        return self._by_intent.get(intent)

    def lookup_exact(self, name: str) -> Optional[Harness]:
        """Look up a specific version by full name (e.g. ``"get_user_details-v1"``)."""
        return self._by_name.get(name)

    # ------------------------------------------------------------------
    # execution tracking
    # ------------------------------------------------------------------
    def record_call(self, intent: str, success: bool) -> None:
        """Record a harness execution outcome for stats."""
        if intent not in self._stats:
            self._stats[intent] = {"calls": 0, "successes": 0, "failures": 0}
        self._stats[intent]["calls"] += 1
        if success:
            self._stats[intent]["successes"] += 1
        else:
            self._stats[intent]["failures"] += 1

    def stats(self, intent: Optional[str] = None) -> dict:
        if intent:
            return self._stats.get(intent, {})
        return dict(self._stats)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def intents(self) -> list[str]:
        return list(self._by_intent.keys())

    @property
    def count(self) -> int:
        return len(self._by_intent)

    def list_harnesses(self) -> list[Harness]:
        return list(self._by_intent.values())

    # ------------------------------------------------------------------
    # 语义分块查找（按 effect 分块 + 加权相似度）
    # ------------------------------------------------------------------
    def lookup_weighted(
        self,
        intent: str,
        tool_name: str = "",
        effect: str | None = None,
        available_inputs: set[str] | None = None,
        needed_outputs: set[str] | None = None,
        embed = None,
    ) -> Optional[Harness]:
        """多字段加权查找（语义分块优化）。

        比 ``lookup()`` 更鲁棒：先用 effect 分块缩小搜索空间，
        再用 exact match → capability match → weighted similarity 三级查找。
        """
        # 1. exact match（最快）
        h = self._by_intent.get(intent)
        if h is not None:
            return h

        # 2. capability match（聚类后的语义标签）
        for h in self._by_intent.values():
            if h.capability == intent:
                return h

        # 3. 多字段加权相似度（需要在分块内遍历）
        candidates = self._by_intent.values()
        if effect:
            candidates = [h for h in candidates
                          if getattr(h, 'effect', '') == effect or not getattr(h, 'effect', '')]
        if tool_name:
            candidates = [h for h in candidates
                          if h.task_type == tool_name or h.capability == tool_name]

        if not candidates or not embed:
            return None

        scored = []
        for h in candidates:
            score = self._weighted_similarity(
                intent, tool_name, effect,
                h, available_inputs, needed_outputs, embed,
            )
            if score >= 0.6:
                scored.append((h, score))

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
        return None

    @staticmethod
    def _weighted_similarity(
        query_intent: str,
        query_tool: str,
        query_effect: str | None,
        harness: Harness,
        available_inputs: set[str] | None,
        needed_outputs: set[str] | None,
        embed,
    ) -> float:
        """多字段加权相似度。"""
        weights = {"intent": 0.35, "tool_name": 0.25,
                   "io_signature": 0.25, "effect": 0.15}
        scores: dict[str, float] = {}

        # Intent: embedding cosine
        try:
            h_text = harness.capability or harness.description or harness.task_type
            q_vec = embed.embed(query_intent)
            h_vec = embed.embed(h_text)
            scores["intent"] = embed._cosine(q_vec, h_vec) if hasattr(embed, '_cosine') else 0.5
        except Exception:
            scores["intent"] = 0.5

        # Tool name: exact > partial > default
        if query_tool and query_tool == harness.task_type:
            scores["tool_name"] = 1.0
        elif query_tool and harness.task_type and query_tool in harness.task_type:
            scores["tool_name"] = 0.6
        else:
            scores["tool_name"] = 0.3

        # I/O signature: overlap score
        if available_inputs is not None and needed_outputs is not None:
            h_inputs = set(harness.input_schema.get("requires", []) if isinstance(harness.input_schema, dict) else [])
            h_outputs = set(harness.output_schema.get("produces", []) if isinstance(harness.output_schema, dict) else [])
            in_overlap = len(available_inputs & h_inputs) / max(1, len(h_inputs)) if h_inputs else 0.5
            out_overlap = len(needed_outputs & h_outputs) / max(1, len(h_outputs)) if h_outputs else 0.5
            scores["io_signature"] = (in_overlap + out_overlap) / 2
        else:
            scores["io_signature"] = 0.5

        # Effect: exact match bonus
        if query_effect and query_effect == getattr(harness, 'effect', ''):
            scores["effect"] = 1.0
        else:
            scores["effect"] = 0.3

        return sum(weights[k] * scores[k] for k in weights)

    def all_active(self) -> list[Harness]:
        """返回所有已注册的 harness（兼容 composite 模块）。"""
        return list(self._by_intent.values())
