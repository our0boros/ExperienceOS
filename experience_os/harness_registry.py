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
