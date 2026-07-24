"""Four-layer Experience Repository with a version DAG.

Layer 0 — :class:`~experience_os.models.Trajectory`      (raw, append-only)
Layer 1 — :class:`~experience_os.models.ExperienceRecord` (induced summaries)
Layer 2 — :class:`~experience_os.models.Harness`         (executable code)
Layer 3 — :class:`~experience_os.models.TaskTypeStats`    (meta / utility)

The version DAG (§3.6 of the RP) lives inside Layer 2: each harness records its
``parent_id`` and the repository can traverse the ancestry / siblings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from experience_os.config import Config
from experience_os.models import (
    ExperienceRecord,
    FailureType,
    Harness,
    HarnessStatus,
    TaskTypeStats,
    Trajectory,
)

log = logging.getLogger(__name__)


def _as_dict(obj) -> dict:
    """Serialise a dataclass to a plain dict (recursively)."""
    from enum import Enum

    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dict__"):
        return {k: _as_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    return obj


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_dict(obj), ensure_ascii=False, indent=2))


class Repository:
    """Persistent storage for all four experience layers."""

    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_dirs()
        self._trajectories: dict[str, Trajectory] = {}
        self._records: dict[str, ExperienceRecord] = {}
        self._harnesses: dict[str, Harness] = {}
        self._stats: dict[str, TaskTypeStats] = {}
        self._load()

    # ==================================================================
    # persistence
    # ==================================================================
    def _load(self) -> None:
        base = self.config.data_dir
        for p in (base / "trajectories").glob("*.json"):
            self._trajectories[p.stem] = _trajectory_from_json(p.read_text())
        for p in (base / "records").glob("*.json"):
            self._records[p.stem] = _record_from_json(p.read_text())
        for p in (base / "harnesses").glob("*.json"):
            self._harnesses[p.stem] = _harness_from_json(p.read_text())
        for p in (base / "stats").glob("*.json"):
            self._stats[p.stem] = _stats_from_json(p.read_text())
        log.info(
            "Loaded %d trajectories, %d records, %d harnesses, %d task-type stats",
            len(self._trajectories), len(self._records), len(self._harnesses), len(self._stats),
        )

    def _save_trajectory(self, t: Trajectory) -> None:
        _save_json(t, self.config.data_dir / "trajectories" / f"{t.id}.json")

    def _save_record(self, r: ExperienceRecord) -> None:
        _save_json(r, self.config.data_dir / "records" / f"{r.id}.json")

    def _save_harness(self, h: Harness) -> None:
        _save_json(h, self.config.data_dir / "harnesses" / f"{h.id}.json")

    def _save_stats(self, s: TaskTypeStats) -> None:
        _save_json(s, self.config.data_dir / "stats" / f"{s.task_type}.json")

    # ==================================================================
    # Layer 0 — trajectories
    # ==================================================================
    def add_trajectory(self, t: Trajectory) -> None:
        self._trajectories[t.id] = t
        self._save_trajectory(t)
        log.debug("Logged trajectory %s (task=%s, outcome=%s)", t.id, t.task_type, t.outcome)

    def trajectories_for_type(self, task_type: str, success_only: bool = True) -> list[Trajectory]:
        return [
            t for t in self._trajectories.values()
            if t.task_type == task_type
            and (not success_only or t.outcome == "success")
        ]

    # ==================================================================
    # Layer 1 — experience records
    # ==================================================================
    def add_record(self, r: ExperienceRecord) -> None:
        self._records[r.id] = r
        self._save_record(r)

    def records_for_type(self, task_type: str) -> list[ExperienceRecord]:
        return [r for r in self._records.values() if r.task_type == task_type]

    # ==================================================================
    # Layer 2 — harnesses / version DAG
    # ==================================================================
    def add_harness(self, h: Harness) -> None:
        self._harnesses[h.id] = h
        self._save_harness(h)
        stats = self.get_stats(h.task_type)
        stats.current_harness_id = h.id
        stats.last_induction_time = h.created_at
        self._save_stats(stats)

    def get_harness(self, hid: str) -> Optional[Harness]:
        return self._harnesses.get(hid)

    def active_harnesses(self) -> list[Harness]:
        return [h for h in self._harnesses.values() if h.status == HarnessStatus.ACTIVE]

    def active_harnesses_for_type(self, task_type: str) -> list[Harness]:
        return [
            h for h in self._harnesses.values()
            if h.task_type == task_type and h.status == HarnessStatus.ACTIVE
        ]

    # --- version DAG traversal ----------------------------------------
    def children(self, hid: str) -> list[Harness]:
        return [h for h in self._harnesses.values() if h.parent_id == hid]

    def ancestry(self, hid: str) -> list[Harness]:
        chain: list[Harness] = []
        current = self._harnesses.get(hid)
        while current and current.parent_id:
            parent = self._harnesses.get(current.parent_id)
            if parent:
                chain.append(parent)
                current = parent
            else:
                break
        return chain

    def deprecate(self, hid: str) -> None:
        h = self._harnesses.get(hid)
        if h:
            h.status = HarnessStatus.DEPRECATED
            h.updated_at = __import__("time").time()
            self._save_harness(h)

    # --- failure tracking ---------------------------------------------
    def record_failure(self, hid: str, ftype: FailureType) -> None:
        h = self._harnesses.get(hid)
        if not h:
            return
        key = ftype.value
        h.failure_counts[key] = h.failure_counts.get(key, 0) + 1
        h.updated_at = __import__("time").time()
        self._save_harness(h)

    # ==================================================================
    # Layer 3 — task-type statistics
    # ==================================================================
    def get_stats(self, task_type: str) -> TaskTypeStats:
        if task_type not in self._stats:
            self._stats[task_type] = TaskTypeStats(task_type=task_type)
        return self._stats[task_type]

    def save_stats(self, task_type: str) -> None:
        if task_type in self._stats:
            self._save_stats(self._stats[task_type])

    # ==================================================================
    # queries used by the induction trigger
    # ==================================================================
    def support_count(self, task_type: str) -> int:
        """Number of *successful agent* trajectories of this type."""
        return len(self.trajectories_for_type(task_type, success_only=True))

    def all_task_types(self) -> list[str]:
        return list({t.task_type for t in self._trajectories.values() if t.task_type})


# ======================================================================
# deserialisation helpers
# ======================================================================
def _trajectory_from_json(text: str) -> Trajectory:
    from experience_os.models import EnvironmentSnapshot, Step, StructuredCoT
    d = json.loads(text)
    steps = [Step(**s) for s in d.pop("steps", [])]
    cot = StructuredCoT(**d.pop("structured_cot", {}))
    env = EnvironmentSnapshot(attributes=d.pop("env_snapshot", {}).get("attributes", {}))
    return Trajectory(**d, steps=steps, structured_cot=cot, env_snapshot=env)


def _record_from_json(text: str) -> ExperienceRecord:
    from experience_os.models import ParamStep
    d = json.loads(text)
    steps = [ParamStep(**s) for s in d.pop("param_steps", [])]
    return ExperienceRecord(**d, param_steps=steps)


def _harness_from_json(text: str) -> Harness:
    from experience_os.models import VerificationMeta
    d = json.loads(text)
    status = d.pop("status", "active")
    if isinstance(status, str):
        status = HarnessStatus(status)
    ver = d.pop("verification", {})
    return Harness(**d, status=status, verification=VerificationMeta(**ver))


def _stats_from_json(text: str) -> TaskTypeStats:
    return TaskTypeStats(**json.loads(text))
