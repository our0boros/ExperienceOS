"""四层经验仓库 — SQLite 主存储。

Layer 0 — :class:`~experience_os.models.Trajectory`      (raw, append-only)
Layer 1 — :class:`~experience_os.models.ExperienceRecord` (induced summaries)
Layer 2 — :class:`~experience_os.models.Harness`         (executable code)
Layer 3 — :class:`~experience_os.models.TaskTypeStats`    (meta / utility)

底层通过 :class:`~experience_os.storage.Storage` (SQLite) 持久化，替代原来的
纯 JSON 文件存储。内存中保留 dict 缓存（``_trajectories`` 等）以保证
compiler.py / runtime.py 直接访问 ``repo._trajectories`` 的向后兼容性。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from experience_os.config import Config
from experience_os.models import (
    ExperienceRecord,
    FailureType,
    Harness,
    HarnessStatus,
    ParamStep,
    Step,
    StructuredCoT,
    TaskTypeStats,
    Trajectory,
    VerificationMeta,
    EnvironmentSnapshot,
)
from experience_os.storage import Storage, _unpack_vector

log = logging.getLogger(__name__)


def _as_dict(obj) -> dict:
    """序列化 dataclass 为 plain dict（递归）。"""
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


# ======================================================================
# 反序列化：从 SQLite dict → dataclass
# ======================================================================
def _trajectory_from_row(row: dict) -> Trajectory:
    """从 SQLite 行 dict 重建 Trajectory。"""
    steps_raw = row.get("steps_json") or "[]"
    cot_raw = row.get("cot_json") or "{}"
    env_raw = row.get("env_snapshot_json") or "{}"

    steps_data = json.loads(steps_raw) if isinstance(steps_raw, str) else steps_raw
    cot_data = json.loads(cot_raw) if isinstance(cot_raw, str) else cot_raw
    env_data = json.loads(env_raw) if isinstance(env_raw, str) else env_raw

    steps = [Step(**s) for s in steps_data]
    cot = StructuredCoT(**cot_data)
    env_attrs = env_data.get("attributes", env_data) if isinstance(env_data, dict) else {}
    env = EnvironmentSnapshot(attributes=env_attrs)

    return Trajectory(
        id=row.get("id", ""),
        task_id=row.get("task_id", ""),
        task_description=row.get("task_description", ""),
        task_type=row.get("task_type", ""),
        steps=steps,
        structured_cot=cot,
        env_snapshot=env,
        outcome=row.get("outcome", "failure"),
        tokens_used=row.get("tokens_used", 0) or 0,
        latency_seconds=row.get("latency_seconds", 0.0) or 0.0,
        timestamp=row.get("created_at", time.time()) or time.time(),
        phase=row.get("phase", ""),
    )


def _record_from_row(row: dict) -> ExperienceRecord:
    """从 SQLite 行 dict 重建 ExperienceRecord。"""
    pre_raw = row.get("preconditions_json") or "{}"
    steps_raw = row.get("canonical_steps_json") or "[]"
    inv_raw = row.get("invariants_json") or "[]"
    src_raw = row.get("source_trajectories_json") or "[]"

    pre = json.loads(pre_raw) if isinstance(pre_raw, str) else pre_raw
    steps_data = json.loads(steps_raw) if isinstance(steps_raw, str) else steps_raw
    inv = json.loads(inv_raw) if isinstance(inv_raw, str) else inv_raw
    src = json.loads(src_raw) if isinstance(src_raw, str) else src_raw

    param_steps = [ParamStep(**s) for s in steps_data]

    return ExperienceRecord(
        id=row.get("id", ""),
        task_type=row.get("task_type", ""),
        candidate_preconditions=pre,
        param_steps=param_steps,
        invariants=inv,
        terminal_verifier=row.get("terminal_verifier", ""),
        source_trajectory_ids=src,
        support_count=row.get("support_count", 0) or 0,
    )


def _harness_from_row(row: dict) -> Harness:
    """从 SQLite 行 dict 重建 Harness。"""
    pre_raw = row.get("preconditions_json") or "{}"
    inv_raw = row.get("invariants_json") or "[]"
    params_raw = row.get("params_json") or "[]"
    ver_raw = row.get("verification_json") or "{}"
    fc_raw = row.get("failure_counts_json") or "{}"

    pre = json.loads(pre_raw) if isinstance(pre_raw, str) else pre_raw
    inv = json.loads(inv_raw) if isinstance(inv_raw, str) else inv_raw
    params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
    ver = json.loads(ver_raw) if isinstance(ver_raw, str) else ver_raw
    fc = json.loads(fc_raw) if isinstance(fc_raw, str) else fc_raw

    status_val = row.get("status", "active")
    if isinstance(status_val, str):
        status = HarnessStatus(status_val)
    else:
        status = status_val

    # embedding BLOB
    embed_blob = row.get("embedding")
    embedding = _unpack_vector(embed_blob) if embed_blob else None

    return Harness(
        id=row.get("id", ""),
        name=row.get("name", ""),
        version=row.get("version", 1) or 1,
        parent_id=row.get("parent_id"),
        status=status,
        task_type=row.get("task_type", ""),
        capability=row.get("capability", ""),
        preconditions=pre,
        procedure_code=row.get("procedure_code", ""),
        invariants=inv,
        params=params,
        verification=VerificationMeta(**ver),
        failure_counts=fc,
        embedding=embedding,
        created_at=row.get("created_at", time.time()) or time.time(),
        updated_at=row.get("updated_at", time.time()) or time.time(),
    )


def _stats_from_row(row: dict) -> TaskTypeStats:
    """从 SQLite 行 dict 重建 TaskTypeStats。"""
    raw = row.get("stats_json") or "{}"
    d = json.loads(raw) if isinstance(raw, str) else raw
    return TaskTypeStats(**d)


# ======================================================================
# 旧版 JSON 反序列化（保留以兼容 migrate_from_json）
# ======================================================================
def _trajectory_from_json(text: str) -> Trajectory:
    d = json.loads(text)
    steps = [Step(**s) for s in d.pop("steps", [])]
    cot = StructuredCoT(**d.pop("structured_cot", {}))
    env = EnvironmentSnapshot(attributes=d.pop("env_snapshot", {}).get("attributes", {}))
    return Trajectory(**d, steps=steps, structured_cot=cot, env_snapshot=env)


def _record_from_json(text: str) -> ExperienceRecord:
    d = json.loads(text)
    steps = [ParamStep(**s) for s in d.pop("param_steps", [])]
    return ExperienceRecord(**d, param_steps=steps)


def _harness_from_json(text: str) -> Harness:
    d = json.loads(text)
    status = d.pop("status", "active")
    if isinstance(status, str):
        status = HarnessStatus(status)
    ver = d.pop("verification", {})
    return Harness(**d, status=status, verification=VerificationMeta(**ver))


def _stats_from_json(text: str) -> TaskTypeStats:
    return TaskTypeStats(**json.loads(text))


def _save_json(obj, path: Path) -> None:
    """旧版 JSON 保存（仅用于迁移）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_dict(obj), ensure_ascii=False, indent=2))


class Repository:
    """四层经验仓库，SQLite 主存储 + 内存缓存。

    公共 API 与旧版 JSON Repository 完全一致，底层持久化切换为 SQLite。
    ``_trajectories`` / ``_harnesses`` 等 dict 保持内存缓存，供
    compiler.py / runtime.py 直接访问。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_dirs()
        # SQLite 存储层
        self.storage = Storage(config)
        # 内存缓存（向后兼容 compiler.py / runtime.py 直接访问）
        self._trajectories: dict[str, Trajectory] = {}
        self._records: dict[str, ExperienceRecord] = {}
        self._harnesses: dict[str, Harness] = {}
        self._stats: dict[str, TaskTypeStats] = {}
        self._load()

    # ==================================================================
    # 加载：从 SQLite 读取到内存缓存
    # ==================================================================
    def _load(self) -> None:
        # Layer 0: trajectories
        for row in self.storage.load_trajectories():
            try:
                t = _trajectory_from_row(row)
                self._trajectories[t.id] = t
            except Exception as exc:
                log.warning("加载 trajectory %s 失败: %s", row.get("id", "?"), exc)

        # Layer 1: records
        for row in self.storage.load_records():
            try:
                r = _record_from_row(row)
                self._records[r.id] = r
            except Exception as exc:
                log.warning("加载 record %s 失败: %s", row.get("id", "?"), exc)

        # Layer 2: harnesses
        for row in self.storage.load_harnesses():
            try:
                h = _harness_from_row(row)
                self._harnesses[h.id] = h
            except Exception as exc:
                log.warning("加载 harness %s 失败: %s", row.get("id", "?"), exc)

        # Layer 3: stats — SQLite stats 表存的是 stats_json
        for row in self.storage.load_all_stats():
            try:
                s = _stats_from_row(row)
                self._stats[s.task_type] = s
            except Exception as exc:
                log.warning("加载 stats 失败: %s", exc)

        log.info(
            "从 SQLite 加载: %d trajectories, %d records, %d harnesses, %d stats",
            len(self._trajectories), len(self._records),
            len(self._harnesses), len(self._stats),
        )

    # ==================================================================
    # Layer 0 — trajectories
    # ==================================================================
    def add_trajectory(self, t: Trajectory) -> None:
        self._trajectories[t.id] = t
        self.storage.save_trajectory(t)
        log.debug("记录 trajectory %s (task=%s, outcome=%s)", t.id, t.task_type, t.outcome)

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
        self.storage.save_record(r)

    def records_for_type(self, task_type: str) -> list[ExperienceRecord]:
        return [r for r in self._records.values() if r.task_type == task_type]

    # ==================================================================
    # Layer 2 — harnesses / version DAG
    # ==================================================================
    def add_harness(self, h: Harness) -> None:
        self._harnesses[h.id] = h
        self.storage.save_harness(h, embedding=h.embedding)
        stats = self.get_stats(h.task_type)
        stats.current_harness_id = h.id
        stats.last_induction_time = h.created_at
        self.save_stats(h.task_type)

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
            h.updated_at = time.time()
            self.storage.save_harness(h, embedding=h.embedding)

    # --- failure tracking ---------------------------------------------
    def record_failure(self, hid: str, ftype: FailureType) -> None:
        h = self._harnesses.get(hid)
        if not h:
            return
        key = ftype.value
        h.failure_counts[key] = h.failure_counts.get(key, 0) + 1
        h.updated_at = time.time()
        self.storage.save_harness(h, embedding=h.embedding)

    # ==================================================================
    # Layer 3 — task-type statistics
    # ==================================================================
    def get_stats(self, task_type: str) -> TaskTypeStats:
        if task_type not in self._stats:
            self._stats[task_type] = TaskTypeStats(task_type=task_type)
        return self._stats[task_type]

    def save_stats(self, task_type: str) -> None:
        if task_type in self._stats:
            self.storage.save_stats(task_type, _as_dict(self._stats[task_type]))

    # ==================================================================
    # queries used by the induction trigger
    # ==================================================================
    def support_count(self, task_type: str) -> int:
        """Number of *successful agent* trajectories of this type."""
        return len(self.trajectories_for_type(task_type, success_only=True))

    def all_task_types(self) -> list[str]:
        return list({t.task_type for t in self._trajectories.values() if t.task_type})

    # ==================================================================
    # 迁移：从旧版 JSON 文件导入 SQLite
    # ==================================================================
    def migrate_from_json(self) -> dict:
        """从旧版 JSON 文件迁移数据到 SQLite（一次性）。"""
        return self.storage.migrate_from_json(self.config.data_dir)
