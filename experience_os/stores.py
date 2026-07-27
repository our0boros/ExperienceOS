"""明确 raw trace、经验记录和 artifact 之间边界的存储 facade。"""

from __future__ import annotations

import json
from typing import Any, Protocol

from experience_os.experience_library import (
    ExperienceLibrary,
    SubStepRecord,
    TrajectoryRecord,
)


class TraceStore(Protocol):
    """只保存原始执行事实；写入轨迹不会触发归纳或 artifact 生成。"""

    def append(self, trajectory: TrajectoryRecord) -> int: ...

    def append_substep(self, substep: SubStepRecord) -> int: ...

    def query(self, **filters: Any) -> list[dict]: ...


class ExperienceStore(Protocol):
    """保存受控归纳结果及子步骤统计，不直接把 raw trace 变成 artifact。"""

    def log_record(self, **kwargs: Any) -> int: ...

    def get_records(self, task_type: str, current_only: bool = True) -> list[dict]: ...

    def aggregate_substep_patterns(self, **kwargs: Any) -> list[dict]: ...

    def consolidate_substeps(
        self, experiment_id: str = "", min_support: int = 3,
        success_only: bool = True, max_candidates: int = 50,
        min_success_rate: float = 0.5, min_bayesian_score: float = 0.5,
    ) -> list[dict]: ...

    def compact_records(self, task_type: str = "", max_records: int = 50) -> list[dict]: ...

    def candidate_stats(self) -> dict: ...

    def record_candidate(self, **kwargs: Any) -> int: ...


class ArtifactStore(Protocol):
    """只保存显式提交的编译产物。"""

    def log_artifact(self, **kwargs: Any) -> int: ...

    def get_artifacts(self, **kwargs: Any) -> list[dict]: ...


class _LibraryFacade:
    def __init__(self, library: ExperienceLibrary) -> None:
        self.library = library

    def close(self) -> None:
        self.library.close()


class LibraryTraceStore(_LibraryFacade):
    """基于现有 SQLite ``ExperienceLibrary`` 的 TraceStore 实现。"""

    def append(self, trajectory: TrajectoryRecord) -> int:
        return self.library.log_trajectory(trajectory)

    def append_substep(self, substep: SubStepRecord) -> int:
        return self.library.log_substep(substep)

    def query(self, **filters: Any) -> list[dict]:
        return self.library.query_trajectories(**filters)


class LibraryExperienceStore(_LibraryFacade):
    """基于 SQLite 的受控经验归纳 facade。

    聚合方法只读取底层事实并返回 bounded candidates；只有显式调用
    ``record_candidate`` 才会写入 records，artifact 始终由 ArtifactStore 写入。
    """

    def __init__(self, library: ExperienceLibrary) -> None:
        super().__init__(library)
        self._last_candidate_stats: dict[str, int] = {
            "candidate_count": 0,
            "accepted_count": 0,
            "dedup_count": 0,
            "filtered_count": 0,
        }

    def log_record(self, **kwargs: Any) -> int:
        return self.library.log_record(**kwargs)

    def get_records(self, task_type: str, current_only: bool = True) -> list[dict]:
        return self.library.get_records(task_type, current_only=current_only)

    def aggregate_substep_patterns(self, **kwargs: Any) -> list[dict]:
        return self.library.aggregate_substep_patterns(**kwargs)

    def consolidate_substeps(
        self, experiment_id: str = "", min_support: int = 3,
        success_only: bool = True, max_candidates: int = 50,
        min_success_rate: float = 0.5, min_bayesian_score: float = 0.5,
    ) -> list[dict]:
        """Return bounded, read-only sub-step pattern candidates.

        A support unit is a distinct trajectory.  Duplicate observations from
        one trajectory are evidence for the same pattern, not extra support.
        Failed sub-steps remain in the statistics when ``success_only`` is
        false, but are never silently persisted as experience records.
        """
        if max_candidates <= 0:
            self._last_candidate_stats = {
                "candidate_count": 0, "accepted_count": 0,
                "dedup_count": 0, "filtered_count": 0,
            }
            return []

        sql = """SELECT trajectory_id, experiment_id, intent, tool_name, effect,
                         success, parent_task_success, meta_json
                  FROM substeps WHERE 1=1"""
        params: list[Any] = []
        if experiment_id:
            sql += " AND experiment_id=?"
            params.append(experiment_id)
        rows = self.library.conn.execute(sql, params).fetchall()

        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        raw_observations = len(rows)
        duplicate_observations = 0
        for trajectory_id, exp_id, intent, tool_name, effect, success, parent_ok, meta_json in rows:
            key = ((intent or "").strip(), (tool_name or "").strip(), (effect or "").strip())
            if not key[0] or not key[1]:
                continue
            group = grouped.setdefault(key, {
                "intent": key[0], "tool_name": key[1], "effect": key[2],
                "experiment_ids": set(), "trajectory_ids": set(),
                "successful_trajectory_ids": set(),
                "successful_full_task_ids": set(), "total_full_task_ids": set(),
                # Phase A: 预测契约统计
                "prediction_accuracies": [],
                "quality_labels": [],
            })
            group["experiment_ids"].add(exp_id)

            # Phase A: 解析预测契约元数据
            pred_acc = 1.0
            qual_label = ""
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                    if isinstance(meta, dict):
                        pred_acc = float(meta.get("prediction_accuracy", 1.0))
                        qual_label = meta.get("quality_label", "")
                except Exception:
                    pass
            group["prediction_accuracies"].append(pred_acc)
            if qual_label:
                group["quality_labels"].append(qual_label)

            if trajectory_id in group["trajectory_ids"]:
                duplicate_observations += 1
                if success:
                    group["successful_trajectory_ids"].add(trajectory_id)
                if parent_ok:
                    group["successful_full_task_ids"].add(trajectory_id)
                continue
            group["trajectory_ids"].add(trajectory_id)
            group["total_full_task_ids"].add(trajectory_id)
            if success:
                group["successful_trajectory_ids"].add(trajectory_id)
            if parent_ok:
                group["successful_full_task_ids"].add(trajectory_id)

        candidates: list[dict[str, Any]] = []
        filtered_count = 0
        for group in grouped.values():
            evidence_ids = sorted(group["trajectory_ids"])
            successful_ids = sorted(group["successful_trajectory_ids"])
            support_ids = successful_ids if success_only else evidence_ids
            support = len(support_ids)
            total = len(evidence_ids)
            successful_full = len(group["successful_full_task_ids"])
            success_rate = len(successful_ids) / total if total else 0.0
            bayesian_score = (1.0 + successful_full) / (2.0 + total) if total else 0.0

            # Phase A: 预测准确率调整贝叶斯评分
            pred_accs = group["prediction_accuracies"]
            avg_pred_acc = sum(pred_accs) / len(pred_accs) if pred_accs else 1.0
            # 调整：预测准确率作为质量乘数（0.5–1.0 范围，不惩罚过重）
            pred_multiplier = 0.5 + 0.5 * avg_pred_acc
            adjusted_bayesian = bayesian_score * pred_multiplier

            # 质量标签分布
            from collections import Counter
            label_counts = Counter(group["quality_labels"])
            lucky_ratio = label_counts.get("lucky_success", 0) / max(1, len(pred_accs))

            # 侥幸成功过多 → 降低评分
            if lucky_ratio > 0.3:
                adjusted_bayesian *= 0.7

            if (
                support < min_support
                or success_rate < min_success_rate
                or adjusted_bayesian < min_bayesian_score
            ):
                filtered_count += 1
                continue
            candidates.append({
                "intent": group["intent"],
                "tool_name": group["tool_name"],
                "action_name": group["tool_name"],
                "effect": group["effect"],
                "support_count": support,
                "success_count": len(successful_ids),
                "success_rate": success_rate,
                "success_in_full_tasks": successful_full,
                "total_appearances": total,
                "bayesian_score": bayesian_score,
                "adjusted_bayesian": adjusted_bayesian,
                "score": adjusted_bayesian,
                "prediction_accuracy": avg_pred_acc,
                "quality_label_dist": dict(label_counts),
                "source": "substeps",
                "evidence": {
                    "trajectory_ids": support_ids,
                    "all_trajectory_ids": evidence_ids,
                    "experiment_ids": sorted(group["experiment_ids"]),
                },
                "reason": (
                    f"support={support} distinct trajectories; "
                    f"success_rate={success_rate:.3f}; "
                    f"bayesian={bayesian_score:.3f} → adjusted={adjusted_bayesian:.3f}"
                    f" (pred_acc={avg_pred_acc:.2f}, lucky={lucky_ratio:.2f})"
                ),
            })

        candidates.sort(
            key=lambda item: (item["score"], item["support_count"], item["intent"]),
            reverse=True,
        )
        candidate_count = len(grouped)
        overflow = max(0, len(candidates) - max_candidates)
        candidates = candidates[:max_candidates]
        self._last_candidate_stats = {
            "candidate_count": candidate_count,
            "accepted_count": len(candidates),
            "dedup_count": duplicate_observations,
            "filtered_count": filtered_count + overflow,
            "raw_count": raw_observations,
        }
        return candidates

    def compact_records(self, task_type: str = "", max_records: int = 50) -> list[dict]:
        """Return a read-only view merging equivalent records' evidence."""
        if max_records <= 0:
            self._last_candidate_stats = {
                "candidate_count": 0, "accepted_count": 0,
                "dedup_count": 0, "filtered_count": 0,
            }
            return []
        sql = "SELECT * FROM records"
        params: list[Any] = []
        if task_type:
            sql += " WHERE task_type=?"
            params.append(task_type)
        sql += " ORDER BY seq DESC"
        rows = self.library.conn.execute(sql, params).fetchall()
        columns = [d[0] for d in self.library.conn.execute(
            "SELECT * FROM records LIMIT 0"
        ).description]

        grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            record = dict(zip(columns, row))
            def decoded(name: str, fallback: Any) -> Any:
                try:
                    value = json.loads(record[name] or "")
                    return value
                except (TypeError, ValueError, json.JSONDecodeError):
                    return fallback

            preconditions = decoded("preconditions_json", {})
            param_steps = decoded("param_steps_json", [])
            invariants = decoded("invariants_json", [])
            source_ids = decoded("source_trajectory_ids_json", [])
            key = (
                record["task_type"], json.dumps(preconditions, sort_keys=True, ensure_ascii=False),
                json.dumps(param_steps, sort_keys=True, ensure_ascii=False),
                json.dumps(invariants, sort_keys=True, ensure_ascii=False),
                record["terminal_verifier"] or "",
            )
            group = grouped.setdefault(key, {
                "task_type": record["task_type"],
                "preconditions": preconditions, "param_steps": param_steps,
                "invariants": invariants, "terminal_verifier": record["terminal_verifier"] or "",
                "record_ids": [], "trajectory_ids": set(), "support_count": 0,
            })
            group["record_ids"].append(record["seq"])
            group["trajectory_ids"].update(source_ids if isinstance(source_ids, list) else [])
            group["support_count"] = max(group["support_count"], record["support_count"] or 0)

        candidates = []
        for group in grouped.values():
            evidence_ids = sorted(group["trajectory_ids"])
            candidates.append({
                **{key: value for key, value in group.items() if key not in {"trajectory_ids"}},
                "source": "records",
                "evidence": {"trajectory_ids": evidence_ids, "record_ids": group["record_ids"]},
                "score": float(group["support_count"]),
                "reason": f"merged {len(group['record_ids'])} equivalent record(s)",
            })
        candidates.sort(key=lambda item: (item["score"], item["task_type"]), reverse=True)
        overflow = max(0, len(candidates) - max_records)
        candidates = candidates[:max_records]
        self._last_candidate_stats = {
            "candidate_count": len(grouped),
            "accepted_count": len(candidates),
            "dedup_count": len(rows) - len(grouped),
            "filtered_count": overflow,
            "raw_count": len(rows),
        }
        return candidates

    def candidate_stats(self) -> dict[str, int]:
        """Return statistics for the most recent read-only candidate operation."""
        return dict(self._last_candidate_stats)

    def record_candidate(self, **kwargs: Any) -> int:
        """将显式筛选/归纳出的 candidate 记录为 experience record。

        这是 raw trace 进入受控经验层的边界；它只写入中层 record，绝不
        自动创建或提交 artifact。调用方必须显式调用 ``ArtifactStore``。
        """
        return self.library.log_record(**kwargs)

    def consolidate(self, **kwargs: Any) -> int:
        """显式 consolidate 入口，当前委托 ``record_candidate`` 持久化候选。"""
        return self.record_candidate(**kwargs)


class LibraryArtifactStore(_LibraryFacade):
    """基于现有 SQLite ``ExperienceLibrary`` 的 ArtifactStore 实现。"""

    def log_artifact(self, **kwargs: Any) -> int:
        return self.library.log_artifact(**kwargs)

    def get_artifacts(self, **kwargs: Any) -> list[dict]:
        return self.library.get_artifacts(**kwargs)


def stores_for(library: ExperienceLibrary) -> tuple[LibraryTraceStore, LibraryExperienceStore, LibraryArtifactStore]:
    """为同一个数据库返回三层 facade；不会创建新的 SQLite schema。"""
    return (
        LibraryTraceStore(library),
        LibraryExperienceStore(library),
        LibraryArtifactStore(library),
    )


__all__ = [
    "TraceStore",
    "ExperienceStore",
    "ArtifactStore",
    "LibraryTraceStore",
    "LibraryExperienceStore",
    "LibraryArtifactStore",
    "stores_for",
]
