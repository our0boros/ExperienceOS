"""统一实验运行器协议。

将验证框架从工程内核中分离——验证框架定义"要测什么"
（TaskSource / SplitPolicy / MethodRunner / MetricsRecorder），
工程内核提供"怎么跑"（TraceStore / ExperienceStore / ArtifactStore / Services）。

协议抽象：
    * :class:`TaskSource`   — 加载任务（tau2 / KSI JSONL / 自定义）
    * :class:`SplitPolicy`  — 将任务划分为 warmup / eval
    * :class:`MethodRunner` — 运行单个方法（vanilla / react / skillopt / coe / ksi）
    * :class:`MetricsRecorder` — 记录和聚合实验指标

内建实现：
    * :class:`Tau2TaskSource`      — 从 τ-bench 域加载
    * :class:`TypeSplitPolicy`     — 按 task_type 分组划分
    * :class:`TrainTestSplitPolicy`— tau2 原生 train/test 划分
    * :class:`ReplaySplitPolicy`   — warmup/eval 复用相同任务
    * :class:`CrossDomainSplitPolicy` — 跨域积累
    * :class:`VanillaRunner` / :class:`ReActRunner` / :class:`SkillOptRunner` / :class:`CoERunner`
    * :class:`ExperimentMetrics`   — 可序列化的指标记录
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from experience_os.experience_library import (
    ExperienceLibrary,
    TrajectoryRecord,
    serialize_messages,
)
from experience_os.stores import TraceStore, stores_for

log = logging.getLogger(__name__)


# ======================================================================
# 运行模式
# ======================================================================


class ExperimentMode(str, Enum):
    """实验模式 — 实验层的一等公民，不是 runtime 副产物。

    * **ACCUMULATION** — 只采集轨迹，不使用 artifact（warmup 专属）
    * **ONLINE_ACCUMULATION** — 边采集边构建边使用（在线学习）
    * **DEPLOYMENT** — 只使用预先积累好的 artifact（eval 专属）
    """

    ACCUMULATION = "accumulation"
    ONLINE_ACCUMULATION = "online_accumulation"
    DEPLOYMENT = "deployment"


# ======================================================================
# TaskSource
# ======================================================================


@dataclass
class TaskBundle:
    """TaskSource 返回的任务包，含元数据。"""

    tasks: list[Any]
    domain: str = ""
    task_types: list[str] = field(default_factory=list)
    total_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.total_count = len(self.tasks)


class TaskSource(ABC):
    """抽象任务源：加载任务列表。"""

    @abstractmethod
    def load(self, **kwargs: Any) -> TaskBundle:
        """加载任务并返回 TaskBundle。"""
        ...


class Tau2TaskSource(TaskSource):
    """从 τ-bench 域加载任务。

    Args:
        domain: τ-bench domain（retail / airline）
        split: "base" | "train" | "test"
    """

    def __init__(self, domain: str = "retail", split: str = "base") -> None:
        self.domain = domain
        self.split = split

    def load(self, **kwargs: Any) -> TaskBundle:
        from experience_os.tau2_adapter import infer_task_type

        domain = kwargs.get("domain", self.domain)
        split = kwargs.get("split", self.split)

        mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
        tasks = mod.get_tasks(split)

        task_types = sorted({infer_task_type(t) for t in tasks})
        return TaskBundle(
            tasks=tasks,
            domain=domain,
            task_types=task_types,
            metadata={"split": split, "source": f"tau2-bench/{domain}"},
        )


class Tau2TrainTestSource(TaskSource):
    """加载 τ-bench 原生 train/test split。

    Returns:
        TaskBundle with metadata["train"] and metadata["test"] lists.
    """

    def __init__(self, domain: str = "retail") -> None:
        self.domain = domain

    def load(self, **kwargs: Any) -> TaskBundle:
        from experience_os.experiments.compare import load_train_test_split

        domain = kwargs.get("domain", self.domain)
        train, test = load_train_test_split(domain)
        from experience_os.tau2_adapter import infer_task_type

        all_task_types = sorted({infer_task_type(t) for t in (train + test)})
        return TaskBundle(
            tasks=train + test,
            domain=domain,
            task_types=all_task_types,
            metadata={
                "split": "train_test",
                "source": f"tau2-bench/{domain}",
                "train": train,
                "test": test,
                "train_count": len(train),
                "test_count": len(test),
            },
        )


# ======================================================================
# SplitPolicy
# ======================================================================


@dataclass
class SplitResult:
    """划分结果。"""

    warmup: list[Any]
    eval: list[Any]
    policy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SplitPolicy(ABC):
    """抽象划分策略。"""

    @abstractmethod
    def split(self, bundle: TaskBundle, **kwargs: Any) -> SplitResult:
        """将 TaskBundle 划分为 warmup / eval。"""
        ...


class TypeSplitPolicy(SplitPolicy):
    """按 task_type 分组划分。

    同类任务前 K 个做 warmup，剩余做 eval。
    不足 min_support 的类型全部放入 warmup。
    """

    def split(self, bundle: TaskBundle, **kwargs: Any) -> SplitResult:
        warmup_count = kwargs.get("warmup", 3)
        eval_count = kwargs.get("eval_size", 5)
        task_type = kwargs.get("task_type", "")
        from experience_os.tau2_adapter import split_tasks

        min_support = max(warmup_count, 3)
        _, _, groups = split_tasks(bundle.tasks, min_support=min_support)
        if not groups:
            return SplitResult(
                warmup=bundle.tasks[:warmup_count],
                eval=bundle.tasks[warmup_count : warmup_count + eval_count],
                policy="type_split",
                metadata={"groups": {}, "selected_type": "unknown"},
            )

        # 选最大组或指定类型
        if task_type and task_type in groups:
            selected = groups[task_type]
        else:
            selected = max(groups.values(), key=len)
            task_type = max(groups, key=lambda k: len(groups[k]))

        return SplitResult(
            warmup=selected[:warmup_count],
            eval=selected[warmup_count : warmup_count + eval_count],
            policy="type_split",
            metadata={
                "groups": {k: len(v) for k, v in groups.items()},
                "selected_type": task_type,
                "selected_total": len(selected),
            },
        )


class TrainTestSplitPolicy(SplitPolicy):
    """tau2 原生 train/test 划分。

    需要 TaskBundle.metadata 中有 "train" 和 "test" 键。
    """

    def split(self, bundle: TaskBundle, **kwargs: Any) -> SplitResult:
        warmup_count = kwargs.get("warmup", 3)
        eval_count = kwargs.get("eval_size", 5)
        task_type = kwargs.get("task_type", "")

        train = bundle.metadata.get("train", bundle.tasks)
        test = bundle.metadata.get("test", bundle.tasks)

        from experience_os.tau2_adapter import infer_task_type

        if task_type:
            train = [t for t in train if infer_task_type(t) == task_type]
            test = [t for t in test if infer_task_type(t) == task_type]

        return SplitResult(
            warmup=train[:warmup_count],
            eval=test[:eval_count],
            policy="train_test",
            metadata={
                "train_total": len(bundle.metadata.get("train", [])),
                "test_total": len(bundle.metadata.get("test", [])),
                "task_type": task_type or "all",
                "filtered_train": len(train),
                "filtered_test": len(test),
            },
        )


class ReplaySplitPolicy(SplitPolicy):
    """warmup/eval 复用相同任务（回放验证）。"""

    def split(self, bundle: TaskBundle, **kwargs: Any) -> SplitResult:
        warmup_count = kwargs.get("warmup", 3)
        eval_count = kwargs.get("eval_size", 5)
        return SplitResult(
            warmup=bundle.tasks[:warmup_count],
            eval=bundle.tasks[:eval_count],
            policy="replay",
            metadata={"note": "eval reuses warmup tasks"},
        )


class CrossDomainSplitPolicy(SplitPolicy):
    """跨域积累：warmup 来自 cross_domain，eval 来自本域。"""

    def split(self, bundle: TaskBundle, **kwargs: Any) -> SplitResult:
        warmup_count = kwargs.get("warmup", 3)
        eval_count = kwargs.get("eval_size", 5)
        cross_domain = kwargs.get("cross_domain", "")
        task_type = kwargs.get("task_type", "")

        if not cross_domain:
            # 回退到 type_split
            return TypeSplitPolicy().split(bundle, **kwargs)

        cd_source = Tau2TaskSource(domain=cross_domain)
        cd_bundle = cd_source.load()
        from experience_os.tau2_adapter import split_tasks

        _, _, cd_groups = split_tasks(cd_bundle.tasks, min_support=warmup_count)
        _, _, local_groups = split_tasks(bundle.tasks, min_support=warmup_count)

        cd_selected = max(cd_groups.values(), key=len) if cd_groups else cd_bundle.tasks
        groups = local_groups
        local_selected = (
            groups[task_type]
            if task_type and task_type in groups
            else max(groups.values(), key=len) if groups else bundle.tasks
        )

        return SplitResult(
            warmup=cd_selected[:warmup_count],
            eval=local_selected[:eval_count],
            policy="cross_domain",
            metadata={
                "cross_domain": cross_domain,
                "target_domain": bundle.domain,
                "cd_task_count": len(cd_bundle.tasks),
            },
        )


# ======================================================================
# MethodRunner
# ======================================================================


@dataclass
class RunResult:
    """单个方法运行的结果。"""

    method: str
    results: list[Any]  # list[TaskResult]
    experiment_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MethodRunner(ABC):
    """抽象方法运行器。"""

    @abstractmethod
    def run(
        self, warmup: list[Any], eval: list[Any], **kwargs: Any
    ) -> RunResult:
        """运行方法并返回结果。"""
        ...


class VanillaRunner(MethodRunner):
    """Vanilla LLM baseline runner。"""

    def __init__(self, model: str, domain: str, max_steps: int = 30,
                 solo_mode: bool = False, **kwargs: Any) -> None:
        self.model = model
        self.domain = domain
        self.max_steps = max_steps
        self.solo_mode = solo_mode

    def run(self, warmup: list[Any], eval: list[Any], **kwargs: Any) -> RunResult:
        from experience_os.experiments.compare import run_vanilla, ExperimentResult

        inter_task_delay = kwargs.get("inter_task_delay", 0.0)
        experiment_id = kwargs.get("experiment_id", "")
        eid = experiment_id or f"vanilla-{self.domain}-{uuid.uuid4().hex[:8]}"

        results = []
        stream = list(warmup) + list(eval)
        warmup_n = len(warmup)
        for i, task in enumerate(stream, 1):
            phase = "warmup" if i <= warmup_n else "eval"
            r = run_vanilla(task, self.domain, self.model, self.max_steps,
                            self.solo_mode, seed=42 + i)
            r.idx = i
            r.phase = phase
            results.append(r)
            if inter_task_delay and i < len(stream):
                time.sleep(inter_task_delay)

        return RunResult(method="vanilla", results=results, experiment_id=eid)


class ReActRunner(MethodRunner):
    """ReAct agent baseline runner。"""

    def __init__(self, model: str, domain: str, max_steps: int = 30,
                 solo_mode: bool = False, **kwargs: Any) -> None:
        self.model = model
        self.domain = domain
        self.max_steps = max_steps
        self.solo_mode = solo_mode

    def run(self, warmup: list[Any], eval: list[Any], **kwargs: Any) -> RunResult:
        from experience_os.experiments.compare import run_react, ExperimentResult

        inter_task_delay = kwargs.get("inter_task_delay", 0.0)
        experiment_id = kwargs.get("experiment_id", "")
        eid = experiment_id or f"react-{self.domain}-{uuid.uuid4().hex[:8]}"

        results = []
        stream = list(warmup) + list(eval)
        warmup_n = len(warmup)
        for i, task in enumerate(stream, 1):
            phase = "warmup" if i <= warmup_n else "eval"
            r = run_react(task, self.domain, self.model, self.max_steps,
                          self.solo_mode, seed=42 + i)
            r.idx = i
            r.phase = phase
            results.append(r)
            if inter_task_delay and i < len(stream):
                time.sleep(inter_task_delay)

        return RunResult(method="react", results=results, experiment_id=eid)


class SkillOptRunner(MethodRunner):
    """SkillOpt baseline runner。"""

    def __init__(self, model: str, domain: str, max_steps: int = 30,
                 solo_mode: bool = False, skill_path: str = "", **kwargs: Any) -> None:
        self.model = model
        self.domain = domain
        self.max_steps = max_steps
        self.solo_mode = solo_mode
        self.skill_path = skill_path

    def run(self, warmup: list[Any], eval: list[Any], **kwargs: Any) -> RunResult:
        from experience_os.experiments.compare import run_skillopt, ExperimentResult

        inter_task_delay = kwargs.get("inter_task_delay", 0.0)
        experiment_id = kwargs.get("experiment_id", "")
        eid = experiment_id or f"skillopt-{self.domain}-{uuid.uuid4().hex[:8]}"

        skill_path = kwargs.get("skill_path", self.skill_path)
        if not skill_path:
            skill_path = "SkillOpt/skillopt/envs/tau2/skills/initial.md"
        skill_text = Path(skill_path).read_text(encoding="utf-8") if Path(skill_path).exists() else ""

        results = []
        stream = list(warmup) + list(eval)
        warmup_n = len(warmup)
        for i, task in enumerate(stream, 1):
            phase = "warmup" if i <= warmup_n else "eval"
            r = run_skillopt(task, self.domain, self.model, self.max_steps,
                             self.solo_mode, skill_text, seed=42 + i)
            r.idx = i
            r.phase = phase
            results.append(r)
            if inter_task_delay and i < len(stream):
                time.sleep(inter_task_delay)

        return RunResult(method="skillopt", results=results, experiment_id=eid)


class CoERunner(MethodRunner):
    """ExperienceOS CoE runner。

    支持两种模式：
    - ``DEPLOYMENT`` / ``ACCUMULATION``：传统 warmup→批量归纳→eval
    - ``ONLINE_ACCUMULATION``：边执行边归纳，每完成一个任务就尝试归纳，
      新 harness 立即可被后续任务使用。所有任务都在一个流中，无 warmup/eval 区分。
    """

    def __init__(self, model: str, domain: str, max_steps: int = 30,
                 solo_mode: bool = False, **kwargs: Any) -> None:
        self.model = model
        self.domain = domain
        self.max_steps = max_steps
        self.solo_mode = solo_mode

    def run(self, warmup: list[Any], eval: list[Any], **kwargs: Any) -> RunResult:
        from experience_os.experiments.compare import run_coe, run_coe_online

        experiment_id = kwargs.get("experiment_id", "")
        skip_validation = kwargs.get("skip_validation", False)
        no_versioning = kwargs.get("no_versioning", False)
        variant = kwargs.get("variant", "type_split")
        mode = kwargs.get("mode", "deployment")
        eid = experiment_id or f"coe-{self.domain}-{variant}-{uuid.uuid4().hex[:8]}"

        lts = ExperienceLibrary.persistent()
        lts_trace_store, _, _ = stores_for(lts)

        if mode == ExperimentMode.ONLINE_ACCUMULATION:
            # 在线模式：所有任务流式处理，边归纳边使用
            all_tasks = list(eval) if eval else list(warmup) + list(eval)
            results = run_coe_online(
                all_tasks, self.domain, self.model,
                self.max_steps, self.solo_mode,
                experiment_id=eid, trace_store=lts_trace_store, library=lts,
            )
        else:
            # 传统模式：warmup 积累 → 批量归纳 → eval 部署
            group = list(warmup) + list(eval)
            results = run_coe(
                group, self.domain, self.model, len(warmup), len(eval),
                self.max_steps, self.solo_mode,
                skip_validation=skip_validation, no_versioning=no_versioning,
                warmup_tasks=warmup, eval_tasks=eval,
                experiment_id=eid, trace_store=lts_trace_store, library=lts,
            )
        lts.close()
        return RunResult(method="coe", results=results, experiment_id=eid)


# ======================================================================
# MetricsRecorder
# ======================================================================


@dataclass
class ExperimentMetrics:
    """实验级别的汇总指标。"""

    method: str
    model: str
    domain: str
    experiment_id: str = ""
    total_tasks: int = 0
    successes: int = 0
    success_rate: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    avg_latency: float = 0.0
    warmup_sr: float = 0.0
    warmup_tokens: int = 0
    eval_sr: float = 0.0
    eval_tokens: int = 0
    path_distribution: dict[str, int] = field(default_factory=dict)
    harness_hit_rate: float = 0.0
    harness_execute_success_rate: float = 0.0
    fallback_rate: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model": self.model,
            "domain": self.domain,
            "experiment_id": self.experiment_id,
            "total_tasks": self.total_tasks,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_latency": round(self.avg_latency, 2),
            "warmup_sr": round(self.warmup_sr, 4),
            "warmup_tokens": self.warmup_tokens,
            "eval_sr": round(self.eval_sr, 4),
            "eval_tokens": self.eval_tokens,
            "path_distribution": self.path_distribution,
            "harness_hit_rate": round(self.harness_hit_rate, 4),
            "harness_execute_success_rate": round(self.harness_execute_success_rate, 4),
            "fallback_rate": round(self.fallback_rate, 4),
        }


class MetricsRecorder:
    """从 RunResult 提取汇总指标。"""

    @staticmethod
    def record(run: RunResult, model: str = "", domain: str = "") -> ExperimentMetrics:
        results = run.results
        if not results:
            return ExperimentMetrics(
                method=run.method, model=model, domain=domain,
                experiment_id=run.experiment_id,
            )

        warmup = [r for r in results if getattr(r, "phase", "") == "warmup"]
        eval_r = [r for r in results if getattr(r, "phase", "") == "eval"]

        # 路径分布
        path_dist: dict[str, int] = {}
        for r in results:
            p = getattr(r, "path", "unknown")
            path_dist[p] = path_dist.get(p, 0) + 1

        total = len(results)
        successes = sum(1 for r in results if getattr(r, "success", False))
        total_tokens = sum(getattr(r, "tokens", 0) for r in results)
        prompt_tok = sum(getattr(r, "prompt_tokens", 0) for r in results)
        completion_tok = sum(getattr(r, "completion_tokens", 0) for r in results)
        avg_lat = sum(getattr(r, "latency", 0.0) for r in results) / total if total else 0.0

        w_sr = sum(1 for r in warmup if getattr(r, "success", False)) / len(warmup) if warmup else 0.0
        w_tok = sum(getattr(r, "tokens", 0) for r in warmup)
        e_sr = sum(1 for r in eval_r if getattr(r, "success", False)) / len(eval_r) if eval_r else 0.0
        e_tok = sum(getattr(r, "tokens", 0) for r in eval_r)

        # harness 相关指标
        harness_attempts = path_dist.get("harness", 0) + path_dist.get("harness+agent", 0)
        harness_only = path_dist.get("harness", 0)
        harness_total = sum(1 for r in results if getattr(r, "path", "") in ("harness", "harness+agent"))
        harness_success = sum(1 for r in results if getattr(r, "path", "") == "harness" and getattr(r, "success", False))

        return ExperimentMetrics(
            method=run.method,
            model=model,
            domain=domain,
            experiment_id=run.experiment_id,
            total_tasks=total,
            successes=successes,
            success_rate=successes / total if total else 0.0,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            avg_latency=avg_lat,
            warmup_sr=w_sr,
            warmup_tokens=w_tok,
            eval_sr=e_sr,
            eval_tokens=e_tok,
            path_distribution=path_dist,
            harness_hit_rate=harness_total / total if total else 0.0,
            harness_execute_success_rate=harness_success / harness_total if harness_total else 0.0,
            fallback_rate=path_dist.get("harness+agent", 0) / total if total else 0.0,
            raw={"results": [getattr(r, "__dict__", {}) for r in results]},
        )


# ======================================================================
# 统一 ExperimentRunner
# ======================================================================


@dataclass
class ExperimentConfig:
    """一次实验的完整配置。

    Args:
        mode: 运行模式（accumulation / online_accumulation / deployment）
        method: 方法（vanilla / react / skillopt / coe）
        domain: τ-bench domain
        model: LLM 模型标识
        warmup: warmup 任务数
        eval_size: eval 任务数
        max_steps: 最大 agent 步数
        split_policy: 划分策略名（type_split / train_test / replay / cross_domain）
        task_type: 筛选特定任务类型（空 = 自动选最大组）
        cross_domain: 跨域积累时的源 domain
        solo_mode: τ-bench solo 模式
        experiment_id: 实验 ID（自动生成）
        skill_path: skillopt 的 skill 文本路径
        skip_validation: 跳过 harness 验证
        no_versioning: 不启用版本管理
        inter_task_delay: 任务间间隔秒数
    """

    mode: str = "deployment"  # ExperimentMode value
    method: str = "react"
    domain: str = "retail"
    model: str = "ollama/qwen2.5:7b"
    warmup: int = 3
    eval_size: int = 5
    max_steps: int = 30
    split_policy: str = "type_split"
    task_type: str = ""
    cross_domain: str = ""
    solo_mode: bool = False
    experiment_id: str = ""
    skill_path: str = ""
    skip_validation: bool = False
    no_versioning: bool = False
    inter_task_delay: float = 0.0


class ExperimentRunner:
    """统一的实验运行器。

    将 TaskSource / SplitPolicy / MethodRunner / MetricsRecorder
    组合为可复用的实验流程。

    用法::

        config = ExperimentConfig(method="coe", domain="retail", warmup=5, eval_size=5)
        runner = ExperimentRunner(config)
        metrics = runner.execute()

    或以组件方式::

        runner = ExperimentRunner(config, task_source=..., split_policy=..., method_runner=...)
        metrics = runner.execute()
    """

    # Split policy 注册表
    SPLIT_POLICIES: dict[str, type[SplitPolicy]] = {
        "type_split": TypeSplitPolicy,
        "train_test": TrainTestSplitPolicy,
        "replay": ReplaySplitPolicy,
        "cross_domain": CrossDomainSplitPolicy,
    }

    # Method runner 工厂
    METHOD_RUNNERS: dict[str, type[MethodRunner]] = {
        "vanilla": VanillaRunner,
        "react": ReActRunner,
        "skillopt": SkillOptRunner,
        "coe": CoERunner,
    }

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        task_source: Optional[TaskSource] = None,
        split_policy: Optional[SplitPolicy] = None,
        method_runner: Optional[MethodRunner] = None,
        metrics_recorder: Optional[MetricsRecorder] = None,
    ) -> None:
        self.config = config
        self._task_source = task_source
        self._split_policy = split_policy
        self._method_runner = method_runner
        self._metrics = metrics_recorder or MetricsRecorder()

    # ── 属性（lazy build from config if not injected）─────────────────

    @property
    def task_source(self) -> TaskSource:
        if self._task_source is None:
            if self.config.split_policy == "train_test":
                self._task_source = Tau2TrainTestSource(domain=self.config.domain)
            elif self.config.split_policy == "cross_domain":
                self._task_source = Tau2TaskSource(domain=self.config.domain)
            else:
                self._task_source = Tau2TaskSource(domain=self.config.domain)
        return self._task_source

    @property
    def split_policy(self) -> SplitPolicy:
        if self._split_policy is None:
            policy_cls = self.SPLIT_POLICIES.get(
                self.config.split_policy, TypeSplitPolicy
            )
            self._split_policy = policy_cls()
        return self._split_policy

    @property
    def method_runner(self) -> MethodRunner:
        if self._method_runner is None:
            runner_cls = self.METHOD_RUNNERS.get(
                self.config.method, ReActRunner
            )
            self._method_runner = runner_cls(
                model=self.config.model,
                domain=self.config.domain,
                max_steps=self.config.max_steps,
                solo_mode=self.config.solo_mode,
                skill_path=self.config.skill_path,
            )
        return self._method_runner

    @property
    def metrics(self) -> MetricsRecorder:
        return self._metrics

    # ── 执行 ────────────────────────────────────────────────────────

    def execute(self) -> ExperimentMetrics:
        """执行完整实验流程并返回指标。"""
        # DeepInfra 自动顺序
        inter_task_delay = self.config.inter_task_delay
        if inter_task_delay == 0.0 and self.config.model.startswith("deepinfra/"):
            inter_task_delay = 3.0

        eid = self.config.experiment_id or (
            f"{self.config.method}-{self.config.domain}"
            f"-{self.config.split_policy}-{uuid.uuid4().hex[:8]}"
        )

        print(f"\n{'='*60}")
        print(f"  实验: {self.config.method}  mode={self.config.mode}")
        print(f"  model={self.config.model}  domain={self.config.domain}")
        print(f"  warmup={self.config.warmup} eval={self.config.eval_size}")
        print(f"  split={self.config.split_policy}  max_steps={self.config.max_steps}")
        print(f"{'='*60}\n")

        # 1. 加载任务
        bundle = self.task_source.load(
            domain=self.config.domain,
            split="base" if self.config.split_policy != "train_test" else "",
        )
        print(f"  已加载 {bundle.total_count} 个任务 ({len(bundle.task_types)} 种类型)")

        # 2. 划分
        split = self.split_policy.split(
            bundle,
            warmup=self.config.warmup,
            eval_size=self.config.eval_size,
            task_type=self.config.task_type,
            cross_domain=self.config.cross_domain,
        )
        print(f"  划分策略: {split.policy}  warmup={len(split.warmup)} eval={len(split.eval)}")

        # 3. 运行
        run = self.method_runner.run(
            list(split.warmup), list(split.eval),
            experiment_id=eid,
            inter_task_delay=inter_task_delay,
            skip_validation=self.config.skip_validation,
            no_versioning=self.config.no_versioning,
            variant=self.config.split_policy,
            skill_path=self.config.skill_path,
            mode=self.config.mode,
        )

        # 4. 记录指标
        metrics = self.metrics.record(run, model=self.config.model, domain=self.config.domain)
        self._print_metrics(metrics)

        # 5. 持久化到 LTS + 实验库
        self._persist(run, eid, bundle)

        return metrics

    # ── 内部 ────────────────────────────────────────────────────────

    def _print_metrics(self, m: ExperimentMetrics) -> None:
        print(f"\n{'='*60}")
        print(f"  汇总: {m.method}")
        print(f"{'='*60}")
        print(f"  总任务:   {m.total_tasks}")
        print(f"  成功率:   {m.successes}/{m.total_tasks} ({m.success_rate:.1%})")
        print(f"  总 Token: {m.total_tokens:,}")
        print(f"  平均延迟: {m.avg_latency:.1f}s")
        print(f"  [Warmup] SR: {m.warmup_sr:.1%}  Token: {m.warmup_tokens:,}")
        print(f"  [Eval]   SR: {m.eval_sr:.1%}  Token: {m.eval_tokens:,}")
        print(f"  路径分布: {m.path_distribution}")
        print(f"  Harness 使用率: {m.harness_hit_rate:.1%}")
        print(f"  experiment_id: {m.experiment_id}")
        print(f"{'='*60}\n")

    def _persist(self, run: RunResult, eid: str, bundle: TaskBundle) -> None:
        try:
            lts = ExperienceLibrary.persistent()
            exp_lib = ExperienceLibrary.experiment(eid)
            lts_trace, _, _ = stores_for(lts)
            exp_trace, _, _ = stores_for(exp_lib)

            for r in run.results:
                rec = TrajectoryRecord(
                    experiment_id=eid,
                    method=run.method,
                    domain=bundle.domain,
                    task_id=getattr(r, "task_id", ""),
                    task_type=getattr(r, "task_type", ""),
                    idx=getattr(r, "idx", 0),
                    phase=getattr(r, "phase", ""),
                    success=getattr(r, "success", False),
                    reward=getattr(r, "reward", 0.0),
                    tokens=getattr(r, "tokens", 0),
                    latency=getattr(r, "latency", 0.0),
                    path=getattr(r, "path", "agent"),
                    messages_json=getattr(r, "messages_json", ""),
                )
                lts_trace.append(rec)
                exp_trace.append(rec)

            print(f"  LTS trajs: {len(lts.query_trajectories(experiment_id=eid))} 条")
            print(f"  实验库: {exp_lib.db_path}")
            lts.close()
            exp_lib.close()
        except Exception as exc:
            log.warning("持久化轨迹失败: %s", exc)


# ======================================================================
# 便捷函数
# ======================================================================


def run_experiment_v2(
    method: str,
    model: str,
    *,
    domain: str = "retail",
    warmup: int = 3,
    eval_size: int = 5,
    max_steps: int = 30,
    mode: str = "deployment",
    split_policy: str = "type_split",
    task_type: str = "",
    cross_domain: str = "",
    solo_mode: bool = False,
    experiment_id: str = "",
    skill_path: str = "",
    skip_validation: bool = False,
    no_versioning: bool = False,
) -> ExperimentMetrics:
    """便捷函数：一行配置即可运行实验。

    这是统一 runner 协议的对外入口，逐步替代 compare.py 中
    的 ``run_experiment()``。
    """
    config = ExperimentConfig(
        mode=mode,
        method=method,
        domain=domain,
        model=model,
        warmup=warmup,
        eval_size=eval_size,
        max_steps=max_steps,
        split_policy=split_policy,
        task_type=task_type,
        cross_domain=cross_domain,
        solo_mode=solo_mode,
        experiment_id=experiment_id,
        skill_path=skill_path,
        skip_validation=skip_validation,
        no_versioning=no_versioning,
    )
    runner = ExperimentRunner(config)
    return runner.execute()
