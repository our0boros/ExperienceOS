from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from experience_os.experiments.runner import (
    CrossDomainSplitPolicy,
    ExperimentConfig,
    ExperimentMetrics,
    ExperimentRunner,
    MetricsRecorder,
    ReplaySplitPolicy,
    RunResult,
    TaskBundle,
    TypeSplitPolicy,
    run_experiment_v2,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_task_result(idx=1, phase="eval", task_id="t-1", task_type="lookup",
                      method="react", success=True, reward=1.0, tokens=1000,
                      latency=10.0, path="agent", error="",
                      prompt_tokens=900, completion_tokens=100):
    return SimpleNamespace(
        idx=idx, phase=phase, task_id=task_id, task_type=task_type,
        method=method, success=success, reward=reward, tokens=tokens,
        latency=latency, path=path, error=error,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


def _make_tau2_task(task_id="1", action_name="get_order_details"):
    return SimpleNamespace(
        id=task_id,
        evaluation_criteria=SimpleNamespace(
            actions=[SimpleNamespace(name=action_name,
                                     arguments={}, requestor="assistant")],
        ),
    )


# ── TaskBundle ───────────────────────────────────────────────────────


def test_task_bundle_counts_tasks():
    tasks = [_make_tau2_task(str(i)) for i in range(5)]
    bundle = TaskBundle(tasks=tasks, domain="retail")
    assert bundle.total_count == 5
    assert bundle.domain == "retail"


# ── TypeSplitPolicy ──────────────────────────────────────────────────


@patch("experience_os.tau2_adapter.split_tasks")
def test_type_split_policy(mock_split):
    mock_split.return_value = (
        ["t1", "t2", "t3"],  # warmup
        ["t4", "t5", "t6", "t7"],  # eval
        {"get_order_details": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]},
    )
    tasks = [_make_tau2_task(str(i)) for i in range(7)]
    bundle = TaskBundle(tasks=tasks, domain="retail",
                        task_types=["get_order_details"])
    result = TypeSplitPolicy().split(bundle, warmup=3, eval_size=4)
    assert result.policy == "type_split"
    assert result.metadata["selected_type"] == "get_order_details"
    assert result.metadata["selected_total"] == 7


@patch("experience_os.tau2_adapter.split_tasks")
def test_type_split_policy_no_groups(mock_split):
    mock_split.return_value = ([], [], {})
    tasks = [_make_tau2_task(str(i)) for i in range(5)]
    bundle = TaskBundle(tasks=tasks, domain="retail")
    result = TypeSplitPolicy().split(bundle, warmup=2, eval_size=3)
    assert len(result.warmup) == 2
    assert len(result.eval) == 3


# ── ReplaySplitPolicy ────────────────────────────────────────────────


def test_replay_split_policy_reuses_tasks():
    tasks = [_make_tau2_task(str(i)) for i in range(10)]
    bundle = TaskBundle(tasks=tasks, domain="retail")
    result = ReplaySplitPolicy().split(bundle, warmup=3, eval_size=5)
    assert result.policy == "replay"
    assert len(result.warmup) == 3
    assert len(result.eval) == 5
    # 可能重叠（前5个和前3个）
    assert result.warmup[0] is result.eval[0]


# ── CrossDomainSplitPolicy ───────────────────────────────────────────


@patch("experience_os.tau2_adapter.split_tasks")
@patch("experience_os.experiments.runner.Tau2TaskSource")
def test_cross_domain_split_policy_stub(mock_source_cls, mock_split):
    mock_split.return_value = (
        ["w1", "w2", "w3"],   # warmup
        ["e1", "e2", "e3", "e4"],  # eval
        {"lookup": ["w1", "w2", "w3", "e1", "e2", "e3", "e4"]},
    )
    mock_source_cls.return_value.load.return_value = TaskBundle(
        tasks=["cd1", "cd2", "cd3", "cd4", "cd5"],
        domain="airline",
    )
    tasks = [_make_tau2_task(str(i)) for i in range(10)]
    bundle = TaskBundle(tasks=tasks, domain="retail")
    result = CrossDomainSplitPolicy().split(
        bundle, warmup=3, eval_size=4, cross_domain="airline",
    )
    assert result.policy == "cross_domain"
    assert result.metadata["cross_domain"] == "airline"
    assert result.metadata["target_domain"] == "retail"


@patch("experience_os.tau2_adapter.split_tasks")
def test_cross_domain_falls_back_to_type_split_when_no_cross_domain(mock_split):
    mock_split.return_value = (
        ["t1", "t2"], ["t3", "t4", "t5"],
        {"lookup": ["t1", "t2", "t3", "t4", "t5"]},
    )
    tasks = [_make_tau2_task(str(i)) for i in range(5)]
    bundle = TaskBundle(tasks=tasks, domain="retail")
    result = CrossDomainSplitPolicy().split(
        bundle, warmup=2, eval_size=3, cross_domain="",
    )
    assert result.policy in ("type_split", "")


# ── MetricsRecorder ──────────────────────────────────────────────────


def test_metrics_recorder_basic():
    results = [
        _make_task_result(idx=1, phase="warmup", success=True, tokens=500, path="agent"),
        _make_task_result(idx=2, phase="warmup", success=True, tokens=600, path="agent"),
        _make_task_result(idx=3, phase="eval", success=True, tokens=400, path="harness"),
        _make_task_result(idx=4, phase="eval", success=False, tokens=800, path="harness+agent"),
    ]
    run = RunResult(method="coe", results=results, experiment_id="exp-1")
    m = MetricsRecorder.record(run, model="deepseek-v4", domain="retail")

    assert m.method == "coe"
    assert m.total_tasks == 4
    assert m.successes == 3
    assert m.success_rate == 0.75
    assert m.total_tokens == 2300
    assert m.warmup_sr == 1.0
    assert m.eval_sr == 0.5
    assert m.path_distribution == {"agent": 2, "harness": 1, "harness+agent": 1}
    assert m.harness_hit_rate == 0.5  # 2 out of 4
    assert m.fallback_rate == 0.25     # 1 out of 4


def test_metrics_recorder_empty():
    run = RunResult(method="vanilla", results=[], experiment_id="exp-e")
    m = MetricsRecorder.record(run)
    assert m.total_tasks == 0
    assert m.success_rate == 0.0


def test_metrics_recorder_to_dict():
    m = ExperimentMetrics(method="react", model="gpt-4", domain="retail",
                          total_tasks=10, successes=8,
                          success_rate=0.8,  # computed by record(), set explicitly
                          total_tokens=5000,
                          path_distribution={"agent": 10})
    d = m.to_dict()
    assert d["method"] == "react"
    assert d["success_rate"] == 0.8
    assert d["path_distribution"] == {"agent": 10}


# ── ExperimentConfig ─────────────────────────────────────────────────


def test_experiment_config_defaults():
    cfg = ExperimentConfig()
    assert cfg.method == "react"
    assert cfg.domain == "retail"
    assert cfg.warmup == 3
    assert cfg.eval_size == 5
    assert cfg.split_policy == "type_split"
    assert cfg.mode == "deployment"


# ── ExperimentRunner construction ────────────────────────────────────


def test_experiment_runner_registry_has_all_methods():
    assert "vanilla" in ExperimentRunner.METHOD_RUNNERS
    assert "react" in ExperimentRunner.METHOD_RUNNERS
    assert "skillopt" in ExperimentRunner.METHOD_RUNNERS
    assert "coe" in ExperimentRunner.METHOD_RUNNERS


def test_experiment_runner_registry_has_all_splits():
    assert "type_split" in ExperimentRunner.SPLIT_POLICIES
    assert "train_test" in ExperimentRunner.SPLIT_POLICIES
    assert "replay" in ExperimentRunner.SPLIT_POLICIES
    assert "cross_domain" in ExperimentRunner.SPLIT_POLICIES


def test_experiment_runner_lazy_builds_components():
    cfg = ExperimentConfig(method="vanilla", model="test-model")
    runner = ExperimentRunner(cfg)
    # 在 execute() 前访问属性应自动构建
    assert runner.method_runner is not None
    assert runner.method_runner.model == "test-model"
    assert runner.task_source is not None


# ── run_experiment_v2 exists ─────────────────────────────────────────


def test_run_experiment_v2_is_callable():
    # 只验证函数存在且可调用（不实际运行实验）
    assert callable(run_experiment_v2)
