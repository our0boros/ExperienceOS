"""统一对照实验运行器。

在 τ-bench 上以**相同 backbone + 相同 Warm-up/Eval 划分**运行四种方法，
输出结构一致的 per-task 记录，供积累曲线图与消融实验使用。

方法：
    * ``vanilla``      — 单轮 LLM：给任务+工具 schema，一次产出全部工具调用，
                         无多轮反馈。任务难度下界。
    * ``react``        — τ-bench 内置多步 llm_agent（ReAct），无积累。
    * ``autoharness``  — Warm-up 阶段跑 react 积累轨迹 → 触发归纳 → Eval 阶段
                         优先 Harness、回退 Agent。

任务流：对 ``autoharness``，x 轴覆盖 warmup+eval（前 K 个为 agent 路径，
之后为 harness 路径），以呈现"交叉超越"曲线；``vanilla``/``react`` 在全流
上均为 agent，曲线持平。

使用::

    from experience_os.experiments.compare import run_experiment
    res = run_experiment(method="react", model="ollama/qwen2.5:7b",
                         domain="retail", warmup=3, eval=5, max_steps=15)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from experience_os.tau2_adapter import infer_task_type

log = logging.getLogger(__name__)


# ======================================================================
# 结果数据结构
# ======================================================================
@dataclass
class TaskResult:
    idx: int  # 任务在流中的序号（1-based）
    phase: str  # "warmup" | "eval"
    task_id: str
    task_type: str
    method: str
    success: bool
    reward: float
    tokens: int
    latency: float
    path: str  # agent | harness | harness+agent
    error: str = ""


@dataclass
class ExperimentResult:
    method: str
    model: str
    domain: str
    task_type: str
    warmup_size: int
    eval_size: int
    max_steps: int
    results: list[TaskResult] = field(default_factory=list)
    experiment_id: str = ""  # LTS 关联 ID

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = f"{self.method}-{self.domain}-{uuid.uuid4().hex[:8]}"

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    @property
    def avg_latency(self) -> float:
        return sum(r.latency for r in self.results) / self.total if self.total else 0.0

    def eval_results(self) -> list[TaskResult]:
        return [r for r in self.results if r.phase == "eval"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = {
            "total": self.total,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "total_tokens": self.total_tokens,
            "avg_latency": round(self.avg_latency, 2),
            "eval_success_rate": round(
                sum(1 for r in self.eval_results() if r.success) / max(1, len(self.eval_results())),
                4,
            ),
        }
        return d


# ======================================================================
# 任务加载与划分
# ======================================================================
def load_tasks(domain: str = "retail") -> list:
    """加载 τ-bench 指定域的全部 base 任务。"""
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
    return mod.get_tasks("base")


def pick_task_group(tasks: list, task_type: str = "", warmup: int = 3):
    """选择指定任务类型组（或最大组），返回 (group, task_type)。"""
    from experience_os.tau2_adapter import infer_task_type, split_tasks

    _, _, groups = split_tasks(tasks, min_support=warmup)
    if not groups:
        return tasks, task_type or "unknown"
    if task_type and task_type in groups:
        return groups[task_type], task_type
    best = max(groups, key=lambda k: len(groups[k]))
    return groups[best], best


# ======================================================================
# 方法实现
# ======================================================================
def _resolve_tau2_model(model: str) -> tuple[str, str]:
    """把 `ollama/qwen2.5:7b` / `deepinfra/xxx` 拆为 (litellm_model, api_base)。"""
    if model.startswith("ollama/"):
        return model, "http://localhost:11434"
    if model.startswith("deepinfra/"):
        return model, ""  # litellm 经 DEEPINFRA_API_KEY 直连
    return model, ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def run_vanilla(
    task: Any,
    domain: str,
    model: str,
    max_steps: int,
    solo_mode: bool,
) -> TaskResult:
    """单轮 LLM：任务+工具 schema → 一次产出全部工具调用 → 执行 → DB hash 判定。"""
    from experience_os.environment import TaskRequest
    from experience_os.llm import LLMClient
    from experience_os.config import Config
    from experience_os.tau2_adapter import (
        Tau2Environment,
        _extract_task_description,
        infer_task_type,
        extract_task_params,
    )

    cfg = Config()
    # 让 LLMClient 用指定 model（覆盖配置：按 backend 写入对应字段）
    if model.startswith("ollama/"):
        cfg.llm.backend = "ollama"
        cfg.llm.ollama_model = model.split("/", 1)[-1]
    elif model.startswith("deepinfra/"):
        cfg.llm.backend = "deepinfra"
        cfg.llm.deepinfra_model = model.split("/", 1)[-1]
    client = LLMClient(cfg.llm)
    tau2_model, api_base = _resolve_tau2_model(model)

    t0 = time.time()
    task_type = infer_task_type(task)
    desc = _extract_task_description(task)
    try:
        env = Tau2Environment(domain, task, solo_mode=solo_mode)
        tools = env.get_tools()
        tool_names = [t["function"]["name"] if isinstance(t, dict) and "function" in t
                      else t.get("name", "") for t in tools]
        # 精简 schema：只给 name + 参数名，避免 prompt 过长
        schema_lines = []
        for t in tools:
            fn = t["function"] if isinstance(t, dict) and "function" in t else t
            name = fn.get("name", "")
            params = fn.get("parameters", {}).get("properties", {})
            pstr = ", ".join(f'{k}:{v.get("type","str")}' for k, v in params.items())
            schema_lines.append(f"  - {name}({pstr})")
        schema_txt = "\n".join(schema_lines[:20])

        prompt = (
            "You are a customer-service agent. Accomplish the task by emitting a "
            "JSON object {\"calls\": [{\"name\": str, \"arguments\": dict}, ...]} "
            "with the ordered tool calls. Do not include any other text.\n\n"
            f"Available tools:\n{schema_txt}\n\nTask: {desc}\n"
        )
        data = client.chat_json(
            [{"role": "system", "content": "You output only JSON tool-call plans."},
             {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        calls = data.get("calls", []) if isinstance(data, dict) else []
        tokens = _estimate_tokens(json.dumps(data, ensure_ascii=False))
        # 执行调用序列
        for c in calls[:max_steps]:
            name = c.get("name", "")
            args = c.get("arguments", {}) or c.get("args", {})
            if name:
                env.call_tool(name, args)
        reward = 1.0 if env.verify("", "") else 0.0
        success = reward >= 1.0
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="vanilla", success=success, reward=reward, tokens=tokens,
            latency=time.time() - t0, path="agent",
            error="" if success else "no_match_or_bad_calls",
        )
    except Exception as exc:
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="vanilla", success=False, reward=0.0, tokens=0,
            latency=time.time() - t0, path="agent", error=str(exc)[:200],
        )


def run_react(
    task: Any,
    domain: str,
    model: str,
    max_steps: int,
    solo_mode: bool,
    seed: int = 42,
) -> TaskResult:
    """τ-bench 内置多步 ReAct agent。"""
    from experience_os.tau2_adapter import (
        _extract_task_description, convert_simulation,
        infer_task_type, run_tau2_simulation,
    )

    t0 = time.time()
    task_type = infer_task_type(task)
    tau2_model, api_base = _resolve_tau2_model(model)
    try:
        sim = run_tau2_simulation(
            domain=domain, task=task, llm_model=tau2_model,
            llm_api_base=api_base, max_steps=max_steps,
            seed=seed, solo_mode=solo_mode,
        )
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        tokens = int(getattr(sim, "agent_cost", 0) or 0) or _estimate_tokens(
            str(sim.messages) if hasattr(sim, "messages") else ""
        )
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="react", success=reward >= 1.0, reward=reward,
            tokens=tokens, latency=time.time() - t0, path="agent",
        )
    except Exception as exc:
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="react", success=False, reward=0.0, tokens=0,
            latency=time.time() - t0, path="agent", error=str(exc)[:200],
        )


# ======================================================================
# AutoHarness：warmup 积累 → 归纳 → eval 部署
# ======================================================================
def run_autoharness(
    group: list,
    domain: str,
    model: str,
    warmup: int,
    eval_size: int,
    max_steps: int,
    solo_mode: bool,
    *,
    skip_validation: bool = False,
    no_versioning: bool = False,
) -> list[TaskResult]:
    """AutoHarness 方法：warmup 阶段积累，触发归纳，eval 阶段 Harness 优先。"""
    from experience_os.config import Config
    from experience_os.environment import MockEnvironment, TaskRequest
    from experience_os.runtime import Runtime, SystemMode
    from experience_os.tau2_adapter import (
        Tau2Environment, _extract_task_description, convert_simulation,
        extract_task_params, infer_task_type, run_tau2_simulation,
    )

    cfg = Config()
    if cfg.data_dir.exists():
        shutil.rmtree(cfg.data_dir)
    cfg.ensure_dirs()
    if skip_validation:
        cfg.induction.validation_threshold = 0.0
    rt = Runtime(cfg, MockEnvironment())
    tau2_model, api_base = _resolve_tau2_model(model)

    warmup_tasks = group[:warmup]
    eval_tasks = group[warmup: warmup + eval_size]
    results: list[TaskResult] = []
    idx = 1

    # --- warmup: agent 积累 ---
    rt.set_mode(SystemMode.ACCUMULATION)
    for i, task in enumerate(warmup_tasks, 1):
        tt = infer_task_type(task)
        t0 = time.time()
        try:
            sim = run_tau2_simulation(
                domain=domain, task=task, llm_model=tau2_model,
                llm_api_base=api_base, max_steps=max_steps,
                seed=42 + i, solo_mode=solo_mode,
            )
            traj = convert_simulation(sim, task, tt)
            rt.repo.add_trajectory(traj)
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            tokens = int(getattr(sim, "agent_cost", 0) or 0) or max(1, len(traj.steps) * 100)
            stats = rt.repo.get_stats(tt)
            stats.total_executions += 1
            stats.agent_executions += 1
            if reward >= 1.0:
                stats.agent_successes += 1
            rt.repo.save_stats(tt)
            results.append(TaskResult(
                idx=idx, phase="warmup", task_id=task.id, task_type=tt,
                method="autoharness", success=reward >= 1.0, reward=reward,
                tokens=tokens, latency=time.time() - t0, path="agent",
            ))
        except Exception as exc:
            results.append(TaskResult(
                idx=idx, phase="warmup", task_id=task.id, task_type=tt,
                method="autoharness", success=False, reward=0.0, tokens=0,
                latency=time.time() - t0, path="agent", error=str(exc)[:200],
            ))
        idx += 1

    # --- 归纳 ---
    induced = []
    for tt in rt.repo.all_task_types():
        trigger = rt.inductor.check_triggers(tt)
        if not trigger:
            continue
        same = [t for t in warmup_tasks if infer_task_type(t) == tt]
        if not same:
            continue
        try:
            venv = Tau2Environment(domain, same[0])
            h = rt.inductor.induce(tt, venv)
            if h:
                induced.append(h)
        except Exception as exc:
            log.warning("induce %s failed: %s", tt, exc)

    # --- eval: harness 优先，回退 agent ---
    rt.set_mode(SystemMode.DEPLOYMENT)
    for i, task in enumerate(eval_tasks, 1):
        tt = infer_task_type(task)
        desc = _extract_task_description(task)
        params = extract_task_params(task)
        t0 = time.time()
        used_harness = False
        path = "agent"

        # 尝试 harness
        matching = [h for h in induced if h.task_type == tt] or induced
        if matching:
            h = matching[0]
            try:
                tenv = Tau2Environment(domain, task)
                req = TaskRequest(task_id=task.id, task_description=desc,
                                  task_type=tt, params=params, expected_output="")
                r = tenv.execute_harness(h, req)
                if r.success:
                    results.append(TaskResult(
                        idx=idx, phase="eval", task_id=task.id, task_type=tt,
                        method="autoharness", success=True, reward=1.0,
                        tokens=r.tokens_used, latency=time.time() - t0,
                        path="harness",
                    ))
                    idx += 1
                    continue
                used_harness = True
            except Exception as exc:
                log.warning("harness exec failed: %s", exc)
                used_harness = True

        # 回退 agent
        try:
            sim = run_tau2_simulation(
                domain=domain, task=task, llm_model=tau2_model,
                llm_api_base=api_base, max_steps=max_steps,
                seed=100 + i, solo_mode=solo_mode,
            )
            traj = convert_simulation(sim, task, tt)
            rt.repo.add_trajectory(traj)
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            tokens = int(getattr(sim, "agent_cost", 0) or 0) or max(1, len(traj.steps) * 100)
            path = "harness+agent" if used_harness else "agent"
            results.append(TaskResult(
                idx=idx, phase="eval", task_id=task.id, task_type=tt,
                method="autoharness", success=reward >= 1.0, reward=reward,
                tokens=tokens, latency=time.time() - t0, path=path,
            ))
        except Exception as exc:
            results.append(TaskResult(
                idx=idx, phase="eval", task_id=task.id, task_type=tt,
                method="autoharness", success=False, reward=0.0, tokens=0,
                latency=time.time() - t0, path=path, error=str(exc)[:200],
            ))
        idx += 1

    return results


# ======================================================================
# 主入口
# ======================================================================
def run_experiment(
    method: str,
    model: str,
    domain: str = "retail",
    warmup: int = 3,
    eval_size: int = 5,
    max_steps: int = 15,
    task_type: str = "",
    solo_mode: bool = False,
    *,
    skip_validation: bool = False,
    no_versioning: bool = False,
    variant: str = "type_split",
    cross_domain: str = "",
    experiment_id: str = "",
) -> ExperimentResult:
    """运行单方法对照实验。

    Args:
        method: ``vanilla`` | ``react`` | ``autoharness``
        model: litellm 模型名（``ollama/qwen2.5:7b`` / ``deepinfra/...``）
        warmup: warm-up 池大小（仅 autoharness 用于积累）
        eval_size: 评估池大小
        variant: 实验设计变体：
            * ``type_split``  — 同任务类型拆分积累/验证池（默认）
            * ``replay``      — 同任务既积累又验证（重跑，上界）
            * ``cross_domain``— cross_domain 上积累，domain 上验证（跨域迁移）
    """
    print(f"\n{'='*60}")
    print(f"  对照实验: {method}  model={model}  domain={domain}")
    print(f"  warmup={warmup} eval={eval_size} max_steps={max_steps} solo={solo_mode}")
    print(f"  variant={variant}" + (f" cross_domain={cross_domain}" if cross_domain else ""))
    print(f"{'='*60}\n")

    tasks = load_tasks(domain)
    group, chosen_type = pick_task_group(tasks, task_type, warmup)
    print(f"  任务类型: {chosen_type} ({len(group)} 个)")

    # --- 实验设计变体决定 warmup/eval 任务集 ---
    if variant == "replay":
        # 同任务既积累又验证：warmup 和 eval 用相同任务
        warmup_tasks = group[:warmup]
        eval_tasks = group[:eval_size]  # 重跑同一批
    elif variant == "cross_domain" and cross_domain:
        # 跨域：cross_domain 上积累，domain 上验证
        cd_tasks = load_tasks(cross_domain)
        cd_group, cd_type = pick_task_group(cd_tasks, task_type, warmup)
        warmup_tasks = cd_group[:warmup]
        # eval 用目标域同类型任务（类型名需一致，否则退化到任意）
        eval_tasks = group[:eval_size]
        print(f"  跨域积累: {cross_domain}/{cd_type} ({len(warmup_tasks)} 个) → 验证: {domain}")
    else:  # type_split (default)
        warmup_tasks = group[:warmup]
        eval_tasks = group[warmup: warmup + eval_size]

    stream = warmup_tasks + eval_tasks

    eid = experiment_id or f"{method}-{domain}-{variant}-{uuid.uuid4().hex[:8]}"
    # 初始化 LTS（跨实验持久底座）
    from experience_os.lts import LTSStore, LTSEntry
    lts = LTSStore()

    results: list[TaskResult] = []

    if method == "autoharness":
        results = run_autoharness(
            group, domain, model, warmup, eval_size, max_steps, solo_mode,
            skip_validation=skip_validation, no_versioning=no_versioning,
        )
    else:
        phase_warmup = method == "vanilla" or method == "react"
        # vanilla/react 在全流上均为 agent；warmup 段也跑（用于曲线对齐）
        for i, task in enumerate(stream, 1):
            phase = "warmup" if i <= warmup else "eval"
            if method == "vanilla":
                r = run_vanilla(task, domain, model, max_steps, solo_mode)
            else:
                r = run_react(task, domain, model, max_steps, solo_mode, seed=42 + i)
            r.idx = i
            r.phase = phase
            results.append(r)
            lts.log(LTSEntry(
                experiment_id=eid, method=method, domain=domain,
                task_id=task.id, task_type=infer_task_type(task),
                idx=i, phase=phase, success=r.success, reward=r.reward,
                tokens=r.tokens, latency=r.latency, path=r.path,
                meta={"variant": variant, "model": model},
            ))
            tag = "✓" if r.success else "✗"
            print(f"  [{i}/{len(stream)}] {phase} {r.task_id} {tag} "
                  f"reward={r.reward:.2f} tokens={r.tokens} {r.error[:40]}")

    # autoharness 也写入 LTS
    if method == "autoharness":
        from experience_os.tau2_adapter import infer_task_type as _itt
        for r in results:
            lts.log(LTSEntry(
                experiment_id=eid, method=method, domain=domain,
                task_id=r.task_id, task_type=r.task_type,
                idx=r.idx, phase=r.phase, success=r.success, reward=r.reward,
                tokens=r.tokens, latency=r.latency, path=r.path,
                meta={"variant": variant, "model": model},
            ))

    exp = ExperimentResult(
        method=method, model=model, domain=domain, task_type=chosen_type,
        warmup_size=warmup, eval_size=eval_size, max_steps=max_steps,
        results=results, experiment_id=eid,
    )
    _print_summary(exp)
    print(f"  experiment_id: {eid}")
    print(f"  LTS: {lts.query(experiment_id=eid).__len__()} 条记录已持久化")
    lts.close()
    return exp


def _print_summary(exp: ExperimentResult) -> None:
    print(f"\n{'='*60}")
    print(f"  汇总: {exp.method}")
    print(f"{'='*60}")
    print(f"  总任务:   {exp.total}")
    print(f"  成功率:   {exp.successes}/{exp.total} ({exp.success_rate:.1%})")
    print(f"  总 Token: {exp.total_tokens:,}")
    print(f"  平均延迟: {exp.avg_latency:.1f}s")
    ev = exp.eval_results()
    if ev:
        esr = sum(1 for r in ev if r.success) / len(ev)
        print(f"  Eval SR:  {esr:.1%} ({len(ev)} tasks)")
    # 路径分布
    from collections import Counter
    paths = Counter(r.path for r in exp.results)
    print(f"  路径分布: {dict(paths)}")
    print(f"{'='*60}\n")


def save_result(exp: ExperimentResult, output_file: str) -> None:
    p = Path(output_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(exp.to_dict(), ensure_ascii=False, indent=2))
    print(f"结果已保存: {p}")
