"""τ-bench Baseline 评估脚本。

直接在 τ-bench retail 域上运行 LLM agent，不经过 ExperienceOS 的归纳/部署流程，
收集 baseline 分数用于后续与经验方案对比。

使用方式::

    # ollama（验证流程）
    experience-os baseline --model ollama/qwen2.5:7b --domain retail --max-tasks 10

    # DeepInfra
    experience-os baseline --model deepinfra/MiniMaxAI/MiniMax-M2.7 \
        --domain retail --max-tasks 20

    # 通过 Python API
    from experience_os.baseline_eval import run_baseline
    results = run_baseline(model="ollama/qwen2.5:7b", domain="retail", max_tasks=10)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """单次任务评估结果。"""
    task_id: str
    task_type: str
    description: str
    reward: float
    tokens_used: int
    latency_seconds: float
    num_steps: int
    error: str = ""

    @property
    def success(self) -> bool:
        return self.reward >= 1.0


@dataclass
class BaselineSummary:
    """评估汇总。"""
    model: str
    domain: str
    total_tasks: int
    successes: int
    avg_reward: float
    total_tokens: int
    avg_latency: float
    results: list[BaselineResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_tasks if self.total_tasks else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "domain": self.domain,
            "total_tasks": self.total_tasks,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "avg_reward": round(self.avg_reward, 4),
            "total_tokens": self.total_tokens,
            "avg_latency": round(self.avg_latency, 2),
            "task_types": self._task_type_breakdown(),
            "results": [
                {
                    "task_id": r.task_id,
                    "task_type": r.task_type,
                    "reward": r.reward,
                    "tokens": r.tokens_used,
                    "latency": round(r.latency_seconds, 2),
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def _task_type_breakdown(self) -> dict:
        from collections import defaultdict

        by_type: dict[str, list[float]] = defaultdict(list)
        for r in self.results:
            by_type[r.task_type].append(r.reward)
        return {
            tt: {
                "count": len(rewards),
                "success_rate": sum(1 for r in rewards if r >= 1.0) / len(rewards),
                "avg_reward": sum(rewards) / len(rewards),
            }
            for tt, rewards in sorted(by_type.items())
        }


def run_baseline(
    model: str,
    domain: str = "retail",
    max_tasks: int = 10,
    max_steps: int = 30,
    solo_mode: bool = False,
    api_base: str = "",
    api_key: str = "",
    output_file: str = "",
) -> BaselineSummary:
    """运行 baseline 评估。

    Args:
        model: litellm 格式的模型名（如 ollama/qwen2.5:7b）
        domain: τ-bench 域名
        max_tasks: 最大评估任务数（0=全部）
        max_steps: 每个任务最大仿真步数
        solo_mode: 是否使用 solo 模式
        api_base: 自定义 API base URL
        api_key: 自定义 API key
        output_file: 结果保存路径

    Returns:
        评估汇总
    """
    from experience_os.tau2_adapter import run_tau2_simulation, infer_task_type

    # 加载任务（tau2 域模块中有 get_tasks）
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
    tasks = mod.get_tasks("base")
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    # 设置 API key
    if api_key:
        os.environ["DEEPINFRA_API_KEY"] = api_key

    print(f"\n{'='*60}")
    print(f"  Baseline 评估")
    print(f"{'='*60}")
    print(f"  模型:     {model}")
    print(f"  域:       {domain}")
    print(f"  任务数:   {len(tasks)}")
    print(f"  最大步数: {max_steps}")
    print(f"  Solo:     {solo_mode}")
    print(f"{'='*60}\n")

    results: list[BaselineResult] = []

    for i, task in enumerate(tasks):
        task_type = infer_task_type(task)
        raw_desc = getattr(task, "description", "")
        desc = (str(raw_desc) if raw_desc else "")[:80]

        print(f"[{i+1}/{len(tasks)}] task={task.id} type={task_type}")
        print(f"  desc: {desc}")

        t0 = time.time()
        try:
            sim = run_tau2_simulation(
                domain=domain,
                task=task,
                llm_model=model,
                llm_api_base=api_base,
                max_steps=max_steps,
                seed=42 + i,
                solo_mode=solo_mode,
            )

            reward = sim.reward if hasattr(sim, "reward") else 0.0
            tokens = _count_tokens(sim)
            steps = len(sim.messages) if hasattr(sim, "messages") else 0

            latency = time.time() - t0
            result = BaselineResult(
                task_id=task.id,
                task_type=task_type,
                description=desc,
                reward=reward,
                tokens_used=tokens,
                latency_seconds=latency,
                num_steps=steps,
            )
            print(f"  reward={reward:.2f} tokens={tokens} latency={latency:.1f}s steps={steps}")

        except Exception as exc:
            latency = time.time() - t0
            result = BaselineResult(
                task_id=task.id,
                task_type=task_type,
                description=desc,
                reward=0.0,
                tokens_used=0,
                latency_seconds=latency,
                num_steps=0,
                error=str(exc)[:200],
            )
            print(f"  ERROR: {exc}")

        results.append(result)

    # 汇总
    total = len(results)
    successes = sum(1 for r in results if r.success)
    avg_reward = sum(r.reward for r in results) / total if total else 0
    total_tokens = sum(r.tokens_used for r in results)
    avg_latency = sum(r.latency_seconds for r in results) / total if total else 0

    summary = BaselineSummary(
        model=model,
        domain=domain,
        total_tasks=total,
        successes=successes,
        avg_reward=avg_reward,
        total_tokens=total_tokens,
        avg_latency=avg_latency,
        results=results,
    )

    print(f"\n{'='*60}")
    print(f"  评估结果汇总")
    print(f"{'='*60}")
    print(f"  模型:        {model}")
    print(f"  成功率:      {successes}/{total} ({summary.success_rate:.1%})")
    print(f"  平均 reward: {avg_reward:.4f}")
    print(f"  总 Token:   {total_tokens:,}")
    print(f"  平均延迟:    {avg_latency:.1f}s")
    print(f"\n  按任务类型:")
    for tt, stats in summary._task_type_breakdown().items():
        print(f"    {tt}: {stats['count']} tasks, sr={stats['success_rate']:.1%}, avg_r={stats['avg_reward']:.2f}")
    print(f"{'='*60}\n")

    # 保存结果
    if output_file:
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        print(f"结果已保存到 {p}")

    return summary


def _count_tokens(sim: Any) -> int:
    """估算仿真使用的 token 数。"""
    if hasattr(sim, "info") and isinstance(sim.info, dict):
        return sim.info.get("total_tokens", 0)
    # 回退：按消息长度估算
    total = 0
    messages = getattr(sim, "messages", [])
    for msg in messages:
        total += len(str(msg)) // 4
    return total
