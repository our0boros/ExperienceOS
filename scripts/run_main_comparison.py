"""主对比实验：ReAct vs SkillOpt vs CoE (ONLINE_ACCUMULATION)。

三种方法，相同 40 个任务，同一起跑线：
- ReAct: 无经验，纯 agent
- SkillOpt: 静态 SOP，agent 始终有引导
- CoE: 在线积累——从零开始，边执行边归纳边使用

用法:
    python scripts/run_main_comparison.py --eval 20
"""

import json
import os
import sys
import time
from pathlib import Path

# ── 修复 SSL + API key ──────────────────────────────────────────
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
if os.environ.get("DEEPINFRA_TOKEN") and not os.environ.get("DEEPINFRA_API_KEY"):
    os.environ["DEEPINFRA_API_KEY"] = os.environ["DEEPINFRA_TOKEN"]

sys.path.insert(0, str(Path(__file__).parent.parent))

from experience_os.experiments.runner import (
    ExperimentConfig,
    ExperimentMode,
    ExperimentRunner,
)


def run_comparison(eval_size: int = 20, model: str = "deepinfra/deepseek-ai/DeepSeek-V4-Flash"):
    """运行三种方法的主对比实验。"""
    methods = [
        ("react", ExperimentMode.DEPLOYMENT, 0),
        ("skillopt", ExperimentMode.DEPLOYMENT, 0),
        ("coe", ExperimentMode.ONLINE_ACCUMULATION, 0),
    ]

    results_dir = Path("docs/exp_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    all_metrics = {}

    for method, mode, warmup in methods:
        print(f"\n{'#'*60}")
        print(f"# {method.upper()}  mode={mode.value}")
        print(f"{'#'*60}")

        config = ExperimentConfig(
            mode=mode,
            method=method,
            domain="retail",
            model=model,
            warmup=warmup,
            eval_size=eval_size,
            max_steps=30,
            split_policy="train_test",
        )

        try:
            runner = ExperimentRunner(config)
            metrics = runner.execute()
            all_metrics[method] = metrics.to_dict()

            # 保存单独结果
            out = results_dir / f"{method}-online-{timestamp}.json"
            out.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
            print(f"  结果已保存: {out}")
        except Exception as exc:
            print(f"  {method} 失败: {exc}")
            all_metrics[method] = {"error": str(exc)}

    # 对比汇总
    print(f"\n{'='*70}")
    print(f"  主对比结果汇总")
    print(f"{'='*70}")
    print(f"  {'Method':<12} {'SR':>8} {'Tokens':>12} {'Harness%':>10} {'Path'}")
    print(f"  {'-'*60}")
    for method in ["react", "skillopt", "coe"]:
        m = all_metrics.get(method, {})
        if "error" in m:
            print(f"  {method:<12} {'ERROR':>8} {m['error'][:40]}")
        else:
            print(f"  {method:<12} {m['success_rate']:>7.1%} "
                  f"{m['total_tokens']:>10,}  "
                  f"{m['harness_hit_rate']:>9.1%}  "
                  f"{m['path_distribution']}")

    # 保存汇总
    summary = results_dir / f"comparison-online-{timestamp}.json"
    summary.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"\n  汇总已保存: {summary}")
    return all_metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="主对比实验")
    p.add_argument("--eval", type=int, default=20, help="eval 任务数")
    p.add_argument("--model", type=str,
                   default="deepinfra/deepseek-ai/DeepSeek-V4-Flash",
                   help="LLM 模型")
    args = p.parse_args()
    run_comparison(eval_size=args.eval, model=args.model)
