#!/usr/bin/env python
"""τ-bench retail 对比实验脚本。

在 tau2 retail 域上运行 baseline (vanilla/react/skillopt) vs EOS (coe) 对比。
使用 tau2 原生 train/test split（train=74, test=40）。

用法：
    # 需要 Python 3.12+ 且已安装 tau2-bench
    # pip install -e tau2-bench

    # 小规模验证（3 warmup + 5 eval）
    python scripts/run_tau_experiment.py --model ollama/qwen2.5:7b --warmup 3 --eval 5

    # 按 task_type 筛选
    python scripts/run_tau_experiment.py --task-type get_order_details --warmup 3 --eval 5

    # 全量对比（所有 baseline + EOS）
    python scripts/run_tau_experiment.py --methods vanilla react skillopt coe

    # 强模型→弱模型经验迁移（GLM v5.2 积累 → MiniMax M3 部署）
    python scripts/run_tau_experiment.py --method coe --warmup-model glm-4.5 --eval-model minimax-m3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 确保 tau2-bench 在路径中
tau2_src = Path(__file__).parent.parent / "tau2-bench" / "src"
if tau2_src.exists():
    sys.path.insert(0, str(tau2_src))


def main():
    parser = argparse.ArgumentParser(description="τ-bench retail 对比实验")
    parser.add_argument("--model", default="ollama/qwen2.5:7b",
                        help="litellm 模型名 (ollama/qwen2.5:7b, deepinfra/MiniMaxAI/MiniMax-M2.7, 等)")
    parser.add_argument("--methods", nargs="+",
                        default=["vanilla", "react", "coe"],
                        help="运行的方法: vanilla react skillopt coe")
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--warmup", type=int, default=3,
                        help="warmup 任务数（积累阶段）")
    parser.add_argument("--eval", type=int, default=5,
                        help="eval 任务数（评估阶段）")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--task-type", default="",
                        help="筛选特定 task_type（如 get_order_details）")
    parser.add_argument("--variant", default="train_test",
                        choices=["type_split", "replay", "cross_domain", "train_test"],
                        help="实验变体（train_test 用 tau2 原生划分）")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过 harness 验证（快速测试）")
    parser.add_argument("--output", default="",
                        help="结果输出 JSON 路径")

    args = parser.parse_args()

    from experience_os.experiments.compare import run_experiment, _print_summary

    all_results = []
    for method in args.methods:
        print(f"\n{'#'*60}")
        print(f"# 运行方法: {method}")
        print(f"{'#'*60}")

        exp = run_experiment(
            method=method,
            model=args.model,
            domain=args.domain,
            warmup=args.warmup,
            eval_size=args.eval,
            max_steps=args.max_steps,
            task_type=args.task_type,
            variant=args.variant,
            skip_validation=args.skip_validation,
        )
        _print_summary(exp)
        all_results.append(exp)

    # 保存结果
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"results/experiment_{args.domain}_{args.variant}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = []
    for exp in all_results:
        results_data.append({
            "method": exp.method,
            "model": exp.model,
            "domain": exp.domain,
            "warmup_size": exp.warmup_size,
            "eval_size": exp.eval_size,
            "total": exp.total,
            "successes": exp.successes,
            "success_rate": exp.success_rate,
            "total_tokens": exp.total_tokens,
            "avg_latency": exp.avg_latency,
            "experiment_id": exp.experiment_id,
            "results": [
                {
                    "idx": r.idx, "phase": r.phase, "task_id": r.task_id,
                    "task_type": r.task_type, "success": r.success,
                    "reward": r.reward, "tokens": r.tokens,
                    "latency": r.latency, "path": r.path,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                }
                for r in exp.results
            ],
        })

    out_path.write_text(json.dumps(results_data, ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {out_path}")

    # 对比摘要
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  对比摘要")
        print(f"{'='*60}")
        print(f"  {'Method':<15} {'SR':>8} {'Tokens':>10} {'Eval SR':>8}")
        print(f"  {'-'*45}")
        for exp in all_results:
            eval_results = [r for r in exp.results if r.phase == "eval"]
            eval_sr = (sum(1 for r in eval_results if r.success) / len(eval_results)
                       if eval_results else 0)
            print(f"  {exp.method:<15} {exp.success_rate:>7.1%} {exp.total_tokens:>10,} {eval_sr:>7.1%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
