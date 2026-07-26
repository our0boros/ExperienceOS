#!/usr/bin/env python
"""经验迁移实验：强模型积累 → 弱模型部署。

验证 ExperienceOS 的核心主张：强模型（如 GLM v5.2）积累的经验，
可以编译为 Harness，让弱模型（如 MiniMax M3）部署时绕过 LLM 推理，
实现能力提升。

流程：
    1. 用强模型在 train split 上运行 agent，收集轨迹
    2. 归纳编译为 Harness
    3. 用弱模型在 test split 上部署（Harness 优先 + agent fallback）
    4. 对比：弱模型 vanilla vs 弱模型 + Harness

用法：
    python scripts/run_transfer_experiment.py \
        --strong-model deepinfra/MiniMaxAI/MiniMax-M2.7 \
        --weak-model ollama/qwen2.5:7b \
        --warmup 5 --eval 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

tau2_src = Path(__file__).parent.parent / "tau2-bench" / "src"
if tau2_src.exists():
    sys.path.insert(0, str(tau2_src))


def main():
    parser = argparse.ArgumentParser(description="经验迁移实验")
    parser.add_argument("--strong-model", required=True,
                        help="强模型（积累阶段）")
    parser.add_argument("--weak-model", required=True,
                        help="弱模型（部署阶段）")
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--eval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--task-type", default="")
    parser.add_argument("--output", default="")

    args = parser.parse_args()

    from experience_os.experiments.compare import (
        run_experiment, _print_summary, load_train_test_split,
    )
    from experience_os.tau2_adapter import infer_task_type

    print("=" * 60)
    print("  经验迁移实验")
    print(f"  强模型: {args.strong_model} (积累)")
    print(f"  弱模型: {args.weak_model} (部署)")
    print(f"  域: {args.domain}  warmup={args.warmup}  eval={args.eval}")
    print("=" * 60)

    # 1. 强模型积累（coe，用强模型跑 warmup + 归纳）
    print("\n>>> Phase 1: 强模型积累")
    strong_exp = run_experiment(
        method="coe",
        model=args.strong_model,
        domain=args.domain,
        warmup=args.warmup,
        eval_size=0,  # 不跑 eval，只积累
        max_steps=args.max_steps,
        task_type=args.task_type,
        variant="train_test",
        skip_validation=False,
    )
    print(f"  强模型积累完成: {strong_exp.successes}/{strong_exp.total} 成功")

    # 2. 弱模型 vanilla baseline（无 Harness）
    print("\n>>> Phase 2: 弱模型 vanilla baseline")
    weak_vanilla = run_experiment(
        method="vanilla",
        model=args.weak_model,
        domain=args.domain,
        warmup=0,
        eval_size=args.eval,
        max_steps=args.max_steps,
        task_type=args.task_type,
        variant="train_test",
    )
    _print_summary(weak_vanilla)

    # 3. 弱模型 + Harness 部署（Harness 优先，弱模型 fallback）
    print("\n>>> Phase 3: 弱模型 + Harness 部署")
    weak_harness = run_experiment(
        method="coe",
        model=args.weak_model,
        domain=args.domain,
        warmup=0,  # 不再积累，直接用强模型积累的 Harness
        eval_size=args.eval,
        max_steps=args.max_steps,
        task_type=args.task_type,
        variant="train_test",
        skip_validation=True,  # 跳过验证（已验证过）
    )
    _print_summary(weak_harness)

    # 4. 对比
    print("\n" + "=" * 60)
    print("  经验迁移对比")
    print("=" * 60)
    print(f"  {'配置':<30} {'Eval SR':>8} {'Tokens':>10}")
    print(f"  {'-'*50}")

    for label, exp in [
        (f"弱模型 vanilla", weak_vanilla),
        (f"弱模型 + Harness", weak_harness),
    ]:
        eval_r = [r for r in exp.results if r.phase == "eval"]
        eval_sr = sum(1 for r in eval_r if r.success) / len(eval_r) if eval_r else 0
        eval_tok = sum(r.tokens for r in eval_r)
        print(f"  {label:<30} {eval_sr:>7.1%} {eval_tok:>10,}")

    print(f"\n  强模型积累 SR: {strong_exp.success_rate:.1%}")
    print("=" * 60)

    # 保存
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"results/transfer_{args.domain}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "strong_model": args.strong_model,
        "weak_model": args.weak_model,
        "domain": args.domain,
        "strong_accumulation_sr": strong_exp.success_rate,
        "weak_vanilla_sr": weak_vanilla.success_rate,
        "weak_harness_sr": weak_harness.success_rate,
    }, ensure_ascii=False, indent=2))
    print(f"结果保存到: {out_path}")


if __name__ == "__main__":
    main()
