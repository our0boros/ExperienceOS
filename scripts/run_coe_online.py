"""CoE ONLINE_ACCUMULATION 实验。

用法:
    python scripts/run_coe_online.py --eval 40
"""
import json
import os
import sys
import time
from pathlib import Path

# ── 修复 SSL + API key ──
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


def main(eval_size: int = 40, model: str = "deepinfra/deepseek-ai/DeepSeek-V4-Flash"):
    config = ExperimentConfig(
        mode=ExperimentMode.ONLINE_ACCUMULATION,
        method="coe",
        domain="retail",
        model=model,
        warmup=0,
        eval_size=eval_size,
        max_steps=30,
        split_policy="train_test",
    )

    runner = ExperimentRunner(config)
    metrics = runner.execute()

    # 保存结果
    results_dir = Path("docs/exp_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out = results_dir / f"coe-online-{timestamp}.json"
    out.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out}")
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=int, default=40, help="eval 任务数")
    p.add_argument("--model", type=str,
                   default="deepinfra/deepseek-ai/DeepSeek-V4-Flash")
    args = p.parse_args()
    main(eval_size=args.eval, model=args.model)
