"""积累曲线图：跨方法对比滚动成功率。

读取多个 ``save_result`` 产出的 JSON，绘制：
    * x = 任务序号（覆盖 warmup+eval）
    * y = 滚动成功率（窗口=3）
    * 竖线标注 warmup/eval 分界（coe 在 K 处"交叉超越"）

依赖：matplotlib（可选）。无 matplotlib 时输出 CSV + ASCII 火花线。
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def rolling_success(successes: list[bool], window: int = 3) -> list[float]:
    """滑动窗口成功率，边界处窗口缩小。"""
    out = []
    for i in range(len(successes)):
        lo = max(0, i - window + 1)
        chunk = successes[lo: i + 1]
        out.append(sum(1 for x in chunk if x) / len(chunk))
    return out


def load_results(*paths: str) -> list[dict]:
    out = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        out.append(data)
    return out


def plot_curve(result_files: list[str], output: str = "docs/exp_results/curve.png",
               warmup_split: int | None = None, window: int = 3) -> str:
    """绘制积累曲线图。返回输出路径。"""
    series = []
    for rf in result_files:
        d = json.loads(Path(rf).read_text())
        successes = [r["success"] for r in d["results"]]
        series.append({
            "method": d["method"],
            "x": list(range(1, len(successes) + 1)),
            "y": rolling_success(successes, window),
            "warmup_size": d.get("warmup_size", 0),
        })

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for s in series:
            ax.plot(s["x"], s["y"], marker="o", label=s["method"], linewidth=2)
            # 标注 warmup/eval 分界
            if s["warmup_size"] and warmup_split is None:
                w = s["warmup_size"]
                if 0 < w < len(s["x"]):
                    ax.axvline(w + 0.5, color="gray", linestyle="--", alpha=0.4)
                    ax.text(w + 0.6, 0.05, "warmup→eval", fontsize=8, color="gray")
        if warmup_split:
            ax.axvline(warmup_split + 0.5, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Task index (warmup → eval)")
        ax.set_ylabel(f"Rolling success rate (window={window})")
        ax.set_title("Accumulation curve: method comparison")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"曲线图已保存: {out_path}")
    except ImportError:
        log.warning("matplotlib 不可用，输出 CSV 代替")
        csv_path = out_path.with_suffix(".csv")
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["idx"] + [s["method"] for s in series]
            w.writerow(header)
            max_len = max(len(s["y"]) for s in series)
            for i in range(max_len):
                row = [i + 1]
                for s in series:
                    row.append(s["y"][i] if i < len(s["y"]) else "")
                w.writerow(row)
        print(f"CSV 已保存: {csv_path}")
        out_path = csv_path

    # 终端火花线预览
    _sparkline(series)
    return str(out_path)


def plot_cost_curve(
    result_files: list[str],
    output: str = "docs/exp_results/cost_curve.png",
    window: int = 3,
) -> str:
    """绘制成本收敛曲线：累计 token + 滚动平均 token。

    用于回答"经验积累是否让成本收敛"——coe 在 harness 命中后
    token 应骤降，rolling_avg 收敛到低位。
    """
    series = []
    for rf in result_files:
        d = json.loads(Path(rf).read_text())
        tokens = [r["tokens"] for r in d["results"]]
        cum, s = [], 0
        for t in tokens:
            s += t
            cum.append(s)

        def _rolling(vals, w):
            out = []
            for i in range(len(vals)):
                lo = max(0, i - w + 1)
                chunk = vals[lo: i + 1]
                out.append(sum(chunk) / len(chunk))
            return out

        series.append({
            "method": d["method"],
            "x": list(range(1, len(tokens) + 1)),
            "cumulative": cum,
            "rolling_avg": _rolling(tokens, window),
        })

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for s in series:
            ax1.plot(s["x"], s["cumulative"], marker="o", ms=3, label=s["method"])
            ax2.plot(s["x"], s["rolling_avg"], marker="o", ms=3, label=s["method"])
        ax1.set_ylabel("Cumulative tokens")
        ax1.set_title("Cost convergence: cumulative token cost")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Task index")
        ax2.set_ylabel(f"Rolling avg tokens (window={window})")
        ax2.set_title("Per-task token cost (lower = cheaper)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"成本曲线已保存: {out_path}")
    except ImportError:
        log.warning("matplotlib 不可用，成本曲线输出 CSV")
        csv_path = out_path.with_suffix(".csv")
        import csv as _csv
        with csv_path.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["idx", "method", "cumulative_tokens", "rolling_avg_tokens"])
            max_len = max(len(s["cumulative"]) for s in series)
            for i in range(max_len):
                for s in series:
                    if i < len(s["cumulative"]):
                        w.writerow([i + 1, s["method"],
                                   s["cumulative"][i], s["rolling_avg"][i]])
        print(f"成本曲线 CSV: {csv_path}")
        out_path = csv_path
    return str(out_path)


def _sparkline(series: list[dict]) -> None:
    chars = " ▂▃▄▅▆▇█"
    print("\n滚动成功率预览：")
    for s in series:
        ys = s["y"]
        line = "".join(chars[min(7, int(v * 8))] for v in ys)
        print(f"  {s['method']:<12} {line} ({len(ys)} pts)")
