"""Generate baseline comparison report + accumulation curves.

Reads result JSONs from docs/exp_results/ and produces:
1. docs/exp_results/README.md — human-readable report
2. docs/exp_results/curve.png — accumulation curve
3. docs/exp_results/cost_curve.png — token cost convergence

Usage: conda run -n ml python3 _generate_report.py
"""
import os, json, sys
from pathlib import Path

for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(var, None)

RESULT_DIR = Path("docs/exp_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load all result JSONs from the results directory."""
    results = {}
    for f in sorted(RESULT_DIR.glob("*_return.json")):
        method = f.stem.replace("_return", "")
        try:
            data = json.loads(f.read_text())
            results[method] = data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  ⚠ Failed to load {f}: {e}")
    return results


def make_tables(results):
    """Generate comparison tables from results."""
    if not results:
        return "# Baseline Results\n\n_No results found. Run experiments first._"

    lines = [
        "# Baseline Results Report — ExperienceOS",
        "",
        f"_Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Method Summary",
        "",
        "| Method | Total Tasks | Success Rate | Eval SR | Total Tokens | Avg Latency |",
        "|--------|------------:|-------------:|--------:|-------------:|------------:|",
    ]

    for method in ["vanilla", "react", "skillopt", "autoharness"]:
        d = results.get(method)
        if not d:
            lines.append(f"| {method} | — | — | — | — | — |")
            continue
        summary = d.get("summary", {})
        lines.append(
            f"| {method} | {summary.get('total', '?')} "
            f"| {summary.get('success_rate', 0)*100:.1f}% "
            f"| {summary.get('eval_success_rate', 0)*100:.1f}% "
            f"| {summary.get('total_tokens', 0):,} "
            f"| {summary.get('avg_latency', 0):.1f}s |"
        )

    lines += [
        "",
        "## Per-Task Breakdown",
        "",
    ]

    for method in results:
        d = results[method]
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| # | Phase | Task ID | Success | Reward | Tokens | Latency | Path |")
        lines.append("|---|-------|---------|---------|-------|-------|---------|------|")
        for r in d.get("results", []):
            mark = "✅" if r["success"] else "❌"
            lines.append(
                f"| {r['idx']} | {r['phase']} | {r['task_id']} "
                f"| {mark} | {r['reward']} | {r['tokens']:,} "
                f"| {r['latency']:.1f}s | {r['path']} |"
            )
        lines.append("")

    lines += [
        "## Key Observations",
        "",
        "- **Vanilla**: Single-turn LLM needs the task description and tool schema in one prompt.",
        "  Low success rate on tasks requiring multi-step reasoning.",
        "- **ReAct**: Multi-step agent with tool-use per step, no experience reuse.",
        "  Higher token cost per task due to repeated reasoning.",
        "- **SkillOpt**: Text skill injected into system prompt. Same reasoning path as ReAct,",
        "  but guided by the skill document. Modest improvement if skill is informative.",
        "- **AutoHarness**: After warmup accumulation, compiles verified executable harness.",
        "  Eval tasks use harness → agent fallback. Expected to show token reduction",
        "  on harness hits, with success rate >= agent fallback.",
        "",
        "## Accumulation Curve",
        "",
        "See `curve.png` for the rolling success rate over task sequence.",
        "See `cost_curve.png` for token cost convergence.",
        "",
    ]

    report = "\n".join(lines)
    (RESULT_DIR / "README.md").write_text(report)
    print(f"Report saved: {RESULT_DIR / 'README.md'}")
    return report


def generate_curve(results):
    """Generate accumulation curve from results."""
    try:
        from experience_os.experiments.curve import plot_curve, plot_cost_curve

        result_files = [
            str(RESULT_DIR / f"{m}_return.json")
            for m in ["vanilla", "react", "skillopt", "autoharness"]
            if (RESULT_DIR / f"{m}_return.json").exists()
        ]

        if not result_files:
            print("  No result files to plot.")
            return

        print(f"\n  Generating curve from: {result_files}")
        curve_path = plot_curve(result_files, output=str(RESULT_DIR / "curve.png"), window=3)
        print(f"  ✅ Accumulation curve: {curve_path}")

        cost_path = plot_cost_curve(result_files, output=str(RESULT_DIR / "cost_curve.png"), window=3)
        print(f"  ✅ Cost curve: {cost_path}")

    except ImportError as e:
        print(f"  ⚠ Cannot generate curves: {e}")
    except Exception as e:
        print(f"  ⚠ Curve generation error: {e}")


def main():
    results = load_results()
    print(f"\nLoaded {len(results)} result files:")
    for m in results:
        sr = results[m].get("summary", {}).get("success_rate", 0)
        print(f"  {m}: SR={sr*100:.1f}%")

    make_tables(results)
    generate_curve(results)


if __name__ == "__main__":
    main()
