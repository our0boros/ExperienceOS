"""End-to-end demo of the ExperienceOS loop.

Runs the full cycle on the bundled :class:`MockEnvironment`:

    1. ACCUMULATION phase: execute N similar tasks via the agent fallback.
       After ``MIN_SUPPORT`` (3) successful trajectories, the inductor
       auto-fires and compiles a harness.
    2. DEPLOYMENT phase: execute the same task type again — now the runtime
       should *retrieve the harness* and execute it with near-zero token cost.

This demonstrates the core thesis: *after accumulation, the harness replaces
LLM reasoning, cutting token consumption dramatically*.
"""

from __future__ import annotations

import logging
import time

from experience_os.config import Config
from experience_os.environment import MockEnvironment, TaskRequest
from experience_os.runtime import Runtime, SystemMode

log = logging.getLogger(__name__)

# A set of similar "lookup-and-submit" tasks for the mock environment.
DEMO_TASKS = [
    {"id": "task_1", "desc": "Look up the key 'customer' in the store and submit its value.",
     "key": "customer", "expected": "Alice"},
    {"id": "task_2", "desc": "Find the value of 'product' from the store and submit it.",
     "key": "product", "expected": "Widget"},
    {"id": "task_3", "desc": "Retrieve the 'order' entry from the store and submit it.",
     "key": "order", "expected": "ORD-42"},
    {"id": "task_4", "desc": "Look up 'customer' and submit the value.",
     "key": "customer", "expected": "Alice"},
    {"id": "task_5", "desc": "Get the 'product' value from the store and submit.",
     "key": "product", "expected": "Widget"},
    {"id": "task_6", "desc": "Retrieve 'order' and submit it.",
     "key": "order", "expected": "ORD-42"},
]


def run_demo(config: Config) -> None:
    """Run the accumulation → induction → deployment demo."""
    print("=" * 60)
    print("  ExperienceOS — End-to-End Demo")
    print(f"  Backend: {config.llm.backend}  Model: {config.llm.model}")
    print("=" * 60)

    # fresh state
    import shutil
    if config.data_dir.exists():
        shutil.rmtree(config.data_dir)
    config.ensure_dirs()

    env = MockEnvironment(
        store={"customer": "Alice", "product": "Widget", "order": "ORD-42"}
    )
    rt = Runtime(config, env)

    # ---- ping ----
    print("\n[0] Pinging LLM backend...")
    if not rt.llm.ping():
        print("[X] LLM unreachable. Check EOS_LLM_BACKEND / ollama running.")
        print("  (try: ollama serve)")
        return
    print("[OK] LLM reachable")

    # ---- ACCUMULATION ----
    print("\n[1] ACCUMULATION phase — executing tasks via agent fallback")
    rt.set_mode(SystemMode.ACCUMULATION)
    accum_results = []
    for i, task in enumerate(DEMO_TASKS[:3], 1):  # first 3 → accumulation
        req = TaskRequest(
            task_id=task["id"],
            task_description=task["desc"],
            task_type="lookup_and_submit",
            params={"key": task["key"]},
            expected_output=task["expected"],
        )
        t0 = time.time()
        result = rt.execute(req)
        elapsed = time.time() - t0
        accum_results.append(result)
        status = "[OK]" if result.success else "[X]"
        print(f"  {status} task {i}: success={result.success} "
              f"path={result.path} tokens={result.tokens_used} "
              f"latency={elapsed:.1f}s")

    # show induced harness
    print("\n[2] Checking for induced harnesses...")
    harnesses = rt.repo.active_harnesses()
    if harnesses:
        for h in harnesses:
            print(f"  [OK] {h.full_name} (replay sr={h.verification.success_rate:.2f}, "
                  f"code={len(h.procedure_code)} chars)")
    else:
        print("  (no harnesses yet — checking triggers manually...")
        h = rt.maybe_induce()
        if h:
            print(f"  [OK] Induced {h.full_name}")
        else:
            print("  [X] Induction did not fire (may need more support)")

    # ---- DEPLOYMENT ----
    print("\n[3] DEPLOYMENT phase — should now use harness (zero tokens)")
    rt.set_mode(SystemMode.DEPLOYMENT)
    deploy_results = []
    for i, task in enumerate(DEMO_TASKS[3:], 4):
        req = TaskRequest(
            task_id=task["id"],
            task_description=task["desc"],
            task_type="lookup_and_submit",
            params={"key": task["key"]},
            expected_output=task["expected"],
        )
        t0 = time.time()
        result = rt.execute(req)
        elapsed = time.time() - t0
        deploy_results.append(result)
        status = "[OK]" if result.success else "[X]"
        print(f"  {status} task {i}: success={result.success} "
              f"path={result.path} tokens={result.tokens_used} "
              f"latency={elapsed:.1f}s")

    # ---- summary ----
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    accum_tokens = sum(r.tokens_used for r in accum_results)
    deploy_tokens = sum(r.tokens_used for r in deploy_results)
    print(f"  Accumulation: {len(accum_results)} tasks, {accum_tokens} tokens")
    print(f"  Deployment:   {len(deploy_results)} tasks, {deploy_tokens} tokens")
    if deploy_tokens < accum_tokens and accum_tokens:
        saving = (1 - deploy_tokens / accum_tokens) * 100
        print(f"  Token reduction: {saving:.0f}%")
    print(f"\n  Status: {json.dumps(rt.status(), indent=2)}")

    # ---- show harness code ----
    for h in rt.repo.active_harnesses():
        print(f"\n  Harness {h.full_name} code:")
        print("  " + "-" * 56)
        for line in h.procedure_code.splitlines():
            print(f"    {line}")


import json  # noqa: E402  (used in summary above)
