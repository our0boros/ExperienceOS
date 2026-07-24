"""ExperienceOS command-line interface.

Usage::

    # check LLM backend connectivity
    experience-os ping

    # run the built-in mock demo (accumulation → induction → deployment)
    experience-os demo

    # show repository status
    experience-os status

    # list harnesses
    experience-os harnesses

Environment variables control the backend (see ``.env.example``):
    EOS_LLM_BACKEND=ollama   # or deepinfra
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from experience_os.config import Config
from experience_os.environment import MockEnvironment
from experience_os.models import TaskTypeStats
from experience_os.runtime import Runtime, SystemMode


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experience-os", description="ExperienceOS runtime")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ping", help="check LLM backend connectivity")
    sub.add_parser("demo", help="run the mock end-to-end demo")
    sub.add_parser("status", help="show repository status")
    sub.add_parser("harnesses", help="list compiled harnesses")
    sub.add_parser("env-info", help="collect and print environment info")

    bl_parser = sub.add_parser("baseline", help="run τ-bench baseline evaluation")
    bl_parser.add_argument("--model", required=True, help="litellm model name (e.g. ollama/qwen2.5:7b)")
    bl_parser.add_argument("--domain", default="retail", help="τ-bench domain")
    bl_parser.add_argument("--max-tasks", type=int, default=10, help="max tasks to evaluate (0=all)")
    bl_parser.add_argument("--max-steps", type=int, default=30, help="max simulation steps per task")
    bl_parser.add_argument("--solo", action="store_true", help="use solo mode")
    bl_parser.add_argument("--api-base", default="", help="custom API base URL")
    bl_parser.add_argument("--api-key", default="", help="custom API key")
    bl_parser.add_argument("--output", default="", help="output JSON file path")

    tau2_parser = sub.add_parser("tau2-demo", help="run τ-bench integration demo")
    tau2_parser.add_argument("--domain", default="retail", help="τ-bench domain")
    tau2_parser.add_argument("--warmup", type=int, default=3, help="warm-up pool size")
    tau2_parser.add_argument("--eval", type=int, default=3, help="evaluation pool size")
    tau2_parser.add_argument("--max-steps", type=int, default=15, help="max steps per simulation")
    tau2_parser.add_argument("--llm-model", default="", help="override tau2 LLM model")
    tau2_parser.add_argument("--llm-api-base", default="", help="override tau2 LLM api_base")
    tau2_parser.add_argument("--task-type", default="", help="filter by task type (first golden action name)")
    tau2_parser.add_argument("--solo", action="store_true", help="use solo mode (dummy user instead of user simulator)")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    config = Config()

    if args.command == "ping":
        return _cmd_ping(config)
    if args.command == "demo":
        from experience_os.demo import run_demo

        run_demo(config)
        return 0
    if args.command == "tau2-demo":
        from experience_os.tau2_demo import run_tau2_demo

        run_tau2_demo(
            config,
            domain=args.domain,
            warmup_size=args.warmup,
            eval_size=args.eval,
            max_steps=args.max_steps,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            task_type=args.task_type,
            solo_mode=args.solo,
        )
        return 0
    if args.command == "env-info":
        from experience_os.env_info import collect_env_info, print_env_info, save_env_info

        info = collect_env_info()
        print_env_info(info)
        save_env_info(info)
        return 0
    if args.command == "baseline":
        from experience_os.baseline_eval import run_baseline

        run_baseline(
            model=args.model,
            domain=args.domain,
            max_tasks=args.max_tasks,
            max_steps=args.max_steps,
            solo_mode=args.solo,
            api_base=args.api_base,
            api_key=args.api_key,
            output_file=args.output,
        )
        return 0
    if args.command == "status":
        return _cmd_status(config)
    if args.command == "harnesses":
        return _cmd_harnesses(config)
    return 1


# ======================================================================
# commands
# ======================================================================
def _cmd_ping(config: Config) -> int:
    from experience_os.llm import LLMClient

    client = LLMClient(config.llm)
    print(f"Backend: {config.llm.backend}")
    print(f"Model:   {config.llm.model}")
    print(f"Base URL: {config.llm.base_url}")
    if client.ping():
        print("✓ LLM backend reachable")
        # also test embeddings
        try:
            vec = client.embed("hello")
            print(f"✓ Embeddings OK (dim={len(vec)})")
        except Exception as exc:
            print(f"✗ Embeddings failed: {exc}")
        return 0
    print("✗ LLM backend unreachable")
    return 1


def _cmd_status(config: Config) -> int:
    from experience_os.repository import Repository

    repo = Repository(config)
    print(f"Data dir:        {config.data_dir}")
    print(f"Trajectories:   {len(repo._trajectories)}")  # noqa: SLF001
    print(f"Records:         {len(repo._records)}")  # noqa: SLF001
    print(f"Harnesses:       {len(repo._harnesses)}")  # noqa: SLF001
    active = repo.active_harnesses()
    print(f"  Active:        {len(active)}")
    for h in active:
        print(f"    {h.full_name} (task={h.task_type}, sr={h.verification.success_rate:.2f})")
    print(f"Task types:      {len(repo.all_task_types())}")
    for tt in repo.all_task_types():
        s = repo.get_stats(tt)
        print(
            f"    {tt}: total={s.total_executions} "
            f"harness_sr={s.harness_success_rate:.2f} "
            f"agent_sr={s.agent_success_rate:.2f} "
            f"saved≈{s.estimated_token_savings}"
        )
    return 0


def _cmd_harnesses(config: Config) -> int:
    from experience_os.repository import Repository

    repo = Repository(config)
    for h in repo._harnesses.values():  # noqa: SLF001
        print(f"\n{'='*60}")
        print(f"  {h.full_name}  [{h.status.value}]")
        print(f"  task_type:   {h.task_type}")
        print(f"  capability:  {h.capability}")
        print(f"  version:     v{h.version} (parent={h.parent_id or '-'})")
        print(f"  preconditions: {h.preconditions}")
        print(f"  invariants:  {h.invariants}")
        print(f"  params:      {h.params}")
        print(f"  verification: sr={h.verification.success_rate:.2f} "
              f"tests={h.verification.test_count}")
        print(f"  failures:    {h.failure_counts}")
        print(f"  code ({len(h.procedure_code)} chars):")
        for line in h.procedure_code.splitlines()[:15]:
            print(f"    {line}")
        if len(h.procedure_code.splitlines()) > 15:
            print("    ...")
    if not repo._harnesses:  # noqa: SLF001
        print("(no harnesses compiled yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
