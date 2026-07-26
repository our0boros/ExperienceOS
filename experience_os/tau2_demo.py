"""τ-bench 集成 demo：在真实 τ-bench 任务上跑通完整流程。

流程：
    1. 加载 τ-bench retail 任务
    2. 按任务类型分组，划分 Warm-up / Evaluation 池
    3. ACCUMULATION：用 tau2 内置 llm_agent 跑仿真，记录轨迹
    4. 自动触发 Harness 归纳（LLM 合成 + 沙盒验证）
    5. DEPLOYMENT：对 Evaluation 任务优先使用 Harness，回退 Agent
    6. 汇报成功率 + Token 对比

使用方式::

    # 先安装 tau2（一次即可）
    uv pip install -e ./tau2-bench

    # 运行 demo
    experience-os tau2-demo --domain retail --max-steps 15
"""

from __future__ import annotations

import argparse
import logging
import time

from experience_os.config import Config
from experience_os.environment import TaskRequest
from experience_os.models import Trajectory
from experience_os.runtime import Runtime, SystemMode
from experience_os.tau2_adapter import (
    Tau2Environment,
    _extract_task_description,
    convert_simulation,
    extract_task_params,
    infer_task_type,
    run_tau2_simulation,
    split_tasks,
)

log = logging.getLogger(__name__)


def check_tau2() -> bool:
    """检查 tau2 是否可导入。"""
    try:
        import tau2  # noqa: F401
        return True
    except ImportError:
        return False


def run_tau2_demo(
    config: Config,
    domain: str = "retail",
    warmup_size: int = 3,
    eval_size: int = 3,
    max_steps: int = 15,
    llm_model: str = "",
    llm_api_base: str = "",
    task_type: str = "",
    solo_mode: bool = False,
) -> None:
    """运行 τ-bench 端到端 demo。

    Args:
        task_type: 指定任务类型（按首条参考动作名筛选），空则自动选最大的
        solo_mode: 使用 solo 模式（dummy_user 替代 user_simulator）
    """

    # ================================================================
    # 0. 前置检查
    # ================================================================
    print("=" * 64)
    print("  ExperienceOS × τ-bench 集成 Demo")
    print("=" * 64)

    if not check_tau2():
        print("\n[X] tau2 未安装。请先执行：")
        print("  cd /home/our0boros/Project/ExecutableExperience")
        print("  source .venv/bin/activate && uv pip install -e ./tau2-bench")
        return

    # 确定 LLM 配置
    if config.llm.backend == "ollama":
        # litellm 需要 ollama/ 前缀，且模型名需与 ollama 中的完全一致
        ollama_model = config.llm.ollama_model  # e.g. "qwen2.5:7b"
        tau2_model = llm_model or f"ollama/{ollama_model}"
        tau2_api_base = llm_api_base or config.llm.ollama_base_url.replace("/v1", "")
    else:
        # DeepInfra 不直接支持 litellm 的 ollama 格式，
        # tau2 的 litellm 需要用 deepinfra/ 前缀
        tau2_model = llm_model or f"deepinfra/{config.llm.deepinfra_model}"
        tau2_api_base = llm_api_base or ""

    print(f"  Backend:      {config.llm.backend}")
    print(f"  τ2 LLM:       {tau2_model}")
    print(f"  EOS LLM:      {config.llm.model}")
    print(f"  Domain:       {domain}")
    print(f"  Task type:    {task_type or '(auto: largest)'}")
    print(f"  Solo mode:    {solo_mode}")
    print(f"  Warm-up:      {warmup_size} tasks")
    print(f"  Evaluation:   {eval_size} tasks")
    print(f"  Max steps:    {max_steps}")

    # ================================================================
    # 1. 加载任务 + 数据划分
    # ================================================================
    print(f"\n[1] 加载 {domain} 域任务...")
    # 动态导入域的 get_tasks
    try:
        mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
        all_tasks = mod.get_tasks("base")
    except Exception as exc:
        log.error("无法加载 %s 域任务: %s", domain, exc)
        return

    print(f"  共 {len(all_tasks)} 个任务")

    warmup_typed, evaluation_typed, groups = split_tasks(all_tasks, min_support=warmup_size)
    print(f"  任务类型分组: {len(groups)} 种")
    for tt, group in sorted(groups.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"    {tt}: {len(group)} 个")

    # 选择指定类型或最大的一个类型做实验
    if groups:
        if task_type and task_type in groups:
            best_type = task_type
        else:
            best_type = max(groups, key=lambda k: len(groups[k]))
        best_group = groups[best_type]
        warmup = best_group[:warmup_size]
        evaluation = best_group[warmup_size : warmup_size + eval_size]
        print(f"\n  选择类型: {best_type} ({len(best_group)} 个)")
        print(f"  Warm-up: {len(warmup)} 个, Evaluation: {len(evaluation)} 个")
    else:
        warmup = all_tasks[:warmup_size]
        evaluation = all_tasks[warmup_size : warmup_size + eval_size]
        print(f"  Warm-up: {len(warmup)} 个, Evaluation: {len(evaluation)} 个")

    # ================================================================
    # 2. 初始化 Runtime + ping
    # ================================================================
    print("\n[2] 初始化 Runtime...")
    # 使用 MockEnvironment 作为 harness 执行的通用环境接口
    # (tau2 环境由 Tau2Environment 适配器管理)
    from experience_os.environment import MockEnvironment

    # 清空旧数据
    import shutil

    if config.data_dir.exists():
        shutil.rmtree(config.data_dir)
    config.ensure_dirs()

    env = MockEnvironment()  # placeholder, tau2 环境按 task 独立构建
    rt = Runtime(config, env)

    if not rt.llm.ping():
        print("[X] EOS LLM 不可达")
        return
    print("[OK] EOS LLM 可达")

    # ================================================================
    # 3. ACCUMULATION：跑 tau2 仿真，记录轨迹
    # ================================================================
    print(f"\n[3] ACCUMULATION — 跑 {len(warmup)} 个 tau2 仿真...")
    rt.set_mode(SystemMode.ACCUMULATION)

    accum_tokens = 0
    accum_successes = 0
    for i, task in enumerate(warmup, 1):
        task_type = infer_task_type(task)
        task_desc = _extract_task_description(task)

        print(f"\n  --- Warm-up {i}/{len(warmup)}: task={task.id} type={task_type} ---")
        print(f"      {task_desc[:80]}")
        t0 = time.time()

        try:
            sim = run_tau2_simulation(
                domain=domain,
                task=task,
                llm_model=tau2_model,
                llm_api_base=tau2_api_base,
                max_steps=max_steps,
                seed=42 + i,
                solo_mode=solo_mode,
            )
            elapsed = time.time() - t0

            # 转换轨迹
            traj = convert_simulation(sim, task, task_type)
            rt.repo.add_trajectory(traj)

            reward = sim.reward_info.reward if sim.reward_info else 0.0
            success = reward >= 1.0
            tokens = int(sim.agent_cost or 0) or len(str(traj.steps)) * 100  # 估算
            accum_tokens += tokens
            if success:
                accum_successes += 1

            # 更新统计
            stats = rt.repo.get_stats(task_type)
            stats.total_executions += 1
            stats.agent_executions += 1
            if success:
                stats.agent_successes += 1
            rt.repo.save_stats(task_type)

            print(f"      reward={reward:.2f} steps={len(traj.steps)} "
                  f"tokens≈{tokens} latency={elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"      [X] 仿真失败 ({elapsed:.1f}s): {exc}")
            # 记录失败轨迹
            traj = Trajectory(
                task_id=task.id,
                task_description=task_desc,
                task_type=task_type,
                steps=[],
                outcome="failure",
                latency_seconds=elapsed,
            )
            rt.repo.add_trajectory(traj)

    print(f"\n  积累完成: {accum_successes}/{len(warmup)} 成功, ≈{accum_tokens} tokens")

    # ================================================================
    # 4. 归纳 Harness
    # ================================================================
    print("\n[4] 检查归纳触发...")
    induced = []
    for task_type in rt.repo.all_task_types():
        trigger = rt.inductor.check_triggers(task_type)
        support = rt.repo.support_count(task_type)
        print(f"  {task_type}: support={support} trigger={trigger or 'none'}")
        if trigger:
            print(f"  → 触发归纳 ({trigger})...")
            # 构建 Tau2Environment 用于沙盒验证
            try:
                # 用 warmup 中的第一个同类型任务构建验证环境
                same_type_tasks = [
                    t for t in warmup if infer_task_type(t) == task_type
                ]
                if same_type_tasks:
                    validate_env = Tau2Environment(domain, same_type_tasks[0])
                    harness = rt.inductor.induce(task_type, validate_env)
                    if harness:
                        print(f"  [OK] {harness.full_name} APPROVED "
                              f"(replay sr={harness.verification.success_rate:.2f})")
                        induced.append(harness)
                    else:
                        print(f"  [X] 归纳失败或被拒绝")
                else:
                    # 用 MockEnvironment 验证（退化）
                    harness = rt.inductor.induce(task_type, env)
                    if harness:
                        print(f"  [OK] {harness.full_name} APPROVED (mock validation)")
                        induced.append(harness)
            except Exception as exc:
                print(f"  [X] 归纳异常: {exc}")
                import traceback

                traceback.print_exc()

    if not induced:
        print("  (无 Harness 被批准，Deployment 将全部走 Agent fallback)")

    # ================================================================
    # 5. DEPLOYMENT
    # ================================================================
    print(f"\n[5] DEPLOYMENT — {len(evaluation)} 个评估任务")
    rt.set_mode(SystemMode.DEPLOYMENT)

    deploy_tokens = 0
    deploy_successes = 0
    harness_hits = 0

    for i, task in enumerate(evaluation, 1):
        task_type = infer_task_type(task)
        task_desc = _extract_task_description(task)
        params = extract_task_params(task)

        print(f"\n  --- Eval {i}/{len(evaluation)}: task={task.id} type={task_type} ---")
        print(f"      {task_desc[:80]}")

        t0 = time.time()

        # 先尝试 Harness
        used_harness = False
        if induced:
            # 找匹配的 harness
            matching = [h for h in induced if h.task_type == task_type]
            if not matching:
                matching = induced  # 用任意一个试试

            if matching:
                harness = matching[0]
                try:
                    tau_env = Tau2Environment(domain, task)
                    request = TaskRequest(
                        task_id=task.id,
                        task_description=task_desc,
                        task_type=task_type,
                        params=params,
                        expected_output="",
                    )
                    result = tau_env.execute_harness(harness, request)
                    elapsed = time.time() - t0
                    deploy_tokens += result.tokens_used

                    if result.success:
                        deploy_successes += 1
                        harness_hits += 1
                        print(f"      [OK] Harness 成功 tokens={result.tokens_used} "
                              f"latency={elapsed:.1f}s")
                        continue
                    else:
                        print(f"      [X] Harness 失败 (F{result.failure_type or '?'}) "
                              f"→ fallback to agent")
                        used_harness = True
                except Exception as exc:
                    print(f"      [X] Harness 异常: {exc} → fallback to agent")

        # Agent fallback: 跑 tau2 仿真
        try:
            sim = run_tau2_simulation(
                domain=domain,
                task=task,
                llm_model=tau2_model,
                llm_api_base=tau2_api_base,
                max_steps=max_steps,
                seed=100 + i,
                solo_mode=solo_mode,
            )
            elapsed = time.time() - t0
            traj = convert_simulation(sim, task, task_type)
            rt.repo.add_trajectory(traj)

            reward = sim.reward_info.reward if sim.reward_info else 0.0
            tokens = int(sim.agent_cost or 0) or len(str(traj.steps)) * 100
            deploy_tokens += tokens
            if reward >= 1.0:
                deploy_successes += 1

            path = "harness+agent" if used_harness else "agent"
            print(f"      {'[OK]' if reward >= 1.0 else '[X]'} Agent "
                  f"reward={reward:.2f} tokens≈{tokens} "
                  f"path={path} latency={elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"      [X] Agent 仿真失败 ({elapsed:.1f}s): {exc}")

    # ================================================================
    # 6. 汇总
    # ================================================================
    print("\n" + "=" * 64)
    print("  汇总")
    print("=" * 64)
    print(f"  Warm-up:    {accum_successes}/{len(warmup)} 成功, ≈{accum_tokens} tokens")
    print(f"  Deployment: {deploy_successes}/{len(evaluation)} 成功, ≈{deploy_tokens} tokens")
    print(f"  Harness命中: {harness_hits}/{len(evaluation)}")
    if accum_tokens > 0 and deploy_tokens < accum_tokens:
        saving = (1 - deploy_tokens / accum_tokens) * 100
        print(f"  Token变化:  {saving:+.0f}%")
    print(f"  Induced Harnesses: {len(induced)}")

    # 显示 harness 代码
    for h in induced:
        print(f"\n  Harness {h.full_name}:")
        print("  " + "-" * 58)
        for line in h.procedure_code.splitlines()[:20]:
            print(f"    {line}")
