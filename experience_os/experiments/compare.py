"""统一对照实验运行器。

在 τ-bench 上以**相同 backbone + 相同 Warm-up/Eval 划分**运行四种方法，
输出结构一致的 per-task 记录，供积累曲线图与消融实验使用。

方法：
    * ``vanilla``      — 单轮 LLM：任务+工具 schema，一次产出全部工具调用。
    * ``react``        — τ-bench 内置多步 ReAct agent，无积累。
    * ``coe``          — Compilation of Experience: Warm-up 积累 → 归纳 → Eval 部署（Harness 优先）。
    * ``skillopt``     — 读取训练好的 skill 文本注入 agent system prompt。

所有轨迹（完整对话/prompt/回复）写入 **LTS 经验库**（持久底座）+
实验库（临时）。DeepInfra 后端自动顺序运行（不并行，加间隔）。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from experience_os.experience_library import (
    ExperienceLibrary,
    TrajectoryRecord,
    serialize_messages,
)
from experience_os.stores import TraceStore, stores_for
from experience_os.tau2_adapter import infer_task_type

log = logging.getLogger(__name__)


# ======================================================================
# 结果数据结构
# ======================================================================
@dataclass
class TaskResult:
    idx: int
    phase: str
    task_id: str
    task_type: str
    method: str
    success: bool
    reward: float
    tokens: int                # 总 token = prompt + completion
    latency: float
    path: str
    error: str = ""
    messages_json: str = ""  # 完整对话（prompt + 回复）
    prompt_tokens: int = 0     # 输入 token 数（从 API response 获取）
    completion_tokens: int = 0 # 输出 token 数


@dataclass
class ExperimentResult:
    method: str
    model: str
    domain: str
    task_type: str
    warmup_size: int
    eval_size: int
    max_steps: int
    results: list[TaskResult] = field(default_factory=list)
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = f"{self.method}-{self.domain}-{uuid.uuid4().hex[:8]}"

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    @property
    def avg_latency(self) -> float:
        return sum(r.latency for r in self.results) / self.total if self.total else 0.0

    def eval_results(self) -> list[TaskResult]:
        return [r for r in self.results if r.phase == "eval"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = {
            "total": self.total,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": sum(r.prompt_tokens for r in self.results),
            "total_completion_tokens": sum(r.completion_tokens for r in self.results),
            "avg_latency": round(self.avg_latency, 2),
            "eval_success_rate": round(
                sum(1 for r in self.eval_results() if r.success)
                / max(1, len(self.eval_results())), 4,
            ),
        }
        return d


# ======================================================================
# 任务加载与划分
# ======================================================================
def load_tasks(domain: str = "retail", split: str = "base") -> list:
    """加载 tau2 任务。

    Args:
        split: "base" (全部) | "train" | "test"
               tau2 retail 内置 train(74)/test(40)/base 划分。
    """
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
    return mod.get_tasks(split)


def load_train_test_split(domain: str = "retail") -> tuple[list, list]:
    """加载 tau2 原生 train/test 划分。

    返回 (train_tasks, test_tasks)。
    如果该域不支持 split，则回退到 base 全量 + 按 task_type 分组划分。
    """
    try:
        mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
        train = mod.get_tasks("train")
        test = mod.get_tasks("test")
        if train and test:
            return train, test
    except Exception as exc:
        log.warning("域 %s 不支持 train/test split: %s，回退到 base", domain, exc)
    # 回退：用 base 全量，前 N 条做 warmup
    all_tasks = load_tasks(domain, "base")
    return all_tasks, all_tasks


def pick_task_group(tasks: list, task_type: str = "", warmup: int = 3):
    from experience_os.tau2_adapter import split_tasks
    _, _, groups = split_tasks(tasks, min_support=warmup)
    if not groups:
        return tasks, task_type or "unknown"
    if task_type and task_type in groups:
        return groups[task_type], task_type
    best = max(groups, key=lambda k: len(groups[k]))
    return groups[best], best


def _resolve_tau2_model(model: str) -> tuple[str, str]:
    if model.startswith("ollama/"):
        return model, "http://localhost:11434"
    return model, ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_token_usage(messages_json: str) -> tuple[int, int, int]:
    """从序列化的 messages JSON 中提取 token 用量。

    返回 (prompt_tokens, completion_tokens, total_tokens)。
    τ-bench 的 Message.usage 字段记录了每一步的 API 实际用量。
    如果 usage 不可用（如 fallback LLM），则从文本估算。
    """
    if not messages_json or len(messages_json) < 20:
        return 0, 0, 0
    try:
        msgs = json.loads(messages_json)
    except json.JSONDecodeError:
        total = _estimate_tokens(messages_json)
        return 0, 0, total

    prompt = 0
    completion = 0
    for msg in msgs:
        usage = msg.get("usage") if isinstance(msg, dict) else {}
        if usage:
            prompt += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            completion += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        else:
            # 每轮对话的文本估算
            content = str(msg.get("content", "")) if isinstance(msg, dict) else ""
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            est = max(1, len(content) // 4)
            if role in ("assistant", "tool"):
                completion += est
            else:
                prompt += est

    total = prompt + completion
    if total == 0:
        total = _estimate_tokens(messages_json)
    return prompt, completion, total or max(1, prompt + completion)


def _is_deepinfra(model: str) -> bool:
    return model.startswith("deepinfra/")


def _serialize_task(task: Any) -> str:
    """序列化完整任务对象。"""
    try:
        return json.dumps(
            {"id": task.id, "task_id": task.task_id,
             "description": getattr(task, "description", ""),
             "evaluation_criteria": str(getattr(task, "evaluation_criteria", ""))[:2000],
             "initial_state": str(getattr(task, "initial_state", ""))[:2000]},
            ensure_ascii=False, default=str,
        )
    except Exception:
        return str(task)[:2000]


# ======================================================================
# 方法：vanilla（单轮 LLM）
# ======================================================================
def run_vanilla(task, domain, model, max_steps, solo_mode, seed=42) -> TaskResult:
    """Vanilla baseline：用 τ-bench 标准 simulation 跑 LLM agent（无额外 skill）。

    与 run_react 使用相同的 τ-bench orchestrator/simulation 框架，
    区别是 vanilla 不做任何 prompt 增强。
    """
    from experience_os.tau2_adapter import run_tau2_simulation

    t0 = time.time()
    task_type = infer_task_type(task)
    tau2_model, api_base = _resolve_tau2_model(model)
    try:
        sim = run_tau2_simulation(
            domain=domain, task=task, llm_model=tau2_model,
            llm_api_base=api_base, max_steps=max_steps,
            seed=seed, solo_mode=solo_mode,
        )
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
        messages_json = serialize_messages(msgs)
        pt, ct, tt = _extract_token_usage(messages_json)
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="vanilla", success=reward >= 1.0, reward=reward,
            tokens=tt, prompt_tokens=pt, completion_tokens=ct,
            latency=time.time() - t0, path="agent",
            messages_json=messages_json,
        )
    except Exception as exc:
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="vanilla", success=False, reward=0.0, tokens=0,
            latency=time.time() - t0, path="agent", error=str(exc)[:200],
        )


# ======================================================================
# 方法：react（多步 ReAct）
# ======================================================================
def run_react(task, domain, model, max_steps, solo_mode, seed=42) -> TaskResult:
    from experience_os.tau2_adapter import run_tau2_simulation

    t0 = time.time()
    task_type = infer_task_type(task)
    tau2_model, api_base = _resolve_tau2_model(model)
    try:
        sim = run_tau2_simulation(
            domain=domain, task=task, llm_model=tau2_model,
            llm_api_base=api_base, max_steps=max_steps,
            seed=seed, solo_mode=solo_mode,
        )
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
        messages_json = serialize_messages(msgs)
        pt, ct, tt = _extract_token_usage(messages_json)
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="react", success=reward >= 1.0, reward=reward,
            tokens=tt, prompt_tokens=pt, completion_tokens=ct,
            latency=time.time() - t0, path="agent",
            messages_json=messages_json,
        )
    except Exception as exc:
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="react", success=False, reward=0.0, tokens=0,
            latency=time.time() - t0, path="agent", error=str(exc)[:200],
        )


# ======================================================================
# 方法：skillopt（skill 文本注入）
# ======================================================================
def run_skillopt(task, domain, model, max_steps, solo_mode, skill_text, seed=42) -> TaskResult:
    """读取训练好的 skill 文本，注入 agent system prompt 后跑 τ-bench 仿真。

    skill_text 被 prepend 到 agent 的 system_messages，与 SkillOpt 训练一致。
    """
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.build import build_orchestrator
    from tau2.runner.simulation import run_simulation

    t0 = time.time()
    task_type = infer_task_type(task)
    tau2_model, api_base = _resolve_tau2_model(model)
    llm_args = {"temperature": 0.0, "max_tokens": 4096}
    if api_base:
        llm_args["api_base"] = api_base
    try:
        config = TextRunConfig(
            domain=domain, agent="llm_agent", llm_agent=tau2_model,
            llm_args_agent=llm_args, user="user_simulator",
            llm_user=tau2_model, llm_args_user=llm_args,
            max_steps=max_steps, num_trials=1, seed=seed,
        )
        orchestrator = build_orchestrator(config, task, seed=seed)
        # 注入 skill
        if skill_text.strip():
            _inject_skill(orchestrator.agent, skill_text)
        sim = run_simulation(orchestrator)
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
        messages_json = serialize_messages(msgs)
        pt, ct, tt = _extract_token_usage(messages_json)
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="skillopt", success=reward >= 1.0, reward=reward,
            tokens=tt, prompt_tokens=pt, completion_tokens=ct,
            latency=time.time() - t0, path="agent",
            messages_json=messages_json,
        )
    except Exception as exc:
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="skillopt", success=False, reward=0.0, tokens=0,
            latency=time.time() - t0, path="agent", error=str(exc)[:200],
        )


def _inject_skill(agent, skill_content: str) -> None:
    """Prepend skill text to τ-bench agent's system messages."""
    skill = skill_content.strip()
    if not skill:
        return
    original = agent.get_init_state

    def _patched(message_history=None):
        state = original(message_history)
        from tau2.data_model.message import SystemMessage
        state.system_messages = [SystemMessage(role="system",
                                                content=f"## Skill\n{skill}")] + list(state.system_messages)
        return state

    agent.get_init_state = _patched


# ======================================================================
# 方法：coe (Compilation of Experience)
# ======================================================================
def run_coe(group, domain, model, warmup, eval_size, max_steps, solo_mode,
                    *, skip_validation=False, no_versioning=False,
                    warmup_tasks=None, eval_tasks=None,
                    experiment_id: str = "", trace_store: Optional[TraceStore] = None,
                    library: Optional[ExperienceLibrary] = None) -> list[TaskResult]:
    from experience_os.config import Config
    from experience_os.environment import MockEnvironment
    from experience_os.runtime import Runtime, SystemMode
    from experience_os.tau2_adapter import (
        Tau2Environment, _extract_task_description,
        convert_simulation, run_tau2_simulation,
    )

    cfg = Config()
    # 不清理 data_dir——LTS 数据库是永久底座，追加不删除。
    # Runtime 的 Repository 创建 JSON 文件时会自动覆盖旧的。
    # 配置归纳用 LLM 后端（与实验模型一致）
    if model.startswith("deepinfra/"):
        cfg.llm.backend = "deepinfra"
        cfg.llm.deepinfra_model = model.split("/", 1)[-1]
    elif model.startswith("ollama/"):
        cfg.llm.backend = "ollama"
        cfg.llm.ollama_model = model.split("/", 1)[-1]
    cfg.ensure_dirs()
    if skip_validation:
        cfg.induction.validation_threshold = 0.0
    rt = Runtime(cfg, MockEnvironment(), library=library)
    trace_store = trace_store or rt.trace_store
    tau2_model, api_base = _resolve_tau2_model(model)
    sequential = _is_deepinfra(model)

    warmup_tasks = warmup_tasks if warmup_tasks is not None else group[:warmup]
    eval_tasks = eval_tasks if eval_tasks is not None else group[warmup: warmup + eval_size]
    from experience_os.input_resolver import ArtifactInputResolver

    input_resolver = ArtifactInputResolver()
    results: list[TaskResult] = []
    warmup_results: list[TaskResult] = []  # 收集用于子步骤提取
    idx = 1

    rt.set_mode(SystemMode.ACCUMULATION)
    rt.set_phase("warmup")
    for i, task in enumerate(warmup_tasks, 1):
        tt = infer_task_type(task)
        t0 = time.time()
        try:
            sim = run_tau2_simulation(
                domain=domain, task=task, llm_model=tau2_model,
                llm_api_base=api_base, max_steps=max_steps,
                seed=42 + i, solo_mode=solo_mode,
            )
            traj = convert_simulation(sim, task, tt)
            rt.repo.add_trajectory(traj)
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
            messages_json = serialize_messages(msgs)
            pt, ct, tt_tok = _extract_token_usage(messages_json)
            stats = rt.repo.get_stats(tt)
            stats.total_executions += 1
            stats.agent_executions += 1
            if reward >= 1.0:
                stats.agent_successes += 1
            rt.repo.save_stats(tt)
            wr = TaskResult(
                idx=idx, phase="warmup", task_id=task.id, task_type=tt,
                method="coe", success=reward >= 1.0, reward=reward,
                tokens=tt_tok, prompt_tokens=pt, completion_tokens=ct,
                latency=time.time() - t0, path="agent",
                messages_json=messages_json,
            )
            results.append(wr)
            warmup_results.append(wr)
        except Exception as exc:
            wr = TaskResult(
                idx=idx, phase="warmup", task_id=task.id, task_type=tt,
                method="coe", success=False, reward=0.0, tokens=0,
                latency=time.time() - t0, path="agent", error=str(exc)[:200],
            )
            results.append(wr)
            warmup_results.append(wr)
        idx += 1

        # ── 在线检测：每个成功任务后，提取 tool calls 写入 substeps → 检查触发 ──
        if reward >= 1.0 and messages_json:
            try:
                _log_tool_calls_as_substeps(wr, experiment_id, trace_store,
                                            source_method="coe")
            except Exception:
                pass

        if sequential and i < len(warmup_tasks):
            time.sleep(3)

    # 归纳（双级：全任务 + 子步骤模式） — 批量补充（可能已被在线检测覆盖）
    induced = []
    warmup_map = {t.id: t for t in warmup_tasks}

    def _env_builder(traj):
        t = warmup_map.get(traj.task_id)
        if t is not None:
            return Tau2Environment(domain, t, solo_mode=solo_mode)
        return Tau2Environment(domain, warmup_tasks[0], solo_mode=solo_mode)

    # 从 warmup 轨迹的 messages_json 中直接提取 tool calls 写入 substeps 表
    print(f"  [substep] extracting from {len(warmup_results)} warmup results...")
    try:
        from experience_os.experience_library import SubStepRecord
        import json as _json
        substep_store = trace_store or rt.trace_store
        n_sub = 0
        for r in warmup_results:
            if not r.messages_json:
                continue
            try:
                msgs = _json.loads(r.messages_json) if isinstance(r.messages_json, str) else r.messages_json
            except Exception:
                continue
            if not isinstance(msgs, list):
                continue
            plan_idx = 0
            for msg in msgs:
                tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else []
                if not isinstance(tool_calls, list):
                    continue
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    # 兼容两种格式：OpenAI ({function: {name, arguments}}) 和扁平 ({name, arguments})
                    fn = tc.get("function", tc)
                    tool_name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try: args = _json.loads(args)
                        except Exception: args = {"raw": args}
                    intent = tool_name  # TODO: LLM infer better intent later
                    rec = SubStepRecord(
                        trajectory_id=r.task_id,
                        experiment_id=experiment_id or "unknown",
                        plan_idx=plan_idx,
                        intent=intent,
                        tool_name=tool_name,
                        params_json=_json.dumps(args, ensure_ascii=False),
                        success=True,  # tool call was made; actual success TBD by result
                        execution_mode="agent",
                        source="react",
                        parent_task_type=r.task_type,
                        parent_task_success=r.success,
                    )
                    substep_store.append_substep(rec)
                    plan_idx += 1
                    n_sub += 1
        if n_sub:
            print(f"  [substep] extracted {n_sub} tool calls as substeps")
            log.info("Extracted %d substeps from warmup messages", n_sub)
    except Exception as exc:
        log.warning("Substep extraction skipped: %s", exc)

    # 全任务级 + 子步骤级归纳
    # 限定子步骤发现范围到当前实验（避免跨实验污染旧轨迹）
    rt.inductor._current_experiment_id = experiment_id
    for tt in rt.repo.all_task_types():
        triggers = rt.inductor.check_triggers(tt)  # 新签名：返回 list[(trigger, pattern)]
        if not triggers:
            continue
        same = [t for t in warmup_tasks if infer_task_type(t) == tt]
        if not same:
            continue
        for trigger, pattern in triggers:
            try:
                venv = Tau2Environment(domain, same[0], solo_mode=solo_mode)
                if pattern is not None:
                    h = rt.inductor.induce(
                        tt, venv, trigger,
                        substep_pattern=pattern, env_builder=_env_builder,
                    )
                else:
                    h = rt.inductor.induce(tt, venv, trigger, env_builder=_env_builder)
                if h:
                    induced.append(h)
            except Exception as exc:
                log.warning("induce %s failed: %s", tt, exc)

    # eval: harness 优先
    rt.set_mode(SystemMode.DEPLOYMENT)
    rt.set_phase("eval")
    for i, task in enumerate(eval_tasks, 1):
        tt = infer_task_type(task)
        desc = _extract_task_description(task)
        params = input_resolver.resolve(task, []).params
        t0 = time.time()
        used_harness = False
        path = "agent"

        matching = [h for h in induced if h.task_type == tt] or induced
        if matching:
            h = matching[0]
            try:
                tenv = Tau2Environment(domain, task)
                req = type("R", (), {"task_id": task.id, "task_description": desc,
                                     "task_type": tt, "params": params,
                                     "expected_output": ""})()
                r = tenv.execute_harness(h, req)
                if r.success:
                    results.append(TaskResult(
                        idx=idx, phase="eval", task_id=task.id, task_type=tt,
                        method="coe", success=True, reward=1.0,
                        tokens=r.tokens_used, latency=time.time() - t0,
                        path="harness",
                    ))
                    idx += 1
                    continue
                used_harness = True
            except Exception as exc:
                log.warning("harness exec failed: %s", exc)
                used_harness = True

        try:
            sim = run_tau2_simulation(
                domain=domain, task=task, llm_model=tau2_model,
                llm_api_base=api_base, max_steps=max_steps,
                seed=100 + i, solo_mode=solo_mode,
            )
            traj = convert_simulation(sim, task, tt)
            rt.repo.add_trajectory(traj)
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
            messages_json = serialize_messages(msgs)
            pt, ct, tt_tok = _extract_token_usage(messages_json)
            path = "harness+agent" if used_harness else "agent"
            results.append(TaskResult(
                idx=idx, phase="eval", task_id=task.id, task_type=tt,
                method="coe", success=reward >= 1.0, reward=reward,
                tokens=tt_tok, prompt_tokens=pt, completion_tokens=ct,
                latency=time.time() - t0, path=path,
                messages_json=messages_json,
            ))
        except Exception as exc:
            results.append(TaskResult(
                idx=idx, phase="eval", task_id=task.id, task_type=tt,
                method="coe", success=False, reward=0.0, tokens=0,
                latency=time.time() - t0, path=path, error=str(exc)[:200],
            ))

        # P2.2 eval 在线反馈：agent/harness+agent 路径的轨迹也提取子步骤
        _last = results[-1]
        if _last.path != "harness" and _last.messages_json and _last.success:
            try:
                _log_tool_calls_as_substeps(_last, experiment_id, trace_store,
                                            source_method="coe")
            except Exception:
                pass

        idx += 1
        if sequential and i < len(eval_tasks):
            time.sleep(3)

    rt.close()
    return results


# ======================================================================
# 方法：coe online（在线积累模式）
# ======================================================================
def run_coe_online(
    tasks: list,
    domain: str,
    model: str,
    max_steps: int = 30,
    solo_mode: bool = False,
    *,
    experiment_id: str = "",
    trace_store: Optional[TraceStore] = None,
    library: Optional[ExperienceLibrary] = None,
) -> list[TaskResult]:
    """CoE 在线积累模式：边执行边归纳，新 harness 立即可用。

    与传统的 warmup→批量归纳→eval 不同，此模式将所有任务放在
    一个顺序流中。每完成一个任务：
      1. 尝试已归纳的 harness（如有匹配）
      2. 若 harness 命中 → 直接执行（绕过 LLM）
      3. 若 harness 失败/无匹配 → agent 执行
      4. 提取 substeps → 检查触发条件 → 立即归纳
      5. 新 harness 加入活跃集合 → 下一个任务可用

    这是 ExperienceOS 的核心创新验证模式：**不依赖预积累**，
    从零开始，通过在线学习逐步提升性能。
    """
    from experience_os.config import Config
    from experience_os.environment import MockEnvironment
    from experience_os.runtime import Runtime, SystemMode
    from experience_os.tau2_adapter import (
        Tau2Environment,
        _extract_task_description,
        convert_simulation,
        infer_task_type,
        run_tau2_simulation,
    )
    from experience_os.input_resolver import ArtifactInputResolver

    cfg = Config()
    # 配置归纳用 LLM 后端
    if model.startswith("deepinfra/"):
        cfg.llm.backend = "deepinfra"
        cfg.llm.deepinfra_model = model.split("/", 1)[-1]
    elif model.startswith("ollama/"):
        cfg.llm.backend = "ollama"
        cfg.llm.ollama_model = model.split("/", 1)[-1]
    cfg.ensure_dirs()

    rt = Runtime(cfg, MockEnvironment(), library=library)
    trace_store = trace_store or rt.trace_store
    tau2_model, api_base = _resolve_tau2_model(model)
    sequential = _is_deepinfra(model)
    input_resolver = ArtifactInputResolver()
    results: list[TaskResult] = []

    def _build_env(t):
        """为指定任务构建独立 Tau2Environment。"""
        return Tau2Environment(domain, t, solo_mode=solo_mode)

    rt.set_mode(SystemMode.DEPLOYMENT)  # 在线模式：有 harness 就用，没有就走 agent
    rt.set_phase("online")

    # 清理旧实验的 ACTIVE harness，确保在线模式从零开始
    # P1.1 去重会阻止对已有 harness 的 pattern 重新归纳
    for h in list(rt.repo.active_harnesses()):
        rt.repo.deprecate(h.id)

    # 初始化 HarnessRegistry（后续 agent 工具调用会被拦截执行 harness）
    rt.registry.load_all()

    print(f"  [online] 开始在线积累: {len(tasks)} 个任务, min_support={cfg.induction.min_support}")

    for i, task in enumerate(tasks, 1):
        tt = infer_task_type(task)
        desc = _extract_task_description(task)
        params = input_resolver.resolve(task, []).params
        t0 = time.time()
        used_harness = False
        path = "agent"

        # ── Agent 执行（registry 拦截工具调用，命中则走 harness）──
        try:
            sim = run_tau2_simulation(
                domain=domain, task=task, llm_model=tau2_model,
                llm_api_base=api_base, max_steps=max_steps,
                seed=42 + i, solo_mode=solo_mode,
                harness_registry=rt.registry,
            )
            traj = convert_simulation(sim, task, tt)
            rt.repo.add_trajectory(traj)
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
            messages_json = serialize_messages(msgs)
            pt, ct, tt_tok = _extract_token_usage(messages_json)
            intercepts = getattr(sim, '_harness_intercept_count', 0)
            path = f"harness+agent({intercepts})" if intercepts > 0 else "agent"
        except Exception as exc:
            results.append(TaskResult(
                idx=i, phase="online", task_id=task.id, task_type=tt,
                method="coe", success=False, reward=0.0, tokens=0,
                latency=time.time() - t0, path=path, error=str(exc)[:200],
            ))
            if sequential:
                time.sleep(3)
            continue

        tr = TaskResult(
            idx=i, phase="online", task_id=task.id, task_type=tt,
            method="coe", success=reward >= 1.0, reward=reward,
            tokens=tt_tok, prompt_tokens=pt, completion_tokens=ct,
            latency=time.time() - t0, path=path,
            messages_json=messages_json,
        )
        results.append(tr)

        # ── Step 3: 提取 substeps ──
        if reward >= 1.0 and messages_json:
            try:
                _log_tool_calls_as_substeps(tr, experiment_id, trace_store,
                                            source_method="coe")
            except Exception:
                pass

        # ── Step 4: 检查触发条件 + 在线归纳 ──
        rt.inductor._current_experiment_id = experiment_id
        for check_tt in rt.repo.all_task_types():
            triggers = rt.inductor.check_triggers(check_tt)
            if not triggers:
                continue
            same = [t for t in tasks[:i] if infer_task_type(t) == check_tt]
            if not same:
                continue
            for trigger, pattern in triggers:
                try:
                    tenv = Tau2Environment(domain, same[0], solo_mode=solo_mode)
                    h = rt.inductor.induce(
                        check_tt, tenv, trigger,
                        substep_pattern=pattern,
                        env_builder=lambda traj, t=same[0]: Tau2Environment(
                            domain, t, solo_mode=solo_mode,
                        ),
                    )
                    # 从 repo 重载 registry（无论 induce 返回什么格式）
                    before = rt.registry.count
                    rt.registry.load_all()
                    new_count = rt.registry.count - before
                    if new_count > 0:
                        print(f"  [online] #{i} induced {new_count} harness(es) "
                              f"(total={rt.registry.count})")
                except Exception as exc:
                    log.debug("online induce %s failed: %s", check_tt, exc)

        tag = "[OK]" if reward >= 1.0 else "[X]"
        print(f"  [{i}/{len(tasks)}] {task.id} {tag} path={path} "
              f"tokens={tt_tok} harnesses={rt.registry.count}")

        if sequential and i < len(tasks):
            time.sleep(3)

    rt.close()
    print(f"  [online] 完成: {len(tasks)} 任务, "
          f"归纳 {rt.registry.count} 个 harness, "
          f"SR={sum(1 for r in results if r.success)}/{len(tasks)}")
    return results


# ======================================================================
# 主入口
# ======================================================================
def run_experiment(
    method: str,
    model: str,
    domain: str = "retail",
    warmup: int = 3,
    eval_size: int = 5,
    max_steps: int = 15,
    task_type: str = "",
    solo_mode: bool = False,
    *,
    skip_validation: bool = False,
    no_versioning: bool = False,
    variant: str = "type_split",
    cross_domain: str = "",
    experiment_id: str = "",
    skill_path: str = "",
    inter_task_delay: float = 0.0,
) -> ExperimentResult:
    """运行单方法对照实验。

    Args:
        method: ``vanilla`` | ``react`` | ``coe`` | ``skillopt``
        model: litellm 模型名
        variant: ``type_split`` | ``replay`` | ``cross_domain`` | ``train_test``
        skill_path: skillopt 方法的 skill 文本路径
        inter_task_delay: 任务间间隔秒数（DeepInfra 自动设 3s）

    variant 说明:
        - ``type_split``: 按 task_type 分组，同组前 N 做 warmup，后续做 eval
        - ``replay``: warmup 和 eval 用相同任务（回放验证）
        - ``cross_domain``: 跨域积累（cross_domain 做 warmup，本域做 eval）
        - ``train_test``: 使用 tau2 原生 train/test split（retail: train=74, test=40）
    """
    # DeepInfra 自动顺序
    if _is_deepinfra(model) and inter_task_delay == 0.0:
        inter_task_delay = 3.0

    print(f"\n{'='*60}")
    print(f"  对照实验: {method}  model={model}  domain={domain}")
    print(f"  warmup={warmup} eval={eval_size} max_steps={max_steps} solo={solo_mode}")
    print(f"  variant={variant}  delay={inter_task_delay}s"
          + (f"  cross_domain={cross_domain}" if cross_domain else "")
          + (f"  skill={skill_path}" if skill_path else ""))
    print(f"{'='*60}\n")

    # 实验设计变体
    if variant == "train_test":
        # 使用 tau2 原生 train/test split
        train_tasks, test_tasks = load_train_test_split(domain)
        print(f"  tau2 split: train={len(train_tasks)} test={len(test_tasks)}")
        if task_type:
            from experience_os.tau2_adapter import infer_task_type
            train_tasks = [t for t in train_tasks if infer_task_type(t) == task_type]
            test_tasks = [t for t in test_tasks if infer_task_type(t) == task_type]
            print(f"  筛选 task_type={task_type}: train={len(train_tasks)} test={len(test_tasks)}")
        warmup_tasks = train_tasks[:warmup]
        eval_tasks = test_tasks[:eval_size]
        chosen_type = task_type or "all"
        group = train_tasks  # coe 用 train 全量做积累
    else:
        tasks = load_tasks(domain)
        group, chosen_type = pick_task_group(tasks, task_type, warmup)
        print(f"  任务类型: {chosen_type} ({len(group)} 个)")

        if variant == "replay":
            warmup_tasks = group[:warmup]
            eval_tasks = group[:eval_size]
        elif variant == "cross_domain" and cross_domain:
            cd_tasks = load_tasks(cross_domain)
            cd_group, _ = pick_task_group(cd_tasks, task_type, warmup)
            warmup_tasks = cd_group[:warmup]
            eval_tasks = group[:eval_size]
            print(f"  跨域积累: {cross_domain} → 验证: {domain}")
        else:
            warmup_tasks = group[:warmup]
            eval_tasks = group[warmup: warmup + eval_size]

    stream = warmup_tasks + eval_tasks

    eid = experiment_id or f"{method}-{domain}-{variant}-{uuid.uuid4().hex[:8]}"
    # LTS 持久库 + 实验库
    lts = ExperienceLibrary.persistent()
    exp_lib = ExperienceLibrary.experiment(eid)
    lts_trace_store, _, _ = stores_for(lts)
    exp_trace_store, _, _ = stores_for(exp_lib)

    # 加载 skill（skillopt 方法）
    skill_text = ""
    if method == "skillopt":
        if not skill_path:
            # 用初始 seed skill
            skill_path = "SkillOpt/skillopt/envs/tau2/skills/initial.md"
        skill_text = Path(skill_path).read_text() if Path(skill_path).exists() else ""
        if not skill_text:
            print("  ⚠ skill 文件为空，将用空 skill 运行")

    results: list[TaskResult] = []

    if method == "coe":
        results = run_coe(
            group, domain, model, warmup, eval_size, max_steps, solo_mode,
            skip_validation=skip_validation, no_versioning=no_versioning,
            warmup_tasks=warmup_tasks, eval_tasks=eval_tasks,
            experiment_id=eid, trace_store=lts_trace_store, library=lts,
        )
    else:
        for i, task in enumerate(stream, 1):
            phase = "warmup" if i <= warmup else "eval"
            if method == "vanilla":
                r = run_vanilla(task, domain, model, max_steps, solo_mode)
            elif method == "skillopt":
                r = run_skillopt(task, domain, model, max_steps, solo_mode,
                                 skill_text, seed=42 + i)
            else:  # react
                r = run_react(task, domain, model, max_steps, solo_mode, seed=42 + i)
            r.idx = i
            r.phase = phase
            results.append(r)
            # 写入 LTS + 实验库（完整轨迹）
            rec = _to_trajectory_record(r, eid, domain, task, model, variant)
            lts_trace_store.append(rec)
            exp_trace_store.append(rec)
            tag = "[OK]" if r.success else "[X]"
            print(f"  [{i}/{len(stream)}] {phase} {r.task_id} {tag} "
                  f"reward={r.reward:.2f} tokens={r.tokens} {r.error[:40]}")
            if inter_task_delay and i < len(stream):
                time.sleep(inter_task_delay)

    exp = ExperimentResult(
        method=method, model=model, domain=domain, task_type=chosen_type,
        warmup_size=warmup, eval_size=eval_size, max_steps=max_steps,
        results=results, experiment_id=experiment_id or "unknown",
    )
    _print_summary(exp)
    print(f"  experiment_id: {eid}")
    print(f"  LTS trajs: {len(lts.query_trajectories(experiment_id=eid))} 条（含完整对话）")
    print(f"  实验库: {exp_lib.db_path}")
    lts.close()
    exp_lib.close()
    return exp


def _extract_prediction_accuracy(
    messages: list[dict],
) -> list[dict[str, Any]]:
    """从对话消息中提取每个工具调用的预测准确性。

    对每条 assistant 消息（含 tool_calls），将其 reasoning（content 文本）
    作为隐式"预测契约"，与随后的 tool 消息（实际结果）对比。

    Returns:
        list of dicts with keys: tool_name, prediction_accuracy, quality_label,
        expected_keywords, result_summary.

    参考：docs/ExperienceOS.md §5.1.2 预测质量分层。
    """
    from experience_os.models import _extract_keywords

    results: list[dict[str, Any]] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls", [])
        if not isinstance(tcs, list) or not tcs:
            continue

        # Assistant reasoning = implicit prediction
        reasoning = msg.get("content", "") or ""

        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", tc)
            tool_name = fn.get("name", "")
            if not tool_name:
                continue

            # Extract expected output keywords from reasoning
            expected_keywords = _extract_keywords(reasoning)

            # Find the matching tool result (next tool message for this call_id)
            call_id = tc.get("id", "")
            result_text = ""
            result_success = True

            for j in range(i + 1, min(i + 5, len(messages))):
                nxt = messages[j]
                if not isinstance(nxt, dict):
                    continue
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") == call_id:
                    result_text = str(nxt.get("content", "") or "")
                    break
                # If next assistant message encountered before tool result,
                # the tool call was likely aborted or the result was empty
                if nxt.get("role") == "assistant":
                    break

            # Heuristic: check for error indicators in result
            error_markers = ["error", "Error", "failed", "not found", "invalid",
                            "denied", "unavailable", "cannot", "could not",
                            "does not exist", "no such"]
            has_error = any(m in result_text for m in error_markers)

            if has_error:
                result_success = False

            # Compute prediction accuracy
            if not result_text:
                # No result found → assume accurate (tool call was made)
                prediction_accurate = True
                quality_label = "high_quality" if result_success else "implementation_defect"
            elif has_error:
                # Result contains error → agent predicted success but got error
                prediction_accurate = False if reasoning else True
                quality_label = "implementation_defect"
            elif expected_keywords:
                # Check if expected keywords appear in result
                match_count = sum(
                    1 for kw in expected_keywords
                    if kw.lower() in result_text.lower()
                )
                prediction_accurate = match_count >= max(1, len(expected_keywords) * 0.3)
                quality_label = "high_quality" if prediction_accurate else "lucky_success"
            else:
                # No explicit expectations → assume accurate
                prediction_accurate = True
                quality_label = "high_quality"

            results.append({
                "tool_name": tool_name,
                "prediction_accuracy": 1.0 if prediction_accurate else 0.0,
                "quality_label": quality_label,
                "expected_keywords": expected_keywords,
                "result_summary": result_text[:300] if result_text else "(no result)",
            })

    return results


def _log_tool_calls_as_substeps(
    r: TaskResult, experiment_id: str, trace_store: TraceStore,
    *,
    source_method: str = "",
) -> int:
    """从 TaskResult 的 messages_json 中提取 tool calls，写入 substeps 表。

    Phase A（预测契约）：对每条 tool call 提取 agent reasoning 作为隐式预测，
    与 tool result 对比，记录 prediction_accuracy 和 quality_label。
    """
    import json as _json
    from experience_os.experience_library import SubStepRecord

    if not r.messages_json:
        return 0
    try:
        msgs = _json.loads(r.messages_json) if isinstance(r.messages_json, str) else r.messages_json
    except Exception:
        return 0
    if not isinstance(msgs, list):
        return 0

    # Phase A: 提取预测契约验证结果
    predictions = _extract_prediction_accuracy(msgs)
    pred_by_tool: dict[str, dict] = {}
    for p in predictions:
        tn = p["tool_name"]
        if tn not in pred_by_tool:
            pred_by_tool[tn] = p
        else:
            # 同一工具多次调用，保留最后一条
            pred_by_tool[tn] = p

    n = 0
    plan_idx = 0
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        tcs = msg.get("tool_calls", [])
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", tc)
            tool_name = fn.get("name", "")
            if not tool_name:
                continue
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {"raw": args}

            # Phase A: 查找预测验证结果
            pred_info = pred_by_tool.pop(tool_name, {})
            pred_acc = pred_info.get("prediction_accuracy", 1.0)
            qual_label = pred_info.get("quality_label", "")

            rec = SubStepRecord(
                trajectory_id=r.task_id,
                experiment_id=experiment_id,
                plan_idx=plan_idx,
                intent=tool_name,
                tool_name=tool_name,
                params_json=_json.dumps(args, ensure_ascii=False),
                success=True,
                execution_mode="agent",
                source=source_method or getattr(r, "method", "unknown"),
                parent_task_type=r.task_type,
                parent_task_success=r.success,
                prediction_accuracy=pred_acc,
                quality_label=qual_label,
            )
            trace_store.append_substep(rec)
            plan_idx += 1
            n += 1
    return n


def _to_trajectory_record(r: TaskResult, eid: str, domain: str, task: Any,
                          model: str, variant: str) -> TrajectoryRecord:
    """把 TaskResult + task 对象转为完整轨迹记录。"""
    task_json = _serialize_task(task) if task is not None else ""
    return TrajectoryRecord(
        experiment_id=eid, method=r.method, domain=domain,
        task_id=r.task_id, task_type=r.task_type,
        task_description=str(getattr(task, "description", "")) if task else "",
        idx=r.idx, phase=r.phase, success=r.success, reward=r.reward,
        tokens=r.tokens, latency=r.latency, path=r.path,
        task_json=task_json, messages_json=r.messages_json,
        meta={
            "model": model, "variant": variant, "error": r.error,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
        },
    )


def _print_summary(exp: ExperimentResult) -> None:
    from collections import Counter
    print(f"\n{'='*60}")
    print(f"  汇总: {exp.method}")
    print(f"{'='*60}")
    print(f"  总任务:   {exp.total}")
    print(f"  成功率:   {exp.successes}/{exp.total} ({exp.success_rate:.1%})")
    total_pt = sum(r.prompt_tokens for r in exp.results)
    total_ct = sum(r.completion_tokens for r in exp.results)
    print(f"  总 Token: {exp.total_tokens:,} (prompt={total_pt:,} + completion={total_ct:,})")
    print(f"  平均延迟: {exp.avg_latency:.1f}s")

    # 阶段化统计（warmup vs eval）
    warmup = [r for r in exp.results if r.phase == "warmup"]
    eval_r = [r for r in exp.results if r.phase == "eval"]
    if warmup:
        w_sr = sum(1 for r in warmup if r.success) / len(warmup)
        w_tok = sum(r.tokens for r in warmup)
        print(f"  [Warmup] SR: {w_sr:.1%} ({len(warmup)} tasks)  Token: {w_tok:,}")
    if eval_r:
        e_sr = sum(1 for r in eval_r if r.success) / len(eval_r)
        e_tok = sum(r.tokens for r in eval_r)
        print(f"  [Eval]   SR: {e_sr:.1%} ({len(eval_r)} tasks)  Token: {e_tok:,}")
    paths = Counter(r.path for r in exp.results)
    print(f"  路径分布: {dict(paths)}")
    print(f"{'='*60}\n")


def save_result(exp: ExperimentResult, output_file: str) -> None:
    p = Path(output_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(exp.to_dict(), ensure_ascii=False, indent=2))
    print(f"结果已保存: {p}")
