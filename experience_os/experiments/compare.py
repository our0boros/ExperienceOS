"""统一对照实验运行器。

在 τ-bench 上以**相同 backbone + 相同 Warm-up/Eval 划分**运行四种方法，
输出结构一致的 per-task 记录，供积累曲线图与消融实验使用。

方法：
    * ``vanilla``      — 单轮 LLM：任务+工具 schema，一次产出全部工具调用。
    * ``react``        — τ-bench 内置多步 ReAct agent，无积累。
    * ``autoharness``  — Warm-up 积累 → 归纳 → Eval 部署（Harness 优先）。
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
def load_tasks(domain: str = "retail") -> list:
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
    return mod.get_tasks("base")


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
def run_vanilla(task, domain, model, max_steps, solo_mode) -> TaskResult:
    from experience_os.llm import LLMClient
    from experience_os.config import Config
    from experience_os.tau2_adapter import (
        Tau2Environment, _extract_task_description, extract_task_params,
    )

    cfg = Config()
    if model.startswith("ollama/"):
        cfg.llm.backend = "ollama"
        cfg.llm.ollama_model = model.split("/", 1)[-1]
    elif model.startswith("deepinfra/"):
        cfg.llm.backend = "deepinfra"
        cfg.llm.deepinfra_model = model.split("/", 1)[-1]
    client = LLMClient(cfg.llm)

    t0 = time.time()
    task_type = infer_task_type(task)
    desc = _extract_task_description(task)
    messages_log = []  # 完整 prompt + 回复
    try:
        env = Tau2Environment(domain, task, solo_mode=solo_mode)
        tools = env.get_tools()
        schema_lines = []
        for t in tools:
            fn = t["function"] if isinstance(t, dict) and "function" in t else t
            name = fn.get("name", "")
            params = fn.get("parameters", {}).get("properties", {})
            pstr = ", ".join(f'{k}:{v.get("type","str")}' for k, v in params.items())
            schema_lines.append(f"  - {name}({pstr})")
        schema_txt = "\n".join(schema_lines[:20])

        prompt = (
            "You are a customer-service agent. Emit a JSON object "
            '{"calls": [{"name": str, "arguments": dict}, ...]} with ordered '
            "tool calls. No other text.\n\n"
            f"Available tools:\n{schema_txt}\n\nTask: {desc}\n"
        )
        messages_log.append({"role": "user", "content": prompt})
        data = client.chat_json(
            [{"role": "system", "content": "You output only JSON tool-call plans."},
             {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        messages_log.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
        calls = data.get("calls", []) if isinstance(data, dict) else []
        prompt_text = json.dumps(data, ensure_ascii=False)
        tokens = _estimate_tokens(prompt_text)
        # 估算 input/output 分开
        prompt_tok = _estimate_tokens(desc + schema_txt) + 200  # system + user prompt
        completion_tok = tokens
        for c in calls[:max_steps]:
            name = c.get("name", "")
            args = c.get("arguments", {}) or c.get("args", {})
            if name:
                result = env.call_tool(name, args)
                messages_log.append({"role": "tool", "name": name,
                                     "result": str(result)[:500]})
        reward = 1.0 if env.verify("", "") else 0.0
        messages_json = json.dumps(messages_log, ensure_ascii=False)
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="vanilla", success=reward >= 1.0, reward=reward,
            tokens=tokens, prompt_tokens=prompt_tok, completion_tokens=completion_tok,
            latency=time.time() - t0, path="agent",
            error="" if reward >= 1.0 else "no_match",
            messages_json=messages_json,
        )
    except Exception as exc:
        return TaskResult(
            idx=0, phase="eval", task_id=task.id, task_type=task_type,
            method="vanilla", success=False, reward=0.0, tokens=0,
            latency=time.time() - t0, path="agent", error=str(exc)[:200],
            messages_json=json.dumps(messages_log, ensure_ascii=False),
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
# 方法：autoharness
# ======================================================================
def run_autoharness(group, domain, model, warmup, eval_size, max_steps, solo_mode,
                    *, skip_validation=False, no_versioning=False) -> list[TaskResult]:
    from experience_os.config import Config
    from experience_os.environment import MockEnvironment
    from experience_os.runtime import Runtime, SystemMode
    from experience_os.tau2_adapter import (
        Tau2Environment, _extract_task_description,
        convert_simulation, extract_task_params, run_tau2_simulation,
    )

    cfg = Config()
    # 不清理 data_dir——LTS 数据库是永久底座，追加不删除。
    # Runtime 的 Repository 创建 JSON 文件时会自动覆盖旧的。
    cfg.ensure_dirs()
    if skip_validation:
        cfg.induction.validation_threshold = 0.0
    rt = Runtime(cfg, MockEnvironment())
    tau2_model, api_base = _resolve_tau2_model(model)
    sequential = _is_deepinfra(model)

    warmup_tasks = group[:warmup]
    eval_tasks = group[warmup: warmup + eval_size]
    results: list[TaskResult] = []
    idx = 1

    rt.set_mode(SystemMode.ACCUMULATION)
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
            results.append(TaskResult(
                idx=idx, phase="warmup", task_id=task.id, task_type=tt,
                method="autoharness", success=reward >= 1.0, reward=reward,
                tokens=tt_tok, prompt_tokens=pt, completion_tokens=ct,
                latency=time.time() - t0, path="agent",
                messages_json=messages_json,
            ))
        except Exception as exc:
            results.append(TaskResult(
                idx=idx, phase="warmup", task_id=task.id, task_type=tt,
                method="autoharness", success=False, reward=0.0, tokens=0,
                latency=time.time() - t0, path="agent", error=str(exc)[:200],
            ))
        idx += 1
        if sequential and i < len(warmup_tasks):
            time.sleep(3)

    # 归纳
    induced = []
    for tt in rt.repo.all_task_types():
        trigger = rt.inductor.check_triggers(tt)
        if not trigger:
            continue
        same = [t for t in warmup_tasks if infer_task_type(t) == tt]
        if not same:
            continue
        try:
            venv = Tau2Environment(domain, same[0])
            h = rt.inductor.induce(tt, venv)
            if h:
                induced.append(h)
        except Exception as exc:
            log.warning("induce %s failed: %s", tt, exc)

    # eval: harness 优先
    rt.set_mode(SystemMode.DEPLOYMENT)
    for i, task in enumerate(eval_tasks, 1):
        tt = infer_task_type(task)
        desc = _extract_task_description(task)
        params = extract_task_params(task)
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
                        method="autoharness", success=True, reward=1.0,
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
                method="autoharness", success=reward >= 1.0, reward=reward,
                tokens=tt_tok, prompt_tokens=pt, completion_tokens=ct,
                latency=time.time() - t0, path=path,
                messages_json=messages_json,
            ))
        except Exception as exc:
            results.append(TaskResult(
                idx=idx, phase="eval", task_id=task.id, task_type=tt,
                method="autoharness", success=False, reward=0.0, tokens=0,
                latency=time.time() - t0, path=path, error=str(exc)[:200],
            ))
        idx += 1
        if sequential and i < len(eval_tasks):
            time.sleep(3)

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
        method: ``vanilla`` | ``react`` | ``autoharness`` | ``skillopt``
        model: litellm 模型名
        variant: ``type_split`` | ``replay`` | ``cross_domain``
        skill_path: skillopt 方法的 skill 文本路径
        inter_task_delay: 任务间间隔秒数（DeepInfra 自动设 3s）
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

    tasks = load_tasks(domain)
    group, chosen_type = pick_task_group(tasks, task_type, warmup)
    print(f"  任务类型: {chosen_type} ({len(group)} 个)")

    # 实验设计变体
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

    if method == "autoharness":
        results = run_autoharness(
            group, domain, model, warmup, eval_size, max_steps, solo_mode,
            skip_validation=skip_validation, no_versioning=no_versioning,
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
            lts.log_trajectory(rec)
            exp_lib.log_trajectory(rec)
            tag = "✓" if r.success else "✗"
            print(f"  [{i}/{len(stream)}] {phase} {r.task_id} {tag} "
                  f"reward={r.reward:.2f} tokens={r.tokens} {r.error[:40]}")
            if inter_task_delay and i < len(stream):
                time.sleep(inter_task_delay)

    # autoharness 也写入 LTS + 实验库
    if method == "autoharness":
        warmup_n = len(warmup_tasks)
        for r in results:
            task = (warmup_tasks + eval_tasks)[r.idx - 1] if r.idx <= len(stream) else None
            rec = _to_trajectory_record(r, eid, domain, task, model, variant)
            lts.log_trajectory(rec)
            exp_lib.log_trajectory(rec)

    exp = ExperimentResult(
        method=method, model=model, domain=domain, task_type=chosen_type,
        warmup_size=warmup, eval_size=eval_size, max_steps=max_steps,
        results=results, experiment_id=eid,
    )
    _print_summary(exp)
    print(f"  experiment_id: {eid}")
    print(f"  LTS trajs: {len(lts.query_trajectories(experiment_id=eid))} 条（含完整对话）")
    print(f"  实验库: {exp_lib.db_path}")
    lts.close()
    exp_lib.close()
    return exp


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
    ev = exp.eval_results()
    if ev:
        esr = sum(1 for r in ev if r.success) / len(ev)
        print(f"  Eval SR:  {esr:.1%} ({len(ev)} tasks)")
    paths = Counter(r.path for r in exp.results)
    print(f"  路径分布: {dict(paths)}")
    print(f"{'='*60}\n")


def save_result(exp: ExperimentResult, output_file: str) -> None:
    p = Path(output_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(exp.to_dict(), ensure_ascii=False, indent=2))
    print(f"结果已保存: {p}")
