"""τ-bench 集成适配器。

将 tau2-bench 的 Environment/Orchestrator/SimulationRun 适配为
ExperienceOS 的 BaseEnvironment / Trajectory 格式，实现真实任务上的
积累 → 归纳 → 部署闭环。

核心组件：
    * :class:`Tau2Environment`   — 包装 tau2 Environment，实现 call_tool / verify
    * :func:`convert_simulation`  — SimulationRun → Trajectory 格式转换
    * :func:`infer_task_type`     — 从参考动作推断任务类型
    * :func:`split_tasks`         — Warm-up / Evaluation 数据池划分
    * :func:`run_tau2_simulation` — 编程式运行 tau2 仿真（积累阶段）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from experience_os.environment import BaseEnvironment, TaskRequest
from experience_os.models import (
    EnvironmentSnapshot,
    ExecutionResult,
    Harness,
    Step,
    StructuredCoT,
    Trajectory,
)

log = logging.getLogger(__name__)


# ======================================================================
# τ-bench Environment 适配器
# ======================================================================
class Tau2Environment(BaseEnvironment):
    """将 τ-bench 的 Environment 包装为 ExperienceOS 的 BaseEnvironment。

    每个 task 对应一个独立的 :class:`Tau2Environment` 实例，内部维护
    一个独立的 τ-bench Environment（已执行 ``set_state`` 初始化）。
    """

    def __init__(self, domain: str, task: Any, solo_mode: bool = False) -> None:
        from tau2.runner.build import build_environment

        self.domain = domain
        self.task = task
        self.solo_mode = solo_mode

        # 构建 τ-bench 环境（默认 DB）
        self.tau2_env = build_environment(domain, solo_mode=solo_mode)
        # 用任务的 initial_state 初始化
        self._init_state()

    def _init_state(self) -> None:
        """用 task.initial_state 初始化 τ-bench 环境。"""
        init_state = getattr(self.task, "initial_state", None)
        init_data = getattr(init_state, "initialization_data", None) if init_state else None
        init_actions = getattr(init_state, "initialization_actions", None) if init_state else None
        msg_history = (
            getattr(init_state, "message_history", None) or []
            if init_state
            else []
        )
        try:
            self.tau2_env.set_state(init_data, init_actions, msg_history, strict=False)
        except Exception as exc:
            log.warning("Tau2 set_state failed for task %s: %s", self.task.id, exc)

    # ------------------------------------------------------------------
    # BaseEnvironment 实现
    # ------------------------------------------------------------------
    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            attributes={
                "domain": self.domain,
                "task_id": self.task.id,
                "solo_mode": self.solo_mode,
            }
        )

    def get_tools(self) -> list[dict]:
        tools = self.tau2_env.get_tools()
        return [t.openai_schema for t in tools]

    def call_tool(self, name: str, arguments: dict) -> str:
        """执行工具调用，返回 JSON 字符串结果。"""
        try:
            result = self.tau2_env.make_tool_call(
                name, requestor="assistant", **arguments
            )
            return self.tau2_env.to_json_str(result)
        except Exception as exc:
            log.warning("Tool call %s failed: %s", name, exc)
            return f"Error: {exc}"

    def verify(self, expected_output: str, actual_output: str) -> bool:
        """通过 DB hash 比对验证终态。"""
        try:
            expected_hash = self._compute_expected_db_hash()
            actual_hash = self.tau2_env.get_db_hash()
            match = expected_hash == actual_hash
            if not match:
                log.debug(
                    "DB hash mismatch for task %s: expected=%s actual=%s",
                    self.task.id,
                    expected_hash[:12],
                    (actual_hash or "")[:12],
                )
            return match
        except Exception as exc:
            log.warning("Verify failed for task %s: %s", self.task.id, exc)
            return False

    # ------------------------------------------------------------------
    # DB hash 计算（回放参考动作）
    # ------------------------------------------------------------------
    def _compute_expected_db_hash(self) -> Optional[str]:
        """在新环境中回放参考动作，得到期望 DB hash。"""
        from tau2.runner.build import build_environment

        fresh = build_environment(self.domain, solo_mode=self.solo_mode)
        init_state = getattr(self.task, "initial_state", None)
        init_data = getattr(init_state, "initialization_data", None) if init_state else None
        init_actions = getattr(init_state, "initialization_actions", None) if init_state else None
        msg_history = (
            getattr(init_state, "message_history", None) or []
            if init_state
            else []
        )
        try:
            fresh.set_state(init_data, init_actions, msg_history, strict=False)
        except Exception as exc:
            log.warning("Fresh set_state failed: %s", exc)

        criteria = getattr(self.task, "evaluation_criteria", None)
        if criteria and getattr(criteria, "actions", None):
            for action in criteria.actions:
                try:
                    fresh.make_tool_call(
                        action.name,
                        requestor=getattr(action, "requestor", "assistant"),
                        **action.arguments,
                    )
                except Exception as exc:
                    log.debug("Reference action %s failed: %s", action.name, exc)
        return fresh.get_db_hash()

    # ------------------------------------------------------------------
    # 重写 execute_harness，使用 τ-bench 的 DB hash 验证
    # ------------------------------------------------------------------
    def execute_harness(self, harness: Harness, request: TaskRequest) -> ExecutionResult:
        import time as _time

        start = _time.time()

        def _call_tool(name: str, *args, **kwargs):
            if args and isinstance(args[0], dict) and not kwargs:
                raw = self.call_tool(name, args[0])
            elif kwargs:
                raw = self.call_tool(name, kwargs)
            else:
                raw = self.call_tool(name, {})
            # auto-parse JSON string into dict/list for harness code
            if isinstance(raw, str):
                stripped = raw.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        import json
                        return json.loads(stripped)
                    except (json.JSONDecodeError, ValueError):
                        pass
            return raw

        sandbox: dict[str, Any] = {
            "env": self,
            "call_tool": _call_tool,
            "snapshot": self.snapshot,
            "params": request.params,
            "request": request,
        }
        try:
            local_ns: dict[str, Any] = {}
            exec(harness.procedure_code, sandbox, local_ns)  # noqa: S102
            run_fn = local_ns.get("run") or local_ns.get("main")
            if run_fn is None:
                return ExecutionResult(
                    success=False, path="harness", harness_id=harness.id,
                    failure_type="F2", output="no run() function",
                    latency_seconds=_time.time() - start,
                )
            output = run_fn()
            output_str = str(output) if output is not None else ""
            success = self.verify("", output_str)
            return ExecutionResult(
                success=success, path="harness", harness_id=harness.id,
                tokens_used=0, latency_seconds=_time.time() - start,
                output=output_str,
            )
        except Exception as exc:
            import traceback

            log.warning("Harness %s raised: %s", harness.full_name, exc)
            return ExecutionResult(
                success=False, path="harness", harness_id=harness.id,
                failure_type="F2",
                output=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                latency_seconds=_time.time() - start,
            )


# ======================================================================
# 轨迹格式转换
# ======================================================================
def convert_simulation(sim_run: Any, task: Any, task_type: str = "") -> Trajectory:
    """将 τ-bench SimulationRun 转换为 ExperienceOS Trajectory。

    从 messages 中提取 assistant 的 tool_call + 对应 ToolMessage 结果，
    构建 :class:`Step` 列表。
    """
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolMessage,
    )

    steps: list[Step] = []
    messages = sim_run.get_messages() if hasattr(sim_run, "get_messages") else (sim_run.messages or [])

    pending_calls: list[dict] = []  # 等待结果的 tool calls
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            for tc in (msg.tool_calls or []):
                pending_calls.append({"name": tc.name, "arguments": tc.arguments})
        elif isinstance(msg, (ToolMessage, MultiToolMessage)):
            tool_msgs = msg.tool_messages if isinstance(msg, MultiToolMessage) else [msg]
            for i, tm in enumerate(tool_msgs):
                if i < len(pending_calls):
                    call = pending_calls[i]
                    steps.append(
                        Step(
                            observation="",
                            action=f'{call["name"]}({json.dumps(call["arguments"], default=str)})',
                            result=(tm.content or "")[:200],
                            action_type="write",
                            metadata={
                                "tool_name": call["name"],
                                "arguments": call["arguments"],
                            },
                        )
                    )
            pending_calls = []
        elif isinstance(msg, AssistantMessage) and msg.has_text_content():
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if steps:
                steps[-1].result += f"\n[agent]: {content[:100]}"
            else:
                steps.append(
                    Step(observation=content[:100], action="reasoning",
                         result="", action_type="think")
                )

    # 判定成功
    success = False
    if sim_run.reward_info:
        success = sim_run.reward_info.reward >= 1.0

    # 提取任务描述
    task_desc = _extract_task_description(task)

    # 环境快照
    domain = "unknown"
    if hasattr(sim_run, "info") and isinstance(sim_run.info, dict):
        domain = sim_run.info.get("domain", "unknown")

    return Trajectory(
        task_id=task.id,
        task_description=task_desc,
        task_type=task_type,
        steps=steps,
        structured_cot=StructuredCoT(goal=task_desc),
        env_snapshot=EnvironmentSnapshot(attributes={"domain": domain, "task_id": task.id}),
        outcome="success" if success else "failure",
        tokens_used=int(getattr(sim_run, "agent_cost", 0) or 0),
        latency_seconds=getattr(sim_run, "duration", 0.0) or 0.0,
    )


def _extract_task_description(task: Any) -> str:
    """从 τ-bench Task 提取自然语言描述。"""
    # 尝试 user_scenario.instructions
    scenario = getattr(task, "user_scenario", None)
    if scenario:
        instructions = getattr(scenario, "instructions", None)
        if instructions:
            if hasattr(instructions, "reason_for_call"):
                return instructions.reason_for_call or ""
            if isinstance(instructions, str):
                return instructions
    # 尝试 description.purpose
    desc = getattr(task, "description", None)
    if desc:
        purpose = getattr(desc, "purpose", None)
        if purpose:
            return purpose
    # 尝试 ticket (solo mode)
    ticket = getattr(task, "ticket", None)
    if ticket:
        return ticket
    return f"tau2 task {task.id}"


# ======================================================================
# 任务类型推断 + 数据划分
# ======================================================================
def infer_task_type(task: Any) -> str:
    """从参考动作的第一个 action name 推断任务类型。

    τ-bench 的 retail/airline 域里，同类型任务通常以相同的
    第一步操作开始（如 ``get_order_details`` / ``exchange_or_cancel``）。
    """
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria and getattr(criteria, "actions", None):
        return criteria.actions[0].name
    return "unknown"


def split_tasks(
    tasks: list,
    min_support: int = 3,
) -> tuple[list, list, dict[str, list]]:
    """按任务类型分组，划分 Warm-up / Evaluation 池。

    返回 ``(warmup, evaluation, groups)``。
    只有当某类型的任务数 >= ``min_support`` 时才会被划分；
    不足的类型全部放入 warmup。
    """
    groups: dict[str, list] = {}
    for task in tasks:
        tt = infer_task_type(task)
        groups.setdefault(tt, []).append(task)

    warmup: list = []
    evaluation: list = []
    for tt, group in groups.items():
        if len(group) >= min_support:
            warmup.extend(group[:min_support])
            evaluation.extend(group[min_support:])
        else:
            warmup.extend(group)

    return warmup, evaluation, groups


# ======================================================================
# 编程式运行 τ-bench 仿真（积累阶段）
# ======================================================================
def run_tau2_simulation(
    domain: str,
    task: Any,
    *,
    llm_model: str = "ollama/qwen2.5",
    llm_api_base: str = "http://localhost:11434",
    max_steps: int = 30,
    seed: int = 42,
    solo_mode: bool = False,
) -> Any:
    """运行一次 τ-bench 仿真，返回 SimulationRun。

    使用 tau2 的 build_orchestrator + run_simulation API。
    LLM 通过 litellm 指向 ollama（或 DeepInfra）。
    """
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.build import build_orchestrator
    from tau2.runner.simulation import run_simulation

    agent_name = "llm_agent_solo" if solo_mode else "llm_agent"
    user_name = "dummy_user" if solo_mode else "user_simulator"

    llm_args = {"temperature": 0.0, "api_base": llm_api_base}

    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        llm_agent=llm_model,
        llm_args_agent=llm_args,
        user=user_name,
        llm_user=llm_model,
        llm_args_user=llm_args,
        max_steps=max_steps,
        num_trials=1,
        seed=seed,
    )

    orchestrator = build_orchestrator(config, task, seed=seed)
    result = run_simulation(orchestrator)
    return result


def extract_task_params(task: Any) -> dict:
    """从任务参考动作中提取参数，用于 Harness 执行。

    将第一条参考动作的 arguments 作为 params 传入，
    Harness 代码通过 ``params`` 访问这些值。
    """
    params: dict = {}
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria and getattr(criteria, "actions", None):
        for action in criteria.actions:
            if action.arguments:
                params.update(action.arguments)
    return params
