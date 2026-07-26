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
                # wrap non-JSON error strings as dict so harness .get() won't crash
                if stripped.startswith("Error"):
                    return {"error": stripped}
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
def _get_tool_type_map(domain: str) -> dict[str, str]:
    """从 τ-bench 环境获取工具分类映射（tool_name → read/write/generic）。

    缓存到模块级 dict，避免重复构建环境。
    """
    global _TOOL_TYPE_CACHE
    if domain in _TOOL_TYPE_CACHE:
        return _TOOL_TYPE_CACHE[domain]
    type_map: dict[str, str] = {}
    try:
        from tau2.environment.toolkit import ToolType
        from tau2.runner.build import build_environment
        env = build_environment(domain, solo_mode=False)
        tools = env.get_tools()
        for t in tools:
            name = getattr(t, "name", "") or ""
            tt = getattr(t, "tool_type", ToolType.READ)
            type_map[name] = tt.value if hasattr(tt, "value") else str(tt)
    except Exception as exc:
        log.debug("获取工具分类失败 (domain=%s): %s", domain, exc)
    _TOOL_TYPE_CACHE[domain] = type_map
    return type_map


_TOOL_TYPE_CACHE: dict[str, dict[str, str]] = {}


def _infer_substep_intent(tool_name: str, arguments: dict) -> str:
    """从工具名 + 参数推断子步骤意图（用于跨轨迹模式发现）。

    意图标签 = tool_name（τ-bench 工具名本身就是语义化的）。
    如果工具名缺失则用 "unknown"。
    """
    if not tool_name:
        return "unknown"
    # τ-bench 工具名如 get_order_details / cancel_pending_order / find_user_id_by_email
    # 本身就是语义化的意图标签，直接使用
    return tool_name


def _build_structured_cot(task: Any, task_desc: str) -> StructuredCoT:
    """从任务对象构建结构化推理链（StructuredCoT）。

    填充 goal / constraints / risk / milestones 字段，
    供归纳引擎做意图抽取和不变量挖掘。
    """
    cot = StructuredCoT(goal=task_desc)

    # 从 evaluation_criteria 提取约束
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria:
        # 参考动作作为 milestones
        actions = getattr(criteria, "actions", None)
        if actions:
            cot.milestones = [a.name for a in actions]
        # 评估条件作为 constraints
        common_condition = getattr(criteria, "common_condition", None)
        if common_condition:
            cot.constraints.append(str(common_condition)[:200])

    # 从 user_scenario 提取约束信息
    scenario = getattr(task, "user_scenario", None)
    if scenario:
        instructions = getattr(scenario, "instructions", None)
        if instructions and hasattr(instructions, "reason_for_call"):
            reason = instructions.reason_for_call or ""
            if reason:
                cot.risk = reason[:200]

    # 从 description 提取额外约束
    desc = getattr(task, "description", None)
    if desc:
        # 如果有 special_instructions 等字段
        for attr in ("special_instructions", "constraints", "notes"):
            val = getattr(desc, attr, None)
            if val and isinstance(val, str) and val.strip():
                cot.constraints.append(val[:200])

    return cot


def convert_simulation(sim_run: Any, task: Any, task_type: str = "") -> Trajectory:
    """将 τ-bench SimulationRun 转换为 ExperienceOS Trajectory（全收集）。

    从 messages 中提取完整交互链：
      - assistant 的 tool_call（工具名 + 完整参数）
      - 对应 ToolMessage 的完整结果
      - assistant 的推理文本
      - 每步的 sub_step_intent / action_type / 状态快照
    构建 :class:`Step` 列表，供归纳引擎做 LCS 对齐 + Daikon 不变量挖掘。
    """
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolMessage,
    )

    steps: list[Step] = []
    messages = sim_run.get_messages() if hasattr(sim_run, "get_messages") else (sim_run.messages or [])

    # 领域信息
    domain = "unknown"
    if hasattr(sim_run, "info") and isinstance(sim_run.info, dict):
        domain = sim_run.info.get("domain", "unknown")

    # 工具分类表（从环境中提取，用于 action_type 推断）
    tool_types = _get_tool_type_map(domain)

    pending_calls: list[dict] = []  # 等待结果的 tool calls
    prev_result_summary = ""  # 上一步结果摘要，作为下一步的 observation

    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            for tc in (msg.tool_calls or []):
                pending_calls.append({"name": tc.name, "arguments": tc.arguments})
        elif isinstance(msg, (ToolMessage, MultiToolMessage)):
            tool_msgs = msg.tool_messages if isinstance(msg, MultiToolMessage) else [msg]
            for i, tm in enumerate(tool_msgs):
                if i < len(pending_calls):
                    call = pending_calls[i]
                    tool_name = call["name"]
                    args = call["arguments"] or {}
                    result_text = tm.content or ""

                    # 推断子步骤意图
                    intent = _infer_substep_intent(tool_name, args)
                    # 分类 action_type
                    action_type = tool_types.get(tool_name, "generic")
                    # 完整参数序列化
                    action_str = f'{tool_name}({json.dumps(args, default=str, ensure_ascii=False)})'

                    steps.append(
                        Step(
                            observation=prev_result_summary[:200],
                            action=action_str,
                            result=result_text,  # 完整结果，不截断
                            action_type=action_type,
                            sub_step_intent=intent,
                            metadata={
                                "tool_name": tool_name,
                                "arguments": args,
                                "params": args,  # 双键兼容 compiler 的 params 提取
                                "result_length": len(result_text),
                                "has_error": "error" in result_text.lower() or "Error" in result_text,
                            },
                        )
                    )
                    # 更新 prev_result_summary 供下一步 observation
                    prev_result_summary = result_text
            pending_calls = []
        elif isinstance(msg, AssistantMessage) and msg.has_text_content():
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if steps:
                # 追加到上一步结果（agent 的文本回复）
                steps[-1].result += f"\n[agent]: {content}"
            else:
                steps.append(
                    Step(
                        observation=content[:200],
                        action="reasoning",
                        result=content,
                        action_type="think",
                        sub_step_intent="reasoning",
                    )
                )

    # 判定成功
    success = False
    if sim_run.reward_info:
        success = sim_run.reward_info.reward >= 1.0

    # 标记成功的子步骤（非 error 的步骤）
    success_steps = [
        i for i, s in enumerate(steps)
        if not s.metadata.get("has_error", False)
    ]

    # 提取任务描述 + 构建结构化 CoT
    task_desc = _extract_task_description(task)
    cot = _build_structured_cot(task, task_desc)

    # 环境快照（完整 env_context）
    env_attrs = {
        "domain": domain,
        "task_id": task.id,
        "solo_mode": getattr(sim_run, "info", {}).get("solo_mode", False)
                     if isinstance(getattr(sim_run, "info", None), dict) else False,
    }
    # 补充任务参考动作信息
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria and getattr(criteria, "actions", None):
        env_attrs["num_reference_actions"] = len(criteria.actions)
        env_attrs["reference_tools"] = [a.name for a in criteria.actions]

    return Trajectory(
        task_id=task.id,
        task_description=task_desc,
        task_type=task_type,
        steps=steps,
        structured_cot=cot,
        env_snapshot=EnvironmentSnapshot(attributes=env_attrs),
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
    """从任务中提取参数，用于 Harness 执行。

    从多个来源提取，优先级从低到高：
    1. user_scenario.instructions 文本 — 用户标识信息（email/name/zip）
    2. user_scenario.instructions.known_info — 结构化已知信息（如有）
    3. evaluation_criteria.actions[*].arguments — 参考动作参数（最高优先级）

    Harness 代码通过 ``params`` 访问这些值。
    """
    import re

    params: dict = {}

    # ── 来源 1: user_scenario.instructions 文本 ──
    user_scenario = getattr(task, "user_scenario", None)
    instruction_text = ""
    if user_scenario:
        instructions = getattr(user_scenario, "instructions", None)
        if instructions is not None:
            # 统一转为文本：兼容 StructuredUserInstructions 和纯 str
            if isinstance(instructions, str):
                instruction_text = instructions
            else:
                # StructuredUserInstructions → 拼接所有字段为文本
                parts = []
                for field in ("domain", "reason_for_call", "known_info",
                              "unknown_info", "task_instructions"):
                    val = getattr(instructions, field, None)
                    if val:
                        parts.append(str(val))
                instruction_text = "\n".join(parts)

                # known_info 可能包含结构化键值信息，单独解析
                known_info = getattr(instructions, "known_info", None)
                if known_info and isinstance(known_info, str):
                    _extract_structured_pairs(known_info, params)

    # 从指令文本中提取常见模式
    if instruction_text:
        text_params = _parse_instruction_text(instruction_text)
        for k, v in text_params.items():
            if k not in params:
                params[k] = v

    # ── 来源 2: evaluation_criteria.actions[*].arguments ──
    # （最高优先级，可能覆盖上述文本提取的值）
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria and getattr(criteria, "actions", None):
        for action in criteria.actions:
            if action.arguments:
                params.update(action.arguments)

    # ── 来源 3: initial_state.initialization_actions ──
    initial_state = getattr(task, "initial_state", None)
    if initial_state and getattr(initial_state, "initialization_actions", None):
        for init_action in initial_state.initialization_actions:
            init_args = getattr(init_action, "arguments", None)
            if init_args:
                for k, v in init_args.items():
                    if k not in params:
                        params[k] = v

    # ── 来源 4: initial_state.initialization_data.user_data ──
    if initial_state:
        init_data = getattr(initial_state, "initialization_data", None)
        if init_data:
            user_data = getattr(init_data, "user_data", None)
            if user_data and isinstance(user_data, dict):
                for k, v in user_data.items():
                    if k not in params:
                        params[k] = v

    return params


def _parse_instruction_text(text: str) -> dict:
    """从指令文本中提取常见参数模式。"""
    import re

    params: dict = {}

    # Email: standard pattern
    email_match = re.search(
        r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b', text
    )
    if email_match:
        params["email"] = email_match.group(0)

    # ZIP code: 5-digit (possibly + 4), near "zip" keyword
    zip_match = re.search(
        r'(?:zip\s*(?:code)?\s*[:\s]*)(\d{5}(?:-\d{4})?)', text, re.IGNORECASE
    )
    if zip_match:
        params["zip"] = zip_match.group(1)
    else:
        # Fallback: any 5-digit number in a reasonable context
        zip_match = re.search(r'\b\d{5}(?:-\d{4})?\b', text)
        if zip_match:
            params["zip"] = zip_match.group(0)

    # User ID: tau-bench convention — snake_case with 4+ digit suffix,
    # appearing after "You are" or "I am" (known_info text)
    user_id_match = re.search(
        r'(?:You|I)\s+(?:am|are)\s+([a-z]+_[a-z]+_\d{4,})', text
    )
    if user_id_match:
        params["user_id"] = user_id_match.group(1)

    # Name patterns: "You are X Y" / "Your name is X Y" / "You're X Y"
    name_patterns = [
        r'You\s+are\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
        r'Your\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
        r"You're\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, text)
        if name_match:
            name_parts = name_match.group(1).split()
            if len(name_parts) >= 1:
                params["first_name"] = name_parts[0]
            if len(name_parts) >= 2:
                params["last_name"] = name_parts[-1]
            break

    # Order ID: various formats (#W2378156, ORD-999, etc.)
    # Captures the full identifier including # prefix when present
    order_match = re.search(
        r'order\s+(#[A-Z]*\d{3,})', text, re.IGNORECASE
    )
    if order_match:
        params["order_id"] = order_match.group(1)
    else:
        # Fallback: standalone #NUMBER pattern
        order_match = re.search(r'(#[A-Z]+\d{3,})', text)
        if order_match:
            params["order_id"] = order_match.group(1)
        else:
            order_match = re.search(r'(?:order\s+)([A-Z]+\d{3,})', text, re.IGNORECASE)
            if order_match:
                params["order_id"] = "#" + order_match.group(1)

    return params


def _extract_structured_pairs(text: str, params: dict) -> None:
    """从结构化文本（如 known_info）中提取键值对。

    支持的格式：
    - ``Name: Sophia Silva``
    - ``Email: sophia.silva@example.com``
    - ``key: value`` (通用)
    """
    import re

    # Key: Value lines
    for match in re.finditer(
        r'(email|name|first_name|last_name|zip|user_id|order_id|product_id|'
        r'phone|address)[:\s]+(.+?)(?:\n|$)',
        text,
        re.IGNORECASE,
    ):
        key = match.group(1).lower().strip()
        value = match.group(2).strip()
        if key == "name":
            parts = value.split()
            if len(parts) >= 1:
                params["first_name"] = parts[0]
            if len(parts) >= 2:
                params["last_name"] = parts[-1]
        else:
            params[key] = value
