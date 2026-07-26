"""Plan-then-Execute 任务分解与执行（§3 运行时层）。

将任务分解为子任务序列（Plan），每个子任务按三要素（输入/输出/影响）
检索匹配的 harness 执行，无匹配时回退 agent。

核心组件：
    * :class:`TaskPlanner`   — LLM 将任务分解为 :class:`SubTaskPlan` 序列
    * :class:`PlanExecutor`  — 逐个子任务执行：harness 优先，agent 回退
    * :func:`match_or_compose` — 粒度感知组合：过大子任务用已有 artifacts 组合
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────

@dataclass
class SubTaskPlan:
    """LLM 产出的单个子任务计划。"""
    intent: str              # "lookup_user_by_email"
    expected_tool: str       # "find_user_id_by_email"
    description: str         # "Find the user's ID from their email"
    depends_on: list[int] = field(default_factory=list)  # 依赖的前置子任务索引
    # 三要素检索签名
    input_schema: dict = field(default_factory=dict)   # {"requires":["email"],"from":"task"}
    output_schema: dict = field(default_factory=dict)  # {"produces":["user_id"],"type":"str"}
    effect: str = "read_only"


@dataclass
class DecomposedTask:
    """LLM 对任务的完整分解。"""
    task_id: str
    task_description: str
    subtasks: list[SubTaskPlan]
    raw_plan: str = ""
    tokens_used: int = 0


@dataclass
class SubStepResult:
    """单个子步骤的执行结果。"""
    plan_idx: int
    intent: str
    tool_name: str
    success: bool
    output: Any = None       # tool call result (parsed)
    raw_output: str = ""     # raw tool call response
    execution_mode: str = "agent"  # 'agent' | 'harness' | 'llm_glue'
    artifact_id: str = ""
    artifact_version: int = 0
    tokens_used: int = 0
    error: str = ""


# ──────────────────────────────────────────────────────────────────
# Plan decomposition prompt
# ──────────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """Decompose this task into a sequence of sub-tasks. Each sub-task should be ONE tool call.

Task: {task_description}
Available tools: {tool_list}

For each sub-task, describe three aspects:
- "intent": short snake_case label (e.g. "lookup_user_by_email")
- "expected_tool": the exact tool name to call
- "description": one-line explanation of what this step does
- "input": {{"requires": [...], "from": "..."}} — what fields this step needs
- "output": {{"produces": [...], "type": "..."}} — what fields this step produces
- "effect": "read_only" (no state change), "write" (changes state), or "mixed"
- "depends_on": list of step indices (0-based) this step depends on

Rules:
- Each sub-task must be ONE tool call — not multi-step reasoning
- Put lookup/read steps before write/action steps
- Do NOT hardcode parameter values — use descriptive names
- Mark write steps (exchange, cancel, return) as effect:"write"

Output STRICT JSON: {{"subtasks": [...]}}"""


# ──────────────────────────────────────────────────────────────────
# LLM Glue prompt (artifact composition)
# ──────────────────────────────────────────────────────────────────

GLUE_PROMPT = """Connect these artifacts. Generate ONLY the parameter mapping glue code.

Artifact chain (input → output signatures):
{artifact_chain}

Available context from previous steps:
{context}

Current sub-task needs:
  intent: {intent}
  requires: {requires}
  produces: {produces}

Generate a Python function that:
1. Maps available context fields to artifact input parameters
2. Calls each artifact in sequence
3. Returns the combined output as a dict

Output ONLY the run() function. Do NOT re-implement the artifacts."""


# ──────────────────────────────────────────────────────────────────
# TaskPlanner
# ──────────────────────────────────────────────────────────────────

class TaskPlanner:
    """LLM 将任务分解为子任务序列。"""

    def __init__(self, llm, tool_names: list[str] | None = None) -> None:
        self._llm = llm
        self._tool_names = tool_names or []

    def decompose(self, task_description: str, task_id: str = "",
                  tool_names: list[str] | None = None) -> DecomposedTask:
        """对任务做 Plan 分解。返回 ``DecomposedTask``。"""
        tools = tool_names or self._tool_names
        tool_list = "\n".join(f"  - {t}" for t in tools) if tools else "  (unknown)"

        messages = [
            {"role": "system", "content": "You decompose tasks into single-tool-call sub-tasks. Output strict JSON."},
            {"role": "user", "content": DECOMPOSE_PROMPT.format(
                task_description=task_description,
                tool_list=tool_list,
            )},
        ]

        try:
            data = self._llm.chat_json(messages)
            raw = json.dumps(data, ensure_ascii=False)
            subtasks_raw = data.get("subtasks", [])
        except Exception as exc:
            log.warning("Plan decomposition failed: %s, falling back to single-step", exc)
            return DecomposedTask(
                task_id=task_id,
                task_description=task_description,
                subtasks=[SubTaskPlan(
                    intent="execute_task",
                    expected_tool="",
                    description=task_description,
                )],
                raw_plan=str(exc),
            )

        subtasks = []
        for i, st in enumerate(subtasks_raw):
            subtasks.append(SubTaskPlan(
                intent=st.get("intent", f"step_{i}"),
                expected_tool=st.get("expected_tool", st.get("tool", "")),
                description=st.get("description", ""),
                depends_on=st.get("depends_on", []),
                input_schema=st.get("input", {}),
                output_schema=st.get("output", {}),
                effect=st.get("effect", "read_only"),
            ))

        return DecomposedTask(
            task_id=task_id,
            task_description=task_description,
            subtasks=subtasks,
            raw_plan=raw,
        )


# ──────────────────────────────────────────────────────────────────
# PlanExecutor
# ──────────────────────────────────────────────────────────────────

class PlanExecutor:
    """按 Plan 逐个执行子任务：harness 优先，agent 回退。"""

    def __init__(
        self,
        llm,
        agent,
        registry,          # HarnessRegistry
        router,            # RuntimeRouter (for retrieve_substep_harness)
        env,               # BaseEnvironment
        substep_store=None,  # ExperienceLibrary
    ) -> None:
        self._llm = llm
        self._agent = agent
        self._registry = registry
        self._router = router
        self._env = env
        self._store = substep_store

    def execute(self, task, plan: DecomposedTask,
                task_params: dict | None = None) -> list[SubStepResult]:
        """执行分解后的 Plan，返回每个子步骤的结果。"""
        results: list[SubStepResult] = []
        context: dict[int, Any] = {}  # plan_idx → output

        for i, subtask in enumerate(plan.subtasks):
            t0 = time.time()

            # 解析参数：从 task_params + 前置步骤的 context
            params = self._resolve_params(subtask, task_params or {}, context)

            # 尝试 harness 匹配
            result = self._try_harness(subtask, params, i)

            if result is None:
                # 尝试粒度感知组合
                result = self._try_compose(subtask, params, context, i)

            if result is None:
                # 回退 agent
                result = self._execute_agent(subtask, params, i)

            result.tokens_used = 0  # TODO: track tokens per sub-step
            results.append(result)

            # 向前传递上下文
            if result.success and result.output is not None:
                context[i] = result.output

            # 持久化
            self._log_substep(task, plan, result, i)

        return results

    def _try_harness(self, subtask: SubTaskPlan, params: dict, idx: int) -> SubStepResult | None:
        """尝试用 harness 执行子步骤。"""
        # Layer 1: exact intent match in registry
        h = self._registry.lookup(subtask.intent)
        if h is None:
            # Layer 2: embedding-based retrieval
            try:
                ret = self._router.retrieve_substep_harness(
                    subtask.intent,
                    available_inputs=set(params.keys()),
                    needed_outputs=set(subtask.output_schema.get("produces", [])),
                    effect_constraint=subtask.effect,
                )
                if ret.harness and ret.level.value != "none":
                    h = ret.harness
            except Exception:
                pass

        if h is None:
            return None

        try:
            from experience_os.environment import TaskRequest
            req = TaskRequest(
                task_id="", task_description=subtask.description,
                task_type=subtask.intent, params=params, expected_output="",
            )
            exec_result = self._env.execute_harness(h, req)
            return SubStepResult(
                plan_idx=idx, intent=subtask.intent, tool_name=subtask.expected_tool,
                success=exec_result.success,
                output=exec_result.output,
                execution_mode="harness",
                artifact_id=h.id or "",
                artifact_version=h.version or 1,
                error=exec_result.error or "",
            )
        except Exception as exc:
            return SubStepResult(
                plan_idx=idx, intent=subtask.intent, tool_name=subtask.expected_tool,
                success=False, execution_mode="harness",
                artifact_id=h.id or "",
                artifact_version=h.version or 1,
                error=str(exc)[:200],
            )

    def _try_compose(self, subtask: SubTaskPlan, params: dict,
                     context: dict[int, Any], idx: int) -> SubStepResult | None:
        """P3 粒度感知组合：用 HarnessChainDetector 发现可用的 harness 链。

        如果 harness 链能覆盖当前子任务的需求，告知 LLM 让 LLM 决定是否调用。
        """
        required = set(subtask.input_schema.get("requires", [])) if isinstance(subtask.input_schema, dict) else set()
        produces = set(subtask.output_schema.get("produces", [])) if isinstance(subtask.output_schema, dict) else set()

        if not required and not produces:
            return None

        try:
            from experience_os.composite import HarnessChainDetector
            detector = HarnessChainDetector()
            chains = detector.detect_chains(required, produces, self._registry)
            if chains:
                log.info("P3: Found %d harness chain(s) for subtask '%s' (coverage=%.0f%%)",
                         len(chains), subtask.intent, chains[0].coverage * 100)
                # 记录链信息供后续 LLM 决策（不自动执行）
                return None  # 当前不自动执行链，仅检测并记录
        except Exception:
            pass

        return None

    def _execute_agent(self, subtask: SubTaskPlan, params: dict, idx: int) -> SubStepResult:
        """回退 agent 执行单个子步骤。"""
        try:
            req = type("R", (), {
                "task_id": f"sub_{idx}",
                "task_description": subtask.description,
                "task_type": subtask.intent,
                "params": params,
                "expected_output": "",
            })()
            result = self._agent.run(req, self._env, task_type=subtask.intent)
            return SubStepResult(
                plan_idx=idx, intent=subtask.intent, tool_name=subtask.expected_tool,
                success=result.success,
                output=getattr(result, 'output', None),
                execution_mode="agent",
                tokens_used=getattr(result, 'tokens_used', 0),
                error=getattr(result, 'error', "") or "",
            )
        except Exception as exc:
            return SubStepResult(
                plan_idx=idx, intent=subtask.intent, tool_name=subtask.expected_tool,
                success=False, execution_mode="agent",
                error=str(exc)[:200],
            )

    @staticmethod
    def _resolve_params(subtask: SubTaskPlan, task_params: dict,
                        context: dict[int, Any]) -> dict:
        """从 task_params + context 解析子步骤所需参数。"""
        params: dict = {}

        # 从 task_params 中提取
        input_from = subtask.input_schema.get("from", "") if isinstance(subtask.input_schema, dict) else ""
        requires = subtask.input_schema.get("requires", []) if isinstance(subtask.input_schema, dict) else []

        for key in requires:
            if key in task_params:
                params[key] = task_params[key]
            elif key in input_from:
                params[key] = input_from

        # 从前置步骤 context 中提取
        for dep_idx in subtask.depends_on:
            if dep_idx in context:
                dep_output = context[dep_idx]
                if isinstance(dep_output, dict):
                    for k, v in dep_output.items():
                        if k in requires and k not in params:
                            params[k] = v
                elif isinstance(dep_output, (str, int, float)):
                    # Single value output — use as primary param
                    primary = requires[0] if requires else "value"
                    if primary not in params:
                        params[primary] = dep_output

        # Fallback: copy all task_params
        if not params:
            params = dict(task_params)

        return params

    def _log_substep(self, task, plan: DecomposedTask,
                     result: SubStepResult, idx: int) -> None:
        """写入 substeps 表（全路径历史）。"""
        if self._store is None:
            return
        try:
            from experience_os.experience_library import SubStepRecord
            rec = SubStepRecord(
                trajectory_id=getattr(task, 'id', str(task)),
                experiment_id="",
                plan_idx=idx,
                intent=result.intent,
                tool_name=result.tool_name,
                success=result.success,
                execution_mode=result.execution_mode,
                artifact_id=result.artifact_id,
                artifact_version=result.artifact_version,
                tokens_used=result.tokens_used,
                source="plan_execute",
                parent_task_type=getattr(task, 'task_type', ''),
                parent_task_success=False,  # 全任务成功需在所有子步骤完成后判断
                meta_json=json.dumps({"plan": plan.raw_plan[:2000]}, ensure_ascii=False),
                input_schema=json.dumps(
                    plan.subtasks[idx].input_schema, ensure_ascii=False
                ) if idx < len(plan.subtasks) else "",
                output_schema=json.dumps(
                    plan.subtasks[idx].output_schema, ensure_ascii=False
                ) if idx < len(plan.subtasks) else "",
                effect=plan.subtasks[idx].effect if idx < len(plan.subtasks) else "",
            )
            self._store.log_substep(rec)
        except Exception:
            log.debug("Failed to log sub-step", exc_info=True)
