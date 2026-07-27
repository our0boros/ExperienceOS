"""KSI (Knowledge-centric Self-Improvement) 适配器。

将 τ-bench 任务转换为 KSI 兼容的 TaskSpec / JSONL 格式，生成
KSI run manifest。本模块**不调用 KSI API**——只产出 KSI 可消费的
task file 和 run manifest，用于后续实验对比。

核心接口：
    * :func:`build_ksi_task_spec`  — τ-bench task → KSI TaskSpec dict
    * :func:`build_ksi_run_spec`   — 批量任务 → KSI run manifest
    * :func:`export_ksi_tasks`     — 导出 JSONL task file
    * :func:`export_ksi_run_manifest` — 导出 run manifest JSON

KSI TaskSpec 格式参考:
    https://recursive-knowledge.github.io/KSI/your_own_tasks/
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ======================================================================
# KSI 兼容数据结构
# ======================================================================


@dataclass
class KsiTaskSpec:
    """KSI 兼容的单个任务定义。

    KSI 要求的字段：task_id（必需）、prompt（必需）。
    可选：workspace_dir、files、eval。
    """

    task_id: str
    prompt: str
    workspace_dir: str = ""
    files: dict[str, str] = field(default_factory=dict)
    eval: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ksi_dict(self) -> dict[str, Any]:
        """输出 KSI 兼容的 JSONL 记录 dict（剔除内部 metadata）。"""
        record: dict[str, Any] = {
            "task_id": self.task_id,
            "prompt": self.prompt,
        }
        if self.workspace_dir:
            record["workspace_dir"] = self.workspace_dir
        if self.files:
            record["files"] = self.files
        if self.eval:
            record["eval"] = self.eval
        # metadata 中只保留 KSI custom task JSONL schema 认识的键
        # KSI schema: task_id, prompt, workspace_dir, files, eval
        # 其他键（如 task_source、num_reference_actions）是内部元数据，不应输出
        for key in ("repo_path", "eval_command", "eval_timeout_sec"):
            if key in self.metadata:
                record[key] = self.metadata[key]
        return record


@dataclass
class KsiRunSpec:
    """KSI 运行清单：描述如何对一组任务运行 KSI。

    这不是 KSI 内置类型，而是 ExperienceOS 为实验记录准备的
    自描述 manifest——记录任务来源、划分策略、模型配置等。
    """

    task_source: str = "custom"
    tasks_path: str = ""
    task_count: int = 0
    task_types: list[str] = field(default_factory=list)
    split_policy: str = ""
    warmup_count: int = 0
    eval_count: int = 0
    domain: str = ""
    model: str = ""
    model_provider: str = ""
    evaluator: str = "command"
    max_steps: int = 30
    experiment_id: str = ""
    notes: str = ""


# ======================================================================
# τ-bench → KSI 转换
# ======================================================================


def _extract_ksi_prompt(task: Any) -> str:
    """从 τ-bench 任务提取 KSI prompt 文本。

    KSI prompt 是给 agent 的自然语言指令。对于 τ-bench retail/airline，
    合并 user_scenario 中的多条信息。
    """
    parts: list[str] = []

    scenario = getattr(task, "user_scenario", None)
    if scenario:
        instructions = getattr(scenario, "instructions", None)
        if instructions is not None:
            if hasattr(instructions, "reason_for_call") and instructions.reason_for_call:
                parts.append(f"Reason for call: {instructions.reason_for_call}")
            if hasattr(instructions, "known_info") and instructions.known_info:
                parts.append(f"Known info: {instructions.known_info}")
            if hasattr(instructions, "unknown_info") and instructions.unknown_info:
                parts.append(f"Unknown info: {instructions.unknown_info}")
            if hasattr(instructions, "task_instructions") and instructions.task_instructions:
                parts.append(f"Instructions: {instructions.task_instructions}")
            # 纯字符串 instructions
            if isinstance(instructions, str) and instructions.strip():
                parts.append(instructions)

    # 补充 task description
    desc = getattr(task, "description", None)
    if desc:
        purpose = getattr(desc, "purpose", None)
        if purpose and isinstance(purpose, str):
            parts.append(f"Purpose: {purpose}")

    # solo mode ticket
    ticket = getattr(task, "ticket", None)
    if ticket and isinstance(ticket, str):
        parts.append(f"Ticket: {ticket}")

    if not parts:
        return f"tau2 task {getattr(task, 'id', 'unknown')}"

    return "\n".join(parts)


def _extract_eval_info(task: Any) -> dict | None:
    """从 τ-bench 任务提取评估信息，转换为 KSI eval dict。

    τ-bench 的评估依赖 DB hash 对比，无法直接映射为 KSI 的 shell
    command 评估。因此对 τ-bench 任务，eval 字段为 None，
    表示需要 τ-bench 自己的评估器。
    """
    # τ-bench 任务无法用简单 shell 命令评估，返回 None
    # 调用方应在 run manifest 中说明评估方式
    return None


def build_ksi_task_spec(
    task: Any,
    *,
    task_type: str = "",
    workspace_dir: str = "",
) -> KsiTaskSpec:
    """将单个 τ-bench 任务转换为 KSI TaskSpec。

    Args:
        task: τ-bench Task 对象。
        task_type: 推断的任务类型（如 ``get_order_details``）。
        workspace_dir: 可选的工作目录路径（KSI 会复制到容器内 repo/）。

    Returns:
        KSI 兼容的 TaskSpec。
    """
    from experience_os.tau2_adapter import infer_task_type

    tt = task_type or infer_task_type(task)
    prompt = _extract_ksi_prompt(task)

    # 收集 τ-bench 特定元数据
    metadata: dict[str, Any] = {
        "task_source": "tau2-bench",
        "task_type": tt,
        "domain": "",
        "has_eval_criteria": False,
        "num_reference_actions": 0,
    }

    criteria = getattr(task, "evaluation_criteria", None)
    if criteria:
        metadata["has_eval_criteria"] = True
        actions = getattr(criteria, "actions", None)
        if actions:
            metadata["num_reference_actions"] = len(actions)
            metadata["reference_tools"] = [a.name for a in actions]

    # 提取 domain
    task_id = str(getattr(task, "id", "unknown"))
    # tau2 retail task id 格式通常是数字字符串，airline 也是
    metadata["task_id_raw"] = task_id

    return KsiTaskSpec(
        task_id=f"tau2-{task_id}",
        prompt=prompt,
        workspace_dir=workspace_dir,
        eval=_extract_eval_info(task),
        metadata=metadata,
    )


def build_ksi_task_specs(
    tasks: list[Any],
    *,
    workspace_dir: str = "",
) -> list[KsiTaskSpec]:
    """批量转换 τ-bench 任务为 KSI TaskSpec 列表。

    Args:
        tasks: τ-bench Task 对象列表。
        workspace_dir: 可选的工作目录路径。

    Returns:
        KSI TaskSpec 列表。
    """
    return [build_ksi_task_spec(t, workspace_dir=workspace_dir) for t in tasks]


def build_ksi_run_spec(
    tasks: list[Any],
    *,
    domain: str = "retail",
    model: str = "",
    model_provider: str = "",
    warmup: int = 3,
    eval_size: int = 5,
    split_policy: str = "type_split",
    experiment_id: str = "",
    max_steps: int = 30,
) -> KsiRunSpec:
    """为一组 τ-bench 任务构建 KSI run manifest。

    Args:
        tasks: τ-bench Task 对象列表。
        domain: τ-bench domain（retail / airline）。
        model: 模型标识。
        model_provider: provider（anthropic / openai / deepinfra）。
        warmup: warmup 任务数。
        eval_size: eval 任务数。
        split_policy: 划分策略（type_split / train_test / cross_domain）。
        experiment_id: 实验 ID。
        max_steps: 最大步数。

    Returns:
        KSI run manifest。
    """
    from experience_os.tau2_adapter import infer_task_type

    task_types = sorted({infer_task_type(t) for t in tasks})

    return KsiRunSpec(
        task_source="custom",
        tasks_path="",
        task_count=len(tasks),
        task_types=task_types,
        split_policy=split_policy,
        warmup_count=warmup,
        eval_count=eval_size,
        domain=domain,
        model=model,
        model_provider=model_provider,
        evaluator="tau2-bench",
        max_steps=max_steps,
        experiment_id=experiment_id,
        notes=(
            f"τ-bench {domain} tasks converted to KSI format. "
            f"Evaluation uses τ-bench DB hash comparison, not KSI command evaluator. "
            f"Warmup={warmup}, eval={eval_size}, split={split_policy}."
        ),
    )


# ======================================================================
# 导出
# ======================================================================


def export_ksi_tasks(
    tasks: list[Any],
    output_path: str,
    *,
    workspace_dir: str = "",
) -> str:
    """将 τ-bench 任务导出为 KSI JSONL task file。

    Args:
        tasks: τ-bench Task 对象列表。
        output_path: 输出 JSONL 文件路径。
        workspace_dir: 可选的工作目录路径。

    Returns:
        输出文件的绝对路径。
    """
    specs = build_ksi_task_specs(tasks, workspace_dir=workspace_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for spec in specs:
            record = spec.to_ksi_dict()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(path.resolve())


def export_ksi_run_manifest(
    tasks: list[Any],
    output_path: str,
    *,
    domain: str = "retail",
    model: str = "",
    model_provider: str = "",
    warmup: int = 3,
    eval_size: int = 5,
    split_policy: str = "type_split",
    experiment_id: str = "",
    max_steps: int = 30,
) -> str:
    """导出 KSI run manifest JSON 文件。

    Args:
        tasks: τ-bench Task 对象列表。
        output_path: 输出 JSON 文件路径。
        domain: τ-bench domain。
        model: 模型标识。
        model_provider: provider。
        warmup: warmup 任务数。
        eval_size: eval 任务数。
        split_policy: 划分策略。
        experiment_id: 实验 ID。
        max_steps: 最大步数。

    Returns:
        输出文件的绝对路径。
    """
    spec = build_ksi_run_spec(
        tasks,
        domain=domain,
        model=model,
        model_provider=model_provider,
        warmup=warmup,
        eval_size=eval_size,
        split_policy=split_policy,
        experiment_id=experiment_id,
        max_steps=max_steps,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path.resolve())
