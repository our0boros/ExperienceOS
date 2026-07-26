"""纯算法函数模块 — 不依赖 HarnessInductor 实例状态。

包含 §3.3 各 Phase 使用的纯函数：分段步骤提取、前置条件交集、
Daikon 不变量挖掘、LCS 多序列对齐与类型感知参数化等。

这些函数原先为 ``HarnessInductor`` 的 ``@staticmethod`` 或模块级函数，
拆分后保持签名与下划线前缀不变，仅改为模块级引用。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from experience_os.compiler.prompts import SEGMENT_PROMPT
from experience_os.models import ParamStep, Trajectory

log = logging.getLogger(__name__)


def _find_constant_params(param_sets: list[dict]) -> dict:
    """找出所有 dict 中均存在且值相同的 key-value 对（Daikon 恒定参数）。

    用于 §5.2.5 不变量挖掘中"参数值恒定"模式检测。
    """
    if not param_sets:
        return {}
    common_keys = set(param_sets[0].keys())
    for ps in param_sets[1:]:
        common_keys &= set(ps.keys())
    constant: dict = {}
    for k in common_keys:
        vals = [ps[k] for ps in param_sets]
        # 值需可比较且全等
        try:
            if all(v == vals[0] for v in vals):
                constant[k] = vals[0]
        except Exception:
            continue
    return constant


# ==================================================================
# Phase 2 — precondition extraction (intersection across trajectories)
# ==================================================================
def _intersect_preconditions(trajectories: list[Trajectory]) -> dict:
    if not trajectories:
        return {}
    env_dicts = [t.env_snapshot.attributes for t in trajectories]
    common: dict = {}
    for key in env_dicts[0]:
        # collect values, handling unhashable types (lists, dicts)
        vals = []
        for d in env_dicts:
            if key in d:
                v = d[key]
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, sort_keys=True)
                if v not in vals:
                    vals.append(v)
        if len(vals) == 1:
            val = vals[0]
            # only keep scalar preconditions; skip complex/mutable ones
            if not isinstance(val, str) or val.startswith("[") or val.startswith("{"):
                continue  # skip list/dict-derived values
            common[key] = val
        else:
            # record the set as a list (any-of constraint)
            common[key] = sorted(str(v) for v in vals)
    return common


# ==================================================================
# Phase 2 辅助 — 分段子轨迹提取 & 多段 precondition 合并（修复 1）
# ==================================================================
def _extract_segment_steps(trajectory: Trajectory, step_indices: list[int]) -> Trajectory:
    """从轨迹中提取指定步骤索引子集，构造一条子轨迹。

    保留原轨迹的 env_snapshot / structured_cot / task 元信息，
    仅 steps 为按 *step_indices* 顺序取出的子集。
    """
    sub_steps = [trajectory.steps[i] for i in step_indices if 0 <= i < len(trajectory.steps)]
    return Trajectory(
        task_id=trajectory.task_id,
        task_description=trajectory.task_description,
        task_type=trajectory.task_type,
        steps=sub_steps,
        structured_cot=trajectory.structured_cot,
        env_snapshot=trajectory.env_snapshot,
        outcome=trajectory.outcome,
        tokens_used=trajectory.tokens_used,
        latency_seconds=trajectory.latency_seconds,
    )


def _merge_preconditions(pres: list[dict]) -> dict:
    """合并多段 preconditions：取各段都存在的 key，且值一致。

    - key 必须在所有段中都出现才保留（交集）
    - 若所有段该 key 值一致（同为标量或同为 any-of 列表），取该一致值
    - 否则取并集 any-of 列表（放宽约束，避免分段导致过严）
    """
    if not pres:
        return {}
    if len(pres) == 1:
        return dict(pres[0])
    common_keys = set(pres[0].keys())
    for p in pres[1:]:
        common_keys &= set(p.keys())
    merged: dict = {}
    for key in common_keys:
        vals = []
        for p in pres:
            v = p[key]
            if isinstance(v, list):
                v = json.dumps(v, sort_keys=True)
            if v not in vals:
                vals.append(v)
        if len(vals) == 1:
            merged[key] = pres[0][key]
        else:
            # 各段值不一致，取并集 any-of（放宽）
            union: list[str] = []
            for p in pres:
                v = p[key]
                if isinstance(v, list):
                    union.extend(str(x) for x in v)
                else:
                    union.append(str(v))
            merged[key] = sorted(set(union))
    return merged


# ==================================================================
# Phase 3 — invariant mining (Daikon 风格动态不变量检测, §5.2.5)
# ==================================================================
def _mine_invariants(trajectories: list[Trajectory]) -> list[str]:
    """Daikon 风格动态不变量检测。

    对每条成功轨迹的状态快照序列挖掘持续成立的谓词，覆盖以下模式：
    1. 首步一致（保留原有启发式）
    2. 全成功（保留原有启发式）
    3. 末步一致
    4. 步骤数一致
    5. 工具调用序列模式（忽略参数，仅看工具名序列）
    6. 参数值恒定模式（同一位置同一参数在所有轨迹中取值相同）
    7. 结果模式（无错误 / 返回 JSON 对象等）
    """
    invariants: list[str] = []
    if not trajectories:
        return invariants

    # 1. 首步一致（保留原有）
    first_actions = [t.steps[0].action for t in trajectories if t.steps]
    if first_actions and len(set(first_actions)) == 1:
        invariants.append(f"first action is always: {first_actions[0]}")

    # 2. 全成功（保留原有）
    if all(t.outcome == "success" for t in trajectories):
        invariants.append("outcome must be success")

    # 3. Daikon: 末步一致
    last_actions = [t.steps[-1].action for t in trajectories if t.steps]
    if last_actions and len(set(last_actions)) == 1:
        invariants.append(f"last action is always: {last_actions[0]}")

    # 4. Daikon: 步骤数一致
    step_counts = [len(t.steps) for t in trajectories]
    if step_counts and len(set(step_counts)) == 1:
        invariants.append(f"step count is always: {step_counts[0]}")

    # 5. Daikon: 工具调用序列模式（忽略参数，仅看工具名序列）
    tool_sequences = []
    for t in trajectories:
        seq = tuple(s.action.split("(")[0].strip() for s in t.steps)
        tool_sequences.append(seq)
    if tool_sequences and len(set(tool_sequences)) == 1:
        invariants.append(f"tool sequence is always: {' → '.join(tool_sequences[0])}")

    # 6. Daikon: 参数值恒定模式 — 同一位置参数在所有轨迹中取值相同
    min_len = min(step_counts) if step_counts else 0
    for i in range(min_len):
        param_sets = []
        for t in trajectories:
            if i < len(t.steps):
                args = t.steps[i].metadata.get("arguments")
                if isinstance(args, dict):
                    param_sets.append(args)
        if len(param_sets) >= 2:
            common_params = _find_constant_params(param_sets)
            for k, v in common_params.items():
                invariants.append(f"step {i} param '{k}' is always: {v}")

    # 7. Daikon: 结果模式 — 工具返回值特征
    for i in range(min_len):
        results = []
        for t in trajectories:
            if i < len(t.steps) and t.steps[i].result:
                results.append(t.steps[i].result)
        if len(results) >= 2:
            # 无错误
            if all("Error" not in r for r in results):
                invariants.append(f"step {i} always succeeds (no error)")
            # 返回 JSON 对象
            if all(r.lstrip().startswith("{") for r in results):
                invariants.append(f"step {i} always returns JSON object")

    return invariants


# ==================================================================
# Phase 4 — step abstraction & parameterisation (LCS + 类型感知, §5.2.4)
# ==================================================================
def _abstract_steps(trajectories: list[Trajectory]) -> list[ParamStep]:
    """跨轨迹 LCS 对齐 + 类型感知参数化。

    流程（§5.2.4）：
    1. 解析每条轨迹的工具调用序列为 {tool, args, raw}
    2. 在工具名序列上做多序列 LCS 对齐
    3. 对每个对齐位置收集该位置的所有步骤，以第一条为模板
    4. 类型感知参数化：取参数 keys 交集，按 key 生成 {param} 占位符
    """
    if not trajectories:
        return []

    # 1. 解析每条轨迹的工具调用序列
    sequences: list[list[dict]] = []
    for traj in trajectories:
        seq = [_parse_action(step.action) for step in traj.steps]
        # 补充 action_type 与原始 step 的 action_type 对齐
        for parsed, step in zip(seq, traj.steps):
            parsed.setdefault("action_type", step.action_type)
        sequences.append(seq)

    # 2. 在工具名序列上做 LCS 对齐
    tool_seqs = [[s["tool"] for s in seq] for seq in sequences]
    aligned_indices = _lcs_align(tool_seqs)
    # aligned_indices[pos] = [idx_in_traj0, idx_in_traj1, ...] (None=gap)

    # 3 + 4. 对每个对齐位置做类型感知参数化
    param_steps: list[ParamStep] = []
    for idxs in aligned_indices:
        aligned_steps = []
        for traj_idx, step_idx in enumerate(idxs):
            if step_idx is not None and step_idx < len(sequences[traj_idx]):
                aligned_steps.append(sequences[traj_idx][step_idx])
        if not aligned_steps:
            continue

        template_step = aligned_steps[0]
        tool_name = template_step["tool"]
        all_args = [s["args"] for s in aligned_steps if s["args"]]

        params: list[str] = []
        param_parts: list[str] = []
        if all_args:
            # 取参数 keys 的交集
            common_keys = set(all_args[0].keys())
            for args in all_args[1:]:
                common_keys &= set(args.keys())
            for key in sorted(common_keys):
                pname = _sanitize_param_name(key)
                params.append(pname)
                param_parts.append(f"{pname}={{{pname}}}")

        if params:
            template = f"{tool_name}({', '.join(param_parts)})"
        else:
            template = f"{tool_name}()"

        param_steps.append(ParamStep(
            template=template,
            params=params,
            action_type=template_step.get("action_type", "generic"),
        ))

    return param_steps


# ------------------------------------------------------------------
# Phase 4 辅助方法
# ------------------------------------------------------------------
def _parse_action(action_str: str) -> dict:
    """解析工具调用字符串为 {tool, args, raw, action_type}。

    支持 ``tool_name({"k": "v", ...})`` 与 ``tool_name(k="v", ...)`` 两种形式。
    解析失败时退化为 {tool: action_str, args: {}}。
    """
    if not action_str:
        return {"tool": "", "args": {}, "raw": action_str, "action_type": "generic"}

    m = re.match(r'(\w+)\((.*)\)', action_str, re.DOTALL)
    if not m:
        return {"tool": action_str.strip(), "args": {}, "raw": action_str, "action_type": "generic"}

    tool_name = m.group(1)
    args_str = m.group(2).strip()
    args: dict = {}

    if args_str:
        # 优先尝试 JSON 解析
        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, dict):
                args = {str(k): v for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            # 退化为 key=value 字符串提取
            for pair in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                args[pair.group(1)] = pair.group(2)

    return {"tool": tool_name, "args": args, "raw": action_str, "action_type": "generic"}


def _sanitize_param_name(name: str) -> str:
    """将参数名清理为合法 Python 标识符。"""
    cleaned = re.sub(r'\W', '_', str(name))
    if not cleaned:
        cleaned = "param"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _lcs_align(sequences: list[list[str]]) -> list[list[Optional[int]]]:
    """多序列 LCS 对齐，返回对齐位置列表。

    每个元素是一个 list，表示该对齐位置在各序列中的索引
    （``None`` 表示该序列在此位置为 gap）。

    采用渐进式两两对齐：以第一条为基准，依次并入后续序列。
    """
    if not sequences:
        return []
    if len(sequences) == 1:
        return [[i] for i in range(len(sequences[0]))]

    # 第一条序列作为初始对齐骨架，ref_symbols 为参考符号序列
    aligned: list[list[Optional[int]]] = [[i] for i in range(len(sequences[0]))]
    ref_symbols: list[str] = list(sequences[0])

    for seq_idx in range(1, len(sequences)):
        seq = sequences[seq_idx]
        pairs = _lcs_pairs(ref_symbols, seq)
        new_aligned: list[list[Optional[int]]] = []
        new_ref: list[str] = []
        bi, si = 0, 0
        gap_prefix = [None] * seq_idx
        for bi_match, si_match in pairs:
            # base 侧的 gap 列（已在骨架中的列）
            while bi < bi_match:
                new_aligned.append(aligned[bi] + [None])
                new_ref.append(ref_symbols[bi])
                bi += 1
            # seq 侧的 gap 列（新序列独有的位置）
            while si < si_match:
                new_aligned.append(gap_prefix + [si])
                new_ref.append(seq[si])
                si += 1
            # 匹配列
            new_aligned.append(aligned[bi] + [si])
            new_ref.append(ref_symbols[bi])
            bi += 1
            si += 1
        # base 剩余列
        while bi < len(aligned):
            new_aligned.append(aligned[bi] + [None])
            new_ref.append(ref_symbols[bi])
            bi += 1
        # seq 剩余位置
        while si < len(seq):
            new_aligned.append(gap_prefix + [si])
            new_ref.append(seq[si])
            si += 1
        aligned = new_aligned
        ref_symbols = new_ref

    return aligned


# ==================================================================
# Phase 1 — trajectory segmentation
# ==================================================================
def _segment(trajectories: list[Trajectory], llm) -> list[list[int]]:
    """返回首条（代表性）轨迹的步骤索引分段。

    对于最小实现，若步骤数较少则整条轨迹作为一个分段；
    否则使用 *llm* 寻找语义边界。

    与原 ``HarnessInductor._segment`` 行为一致，但不持有实例状态，
    调用方需自行保存返回值（inductor 中存入 ``self._segments``）。
    """
    if not trajectories:
        return []
    rep = trajectories[0]
    if len(rep.steps) <= 3:
        return [list(range(len(rep.steps)))]
    steps_json = json.dumps(
        [{"i": i, "action": s.action, "result": s.result[:80]} for i, s in enumerate(rep.steps)],
        ensure_ascii=False,
    )
    try:
        data = llm.chat_json([
            {"role": "system", "content": "You segment agent trajectories into semantic sub-tasks."},
            {"role": "user", "content": SEGMENT_PROMPT.format(task=rep.task_description, steps_json=steps_json)},
        ])
        segs = data.get("segments", [])
        return [s.get("steps", []) for s in segs] or [list(range(len(rep.steps)))]
    except Exception as exc:
        log.warning("Segmentation failed (%s), using whole trajectory", exc)
        return [list(range(len(rep.steps)))]


def _lcs_pairs(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """两序列 LCS，返回匹配的 (a_idx, b_idx) 对列表（按顺序）。"""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # 回溯
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ------------------------------------------------------------------
# 公共别名（不带下划线前缀），便于外部按语义化名称导入，
# 例如 ``from experience_os.compiler.algorithms import segment``
# ------------------------------------------------------------------
segment = _segment
mine_invariants = _mine_invariants
lcs_align = _lcs_align
