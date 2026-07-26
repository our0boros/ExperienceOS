"""Composite Harness 检测 + 三要素检索 + LLM 决策接口（P3 提案）。

提供三个核心能力：
    1. :class:`HarnessChainDetector` — 检测已有 harness 的 I/O 签名链
    2. :func:`retrieve_by_io_signature` — 三要素（输入/输出/影响）签名检索
    3. :func:`build_harness_chain_prompt` — 生成 LLM 决策提示
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────

@dataclass
class HarnessChain:
    """一条 harness 组合链。"""
    harnesses: list          # 链中的 harness 序列
    coverage: float = 0.0    # 对 task 需求的覆盖率 (0-1)
    description: str = ""    # 人类可读描述

    @property
    def ids(self) -> list[str]:
        return [h.id for h in self.harnesses]

    @property
    def input_signature(self) -> set[str]:
        """链的整体输入（第一个 harness 的 requires 减去链内产出）。"""
        all_outputs: set[str] = set()
        for h in self.harnesses:
            all_outputs |= set(h.output_schema.get("produces", []) if isinstance(h.output_schema, dict) else [])
        first_inputs = set(self.harnesses[0].input_schema.get("requires", []) if isinstance(self.harnesses[0].input_schema, dict) else [])
        return first_inputs - all_outputs

    @property
    def output_signature(self) -> set[str]:
        """链的整体输出（所有 harness 的 produces 的并集）。"""
        outputs: set[str] = set()
        for h in self.harnesses:
            outputs |= set(h.output_schema.get("produces", []) if isinstance(h.output_schema, dict) else [])
        return outputs


# ──────────────────────────────────────────────────────────────────
# I/O 签名工具
# ──────────────────────────────────────────────────────────────────

def io_signature_text(harness) -> str:
    """生成用于 embedding 检索的签名文本。"""
    inputs = harness.input_schema.get("requires", []) if isinstance(harness.input_schema, dict) else []
    outputs = harness.output_schema.get("produces", []) if isinstance(harness.output_schema, dict) else []
    effect = getattr(harness, 'effect', '') or 'read_only'
    return f"input: {', '.join(sorted(inputs))}. output: {', '.join(sorted(outputs))}. effect: {effect}"


def retrieve_by_io_signature(
    requires: set[str],
    produces: set[str],
    registry,
    effect: str | None = None,
    embed = None,
) -> list[tuple]:
    """按 I/O 签名检索匹配的 harness。

    返回 ``[(harness, score, match_type), ...]`` 按 score 降序。
    match_type: "exact_io" | "embedding_io" | "partial_io"
    """
    candidates: list[tuple] = []

    for h in registry.all_active():
        h_inputs = set(h.input_schema.get("requires", []) if isinstance(h.input_schema, dict) else [])
        h_outputs = set(h.output_schema.get("produces", []) if isinstance(h.output_schema, dict) else [])

        # Exact I/O match
        if requires and produces and requires == h_inputs and produces == h_outputs:
            candidates.append((h, 1.0, "exact_io"))
            continue

        # Partial overlap
        input_overlap = len(requires & h_inputs) / max(1, len(requires)) if requires else 0.5
        output_overlap = len(produces & h_outputs) / max(1, len(produces)) if produces else 0.5
        partial_score = (input_overlap + output_overlap) / 2

        if partial_score >= 0.5:
            candidates.append((h, partial_score, "partial_io"))
            continue

        # Embedding similarity (if embed service available)
        if embed and (requires or produces):
            try:
                query_sig = f"input: {', '.join(sorted(requires))}. output: {', '.join(sorted(produces))}"
                h_sig = io_signature_text(h)
                q_vec = embed.embed(query_sig)
                h_vec = embed.embed(h_sig)
                sim = embed._cosine(q_vec, h_vec)
                if sim >= 0.65:
                    candidates.append((h, sim, "embedding_io"))
            except Exception:
                pass

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


# ──────────────────────────────────────────────────────────────────
# HarnessChainDetector
# ──────────────────────────────────────────────────────────────────

class HarnessChainDetector:
    """检测已有 harness 的 I/O 签名链，找出能覆盖 task 需求的组合。"""

    MAX_CHAIN_LENGTH = 5
    MAX_CHAINS = 3

    def detect_chains(
        self,
        task_requires: set[str],
        task_produces: set[str],
        registry,
        embed = None,
    ) -> list[HarnessChain]:
        """返回能覆盖 task_requires → task_produces 的 harness 链。

        BFS: 从 available inputs 出发，探索所有可能的 harness 组合路径。
        """
        harnesses = registry.all_active()
        if not harnesses:
            return []

        chains: list[HarnessChain] = []

        # BFS: (current_available_outputs, harness_sequence, used_ids)
        from collections import deque
        queue: deque = deque()
        queue.append((set(task_requires), [], set()))

        while queue and len(chains) < self.MAX_CHAINS:
            available, path, used_ids = queue.popleft()

            if len(path) > self.MAX_CHAIN_LENGTH:
                continue

            # 检查是否已满足需求
            coverage = len(task_produces & available) / max(1, len(task_produces)) if task_produces else 0.0
            if coverage >= 0.8 and path:
                chains.append(HarnessChain(
                    harnesses=list(path),
                    coverage=coverage,
                    description=" → ".join(
                        f"{h.name or h.task_type}({', '.join(h.params[:3])})"
                        for h in path
                    ),
                ))
                continue

            # 探索下一步
            for h in harnesses:
                if h.id in used_ids:
                    continue  # 无环

                h_inputs = set(h.input_schema.get("requires", []) if isinstance(h.input_schema, dict) else [])
                h_outputs = set(h.output_schema.get("produces", []) if isinstance(h.output_schema, dict) else [])

                # 这个 harness 的输入是否被当前的 available 满足？
                if h_inputs and not h_inputs.issubset(available):
                    # 部分匹配：有些输入我们还没有 → 仍然探索（LLM 可能填补）
                    if len(h_inputs & available) == 0:
                        continue

                new_available = available | h_outputs
                queue.append((new_available, path + [h], used_ids | {h.id}))

        # 按覆盖率排序
        chains.sort(key=lambda c: c.coverage, reverse=True)
        return chains[: self.MAX_CHAINS]


# ──────────────────────────────────────────────────────────────────
# LLM 决策提示
# ──────────────────────────────────────────────────────────────────

CHAIN_SUGGESTION_PROMPT = """Available compiled harnesses relevant to this task:

{harness_list}

{chain_suggestion}

To use a harness, call: call_harness("harness_capability", param1=value1, ...)
Harness execution costs ZERO LLM tokens — it runs as deterministic Python code.
If the harness fails, you'll receive an error message and can fall back to the regular tool.

Decide:
- If a harness or chain covers your current step, USE IT (prefer harness over regular tools)
- If a harness partially helps, use it for the covered part + regular tools for the rest
- If no harness applies, use regular tools as normal
"""


def build_harness_chain_prompt(
    task_description: str,
    registry,
    chain_detector: HarnessChainDetector | None = None,
    embed = None,
) -> str:
    """构建注入 agent 的 harness 建议 prompt。

    检测可用的 harness 和组合链，生成 LLM 决策提示。
    """
    harnesses = registry.all_active()
    if not harnesses:
        return ""

    # 列出可用 harness
    harness_lines = []
    for h in harnesses[:10]:  # top 10 by usage
        inputs = h.input_schema.get("requires", []) if isinstance(h.input_schema, dict) else []
        outputs = h.output_schema.get("produces", []) if isinstance(h.output_schema, dict) else []
        harness_lines.append(
            f"  [{h.capability or h.task_type}] "
            f"({', '.join(inputs[:3])}) → ({', '.join(outputs[:3])})"
        )

    harness_list = "\n".join(harness_lines) if harness_lines else "  (none)"

    # 检测组合链
    chain_text = ""
    if chain_detector and hasattr(registry, 'all_active'):
        # 从 task_description 提取可能需要的关键词作为 requires/produces
        # 简化版：用常见字段
        common_inputs = {"email", "order_id", "user_id", "first_name", "last_name", "zip",
                         "item_ids", "product_id", "payment_method_id"}
        common_outputs = {"user_id", "order_items", "order_status", "product_details",
                          "exchange_result", "return_result", "user_details"}

        chains = chain_detector.detect_chains(
            task_requires=common_inputs,
            task_produces=common_outputs,
            registry=registry,
            embed=embed,
        )

        if chains:
            chain_lines = []
            for i, chain in enumerate(chains[:3], 1):
                chain_lines.append(
                    f"  Chain {i} (coverage={chain.coverage:.0%}): {chain.description}"
                )
            chain_text = "\n".join([
                "Suggested harness chains for this task:",
                *chain_lines,
                "",
                "You can use chains by calling harnesses in sequence.",
            ])

    return CHAIN_SUGGESTION_PROMPT.format(
        harness_list=harness_list,
        chain_suggestion=chain_text or "(no chain suggestions available)",
    )
