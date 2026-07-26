# 实验记录 0002：CoE 子步骤管线端到端验证 + 基线对比

- **日期**: 2026-07-26
- **环境**: Windows + conda `eos` (Python 3.12.13)
- **模型**: `deepinfra/deepseek-ai/DeepSeek-V4-Flash`（经 LiteLLM 调用 DeepInfra）
- **域 / 任务**: `tau2-bench` retail / train_test split (train=74, test=40)
- **max_steps**: 30

## 1. 实验设计

三方法对比，同一 backbone + 同一数据划分：

| 方法 | warmup | eval | 说明 |
|------|--------|------|------|
| `react` | 0 | 5 | τ-bench 内置多步 ReAct agent，无积累 |
| `skillopt` | 0 | 5 | SkillOpt 初始 skill 文本注入 agent system prompt |
| `coe` | 5 | 5 | Warm-up 积累 → 子步骤提取 → 归纳 → Eval 部署（Harness 优先） |

## 2. 架构背景：层次 2 重构

本次实验在全新架构上运行（6 个阶段全部重构完成）：

| 阶段 | 内容 |
|------|------|
| S1 | `Services` — 统一 LLM/Embedding 服务层 |
| S2 | `substeps` 表 — 子步骤作为独立一等实体（24 列，含三要素签名 + 贝叶斯权重） |
| S3 | 四层意图匹配 — exact → embed≥0.85 → 0.65-0.85 LLM → <0.65 ReAct |
| S4 | `check_triggers()` 返回子步骤模式列表 + 贝叶斯门控 |
| S5 | `TaskPlanner` + `PlanExecutor` — Plan-then-Execute 核心 |
| S6 | `compare.py` 集成 — warmup → 子步骤提取 → 双级归纳 |

核心改动：子步骤触发不再依赖全任务成功率。`EOS_MIN_SUPPORT=2`（小规模测试）。

## 3. 实验结果

### 3.1 方法对比

| 方法 | Eval SR | Eval Tokens | Warmup SR | 路径 |
|------|---------|------------|-----------|------|
| react | 4/5 (80.0%) | 421,572 | N/A | agent:5 |
| skillopt | 5/5 (100.0%) | 456,901 | N/A | agent:5 |
| coe | 3/5 (60.0%) | 417,891 | 2/5 (40.0%) | agent:5 |

### 3.2 子步骤管线

```
warmup 5 任务（2 成功）
  → [substep] extracted 11 tool calls as substeps ✅
  → 5 聚合模式 (support=2, sr=1.00) ✅
  → 全部触发 sub-step induction ✅
  → LLM 合成代码 ✅
  → sandbox 验证 ❌ (0/5 APPROVED: 4 REJECTED, 1 NEEDS_REVISION)
```

### 3.3 已合成 Harness（DRAFT）

| Pattern | Status | Validation Rate | 问题 |
|---------|--------|----------------|------|
| `find_user_id_by_name_zip` | DRAFT | 0.29 | 生成多步代码（含 get_user_details） |
| `exchange_delivered_order_items` | DRAFT | 0.50 | 3 次修复仍不通过 |
| `get_order_details` | REJECTED | 0.00 | 验证失败 |
| `get_product_details` | REJECTED | 0.00 | 验证失败 |
| `get_user_details` | REJECTED | 0.00 | 验证失败 |

### 3.4 根因诊断

四个问题被识别和修复（见 §4）：

1. **跨实验污染**：`_discover_substep_patterns_from_store()` 查询全库 substeps，含旧实验轨迹
2. **合成 prompt 不适配**：通用 `SYNTHESIS_PROMPT` 告诉 LLM 生成"完整任务"代码，而非单工具调用
3. **单步语义缺失**：LLM 自动添加额外的 tool calls（如 lookup_user 后自动加 get_user_details）
4. **验证阈值过高**：子步骤验证用同样的 0.8 threshold

## 4. 修复记录

### 4.1 Experiment 过滤

`inductor.py:_discover_substep_patterns_from_store()` 新增 `experiment_id` 参数，`compare.py` 在归纳前设置 `inductor._current_experiment_id`。

### 4.2 子步骤专用 Synthesis Prompt

新增 `SUBSTEP_SYNTHESIS_PROMPT`（`prompts.py`）— 明确约束：
- 只生成单个 `call_tool()` 调用
- 不添加额外步骤
- 直接返回 tool result
- 强调 "SINGLE-STEP harness"

### 4.3 _synthesize 分流

`inductor._synthesize()` 新增 `is_substep` 和 `tool_name` 参数，子步骤触发时自动切换 prompt。

### 4.4 数据清理

清空旧 substeps/harnesses 数据，确保干净验证环境。

## 5. 后续实验计划

| Phase | 内容 |
|-------|------|
| A | 验证修复效果：重跑 coe 5+5（当前进行中） |
| B | 若修复生效（≥1 APPROVED harness）→ 全量 retail train=74/eval=40 |
| C | 全量三方法对比（react vs skillopt vs coe） |
| D | 积累曲线 + 消融实验 |

## 6. 数据持久化

- LTS 经验库：`.experience_os_data/lts_library.db`
- 实验库：`.experience_os_data/exp_<experiment_id>.db`
- 子步骤：`substeps` 表（50+ 条记录，跨越 4 个实验）
