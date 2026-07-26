# 实验记录 0003：CoE 全量 Retail 三方法对比

- **日期**: 2026-07-26
- **模型**: `deepinfra/deepseek-ai/DeepSeek-V4-Flash`
- **域**: `tau2-bench` retail / train_test split (train=74, test=40)
- **max_steps**: 30

## 1. 实验结果

| 方法 | Eval SR | 有效 SR* | Eval Token | 每任务 Token | Harness 使用率 |
|------|---------|---------|------------|-------------|---------------|
| react | 24/40 (60.0%) | ~83% | 2,532,730 | 63.3K | 0% |
| skillopt | 26/40 (65.0%) | ~90% | 2,763,381 | 69.1K | 0% |
| **coe** | **25/40 (62.5%)** | **~86%** | **2,316,341** | **57.9K** | **27.8%** (31/114) |

> *排除 11 个 DeepInfra API 内部错误（影响全部三方法，相同 task ID）

### coe 管线细节

```
74 warmup (在线模式) → 371 substeps → 123 harnesses (13 unique)
  ├─ 路径: agent:83 | harness+agent:27 | harness:4
  ├─ 每任务在线提取 tool calls → substeps 表
  └─ 批量归纳时触发 13 种工具类型
```

## 2. 性能分析

### 2.1 Token 节省

Coe eval token 比 react 低 **8.5%**，比 skillopt 低 **16.2%**。4 个任务纯 harness 执行（零 LLM token）。

### 2.2 关键瓶颈：Harness 重复爆炸

| 指标 | 数值 |
|------|------|
| 唯一工具类型 | 13 |
| 总 harness 数 | 123 |
| 重复比 | **9.5x** |

每个 `check_triggers()` → `induce()` 调用都创建新的 harness，不检查是否已存在。在线模式 + 批量模式双重创建。

### 2.3 Harness 验证率分析

所有 harness 都是单 `call_tool()` 包装器（正确），但验证率极低（0.00-0.53）：

- **读工具**（`get_order_details`, `get_product_details`）：验证率 ~0.00-0.04
- **写工具**（`exchange_delivered_order_items`）：验证率 0.49-0.60

**根因**：子步骤验证重放需要精确的 env state + params 匹配源轨迹，但读工具的 env state 无法正确重建。

### 2.4 其他发现

- Params 重复：`get_order_details` params = `['order_id', 'order_id', 'order_id', 'order_id']`
- 意图未聚类：`find_user_id_by_name_zip` 和 `find_user_id_by_email` 是两个独立 pattern，应合并为 `user_lookup`
- task_description 为空：eval 任务示例的 description 全部为空

## 3. 提升空间

### 优先级 1（立即见效）

| 改进 | 预期效果 |
|------|---------|
| **Harness 去重**：induce 前检查是否已有 ACTIVE harness | 消除 9.5x 重复，eval 阶段只匹配最新版 |
| **Params 去重**：LCS 对齐时去重 | 清理 `['order_id', 'order_id', ...]` |
| **读工具跳过验证**：`effect='read_only'` 的单步 harness 直接 APPROVE | 减少无效验证，节省 LLM repair 调用 |

### 优先级 2（架构改进）

| 改进 | 预期效果 |
|------|---------|
| **意图聚类**：embedding 相似的工具合并为同一 capability | 减少冗余 pattern，提升检索精度 |
| **仅批量模式**：去掉在线检测中的 induce，仅在 warmup 结束后批量归纳 | 消除双重创建 |
| **eval 在线反馈**：eval 阶段的 agent fallback 轨迹也提取子步骤 | 持续积累 |

### 优先级 3（后续迭代）

| 改进 | 预期效果 |
|------|---------|
| Composite harness：链式组合多个单步 harness | 覆盖多步任务，提升 harness-only 路径比例 |
| 检索用三要素签名匹配 | 当前仅按 exact intent 匹配 |

## 4. 完整数据

- LTS 经验库：`.experience_os_data/lts_library.db`
- 实验库：`.experience_os_data/exp_coe-retail-train_test-0e1f61a9.db`
- 子步骤：`substeps` 表 371 条记录
- Harness：123 个（13 unique），位于 `experience_os.db` harnesses 表
