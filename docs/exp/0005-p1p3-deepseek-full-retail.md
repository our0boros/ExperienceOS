# 实验记录 0005：P1-P3 改进后 DeepSeek 全量 Retail

- **日期**: 2026-07-26–27
- **模型**: `deepseek-ai/DeepSeek-V4-Flash`
- **配置**: retail train_test, warmup=74, eval=40, max_steps=30, min_support=3

## 结果

| 指标 | 结果 |
|------|------|
| **Warmup SR** | 48.6% (74 tasks) |
| **Eval SR** | **62.5%** (40 tasks) |
| **总 Token** | 6,592,530 (prompt=6,287,194 + completion=305,336) |
| **平均延迟** | 80.0s |
| **Warmup Token** | 4,035,482 |
| **Eval Token** | 2,557,048 |
| **路径分布** | agent: 82, harness+agent: 28, harness: 4 |
| **LTS 轨迹** | 114 条（含完整对话） |

## Harness 归纳

- **Substep 提取**: 375 tool calls → 写入 LTS
- **Harness APPROVED**: 116 次（13 种唯一工具，每种 8-9 次重复）
- **去重效果**: 每次 APPROVE 前的 deprecate 生效，但 check_triggers 未做 pre-filter
- **Specialization**: 72 次尝试，全部因 `trigger` 参数冲突失败

## 与 exp-0003 对比

| 指标 | exp-0003 (改进前) | exp-0005 (P1-P3 后) | 变化 |
|------|------------------|---------------------|------|
| Eval SR | 62.5% | 62.5% | 无变化 |
| 路径分布 | agent:82, h+a:28, h:4 | agent:82, h+a:28, h:4 | 无变化 |
| Harness 数 | 123 (13 unique) | 116 (13 unique) | 微降（去重有效但不完整） |

## 关键发现

1. **P1-P3 工程修复不改变 SR 上限** — 62.5% 天花板未突破
2. **去重需要 pre-filter** — 当前只在 APPROVE 后 deprecate 旧 harness，未在 trigger 阶段拦截（已修复于 commit `2cb68fd`）
3. **Specialization 全部失败** — 参数顺序错误（已修复于 commit `2cb68fd`）
4. **实验 DB 全空** — 仅 trajectories，substeps/artifacts 仅存于 LTS（legacy 数据已迁移）

## 下一步

- 方向 1: Sub-agent 主动构建 artifact → 突破 SR 天花板
- 方向 2: 强→弱模型迁移实验 → 验证可迁移性
- 需要代码改造：inductor 双写到 ExperienceLibrary（当前仅写 legacy Repository）
