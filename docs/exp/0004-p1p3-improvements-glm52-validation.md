# 实验记录 0004：P1-P3 改进 + GLM-5.2 小型验证

- **日期**: 2026-07-26
- **模型**: `deepseek-ai/DeepSeek-V4-Flash` + `zai-org/GLM-5.2`

## 1. 改进清单

| 编号 | 改进 | 文件 | 效果 |
|------|------|------|------|
| P1.1 | Harness 去重 | `inductor.py` | APPROVED 前 deprecate 旧 harness |
| P1.2 | 读工具跳过验证 | `inductor.py` | `effect=read_only` → 直接 APPROVE |
| P1.3 | Params 去重 | `inductor.py` | `list(dict.fromkeys(...))` |
| P1.4 | 仅批量模式 | `compare.py` | warmup 只提取 substep，不触发在线 induce |
| P2.1 | 意图聚类 | `inductor.py` | `_cluster_patterns()` — embedding 相似工具合并 |
| P2.2 | Eval 在线反馈 | `compare.py` | eval agent 轨迹也提取子步骤 |
| P3 | Composite + 三要素检索 | `composite.py`(新) | `HarnessChainDetector`, I/O 签名检索 |
| Alg | 语义分块 + 加权相似度 | `harness_registry.py` | `lookup_weighted()` 多字段匹配 |

## 2. 去重效果验证（exp-0003 旧数据）

```
修复前: 123 harnesses / 13 unique tools = 9.5x 重复
修复后: 10 old harnesses deprecated → 1 per tool ✓
```

## 3. GLM-5.2 小型验证

**配置**: GLM-5.2, retail train_test, 10 warmup, 0 eval, max_steps=30, min_support=2

| 指标 | 结果 |
|------|------|
| Warmup SR | 7/10 (70.0%) |
| Substep 提取 | 53 tool calls |
| Harness APPROVED | 7 个 |
| 去重生效 | ✅ 旧 harness deprecated |

### 与 DeepSeek-V4-Flash 对比

| 指标 | GLM-5.2 | DeepSeek-V4-Flash |
|------|---------|-------------------|
| Warmup SR (10 tasks) | **70%** | 40-50% |
| Harness APPROVED | 7 | 4-14 |
| Harness 验证率天花板 | 0.00-0.51 | 0.00-0.53 |

**结论**: GLM-5.2 积累速度更快（更高的 warmup SR），但 harness 验证率天花板相同。验证率低是 replay 机制的固有问题，与模型无关。方向 2 的价值在于「积累速度」——强者更快到达 min_support。

## 4. 方向判断

- P1-P3 是工程修复，不改变 SR 上限
- SR 上限受限于模型基础能力 + harness 验证机制的固有问题
- 方向 2 (强→弱迁移) 价值在积累速度，非 harness 质量
- 方向 1 (sub-agent 主动构建) 可能突破 harness 质量天花板
