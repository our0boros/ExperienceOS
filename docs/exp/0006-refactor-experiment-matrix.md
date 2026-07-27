# 实验记录 0006：重构后实验矩阵与验证计划

- **日期**: 2026-07-27
- **重构提交**: 待创建 git tag `refactor-v1`

## 1. 重构变更摘要

本次重构完成了 TODO.md §11.3 的前三个步骤：

| 步骤 | 内容 | 新增文件 | 测试 |
|------|------|---------|------|
| 1 | KSI adapter — τ-bench → KSI TaskSpec 转换，JSONL 导出 | `experience_os/ksi_adapter.py` | 15 个用例 |
| 2 | 统一 runner 协议 — TaskSource / SplitPolicy / ModeController / MethodRunner / MetricsRecorder | `experience_os/experiments/runner.py` | 14 个用例 |
| 3 | Provider 注册 — DeepInfra / Ollama / OpenAI / Anthropic / local / LiteLLM | `experience_os/services.py` (追加) | 9 个用例 |
| 4 | 弃用代码清理 — `__init__.py` 更新、deprecation docstring、inductor legacy fallback 警告 | 多个文件 | 50 个全量通过 |

**总测试数**: 50 个（27 旧 + 23 新），全部通过。

## 2. 实验矩阵

### 2.1 主对比实验（retail train_test, min_support=3）

| # | 方法 | Model | Warmup | Eval | 关键指标 | 状态 |
|---|------|-------|--------|------|---------|------|
| M1 | react | DeepSeek-V4-Flash | 0 | 40 | SR baseline | ⬜ 待运行 |
| M2 | skillopt | DeepSeek-V4-Flash | 0 | 40 | 文本注入 baseline | ⬜ 待运行 |
| M3 | coe | DeepSeek-V4-Flash | 74 | 40 | harness SR + token + hit rate | ⬜ 待运行 |
| M4 | ksi | DeepSeek-V4-Flash | 74 | 40 | KSI JSONL manifest + 对比 | ⬜ 待运行 |

### 2.2 积累曲线

| # | 内容 | 配置 | 状态 |
|---|------|------|------|
| C1 | coe 累积曲线 | retail, 滚动窗口=5, checkpoints=10/20/40/74 | ⬜ 待运行 |
| C2 | react baseline 曲线 | retail, 相同任务序列 | ⬜ 待运行 |

### 2.3 泛化实验

| # | 内容 | 配置 | 状态 |
|---|------|------|------|
| G1 | cross-domain | airline → retail | ⬜ 待运行 |
| G2 | cross-model (强→弱) | GLM-5.2 warmup → qwen2.5:7b eval | ⬜ 待运行 |
| G3 | in-domain cross-instance | retail, type_split, unseen instances | ⬜ 待运行 |

### 2.4 消融实验

| # | 消融项 | 配置 | 状态 |
|---|--------|------|------|
| A1 | no executable artifact | Passive Retrieval baseline | ⬜ 待运行 |
| A2 | no semantic consolidation | rule-only substep aggregation | ⬜ 待运行 |
| A3 | no validation gate | skip_validation=True | ⬜ 待运行 |
| A4 | MIN_SUPPORT sensitivity | support={1,3,5,7} | ⬜ 待运行 |

## 3. 最小验证运行

重构后应先运行最小对比，确保 CoE 不劣于旧版本（exp-0005：SR 62.5%，57.9K tokens/task，27.8% harness 使用率）。

### 3.1 使用新 runner 协议运行

```bash
# 方式 1：使用 run_experiment_v2 便捷函数
python -c "
from experience_os.experiments.runner import run_experiment_v2, ExperimentConfig, ExperimentRunner
metrics = run_experiment_v2(
    method='coe', model='deepinfra/deepseek-ai/DeepSeek-V4-Flash',
    domain='retail', warmup=74, eval_size=40, max_steps=30,
    split_policy='train_test',
)
print(metrics.to_dict())
"

# 方式 2：使用 ExperimentRunner（更灵活）
python -c "
from experience_os.experiments.runner import ExperimentRunner, ExperimentConfig
config = ExperimentConfig(
    mode='deployment', method='coe', domain='retail',
    model='deepinfra/deepseek-ai/DeepSeek-V4-Flash',
    warmup=74, eval_size=40, max_steps=30, split_policy='train_test',
)
runner = ExperimentRunner(config)
metrics = runner.execute()
"
```

### 3.2 使用旧 compare.py（向后兼容）

```bash
experience-os compare --method coe --model deepinfra/deepseek-ai/DeepSeek-V4-Flash --domain retail --warmup 74 --eval 40 --variant train_test
```

## 4. KSI Baseline 运行步骤

1. 利用 `ksi_adapter.py` 导出 τ-bench 任务为 KSI JSONL：
```bash
python -c "
from experience_os.tau2_adapter import load_tasks
from experience_os.ksi_adapter import export_ksi_tasks, export_ksi_run_manifest
tasks = __import__('tau2.domains.retail.environment', fromlist=['get_tasks']).get_tasks('base')
export_ksi_tasks(tasks, 'ksi_retail_tasks.jsonl')
export_ksi_run_manifest(tasks, 'ksi_retail_manifest.json', domain='retail')
"
```

2. 用 KSI 运行导出的任务（需 KSI 环境）：
```bash
cd KSI
ksi --task-source custom --tasks-path ../ksi_retail_tasks.jsonl --num-generations 1 --num-agents 1
```

3. 对比 KSI 结果与 ExperienceOS 结果。

## 5. 清理检查清单

- [x] `llm.py` 删除，零残留引用
- [x] `embedding.py` 删除，零残留引用
- [x] `__init__.py` docstring 更新为当前模块布局
- [x] `repository.py` 添加 deprecation docstring
- [x] `storage.py` 添加 deprecation docstring
- [x] `experience_library.py` 添加 deprecation docstring
- [x] `inductor.py` legacy fallback 添加 log 警告
- [x] `Services` 只暴露 `chat` + `embedding`（`llm`/`embed` 别名已移除）
- [x] ProviderRegistry 注册 6 个内置 provider
- [ ] 创建 git tag `refactor-v1`（验证实验后）

## 6. 中断点与后续任务

按 TODO.md §11.3 顺序：

1. ✅ KSI adapter + 测试
2. ✅ compare.py 统一 runner 协议重构
3. ✅ DeepInfra provider 注册
4. ⬜ 最小验证实验（需 DeepInfra API key + tau2-bench 环境）
5. ⬜ git tag `refactor-v1`（验证通过后）
6. ⬜ 按实验矩阵补齐 cross-domain / cross-model / 消融
