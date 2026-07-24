# AGENTS.md — ExperienceOS 项目指导

## 项目概述

ExperienceOS 是一个**从经验中自动归纳可执行知识**的框架。核心假设：
LLM agent 在真实任务上运行 → 积累成功轨迹 → 自动归纳 Harness（可执行 artifact）→
部署时用 Harness 替代 LLM 调用 → 降低 Token 成本 → 随经验增长持续优化。

研究提案详见 `docs/Executable Experience RP.md`，设计讨论详见 `docs/Executable Experience Discuss.md`。

## 架构

```
ACCUMULATION 阶段                   DEPLOYMENT 阶段
┌──────────────┐                   ┌──────────────┐
│  LLM Agent    │──轨迹──→          │  Runtime     │
│  (真实任务)   │         │          │  Router      │
└──────────────┘         ▼          │  (检索 Harness)│
                  ┌──────────┐      └──────┬───────┘
                  │Repository│             │
                  │(4层经验) │      ┌──────▼───────┐
                  └────┬─────┘      │  Harness OR  │
                       │            │  Agent Fallback│
                  ┌────▼─────┐      └──────────────┘
                  │ Inductor  │
                  │ (6阶段编译)│
                  └──────────┘
```

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据模型 | `models.py` | Trajectory / Harness / Stats / VerificationMeta |
| 经验仓库 | `repository.py` | 4 层存储（轨迹→记录→Harness→统计）+ 版本 DAG |
| SQLite 存储 | `storage.py` | 结构化查询 + 向量 BLOB 持久化 + JSON 迁移 |
| Embedding | `embedding.py` | 本地 Qwen3-Embedding-8B → ollama → hash 三级回退 |
| 环境信息 | `env_info.py` | OS/Python/硬件/模型/包版本收集 |
| 检索器 | `retriever.py` | 语义检索 + task_type fallback |
| 编译器 | `compiler.py` | 6 阶段归纳：分割→提取→参数化→生成→验证→注册 |
| Agent | `agent.py` | ReAct fallback + F1-F4 失败分类 |
| 运行时 | `runtime.py` | ACCUMULATION / DEPLOYMENT 模式切换 |
| τ-bench 适配 | `tau2_adapter.py` | Tau2Environment + 轨迹转换 + 数据划分 |
| τ-bench Demo | `tau2_demo.py` | 端到端演示 |
| Baseline | `baseline_eval.py` | 不经归纳的直接 LLM 评估 |
| CLI | `cli.py` | ping/demo/status/harnesses/env-info/baseline/tau2-demo |

## 开发环境

### Python 环境

**主开发环境**：conda env `ml`（含 torch + CUDA）

```bash
# 激活
conda activate ml

# 安装项目（开发模式）
cd /home/our0boros/Project/ExecutableExperience
pip install -e . --no-deps

# 验证
experience-os ping
experience-os env-info
```

**备用环境**：项目 `.venv`（不含 torch，用于轻量测试）

```bash
source .venv/bin/activate
experience-os ping
```

### LLM 后端

| 后端 | 用途 | 配置 |
|------|------|------|
| ollama | 本地测试 | `EOS_LLM_BACKEND=ollama`，模型 `qwen2.5:7b` 或 `qwen3.5:9b` |
| DeepInfra | 远程正式 | `EOS_LLM_BACKEND=deepinfra`，需设置 `DEEPINFRA_TOKEN` |

### 本地模型

`models/` 目录（软链接）包含：
- `Qwen3-Embedding-8B/` — 本地 embedding 模型（safetensors，GPU 加速）
- `Qwen2.5-1.5B-Instruct/` — 小型 LLM
- `Magpie-Qwen2-Pro-200K-Chinese/` — 中文模型

## 常用命令

```bash
# 检查 LLM 连通性
experience-os ping

# Mock 端到端 demo
experience-os demo

# τ-bench 集成 demo
experience-os tau2-demo --domain retail --task-type find_user_id_by_email --warmup 3 --eval 2 --max-steps 30

# Baseline 评估
experience-os baseline --model ollama/qwen2.5:7b --domain retail --max-tasks 10 --max-steps 30

# 查看仓库状态
experience-os status

# 查看 Harness 列表
experience-os harnesses

# 环境信息
experience-os env-info
```

### DeepInfra 后端

```bash
export DEEPINFRA_TOKEN=your_token
EOS_LLM_BACKEND=deepinfra experience-os tau2-demo --max-steps 30
EOS_LLM_BACKEND=deepinfra experience-os baseline --model deepinfra/MiniMaxAI/MiniMax-M2.7 --max-tasks 10
```

## 数据存储

- **SQLite**（主存储）：`.experience_os_data/experience_os.db`
- **JSON 文件**（向后兼容）：`.experience_os_data/{trajectories,records,harnesses,stats}/`
- **环境信息**：`.experience_os_data/env_info.json`

### 存储层级

| 层 | 表 | 内容 |
|----|-----|------|
| L0 | `trajectories` | 原始执行轨迹（步骤、CoT、token、延迟） |
| L1 | `records` | 经验记录（前置条件、规范步骤、不变量） |
| L2 | `harnesses` | 可执行 Harness（代码 + 验证 + embedding BLOB） |
| L3 | `stats` | 任务类型统计（成功率、Token 节省） |
| - | `embeddings` | 向量缓存（text_hash → float32 BLOB） |
| - | `env_metadata` | 环境信息快照 |

## 代码约定

- Python 3.13+，使用 `from __future__ import annotations`
- 类型注解：全部公共函数
- 日志：`log = logging.getLogger(__name__)`，不直接 print（CLI 除外）
- 数据模型：pydantic v2 `BaseModel`
- 测试：`pytest`（待补）
- Lint：`ruff`

### Harness 代码格式

Harness 是 `procedure_code` 字段中的 Python 函数源码，格式：

```python
def run():
    # params: dict — 任务参数
    # call_tool(name, **kwargs) — 调用环境工具，返回 str 或自动解析的 dict
    # env.snapshot() — 当前环境快照
    result = call_tool("tool_name", param="value")
    if isinstance(result, dict):
        user_id = result["user_id"]
    return "success"
```

## τ-bench 集成

τ-bench（Sierra 原版）位于 `tau2-bench/` 目录，Python 3.12+ 兼容。

### 验证状态

| 模型 | 任务类型 | 成功率 | 说明 |
|------|---------|--------|------|
| DeepInfra/MiniMax-M2.7 | find_user_id_by_email | 5/5 (100%) | Warm-up 3 + Eval 2 |
| ollama/qwen2.5:7b | find_user_id_by_name_zip | 0/5 (0%) | zip 参数类型问题 |

### 已知问题

1. `find_user_id_by_name_zip` 工具的 `zip` 参数需为字符串，LLM 常传整数导致失败
2. `max_steps` < 30 时多轮对话任务容易超时
3. DeepInfra 有速率限制，需间隔调用

## 关键设计决策

### 为什么 Harness 是 plain text 代码

Harness 的 `procedure_code` 是 Python 函数源码字符串。选择 plain text 而非 AST/字节码：
- LLM 生成的是文本，plain text 最自然
- 沙盒执行用 `exec()` + restricted globals，简单直接
- 版本 diff 可读

### 为什么用三级 embedding 回退

1. 本地 Qwen3-Embedding-8B（GPU）— 语义最佳，无网络依赖
2. ollama API — 有网络时方便，但 ollama 可能未开 embeddings
3. hash 伪向量 — 保证一致性，无语义，仅用于 fallback

## 待完成

- [ ] Baseline 对比框架（Vanilla LLM vs RAG vs AutoHarness）
- [ ] 消融实验（w/o Validation, w/o Versioning）
- [ ] TerminalBench 集成
- [ ] 多层级知识库（personal/org/public）
- [ ] Harness artifact 包格式（可分发）
- [ ] 版本树 DAG 可视化
