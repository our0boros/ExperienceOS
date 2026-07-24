# ExperienceOS 项目结构与差距评估

> 本文档记录当前项目的实际结构、验证环境、以及与 `Executable Experience Discuss.md` (L2089 之后) 中设计的对比差距。

---

## 1. 当前项目结构

```
ExecutableExperience/
├── docs/
│   ├── Executable Experience RP.md          # 研究提案（AutoHarness + ExperienceOS）
│   └── Executable Experience Discuss.md     # 讨论记录（含系统设计、实验方案）
├── tau2-bench/                               # τ-bench 框架（Sierra 原版，Python 3.12）
├── harbor-TerminalBench/                     # TerminalBench 框架（harbor，Python 3.12）
├── models/                                   # 本地模型目录（软链接）
│   ├── Qwen3-Embedding-8B/                   # ★ 本地 embedding 模型（safetensors）
│   ├── Qwen2.5-1.5B-Instruct/                # 小型 LLM
│   └── ...
├── experience_os/                            # ★ 框架核心
│   ├── __init__.py
│   ├── config.py                             # 配置：LLM 后端 + 归纳阈值
│   ├── llm.py                                # LLM 客户端（ollama / DeepInfra）
│   ├── embedding.py                          # ★ 本地 embedding 客户端（Qwen3-Embedding-8B → ollama → hash）
│   ├── models.py                             # 数据模型：Trajectory / Harness / Stats 等
│   ├── repository.py                         # 4 层经验仓库 + 版本 DAG（JSON 持久化）
│   ├── storage.py                             # ★ SQLite 存储层（结构化查询 + 向量持久化）
│   ├── env_info.py                            # ★ 环境信息收集器（OS/内核/Python/硬件/模型）
│   ├── environment.py                         # 环境接口 + MockEnvironment + 沙盒执行
│   ├── retriever.py                           # Runtime Router：语义检索 + 前置条件匹配
│   ├── compiler.py                           # Harness Inductor：6 阶段编译 + 沙盒验证
│   ├── agent.py                               # Agent Fallback（ReAct）+ F1-F4 失败分类
│   ├── runtime.py                             # 主循环：ACCUMULATION / DEPLOYMENT 模式
│   ├── tau2_adapter.py                        # τ-bench 集成：环境适配 + 轨迹转换 + 数据划分
│   ├── tau2_demo.py                           # τ-bench 端到端 demo
│   ├── cli.py                                 # CLI：ping / demo / status / harnesses / env-info / tau2-demo
│   └── demo.py                                # MockEnvironment 端到端演示脚本
├── pyproject.toml
├── .env.example
└── .experience_os_data/                      # 运行时数据
    ├── experience_os.db                      # ★ SQLite 数据库（主存储）
    ├── env_info.json                          # ★ 环境信息快照（最新）
    ├── trajectories/                          # Layer 0：原始轨迹 JSON（向后兼容）
    ├── records/                               # Layer 1：经验记录 JSON（向后兼容）
    ├── harnesses/                             # Layer 2：可执行 Harness JSON（向后兼容）
    ├── embeddings/                            # 向量缓存（预留）
    └── stats/                                 # Layer 3：任务类型统计 JSON（向后兼容）
```

### 模块职责映射

| 文件 | 对应文档设计 | 实现状态 |
|------|-------------|---------|
| `models.py` | §2 基本定义、Hoare Triple `H=<P,steps,I,Q,R>` | ✅ 完整 |
| `repository.py` | §3 三层结构 + §3.6 版本 DAG | ⚠️ 4 层 JSON 存储（向后兼容） |
| `storage.py` | §3 三层结构（SQLite 持久化） | ✅ SQLite + 向量 BLOB + JSON 迁移 |
| `embedding.py` | §3.2 向量检索 embedding 生成 | ✅ 本地 Qwen3-Embedding-8B → ollama → hash |
| `env_info.py` | 环境信息收集（非文档设计，工程需求） | ✅ OS/Python/硬件/模型/包版本 |
| `retriever.py` | §3.2 Harness 检索（粗筛+精筛） | ⚠️ 内存 cosine，可接入 Storage 向量缓存 |
| `compiler.py` | §3.3 六阶段归纳 + §2.2 Bayesian 触发 | ⚠️ 基本流程有，LCS/参数化粗糙 |
| `agent.py` | §3.4 失败分类 F1-F4 + Agent Fallback | ✅ 基本完整 |
| `runtime.py` | §3.1 系统总览 + §4.3 模式分离 | ✅ ACCUMULATION / DEPLOYMENT |

### 存储架构

当前采用**双存储并存**，SQLite 为首选，JSON 文件向后兼容：

| 存储层 | 格式 | 优势 | 用途 |
|--------|------|------|------|
| `experience_os.db` | SQLite | 结构化查询、事务、向量 BLOB | 主存储（轨迹/记录/Harness/统计/embedding/env metadata） |
| `trajectories/*.json` 等 | JSON 文件 | 人类可读、Git 友好 | 向后兼容、调试 |

**SQLite 表结构**：
- `trajectories` — Layer 0，原始轨迹（含 steps JSON）
- `records` — Layer 1，经验记录
- `harnesses` — Layer 2，可执行 Harness + embedding BLOB
- `stats` — Layer 3，任务类型统计
- `embeddings` — 向量缓存（text_hash → float32 BLOB）
- `env_metadata` — 环境信息快照（含 OS/Python/硬件/模型版本）

**Embedding 三级回退**：
1. 本地 Qwen3-Embedding-8B（sentence-transformers + GPU）
2. ollama embeddings API
3. hash 伪向量（保证一致性，无语义）

所有 embedding 通过 SQLite 持久化缓存，避免重复计算。

---

## 2. 当前评估和验证环境

### 2.1 使用的是什么环境？

**两个环境并存：**

1. **MockEnvironment**（[environment.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/environment.py)）：极简内存模拟环境，用于快速验证流程闭环。已验证：积累 → 归纳 → 部署完整跑通，Token 降低 100%。

2. **τ-bench retail 域**（[tau2_adapter.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/tau2_adapter.py)）：已集成 τ-bench 框架，可运行真实客服任务仿真。已验证：tau2 仿真 → 轨迹捕获 → 数据划分 → 归纳触发检查 → 部署，全流程跑通。

### 2.2 τ-bench 集成现状

通过 `experience-os tau2-demo` 命令运行。支持 `--task-type` 选择任务类型、`--solo` solo 模式。

#### 验证 1：ollama / qwen2.5:7b + `find_user_id_by_name_zip`

| 阶段 | 结果 |
|------|------|
| 任务加载 | ✅ 114 个 retail 任务，9 种任务类型 |
| 数据划分 | ✅ Warm-up / Evaluation 池按任务类型分离 |
| tau2 仿真 | ✅ 5 次仿真全部完成 |
| 轨迹捕获 | ✅ SimulationRun → Trajectory 格式转换成功 |
| 归纳触发 | ❌ 未触发（所有 reward=0.0，support_count=0） |
| 部署 | ✅ 2 个评估任务执行（全部 fallback 到 agent） |

**失败原因**：LLM 传递 `zip` 参数为整数 `19122` 而非字符串 `"19122"`，导致 `find_user_id_by_name_zip` 工具返回 "User not found"。参考动作使用字符串类型。

#### 验证 2：DeepInfra / MiniMax-M2.7 + `find_user_id_by_email`（max_steps=30）

| 阶段 | 结果 |
|------|------|
| 任务加载 | ✅ 11 个 `find_user_id_by_email` 任务 |
| 数据划分 | ✅ 3 Warm-up + 2 Evaluation |
| tau2 仿真 | ✅ 5 次仿真全部完成 |
| 轨迹捕获 | ✅ 5 条成功轨迹（6-7 步/条） |
| 归纳触发 | ✅ support_count=3 → `new_harness` |
| Harness 编译 | ✅ `find_user_id_by_email-v1` APPROVED（replay sr=1.00）|
| 部署 | ✅ 2/2 成功（agent fallback，harness 尚未在 demo 中自动部署） |

**关键突破**：
1. 首次在真实 τ-bench 任务上获得 **5/5 成功率**（3 warm-up + 2 eval）
2. 归纳成功触发并 **编译通过** Harness（replay rate=1.0）
3. Harness 代码使用 `call_tool` 自动解析 JSON 结果（修复了 `string indices must be integers` 错误）
4. Harness 已存储在经验仓库中，状态为 ACTIVE

### 2.3 当前验证结果汇总

| 环境 | 模型 | 任务 | 成功率 | 归纳 | Harness sr |
|------|------|------|--------|------|-----------|
| MockEnvironment | ollama/qwen2.5:7b | lookup→submit | 3/3 | ✅ triggered | 1.00 |
| τ-bench retail | DeepInfra/MiniMax-M2.7 | find_user_id_by_email | 5/5 | ✅ triggered | 1.00 |
| τ-bench retail | ollama/qwen2.5:7b | find_user_id_by_name_zip | 0/5 | ❌ not triggered | — |

**结论**：流程闭环在真实 τ-bench 任务上已验证可行。模型质量是关键因素——MiniMax-M2.7 能在 30 步内完成 `find_user_id_by_email` 类任务，而 qwen2.5:7b 在 `find_user_id_by_name_zip` 类任务上因参数类型问题全部失败。

---

## 3. 与文档设计的差距评估

以下逐条对比 `Discuss.md` L2089 之后的核心设计，评估当前实现的差距。

### 3.1 经验和 Artifact 的保存结构

#### 文档设计（L2430-2458）

文档提出**多层级知识库**：

```
Level 0：个人知识库（私有，个性化）
Level 1：组织知识库（企业内部共享）
Level 2：公共知识库（通用软件标准化 Harness，类似 npm）

混用逻辑（优先级覆盖链）：
  个人 Harness > 组织 Harness > 公共 Harness
```

#### 当前实现

| 设计要素 | 当前状态 | 差距 |
|---------|---------|------|
| 4 层存储（轨迹/记录/Harness/统计） | ✅ SQLite + JSON 双存储 | 结构化查询 + 向量 BLOB 持久化 |
| 多层级知识库 | ❌ 未实现 | 只有单层本地存储 |
| 优先级覆盖链 | ❌ 未实现 | 无 personal/org/public 分层 |
| Harness 导出/导入 | ❌ 未实现 | 无法共享 artifact 包 |
| 向量数据库索引 | ⚠️ SQLite BLOB | 无独立向量数据库，但本地 embedding + SQLite 缓存已实现 |
| 环境 metadata | ✅ 已实现 | env_info.py 收集 OS/Python/硬件/模型信息，SQLite 持久化 |

**核心差距**：当前的保存结构已升级为 SQLite + 本地 embedding，支持结构化查询和向量持久化。但文档设想的"公共可执行知识库"仍需要一个可分发、可版本管理、可分层覆盖的 artifact 包格式（类似 pip/npm），当前完全缺失。

---

### 3.2 轨迹（Trajectory）的质量与数量

#### 文档设计（L2274-2286, L2389-2426）

- 每个任务类型需积累 `MIN_SUPPORT=3` 条**成功**轨迹
- 轨迹包含：完整 observation-action 序列 + 环境快照 + 结构化 CoT + 执行结果
- `TaskTypeStats` 维护 `pending_variations`（积累中但未触发归纳的变异）
- 文档明确："3 条足够归纳共性"

#### 当前实现

| 设计要素 | 当前状态 | 差距 |
|---------|---------|------|
| 轨迹格式（obs-action-result） | ✅ 有 | 但 MockEnvironment 的轨迹过于简单（1-2 步） |
| 结构化 CoT | ⚠️ 有字段但内容空 | `StructuredCoT.goal` 只存了 expected_output，无真实推理链 |
| 环境快照 | ✅ 有 | MockEnvironment 只有 `{env: mock, store_keys: [...]}` |
| 轨迹数量触发 | ✅ support_count >= 3 | 正常工作 |
| `pending_variations` | ❌ 未实现 | 无法积累"变异维度"，不支持特化分裂触发 |
| 真实任务轨迹 | ❌ 未接入 | 无 τ-bench / TerminalBench 的真实多步轨迹 |

**核心差距**：当前轨迹来自 MockEnvironment，每条轨迹只有 1-2 步（lookup → submit），过于简单。真实 τ-bench 任务的轨迹包含多轮对话、policy 查询、数据库操作、多步验证等，复杂度高一个数量级。**归纳算法的质量完全取决于轨迹质量**——当前的轨迹不足以验证归纳在真实场景下的效果。

---

### 3.3 归纳算法（Harness Induction）

#### 文档设计（L2270-2357, L2291-2326）

6 阶段：
1. 轨迹分段（LLM 识别语义边界）
2. 前后置条件提取（跨轨迹交集）
3. 不变量挖掘（Daikon 风格动态不变量）
4. 步骤抽象与参数化（LCS 最长公共子序列 + 具体值→变量名）
5. Harness 合成（LLM 生成可执行代码）
6. 沙盒回放验证（success_rate >= 0.8）

#### 当前实现

| 阶段 | 当前状态 | 差距 |
|------|---------|------|
| Phase 1 分段 | ⚠️ 有但退步 | 步骤 ≤3 时跳过分段，直接整体处理 |
| Phase 2 前置条件 | ✅ 交集提取 | 跳过 list/dict 值，只保留标量 |
| Phase 3 不变量 | ⚠️ 粗糙 | 只检测首步一致性和成功率，非 Daikon 风格 |
| Phase 4 参数化 | ⚠️ 粗糙 | 用正则替换引号内字符串，非 LCS |
| Phase 5 合成 | ✅ LLM 生成 | 有 few-shot 示例，代码提取有 markdown fence 处理 |
| Phase 6 验证 | ✅ 沙盒回放 | replay rate >= 0.8 入库 |
| NEEDS_REVISION 重试 | ❌ 未实现 | 存为 DRAFT 后不重试，文档设计应分析失败模式并修复 |

**核心差距**：归纳算法的骨架完整，但 Phase 1/3/4 都是简化版。特别是 Phase 4 的参数化用简单正则而非 LCS，在真实多步轨迹上效果会差。另外缺少 `NEEDS_REVISION` 的修复循环。

---

### 3.4 触发机制

#### 文档设计（L2361-2385）

```
事件流：
  任务执行完成
    ├→ [立即] 记录 Raw Trajectory
    ├→ [立即] 更新 task_type 统计
    ├→ [异步] 检查 Induction 触发条件
    │         - support_count 达阈值 → 新建 Harness
    │         - new_variation → 特化分裂
    └→ [异步] 失败分类 → F2 count >= 2 → Patch
```

#### 当前实现

| 设计要素 | 当前状态 | 差距 |
|---------|---------|------|
| 立即记录轨迹 | ✅ 有 | 同步执行 |
| 立即更新统计 | ✅ 有 | |
| 触发 new_harness | ✅ 有 | support_count >= MIN_SUPPORT |
| 触发 patch | ✅ 有 | F2 >= 2 |
| 触发 specialization | ❌ 未实现 | 无 `new_variation_detected` 逻辑 |
| 异步执行 | ❌ 同步 | 文档设计为异步，当前阻塞主流程 |

**核心差距**：特化分裂触发未实现（需要 `pending_variations` 支持），且归纳是同步阻塞的。

---

### 3.5 实验设计与 Baseline

#### 文档设计（L2815-2855, L2859-2887）

**数据划分**：
- Warm-up Pool：每类 K=3 个实例，用于积累
- Evaluation Pool：剩余实例，与 Warm-up 不重叠
- 所有 Baseline 使用相同 Warm-up 数据

**Baseline 层次**：
- A: Vanilla LLM（无工具）
- B: ReAct Agent（无记忆）
- C: RAG Agent（检索历史轨迹）
- D: AutoHarness w/o Validation
- E: AutoHarness w/o Versioning
- F: AutoHarness w/ Fixed Harness（人工上界）

**主实验指标**：成功率 + Token + 延迟 + Harness Hit Rate

**核心图表**：积累曲线图（x=任务序号, y=滚动成功率）

#### 当前实现

| 设计要素 | 当前状态 | 差距 |
|---------|---------|------|
| Warm-up / Evaluation 划分 | ✅ 已实现 | `tau2_adapter.split_tasks()` 按任务类型分组划分 |
| τ-bench 集成 | ✅ 已实现 | 仿真运行 + 轨迹转换 + DB hash 验证 |
| Vanilla LLM baseline | ❌ 未实现 | |
| ReAct Agent baseline | ⚠️ 有 | agent.py 即 ReAct，但无对比框架 |
| RAG Agent baseline | ❌ 未实现 | |
| 消融实验（w/o Validation 等）| ❌ 未实现 | |
| 积累曲线图 | ❌ 未实现 | |
| TerminalBench 集成 | ❌ 未实现 | harbor 目录在但无适配器 |

**核心差距**：τ-bench 集成的流程已跑通，但缺少 Baseline 对比框架和消融实验。需要更强模型让 agent 成功完成任务以触发归纳。

---

### 3.6 检索质量

#### 文档设计（L2189-2210）

- 用 **向量数据库** 做 semantic retrieval
- Harness 检索向量由 4 个维度构成：`task_type + description + preconditions_summary + example_tasks`

#### 当前实现

| 设计要素 | 当前状态 | 差距 |
|---------|---------|------|
| 向量数据库 | ⚠️ SQLite BLOB | 无独立向量 DB，但本地 embedding + SQLite 缓存已实现 |
| 真实 embedding | ✅ 有 | Qwen3-Embedding-8B 本地模型（GPU 加速），ollama 回退 |
| 检索向量 4 维度 | ⚠️ 3 维度 | 有 task_type + capability + description + preconditions，缺 `example_tasks` |
| 持久化向量索引 | ✅ 有 | embedding 通过 SQLite 持久化，Harness embedding 存为 BLOB |

**核心差距**：已实现本地 embedding（Qwen3-Embedding-8B）和 SQLite 向量持久化，但缺少独立向量数据库（如 FAISS/ChromaDB）在大规模场景下的索引优化。当前靠 `task_type` fallback 也能工作，但真实多任务类型场景下需要真正的语义检索。

---

## 4. 总结：当前验证了什么 vs 还缺什么

### ✅ 已验证

1. **流程闭环可行**：积累 → 归纳 → 验证 → 部署的完整循环能跑通
2. **Token 降低效果**：Harness 部署阶段 Token 从 89 降到 0（MockEnvironment，100%）
3. **归纳触发正确**：support_count >= 3 自动触发，replay 验证门控正常
4. **双后端切换**：ollama（本地测试）和 DeepInfra（远程正式）都能工作
5. **τ-bench 真实任务成功**：DeepInfra/MiniMax-M2.7 在 `find_user_id_by_email` 任务上 5/5 成功
6. **Harness 编译通过**：从真实轨迹归纳的 Harness replay rate=1.0，已存储为 ACTIVE
7. **JSON 自动解析**：`call_tool` 返回 JSON 字符串时自动解析为 dict/list，Harness 代码可直接访问字段

### ❌ 主要缺口（按优先级排序）

| 优先级 | 缺口 | 影响 | 状态 |
|--------|------|------|------|
| ~~P0~~ | ~~无 τ-bench 环境集成~~ | ~~无法在真实任务上验证~~ | ✅ 已实现（tau2_adapter.py） |
| ~~P0~~ | ~~无数据划分（Warm-up/Eval）~~ | ~~实验存在数据泄露风险~~ | ✅ 已实现（split_tasks） |
| P0 | **轨迹质量过低** | ~~MockEnv 的 1-2 步轨迹无法验证归纳在真实多步场景的效果~~ | ✅ 已解决：τ-bench 真实任务上获得 5 条 6-7 步成功轨迹，Harness 编译通过 |
| P1 | **无 Baseline 对比框架** | 无法回答"AutoHarness 是否优于 RAG/ReAct" | ❌ 待实现 |
| P1 | **无积累曲线图** | 无法展示核心假设的"交叉曲线" | ❌ 待实现 |
| P1 | **归纳算法简化** | Phase 1/3/4 退化为简单启发式，真实轨迹上可能失效 | ❌ 待实现 |
| P2 | **无多层级知识库** | "公共知识库"愿景无法验证 | ❌ 待实现 |
| ~~P2~~ | ~~无向量数据库~~ | ~~检索质量在多任务类型下不可靠~~ | ⚠️ 已实现 SQLite BLOB + 本地 embedding，缺独立向量 DB |
| P2 | **无特化分裂触发** | 环境变异时无法自动适配 | ❌ 待实现 |
| P2 | **无 NEEDS_REVISION 修复循环** | 归纳失败后不重试 | ❌ 待实现 |

---

## 5. 存储架构设计决策

### 5.1 SQL 为核心标准，移除 JSON 文件依赖

**问题**：当前 env_metadata 表中 `info_json` 存的是整个 JSON blob，无法直接 SQL 查询字段。
JSON 文件存储（trajectories/*.json 等）也无法做结构化查询。

**决策**：以 SQLite 为核心标准存储。具体原则：

| 数据类型 | 存储方式 | 理由 |
|---------|---------|------|
| 高频查询字段（task_type, outcome, status） | 独立 SQL 列 | 支持 WHERE/INDEX |
| 低频嵌套数据（steps, invariants, preconditions） | JSON 字符串列 | 结构可变，不需要字段级查询 |
| 环境元数据（OS, Python, GPU） | 独立 SQL 列 | 可直接 `SELECT python_version FROM env_metadata` |
| 向量数据 | BLOB 列 | float32 打包，高效存储 |
| 变长列表（ollama_models, tau2_domains） | 逗号分隔 TEXT 列 | 简单可查 |

**env_metadata 表重构**（从 JSON blob → 结构化列）：

```sql
CREATE TABLE env_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    -- OS
    os_system TEXT,
    os_release TEXT,
    os_distro TEXT,
    os_machine TEXT,
    -- Python
    python_version TEXT,
    python_executable TEXT,
    in_venv INTEGER,  -- bool
    -- Hardware
    cpu_count INTEGER,
    memory_gb REAL,
    gpu_name TEXT,
    gpu_memory TEXT,
    -- Models
    ollama_models TEXT,        -- comma-separated
    local_model_names TEXT,   -- comma-separated
    -- τ-bench
    tau2_installed INTEGER,   -- bool
    tau2_version TEXT,
    tau2_domains TEXT,        -- comma-separated
    -- Packages (variable dict, keep as JSON)
    packages_json TEXT,
    -- Env vars
    eos_llm_backend TEXT,
    eos_ollama_model TEXT,
    eos_deepinfra_model TEXT
);
```

**JSON 文件存储**：仅作为导入/导出格式，不再是运行时存储路径。`repository.py` 逐步迁移到通过 `Storage` 层操作 SQLite。

### 5.2 Harness 错误检测与修复迭代

**问题**：Harness 部署后出错，如何判断是 Harness 的问题？如何修复？

**错误检测流程**：

```
Harness 执行
    ├─ 无异常 + verify(match) → ✅ 成功
    ├─ 无异常 + verify(mismatch) → ❌ F1: 前置条件缺口
    │   原因：环境状态与 Harness 预期不匹配（如用户已不存在）
    │   处理：标记 NEEDS_REVISION → 重新归纳，更新前置条件
    ├─ 异常（ToolError/KeyError/...） → ❌ F2: 实现错误
    │   原因：Harness 代码 bug（如访问不存在的字段）
    │   处理：标记 NEEDS_REVISION → 用失败信息重新归纳
    ├─ 异常（环境 API 变化） → ❌ F3: 环境漂移
    │   原因：环境接口变更（如工具名重命名）
    │   处理：标记 DEPRECATED → 创建新版本分支
    └─ 任务超出 Harness 覆盖范围 → ❌ F4: 超出范围
        原因：用户请求了 Harness 不支持的操作
        处理：fallback 到 Agent → 记录为新任务类型
```

**修复迭代机制**：

1. **失败计数**：每次 Harness 失败，`failure_counts[failure_type] += 1`
2. **自动降级**：连续失败 N 次（默认 3）→ 状态从 `ACTIVE` → `NEEDS_REVISION`
3. **重新归纳**：用失败轨迹 + 原成功轨迹重新编译新版本
4. **版本 DAG**：新版本的 `parent_id` 指向旧版本，形成版本链
5. **遗弃与经验存储**：如果重新归纳也失败，标记为 `ABANDONED`，存储为负面经验供未来参考

### 5.3 Harness Artifact 存储与版本树

**问题**：Harness 代码目前是 SQLite 中的 plain text。如何维护版本树、支持分支（如 Chrome/Safari 变体）？

**设计方案：SQLite DAG + 文件系统镜像**

```
.experience_os_data/
├── experience_os.db              # SQLite: 元数据 + 关系 + 查询
├── artifacts/                    # 文件系统: Harness 代码 + 测试数据
│   ├── find_user_id_by_email/
│   │   ├── v1/
│   │   │   ├── harness.py        # procedure_code
│   │   │   ├── meta.json          # preconditions, invariants, params
│   │   │   └── replay_tests/      # 回放验证数据
│   │   ├── v2/                    # 迭代版本（parent=v1）
│   │   │   ├── harness.py
│   │   │   └── meta.json
│   │   └── v1.chrome/            # 分支（环境特化）
│   │       ├── harness.py
│   │       └── meta.json
│   └── return_delivered_order/
│       └── v1/
│           ├── harness.py
│           └── meta.json
```

**SQLite 中的 DAG 关系**：

```sql
-- harnesses 表已有 parent_id 字段
-- 额外增加 branch 标记
ALTER TABLE harnesses ADD COLUMN branch TEXT DEFAULT 'main';
-- 查询版本链
SELECT id, version, parent_id, branch, status
FROM harnesses
WHERE task_type = 'find_user_id_by_email'
ORDER BY version;
```

**版本树操作**：

| 操作 | SQL | 文件系统 |
|------|-----|---------|
| 创建新版本 | `INSERT INTO harnesses (..., parent_id=old_id)` | `mkdir artifacts/{type}/v{n}/` |
| 创建分支 | `INSERT INTO harnesses (..., branch='chrome')` | `mkdir artifacts/{type}/v1.chrome/` |
| 查询版本链 | `WITH RECURSIVE ...` | `ls artifacts/{type}/` |
| 导出 artifact | `SELECT * FROM harnesses WHERE id=?` | `zip artifacts/{type}/v{n}/` |
| 合并分支 | 更新 `parent_id` + `status` | 移动文件 |

**为什么不用 sub-git**：
- Harness 是单个 Python 函数，不是完整项目，sub-git 过重
- SQLite DAG 已能管理版本关系
- 文件系统镜像提供 git-friendly 的 diff 体验
- 需要时可以对 `artifacts/` 目录做 git 追踪

**为什么不用纯文件系统**：
- 需要结构化查询（"找所有 ACTIVE 的 Harness"）
- SQLite 的事务性保证一致性
- 向量 BLOB 需要数据库存储
