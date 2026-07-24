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

---

## 6. Baseline 对照实验设计

### 6.1 对照组总览

在 τ-bench（retail/airline）上对比四条路线，**统一 backbone 模型 + 统一 Warm-up 数据**，仅改变"如何利用历史经验"：

| 代号 | 方法 | 经验形态 | 部署时改变什么 | 公平性约束 |
|------|------|---------|---------------|-----------|
| **A. Vanilla LLM** | 纯 LLM via DeepInfra，无工具/无记忆 | 无 | 输入分布 | 任务难度下界 |
| **B. ReAct Agent** | [agent.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/agent.py) 的 ReAct 工具调用，无积累 | 无 | 工具集 | 当前主流范式 |
| **C. SkillOpt** | 微软 [SkillOpt](https://github.com/microsoft/SkillOpt) 优化的 `best_skill.md` 文本技能 | **文本 skill 文档**（300–2000 token） | LLM 输入 prompt | 最强对照（§7 prior work） |
| **D. AutoHarness (ours)** | ExperienceOS 编译的可执行 `procedure_code` | **可执行代码 artifact** | 计算路径（绕过 LLM 推理） | 本文方案 |

**关键区分**：SkillOpt 优化的是 *skill 文本*（改 LLM 输入），AutoHarness 编译的是 *可执行代码*（改计算路径，Harness 执行阶段绕过 LLM）。两者用同一 backbone（DeepInfra/MiniMax-M2.7）+ 同一 Warm-up 池，对比"经验形态"本身的差异——这是论文相对 RAG/SkillOpt 的核心论证点。

### 6.2 各 Baseline 集成路径

#### A. 纯 LLM (DeepInfra)
- 复用 [baseline_eval.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/baseline_eval.py) 的 `run_baseline()`，固定 `max_steps`、禁用工具调用、直接让 LLM 给出最终答案。
- 已有：DeepInfra 后端配置（`EOS_LLM_BACKEND=deepinfra`）。
- 缺：需在 baseline_eval 增加 `--no-tools` 模式与 reward 统计。

#### B. ReAct Agent
- 即 [agent.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/agent.py) 的 `AgentFallback`，ACCUMULATION 模式全量运行。
- 已具备。只需固定 Warm-up/Eval 池，输出 token + 成功率。

#### C. SkillOpt
- 作为 submodule 引入 `microsoft/SkillOpt`（同 harbor/tau2-bench 处理方式）。
- SkillOpt 的 benchmark 是 `skillopt/envs/<name>/` 包（adapter + data loader + scored rollout + YAML）。τ-bench **不在其内置 6 个 benchmark 内**，需写一个 `skillopt/envs/tau2/` adapter：
  - data loader：复用 [tau2_adapter.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/tau2_adapter.py) 的 `infer_task_type` + `split_tasks`
  - scored rollout：复用 tau2 的 `reward` 判定
  - seed skill：空 `best_skill.md`
- 训练用 DeepInfra/MiniMax-M2.7（与 ExperienceOS 同 backbone），Warm-up 池作为训练数据，产出 `best_skill.md` 后在 Eval 池评测。
- 风险：SkillOpt 优化 skill 文本，可能对 τ-bench 的多轮 policy 推理增益有限——这恰好是 AutoHarness "可执行代码" 的对比卖点。

#### D. AutoHarness
- 即本框架 DEPLOYMENT 模式。需先在 Warm-up 池跑 ACCUMULATION 触发归纳，再在 Eval 池用 Harness + Agent fallback。
- 缺口见 §7。

### 6.3 实验协议（防数据泄露）

1. **数据划分**：复用 [tau2_adapter.split_tasks()](file:///home/our0boros/Project/ExecutableExperience/experience_os/tau2_adapter.py)，按任务类型分组，每类取前 K=3 进 Warm-up，剩余进 Eval，**实例不重叠**。
2. **同 Warm-up 数据**：C/SkillOpt 用 Warm-up 轨迹做 skill 训练语料；D/AutoHarness 用其做归纳素材；B/ReAct 不使用历史。
3. **同 Eval 序列**：四组面对完全相同的 Eval 任务顺序。
4. **指标**：Task Success Rate、Avg Tokens/Task、Avg Latency、Harness Hit Rate（D 独有）。
5. **核心图表**：积累曲线图（x=任务序号，y=滚动成功率），展示 AutoHarness 在第 K+1 个任务后的"交叉超越"。

### 6.4 实验设计变体（归纳证据来源）

核心方法学问题：**归纳的素材来自哪里、验证在哪上面做？** 三种变体 + 临场/预积累维度：

| 变体 | 积累池 | 验证池 | 测什么 | CLI |
|------|--------|--------|--------|-----|
| **type_split**（默认） | 同类型前 K 个 | 同类型剩余 | 类型内泛化（未见实例） | `--variant type_split` |
| **replay** | 同类型前 K 个 | **同一批**重跑 | 记忆 vs 泛化（上界） | `--variant replay` |
| **cross_domain** | domain X（如 airline） | domain Y（如 retail） | 跨域迁移 | `--variant cross_domain --cross-domain airline` |

**临场 vs 预积累维度**：
- **临场（online）**：在 eval 流中边跑边积累，harness 在第 K+1 个任务后"上线"——`autoharness` 方法即此。
- **预积累（pre-accumulated）**：warm-up 阶段先单独跑完并归纳，再进 eval 流——`autoharness` 的 warmup/eval 分离即此。

两种维度可正交组合。`type_split + 预积累` 是主实验；`replay` 给上界；`cross_domain` 测迁移鲁棒性。

### 6.5 层级化经验库

[experience_library.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/experience_library.py) 提供**层级化 SQLite 经验库**，三层结构：

| 层 | 表 | 内容 | 生命周期 |
|----|-----|------|---------|
| 底层 | `trajectories` | **完整轨迹**：任务对象、完整对话（LLM 看到的 prompt 和回复）、tool calls/results、reward、tokens | append-only，永不删除 |
| 中层 | `records` | 经验记录：前置条件、参数化步骤、不变量 | 版本化（superseded_by 链） |
| 上层 | `artifacts` | harnesses/skills（可执行代码 / 文本技能） | 版本 DAG（parent_seq + edge_type） |

**多实例**：
- **LTS 库**（`.experience_os_data/lts_library.db`）— 持久，底层 trajs 永不丢失，上层随版本更新优化总结。所有实验都写入此库。
- **实验库**（`.experience_os_data/exp_<id>.db`）— 临时，服务于单次实验，可丢弃。原始数据仍在 LTS。

与归纳方案解耦：无论上层用什么归纳算法（code/text/AST），都从底层 trajs 读取素材，互不影响。`serialize_messages()` 保留每条消息的 role/content/tool_calls/tool_results 全文。

CLI：`experience-os lts` 列出所有实验汇总。`experience-os compare` 每次运行自动写入 LTS + 实验库。

### 6.6 成本收敛接口

回答"经验积累是否让成本收敛"：
- `LTSStore.cost_curve(experiment_id)` 返回 `{x, rolling_sr, cumulative_tokens, rolling_avg_tokens}`。
- `experience-os curve --cost <files>` 绘制双轴图：累计 token（上）+ 滚动平均 token（下）。AutoHarness 在 harness 命中后 rolling_avg 应骤降。
- 当前 `runtime.py:197` 的 `estimated_token_savings` 用硬编码 1000 估算，LTS 用真实 per-task token 替代。

---

## 7. 差距修复路线图

按"重要性 × 难易度"排序，分三阶段推进。重要性 = 对论文核心论证的贡献；难易度 = 工程量与依赖复杂度。

### 阶段一：打通对照实验（高价值 / 中低难度）

| # | 任务 | 重要性 | 难度 | 依赖 | 产出 |
|---|------|-------|------|------|------|
| 1.1 | baseline_eval 增加 `--no-tools` 纯 LLM 模式 + 统一输出 | 高 | 低 | A | Vanilla 下界 |
| 1.2 | 固化 ReAct Baseline 脚本（Warm-up/Eval 分池 + 指标导出） | 高 | 低 | B | ReAct 对照 |
| 1.3 | 引入 SkillOpt submodule + 写 `skillopt/envs/tau2/` adapter | 高 | 中 | C | 最强对照 |
| 1.4 | 实现积累曲线图（x=任务序号，y=滚动成功率/Token） | 高 | 低 | 1.2/1.3 | 论文核心图 |
| 1.5 | 消融开关：`--no-validation` / `--no-versioning` / `MIN_SUPPORT={1,3,5}` | 高 | 低 | D | 消融表 |

### 阶段二：补强归纳算法（高价值 / 中高难度）

| # | 任务 | 重要性 | 难度 | 现状 | 产出 |
|---|------|-------|------|------|------|
| 2.1 | Phase 1 分段结果真正使用（当前 [compiler.py:349](file:///home/our0boros/Project/ExecutableExperience/experience_os/compiler.py#L349) 丢弃返回值） | 高 | 中 | 占位 | 多步轨迹可分段归纳 |
| 2.2 | Phase 3 不变量挖掘替换为 Daikon 风格跨轨迹谓词交集 | 高 | 高 | 仅"首步一致+全成功" | 真正的不变量 |
| 2.3 | Phase 4 参数化改为 LCS + 类型感知（非正则替换引号串） | 高 | 中 | 正则启发式 | 真实多步轨迹可用 |
| 2.4 | 实现 NEEDS_REVISION 修复循环（失败模式分析 → 重合成） | 中 | 中 | DRAFT 后不重试 | 归纳鲁棒性 |
| 2.5 | 充实 StructuredCoT（[agent.py:115](file:///home/our0boros/Project/ExecutableExperience/experience_os/agent.py#L115) 仅填 goal） | 中 | 中 | 字段全空 | 归纳信号质量 |

### 阶段三：架构完善（中低价值 / 中难度）

| # | 任务 | 重要性 | 难度 | 现状 | 产出 |
|---|------|-------|------|------|------|
| 3.1 | Repository 真正接入 Storage（SQLite），当前仅 embedding.py 用，[repository.py](file:///home/our0boros/Project/ExecutableExperience/experience_os/repository.py) 仍纯 JSON | 中 | 中 | 与 §5 描述不符 | 结构化查询 + 向量 BLOB 落地 |
| 3.2 | 失败分类 F1-F4 从关键字匹配升级为结构化判定 | 中 | 低 | [agent.py:194](file:///home/our0boros/Project/ExecutableExperience/experience_os/agent.py#L194) 关键字 | 分类准确率 |
| 3.3 | 特化分裂触发（F1 累积 + `pending_variations` + 环境变异维度检测） | 中 | 高 | 缺失 | 环境漂移自适配 |
| 3.4 | 版本 DAG 补 `specialization` / `composition` 边（当前仅 `patch`） | 低 | 中 | [repository.py:143](file:///home/our0boros/Project/ExecutableExperience/experience_os/repository.py#L143) | 分支/组合版本 |
| 3.5 | 归纳异步化（当前 [runtime.py:213](file:///home/our0boros/Project/ExecutableExperience/experience_os/runtime.py#L213) 同步阻塞） | 低 | 低 | 同步 | 主流程不阻塞 |
| 3.6 | TerminalBench 适配器（harbor 已 submodule，无 adapter） | 低 | 中 | 缺失 | 跨环境验证 |
| 3.7 | 多层级知识库（personal/org/public 优先级覆盖） + artifact 包格式 | 低 | 高 | 缺失 | 公共知识库愿景 |

### 里程碑

- **M1（阶段一完成）**：四组 Baseline 在 τ-bench retail 上跑通，产出积累曲线图 → 可写实验段。
- **M2（阶段二完成）**：归纳算法在真实多步轨迹上稳定触发，replay 验证通过率 ≥ 0.8 → 可写方法段。
- **M3（阶段三按需）**：架构完善 + 跨环境/跨域迁移实验 → 可写分析段与扩展实验。

### 修正说明

本节 §7 已根据实际代码通读修正 [STRUCTURE.md](file:///home/our0boros/Project/ExecutableExperience/docs/STRUCTURE.md) §3–§4 中偏乐观的描述：
- §3.6 标注 SQLite"✅ 主存储"实为 **Storage 类已实现但未接入 Repository**（仅 embedding 缓存用），见 §7-3.1。
- Phase 1 分段在代码中结果被丢弃，§3.3 标注"⚠️ 有但退步"应改为"占位未生效"，见 §7-2.1。
- StructuredCoT 实际仅填 goal 字段，§3.2 标注"⚠️ 有字段但内容空"已确认，见 §7-2.5。
