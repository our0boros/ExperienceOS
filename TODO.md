# ExperienceOS 重构与实验执行计划

> **最后更新**：2026-07-27（flow.md 融合分析 + ExperienceOS.md 文档更新）
>
> 目标：围绕 ExperienceOS 的根本研究目的重构系统，而不是继续在现有双存储、半服务化结构上堆叠补丁。
>
> 根本问题：**Agent 的原始执行轨迹如何经过受控的经验归纳，形成有限、可复用、可验证的 artifact，并在后续任务中主动执行，从而提升成功率或降低推理成本。**
>
> **当前恢复点**：§12 flow.md 融合方案已确定。下一步实施 P0-1（预测契约）。

## 0. 已确认的设计决策

- [x] `Trajectory` 不等于经验；必须经过 `ExperienceStore` 的归纳/压缩/版本化后，才允许进入 artifact induction。
- [x] `TraceStore` 只保存原始事实，append-only；不得承担经验语义，也不得直接作为 artifact 的无限输入集合。
- [x] `ExperienceStore` 是核心工程层，不是可选缓存。它负责从轨迹生成受控的 ExperienceRecord / SubStepPattern，并提供 support、置信度、覆盖范围、去重与淘汰依据。
- [x] `ArtifactStore` 只保存由 ExperienceStore 产出的候选经验编译结果及其验证、版本、失败反馈；不直接消费无界 raw traces。
- [x] 研究验证框架与工程内核分离：验证框架定义问题、数据切分、方法组和指标；工程内核提供 TraceStore / ExperienceStore / ArtifactStore / Services / Retriever / Executor。
- [x] 所有模型能力通过独立服务接口暴露：Chat、Embedding，未来可扩展 Judge/Rerank；本地模型和 API 模型只是 provider，不影响上层调用。
- [x] 经典文档 RAG 不作为强制 P0 baseline。主被动基线采用 `Passive Experience Retrieval`，并评估 KSI-like 结构化知识注入作为外部/复现型 baseline。
- [x] 参数绑定不再是 tau2 adapter 内的字符串补丁；升级为独立的 `ArtifactInputResolver`，采用 schema + deterministic candidates + LLM completion + deterministic validation 的混合流程。
- [x] 验证框架不等于 sandbox replay。Sandbox replay 是 artifact 生命周期中的一个验证器；benchmark evaluation 是系统级验证，两者必须分别记录。

## 1. 目标架构

```text
                         Experiment / Evaluation Layer
  TaskSource → SplitPolicy → MethodRunner → Metrics/Reports
       │             │             │
       └─────────────┴─────────────┴────── controls scenarios
                                      │
                          ExperienceOS Runtime Kernel
  Task → Router → Artifact Executor ───────────────┐
    │          │                                    │
    │          └── miss/failure → Agent Executor ──┤
    │                                               │
    └────────────── Execution Event ────────────────┘
                                      │
                                  TraceStore
                                      │ raw facts only
                                      ▼
                               ExperienceStore
                    dedup / cluster / summarize / score / gate
                                      │ bounded ExperienceRecords
                                      ▼
                                Inductor
                                      │
                                  ArtifactStore
                         draft → verified → active → deprecated

  Model Services (injected into kernel and experiment runners)
  ChatService | EmbeddingService | optional JudgeService / RerankService
```

### 1.1 TraceStore

职责：保存完整、可审计的执行事实。

保存：

- task、domain、task_type、experiment_id、method、phase
- 完整 messages / tool calls / tool results / usage
- structured steps、环境快照、结果、失败类型、路径
- 原始轨迹的版本与来源信息

不负责：

- 语义聚类
- 经验摘要
- artifact 触发
- 直接为 induction 提供“所有历史记录”

### 1.2 ExperienceStore

职责：把 raw traces 转成有限、可解释、可检索、可编译的经验单元。

建议的中间对象：

- `ExperienceRecord`：全任务或阶段级 canonical procedure
- `SubStepPattern`：跨任务复现的子步骤模式
- `ExperienceEvidence`：支持该经验的轨迹集合、成功率、失败率、环境范围
- `ExperienceVersion`：经验归纳算法/模型/输入集合版本

ExperienceStore 的最小处理流程：

1. **Candidate selection**：只选满足质量和成功条件的轨迹/子步骤。
2. **Canonicalization**：统一 tool name、参数表示、动作类型和结果状态。
3. **Deduplication**：按 task type、intent、tool sequence、I/O signature 去重。
4. **Clustering**：规则聚类为默认路径；Embedding/LLM 只用于语义相近但字面不同的合并。
5. **Summarization**：生成 preconditions、input/output schema、canonical steps、invariants、terminal verifier。
6. **Evidence scoring**：记录 support、成功率、跨实例覆盖、环境范围、失败类型。
7. **Utility gate**：只有预期未来收益超过归纳和维护成本的经验才进入 artifact candidate。
8. **Compaction**：相同经验合并 evidence；旧版本 supersede；不允许每条 trace 生成一个 record/artifact。

### 1.3 ArtifactStore

职责：保存编译产物，而不是保存经验素材。

必须记录：

- artifact 类型：harness / skill / verifier / composite
- source experience IDs 和 source evidence snapshot
- input schema / output schema / preconditions / postconditions
- procedure code 或文本
- validation method、validation score、test cases
- active/deprecated/version DAG
- runtime hit、execution success、fallback、failure F1-F4

ArtifactStore 不负责：

- 从 raw traces 自己归纳经验
- 直接覆盖 ExperienceStore 的证据
- 将未验证草稿视为可执行 artifact

## 2. 当前代码映射与重构顺序

### 阶段 A：先建立边界，不改变实验语义

- [x] 将 `experience_library.py` 的现有层级拆成明确的 TraceStore / ExperienceStore / ArtifactStore facade。
- [x] 保留 SQLite 作为实验和长期存储的实现；Repository 作为 domain facade，Storage 作为 SQLite driver，当前实验通过 `ExperienceLibrary`/Stores 访问分层数据。
- [x] `Repository` 明确为 domain facade；Runtime/Inductor 不新增直接访问旧 JSON API。
- [x] 将 `substeps` 聚合逻辑前移到 ExperienceStore：`consolidate_substeps` / `aggregate_substep_patterns` 提供 support、成功率、Bayesian score、distinct trajectory 去重和 `max_candidates` 门控；候选不自动写 artifact。
- [x] 增加 experience evidence 与 compaction 元数据：候选携带 `source`、`evidence.trajectory_ids`、`support_count`、`success_rate`、`bayesian_score`、`score`、`reason`；`compact_records` 只读聚合 evidence，不删除历史。
- [ ] 所有 record/artifact 写入必须带 `source_experience_ids`、`experiment_id`、`induction_policy`、`model`。

### 阶段 B：统一模型服务

- [x] 定义稳定的 `ChatService` 接口：`complete`、`complete_json`、usage/model/provider metadata。
- [x] 定义稳定的 `EmbeddingService` 接口：`embed`、`embed_batch`、`model_name`、`dimension`。
- [x] 将 ChatService、EmbeddingService、services.py 统一为唯一 provider/service 层，移除旧服务别名。
- [x] Retriever、Inductor、ArtifactInputResolver 不得直接调用 provider SDK 或 `llm.embed`。
- [ ] 支持 local sentence-transformers、OpenAI-compatible API、Ollama、DeepInfra provider。
- [x] embedding cache 以 `(provider, model, normalized_text_hash)` 为 key，避免不同模型向量混用。
- [ ] hash pseudo-vector 仅可在明确的测试配置启用；实验结果必须记录是否使用 fallback。

### 阶段 C：重做 ArtifactInputResolver

- [x] Harness schema 明确声明每个输入的名称、类型、来源、是否必需、约束和示例。
- [x] Resolver 输入：task description、task object、environment、artifact schema、可选轨迹证据。
- [x] 第一层：从 task object / benchmark metadata / 对话中生成 deterministic candidates。
- [x] 第二层：schema-guided LLM completion，只补充缺失或存在歧义的字段。
- [x] 第三层：类型、枚举、必填、交叉字段和 precondition 校验；失败时返回结构化原因，不隐式执行。
- [x] 记录 binding method、confidence、missing fields、validation errors。
- [x] tau2 adapter 只负责提供 task/environment 转换；通用参数解析统一由 ArtifactInputResolver 提供。

### 阶段 D：验证框架与工程内核接线

- [ ] 实验层统一使用 `TaskSource`、`SplitPolicy`、`ModeController`、`MethodRunner`、`MetricsRecorder`。
- [ ] 工程层暴露统一 runtime API：`observe`、`consolidate`、`induce`、`retrieve`、`execute`。
- [ ] `ACCUMULATION`、`ONLINE_ACCUMULATION`、`DEPLOYMENT` 作为实验模式，而不是散落的 if/patch。
- [ ] 每次实验生成独立 experiment DB / manifest；LTS raw trace 可选复制但不影响实验隔离。
- [ ] 每条执行结果同时记录 system path：agent / passive_retrieval / harness / harness+agent，以及 artifact ID 和 experience ID。

## 3. ExperienceStore 是否需要单独实验

结论：**需要，但不是把 ExperienceStore 当成与 ReAct、SkillOpt 同级的 baseline。**

它是 ExperienceOS 的内部机制，应通过以下实验验证其必要性：

### 3.1 必做内部机制实验

| 实验 | 对照 | 目的 | 关键指标 |
|---|---|---|---|
| Trace→Artifact 直接路径 | raw trace 直接 induction vs Trace→Experience→Artifact | 证明中间经验层能防止 artifact 爆炸并提升质量 | artifact 数量、重复率、平均 support、验证通过率 |
| Compaction threshold | support=1/2/3/5 或 utility gate | 找到体量与覆盖率的交叉点 | coverage、SR、artifact 数、induction cost |
| 归纳机制 | rule-only vs rule+embedding vs rule+LLM | 判断 LLM 归纳是否真的必要 | pattern purity、跨实例召回、成本 |
| 全任务/子步骤 | full-task only vs substep only vs hybrid | 验证经验粒度 | hit rate、compose rate、SR、token |
| 质量门控 | 无 gate vs success/support/evidence gate | 防止失败轨迹污染 | 错误 artifact 率、回退率 |

### 3.2 不应做的事情

- 不把每种 compaction 策略都包装成外部 baseline。
- 不用最终 benchmark SR 单独证明 ExperienceStore 好；必须同时报告经验压缩质量和 artifact 供给质量。
- 不让验证框架直接读取全部 raw traces 来“帮助” ExperienceOS；否则会造成数据泄漏和层次混淆。

## 4. 主实验矩阵

### 4.1 主问题：主动执行是否优于被动告知

固定：同一 backbone、同一任务、同一 warmup、同一 budget、同一随机种子策略。

方法：

1. ReAct：无经验。
2. SkillOpt：文本 skill 注入。
3. Passive Experience Retrieval：检索历史经验文本/步骤，注入 prompt，由 agent 自己决定是否使用。
4. KSI-like：结构化知识/经验注入，复现其“共享知识→agent 自主使用”路径；与 KSI 的实现差异和数据构建成本单独记录。
5. ExperienceOS：ExperienceStore → ArtifactStore → 主动检索/执行，失败 fallback agent。

主要指标：

- task success rate（主指标）
- average / p50 / p95 tokens
- cost、latency
- passive retrieval hit / injected token volume
- artifact retrieval hit rate
- artifact execution success rate
- fallback rate
- net utility：成功率收益、token 节省、构建成本的联合指标

### 4.2 三种运行模式

1. **Accumulation**：只由 agent 执行并写 TraceStore；不使用新 artifact。
2. **Online Accumulation**：每完成一批任务，TraceStore→ExperienceStore→ArtifactStore；后续任务可使用新 artifact。
3. **Deployment**：使用独立 accumulation split 产出的经验/artifact；评测 split 禁止反向写入可用 artifact，允许记录失败反馈。

### 4.3 数据切分

按 tau2 的三层泛化关系分别报告：

- **Cross-domain**：零售→航空等不同领域。
- **In-domain / cross-scope**：同域不同任务范围。
- **In-scope / cross-instance**：同范围不同细节实例。

必须避免：

- eval 任务的完整消息进入 induction 输入。
- eval 任务通过 runtime 失败后立即污染同一轮可用 artifact，除非实验明确标为 online。
- 不同 baseline 使用不同 warmup 数据量。

### 4.4 累积曲线

横轴：已处理的 accumulation tasks / successful evidence count。

纵轴至少画四条：

- rolling success rate
- artifact coverage / hit rate
- average tokens per task
- cumulative induction cost 或 amortized cost

不要预先假设一定是 S 曲线；把 S 曲线作为待验证假设。应报告 crossover point（ExperienceOS 超过各 baseline 的最早区间），若不存在也必须保留负结果。

### 4.5 跨 Agent 迁移

在主方法成立后执行：

- 强 Agent accumulation → 弱 Agent deployment：SR、token、artifact reuse。
- 弱 Agent accumulation → 强 Agent deployment：是否仍有节省，是否增加错误路由。
- 不同 Chat provider / model 的 artifact 迁移：区分 artifact execution 与 LLM glue 依赖。
- ExperienceStore 网络效应：共享库与独立库的边际收益、污染率和冲突率。

## 5. 最小消融集合

消融不是越多越好，但以下三组必须保留：

1. **No executable artifact**：只做被动经验注入，证明收益来自可执行性而不只是更多上下文。
2. **No semantic consolidation**：按字符串/工具名直接聚合，证明 ExperienceStore 的归纳质量贡献。
3. **No active retrieval / rule-only retrieval**：证明主动路由和语义匹配的贡献。

可选：

- no validation gate
- no substep induction
- no input schema resolver / raw regex resolver
- no cross-task evidence sharing

## 6. 结果与实验复现要求

每次 DeepInfra 实验必须保存：

- experiment manifest：model、embedding model、provider、temperature、max steps、seed、split、warmup、eval、mode、commit/version。
- 每条任务的完整 messages、tool calls、usage、reward、path、artifact/experience IDs。
- TraceStore、ExperienceStore、ArtifactStore 的快照或数据库路径。
- induction log：输入 evidence IDs、输出经验、压缩/去重原因、模型调用成本。
- validation log：验证环境、测试轨迹、read/write/mixed 类型、失败分类。
- summary JSON 和可绘图 CSV。

推荐最终主结果最小集合：

1. Retail train/test：ReAct、SkillOpt、Passive/KSI-like、ExperienceOS。
2. Retail online accumulation curve：至少多个 accumulation checkpoints。
3. 一个 cross-domain transfer：例如 airline→retail 或 retail→airline。
4. 一个 ExperienceStore compaction 消融。
5. 强/弱 Agent 的 artifact transfer（若前四项结果稳定）。

## 7. 开发任务清单

### P0：边界与数据流

- [x] 新建 Store 接口/协议和统一 execution event schema：`TraceStore` / `ExperienceStore` / `ArtifactStore` 已在 `stores.py` 落地。
- [x] 将现有 `ExperienceLibrary` 重新拆分为三个 facade；保留同一 SQLite 实现，避免立即重复迁移数据。
- [x] Runtime/Inductor/compare.py 已最小接线：共享同一组 Store facade 和 Services，不再各自创建数据库；compare.py 轨迹与子步骤写入通过 TraceStore。
- [ ] `compare.py` 重构为统一 runner 协议（`TaskSource` / `SplitPolicy` / `ModeController` / `MethodRunner` / `MetricsRecorder`），不再在 adapter 中写路径补丁。
- [x] `Repository` 已标记为 domain facade，禁止继续扩展旧 API；`storage.py` 仅作 SQLite driver。

### P0：模型服务

- [x] 统一 ChatService / EmbeddingService；`Services` 只暴露 `chat` 和 `embedding`，移除 `llm` / `embed` 兼容别名。
- [ ] 修正不同 embedding provider/model 的配置、缓存和维度记录。
- [x] Retriever / Inductor / ArtifactInputResolver 已全部改为依赖注入服务；retriever 未注入服务时明确报错，不再隐式调用 `LLMClient.embed`。
- [ ] 支持 local sentence-transformers、OpenAI-compatible API、Ollama、DeepInfra provider（接口已就绪，provider 注册待补全）。

### P0：经验归纳门控

- [x] ExperienceStore 实现候选筛选、canonicalization、去重、support/evidence aggregation.
- [ ] 只有 ExperienceStore 输出才能触发 Inductor（当前 Inductor 已优先调用 facade，但仍保留 legacy fallback）。
- [ ] 增加 artifact budget、min support、utility gate 配置。

### P1：参数绑定与验证

- [x] 实现 ArtifactInputResolver 和 schema validation。
- [x] tau2 adapter 的 `extract_task_params` 已删除；通用参数解析统一由 ArtifactInputResolver 提供。
- [ ] 分离 harness replay validation、input binding validation、benchmark evaluation。
- [ ] read-only artifact 使用 capability/contract validation，不再简单跳过验证。

### P1：实验方法

- [ ] 将 `compare.py` 重构为统一 runner 协议。
- [ ] 添加 Passive Experience Retrieval baseline（当前阶段先忽略，见 §0 决策）。
- [ ] 实现 KSI-like baseline adapter：KSI 已 clone 到 `KSI/`，接口与 TaskSpec 已调研，`ksi_adapter.py` 与测试尚未落地（见 §11）。
- [ ] 添加 online accumulation 和 checkpoint curve。
- [ ] 添加 cross-domain / cross-scope / cross-instance split。

### P2：迁移与网络效应

- [ ] 强/弱 agent transfer。
- [ ] 共享 ExperienceStore 与独立库对比。
- [ ] artifact composition 与更复杂的多步 harness。

## 8. 当前实现中需要特别处理的风险

- `repository.py`、`storage.py`、`experience_library.py` 已标记为 domain facade / SQLite driver / 兼容层；不允许再扩展旧 API，但保留以支撑历史数据和未迁移的实验入口。
- `compare.py` 仍直接创建 `ExperienceLibrary` 用于离线 LTS 复用，需在统一 runner 协议落地后收敛。
- `inductor.py` 已优先调用 ExperienceStore facade，但 legacy fallback 仍存在；需在“只有 ExperienceStore 输出才能触发 Inductor”完成后删除。
- `tau2_adapter.py` 的 `extract_task_params` 已删除；所有参数解析走 `ArtifactInputResolver`。
- `retriever.py` 已移除隐式 `llm.embed` 路径，未注入服务时明确报错。
- DeepInfra API 运行要保持串行、记录 provider/model/usage，并设置可配置延迟；实验不能因为重构丢失历史细节。
- 实验代码不得把 eval 结果写入同一轮部署可见的 ExperienceStore，除非模式明确是 Online Accumulation。

## 9. 外部参考

- KSI：<https://github.com/recursive-knowledge/KSI>
  - 用作“结构化共享知识被动注入、由 agent 自主使用”的对照参考。
  - 不默认假设其与 ExperienceOS 的任务格式、执行器和数据预算完全等价。

## 10. 完成标准

- [ ] 任意新实验只需要选择 `mode`、`method`、`split_policy` 和 `model services`，不再在 adapter 中写路径补丁。
- [ ] 一条 raw trace 不会自动产生一条 experience/artifact；可解释地合并到已有经验或被 gate 拒绝。
- [ ] 同一 experiment 可以完整回答：收集了哪些轨迹、归纳了哪些经验、生成了哪些 artifact、哪些 artifact 被使用、最终是否有效。
- [ ] 运行 DeepInfra 时可以复现至少一组主对比和一条 accumulation curve。
- [ ] 实验结果能区分：模型能力、经验归纳质量、artifact 执行质量、路由质量和参数绑定质量。

## 11. 当前进度与后续任务（恢复点）

### 11.1 已完成并验证

- **Store 三层 facade**：`experience_os/stores.py` 落地 `TraceStore` / `ExperienceStore` / `ArtifactStore`，共享同一 SQLite。
- **ExperienceStore 受控归纳门控**：`consolidate_substeps` / `aggregate_substep_patterns` / `compact_records` / `candidate_stats`；按 intent/tool/effect 聚合，distinct trajectory 去重，support + 成功率 + Bayesian score 门控，候选携带 evidence/source/score/reason，不自动写 artifact。
- **Services 统一**：`Services` 只暴露 `chat` 和 `embedding`；Runtime/Inductor/Retriever/Resolver/CLI/demo 全部依赖注入；`services.llm` / `services.embed` 别名已移除。
- **ArtifactInputResolver**：schema + deterministic candidates + 可选 LLM completion + 校验；`tau2_adapter.extract_task_params` 已删除，旧调用改为委托 resolver。
- **Runtime/Inductor 最小接线**：共享 Store 和 Services，避免各组件各自打开数据库；子步骤发现优先走 ExperienceStore。
- **兼容清理**：`llm.py` / `embedding.py` 已删除，零残留引用；`repository.py` / `storage.py` / `experience_library.py` 标记为 deprecated，添加 docstring 警告。
- **inductor legacy fallback**：`check_triggers` 和 `_discover_substep_patterns_from_store` 中添加 log 警告；构造函数无 ExperienceStore 时记录 info 日志。
- **离线测试**：50 个用例全部通过（`test_stores.py` 4、`test_services.py` 14、`test_input_resolver.py` 3、`test_ksi_adapter.py` 15、`test_runner.py` 14）。
- **未创建 git tag**：等待最小 DeepInfra 实验验证重构后 CoE 不劣于旧版本后再标记。

#### 2026-07-27 新增交付（步骤 1-4）

- ✅ **步骤 1**：`experience_os/ksi_adapter.py` — `build_ksi_task_spec` / `build_ksi_run_spec` / `export_ksi_tasks` / `export_ksi_run_manifest`，15 个测试用例。
- ✅ **步骤 2**：`experience_os/experiments/runner.py` — `TaskSource` / `SplitPolicy`（4 种）/ `MethodRunner`（4 种）/ `MetricsRecorder` / `ExperimentRunner` / `ExperimentConfig` / `ExperimentMode`，14 个测试用例。`compare.py` 保持向后兼容。
- ✅ **步骤 3**：`ProviderRegistry` + `ProviderInfo` — 6 个内置 provider（deepinfra / ollama / openai / anthropic / local / litellm），`Services.from_provider()`、`Services.list_providers()`，9 个测试用例。
- ✅ **步骤 4**：弃用清理 — `__init__.py` 更新、3 个文件添加 deprecation docstring、inductor legacy fallback 添加 log 警告。

### 11.2 中断点：尚未完成

- **最小 DeepInfra 实验未执行**：目标是对齐 `docs/exp/0003-coe-full-retail-comparison.md` 的 retail 全量对比，验证重构后 CoE 不劣于旧版本（SR 62.5%、57.9K tokens/task、27.8% harness 使用率）。详见 [docs/exp/0006-refactor-experiment-matrix.md](docs/exp/0006-refactor-experiment-matrix.md)。
- **git tag `refactor-v1` 未创建**：等待实验验证通过后标记。
- **实验矩阵未运行**：cross-domain / cross-model / 消融实验待补齐。

### 11.3 下一步执行顺序

1. ~~实现 `experience_os/ksi_adapter.py` + 测试~~ ✅
2. ~~重构 `compare.py` 为统一 runner 协议~~ ✅
3. ~~补全 DeepInfra provider 注册~~ ✅
4. 在 retail 全量 split 上运行最小对比：ReAct、SkillOpt、CoE（重构后）；目标是不劣于旧 CoE。
5. 若结果稳定，创建 git tag `refactor-v1`，并扩展 KSI baseline 与 accumulation curve。
6. 按实验矩阵补齐 cross-domain / cross-scope / cross-instance 与跨 agent 迁移。

---

## 12. flow.md × ExperienceOS.md 融合方案（2026-07-27）

### 12.1 融合决策

对比 `docs/flow.md` 与 `docs/ExperienceOS.md` 后确定：**ExperienceOS.md 是架构骨架，flow.md 的四项创新注入到对应阶段，增强但不替换。**

| flow.md 创新 | 注入位置 | 融合效果 | 优先级 |
|-------------|---------|---------|--------|
| **预测-验证契约** | §5.1 路径收集 + §5.2 贝叶斯门控 | 区分有效经验与侥幸成功；PredictionContract → P/Q 映射 | **P0-1** |
| **空间聚类** | §5.2.0 归纳前分组 | 替代 `(intent, tool)` 硬分组；确保 LCS 输入语义相似 | **P0-2** |
| **依赖发现 → Composite** | §5.3.6 编译引擎 | 从原子经验构造工具链；Level 2 Composite 实施路径 | **P1-7** |
| **贝叶斯形式化** | §5.2.3–§5.2.5 | 已高度一致；flow.md 的三分布模型补充但不需要独立实现 | — |

### 12.2 实施路线

```
阶段 A [P0-1] 预测契约（依赖：无）
  ├── StructuredCoT 新增 PredictionContract + PredictionVerification
  ├── inductor 读取预测质量标签，调整贝叶斯权重
  └── 改动范围：models.py + agent.py + inductor.py

阶段 B [P0-2] 空间聚类（依赖：无，可与 A 并行）
  ├── _discover_substep_patterns 重写：特征提取 → 密度聚类
  ├── 引入 sklearn.cluster.DBSCAN 或轻量实现
  └── 改动范围：compiler/inductor.py + compiler/algorithms.py

阶段 C [P0-3] 真正多步 LCS 参数化（依赖：B）
  ├── 替换单步 call_tool() wrapper 为多步参数化序列
  ├── 同簇内轨迹做 LCS 对齐 → 公共骨架 + 变化参数
  └── 改动范围：compiler/algorithms.py Phase 4 + inductor.py _synthesize

阶段 D [P1-7] 依赖发现 → Composite（依赖：C）
  ├── 数据流分析：检测子 Harness 间的参数传递
  ├── 转移概率：P(H_j | H_i.success)
  └── 改动范围：新增 compiler/dependency.py + models.py CompositeHarness
```

### 12.3 与之前计划的变更

- ❌ **废止**：第 11.3 节"先跑实验再修实现"的顺序。应先实施 P0-1 → P0-2 → P0-3（核心论文贡献），再跑实验验证。
- ❌ **废止**：当前单步 `call_tool()` harness 作为最终形态。确认为临时简化，P0-3 替换为真正的多步参数化序列。
- ✅ **保留**：11.1 已完成的重构项全部保留（Store facade、Services 统一、runner 协议、KSI adapter、ProviderRegistry）。
- ✅ **保留**：实验结果（exp-0001/0003/0005）作为 baseline 参考，融合实施后重新跑对比。
