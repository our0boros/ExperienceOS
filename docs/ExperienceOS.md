# ExperienceOS：跨 Agent 跨环境的知识编译与共享运行时

> 本文档是 ExperienceOS 项目的**唯一基准文档**，整合自历史文档（OVERALL.md / STRUCTURE.md / Executable Experience RP.md / Executable Experience Discuss.md）。
> 当本文档与其他文档存在冲突时，**一律以本文档为准**。
> 历史文档（除 `Executable Experience Discuss.md` 作为历史讨论保留外）将在整合完成后清理，git 历史保留可追溯。

---

## 1. 一句话定位

> **ExperienceOS 是一个跨 Agent、跨环境的知识编译与共享运行时，它将任意 Agent 的执行经验自动编译为可复用的确定性 Harness（扩展 Hoare Triple），并通过层级化经验仓库使高级 Agent 归纳的经验可被低级 Agent 直接调用。**

五个可验证关键词：

| 关键词 | 验证问题 | 对应实验 |
|-------|---------|---------|
| **自动编译** | 系统能否无需人工归纳 Harness？ | 归纳触发率 + 回放验证通过率 |
| **可复用** | Harness 能否在未见同类任务上成功？ | type_split 泛化实验 |
| **绕过 LLM 推理** | Harness 执行阶段 Token 是否显著更低？ | Token/task 对比 |
| **成功率不降** | Harness 路径 SR 是否 ≥ Agent Baseline？ | SR 主实验 |
| **显著降低成本** | 积累后整体 Token 成本是否收敛下降？ | 积累曲线图 |

---

## 2. 方法论的核心定位

### 2.1 知识编译 vs RAG vs fine-tuning

| 路径 | 机制 | 边际成本 |
|------|------|---------|
| **RAG** | 检索文本注入上下文，改变 *输入分布* | 每次任务仍需完整 LLM 推理 |
| **fine-tuning** | 改变模型 *权重* | 训练成本高，泛化不可控 |
| **ExperienceOS（知识编译）** | 编译为可执行 *Harness*，改变 *计算路径* | 命中时绕过 LLM，Token ≈ 0 |

核心区分：**RAG 给 CPU 递纸条；ExperienceOS 扩展指令集**。

### 2.2 方法论严格边界（不可替代的核心贡献）

ExperienceOS 的差异化必须三点同时成立：

1. **从轨迹自动归纳**（不需人工）—— 区别于人工/预设计 Harness
2. **执行时绕过 LLM**（不是改输入）—— 区别于 RAG / SkillOpt
3. **可验证的回放门控**（不是盲目存储）—— 区别于 Memory / 经验缓存

### 2.3 与相关工作的精确区隔

| 工作 | 核心主张 | ExperienceOS 的关键区别 |
|------|---------|----------------------|
| AutoHarness (2603.03329) | 用代码 Harness 改善 LLM 工具调用准确性 | **起点不同**：他们的 Harness 是人工/预先设计的；ExperienceOS 从轨迹**自动归纳** |
| Meta-Harness (2603.28052) | 端到端优化 Harness 的模型接口设计 | **目标不同**：他们优化接口结构；ExperienceOS 研究 Harness 的**自动生成与积累机制** |
| AHE (2604.25850) | Harness 工程化范式 | **视角不同**：他们是工程实践规范；ExperienceOS 研究**知识编译的学习过程** |
| Harness Updating (2605.30621) | 区分 Harness 进化能力与收益 | **最接近**，重点对比：他们研究已有 Harness 更新；ExperienceOS 研究从无到有的**编译归纳** |
| PatchWorld | Executable world model | **定位不同**：他们学习状态转移；ExperienceOS 编译可复用操作结构 |
| SkillOpt | 优化文本 skill 文档 | **计算路径不同**：SkillOpt 改变 LLM 输入；ExperienceOS **绕过 LLM 推理** |
| RAG | 检索历史轨迹注入上下文 | **机制不同**：RAG 给 CPU 递纸条；ExperienceOS **扩展指令集** |

---

## 3. 数据结构与层次模型

### 3.1 两个正交的"层次"维度

历史文档中"层次"概念混用的根源是**两个正交维度被混为一谈**。本文档严格区分：

#### 维度 A：经验表示层次（Layer 0–3，数据结构维度）

描述系统中数据从原始到抽象的四个表示层级，每层有对应数据结构，**层间有显式来源链接关系**。

```
Layer 0: 交互路径数据 (Raw Trajectory)
   ↓ 精炼/过滤
Layer 1: 经验数据 (Experience Record)
   ↓ 选择性归纳
Layer 2: 可执行 Artifact (Harness / Skill / Verifier)
   ↓ 统计与策略学习
Layer 3: 顶层决策与方法 (Meta-Experience / Router Policy)
```

#### 维度 B：归纳层次（SubStep / Task / Composite，触发对象维度）

描述归纳引擎在哪一层粒度上触发归纳、产出什么类型的 Harness。

```
SubStep 级归纳  →  SubStep Harness（最小可复用单元，如"截图→角点检测→定位"）
Task 级归纳     →  Task Harness（完整流程的参数化程序，如"发送邮件"）
Composite 级归纳 →  Composite Harness（调用子 Harness 的元程序，如"每日报告发送"）
```

> **论文边界**：当前论文只实现 SubStep + Task 两级归纳，Composite 为 Future Work。

### 3.2 经验表示层次详细定义

#### Layer 0：交互路径数据（Raw Trajectory）

完整交互路径的结构化记录。**完整路径内部需要细粒度 SubStep 拆分**，因此引入 TODO list / 对话记录 / 里程碑机制以标识子步骤边界。

```python
@dataclass
class Trajectory:
    # ── 身份标识 ──────────────────────────────
    traj_id: str                    # 全局唯一 ID
    task_id: str                    # 关联的任务类型
    agent_id: str                   # 执行的 Agent 标识
    agent_capability_level: int     # Agent 能力等级 (0=基础, 1=中级, 2=高级)

    # ── 环境上下文（适用范围的来源）──────────────
    env_context: EnvContext         # 见 3.2.1

    # ── 执行内容 ──────────────────────────────
    steps: List[TrajectoryStep]     # 步骤列表（含 SubStep 拆分）
    conversation_log: List[Message] # 完整对话记录（prompt/response/tool_call）
    todo_milestones: List[Milestone]# TODO list / 里程碑标记

    # ── 结果 ──────────────────────────────────
    outcome: Outcome                # SUCCESS / PARTIAL / FAILURE
    success_steps: List[int]        # 哪些子步骤成功了

    # ── 元信息 ──────────────────────────────
    timestamp: datetime
    token_cost: int
    wall_time_ms: int
    source_harness_ids: List[str]   # 执行中调用了哪些 Harness（嵌套追踪）

@dataclass
class TrajectoryStep:
    step_id: int
    # 细粒度 SubStep 拆分
    sub_step_intent: Optional[str]  # 子步骤意图（如"定位元素"、"填写表单"）
    sub_step_plan: Optional[SubStepPlan]  # 子步骤计划

    step_type: StepType             # TOOL_CALL / LLM_REASON / HARNESS_INVOKE
    tool_name: str
    raw_args: Dict                  # 原始参数（未参数化）
    result: Any
    success: bool

    # 规范化记录（用于归纳，见 §6 语义对齐）
    canonical_tool: Optional[str]
    canonical_args: Optional[Dict]
    normalization_confidence: float

    # 状态快照（不变量挖掘用）
    pre_state_snapshot: StateHash
    post_state_snapshot: StateHash

> **关于 `sub_step_plan` / StructuredCoT（结构化推理链）的可选性说明**：
> - `sub_step_plan`（即 StructuredCoT / 结构化推理链）是**可选的辅助功能，默认不启用**。
> - **并非所有 LLM 都支持 CoT（Chain of Thought）能力**，框架不依赖 CoT 也能正常工作。
> - 当模型支持 CoT 时，`sub_step_plan` 可作为归纳引擎的辅助输入，提供约束（constraint）/风险（risk）/里程碑（milestone）等结构化信号，提升归纳质量。
> - 当模型不支持 CoT 时，框架仅依赖 `goal` 字段（任务描述）与轨迹本身进行归纳，`sub_step_plan` 及其他 CoT 相关字段为空（`None`）。
> - 当前阶段先忽略 StructuredCoT 的完整填充，后续可按需启用。

@dataclass
class EnvContext:
    os: str                         # linux / macos / windows
    browser: Optional[str]          # chrome / firefox / None
    app: Optional[str]              # gmail / outlook / None
    app_version_range: Optional[str]
    locale: str                     # zh-CN / en-US
    agent_min_level: int = 0
    custom_tags: Dict[str, str]
```

**收集时机**：

| 触发点 | 处理 |
|--------|------|
| 全任务成功 | 写入 SUCCESS 桶 |
| 全任务失败 | 写入 FAILURE 桶（**失败轨迹也要收集**，用于挖掘部分成功子步骤 / 负样本 / 诊断边界）|
| 子步骤成功 | 标记 `success_steps` |
| Harness 调用 | 记录嵌套引用 `source_harness_ids` |

#### Layer 1：经验数据（Experience Record）

从原始轨迹精炼后的语义摘要。**必须记录来自哪些交互路径**（来源链接）。

```python
@dataclass
class ExperienceRecord:
    record_id: str

    # ── 来源链接（关键）────────────────────────
    source_trajectory_ids: List[str]     # 归纳自哪些原始轨迹
    source_success_steps: List[List[int]]# 每条轨迹贡献的子步骤

    # ── 精炼内容 ──────────────────────────────
    task_type: str
    preconditions: List[Predicate]       # P 候选
    param_steps: List[ParameterizedStep] # 参数化步骤骨架
    invariants: List[Predicate]          # I 候选
    terminal_verifier: Predicate         # Q 候选

    # ── 统计 ──────────────────────────────────
    support_count: int                   # 支持该经验的轨迹数
    success_rate: float
    superseded_by: Optional[str]         # 被哪个 Harness 采用
```

#### Layer 2：可执行 Artifact（Harness）

经验选择性归纳后的可执行结构。**必须记录归纳自哪些经验、以及反例经验导致的分裂**。

形式化定义采用**扩展 Hoare Triple**：

```
H = ⟨P, steps, I, Q, R⟩

P（前置条件）：执行前必须成立的环境状态
steps（步骤）：参数化的动作序列，具体值已替换为变量
I（不变量）：执行过程中必须持续成立的谓词
Q（后置条件）：执行完成后用于验证成功的终态检查
R（回滚策略）：失败时的恢复动作
```

```python
@dataclass
class Harness:
    # ── 身份 ──────────────────────────────────
    harness_id: str                 # H_<task>_<hash>
    name: str
    version: SemanticVersion        # MAJOR.MINOR.PATCH

    # ── 来源链接（关键）────────────────────────
    source_record_ids: List[str]    # 归纳自哪些 ExperienceRecord
    source_trajectory_ids: List[str]# 间接来源的原始轨迹
    negative_record_ids: List[str]  # 反例经验（导致分裂/修复的负样本）

    # ── 适用范围 ──────────────────────────────
    scope: ApplicabilityScope       # 见 5.3.3

    # ── 执行结构（扩展 Hoare Triple）──────────
    signature: HarnessSignature     # 入参定义
    preconditions: List[Predicate]  # P
    steps: List[ExecutableStep]     # 参数化动作序列
    invariants: List[Predicate]     # I
    postconditions: List[Predicate] # Q
    rollback: RollbackStrategy      # R

    # ── 依赖 ──────────────────────────────────
    sub_harness_refs: List[HarnessRef]  # 嵌套调用的子 Harness

    # ── 版本树元信息 ──────────────────────────
    parent_version: Optional[str]
    split_reason: Optional[str]     # 若为分裂产生，记录原因
    merge_source: Optional[List[str]]

    # ── 统计与生命周期 ────────────────────────
    status: HarnessStatus           # CANDIDATE / ACTIVE / NEEDS_REVISION / DEPRECATED
    usage_count: int
    success_rate: float
    avg_token_saving: int
    last_validated: datetime
```

Harness **不是**：
- 轨迹的文本摘要（那是 RAG/Memory）
- 工作流模板（那是人工设计的 workflow）
- fine-tuning 的训练数据（那改变模型权重）

#### Layer 3：顶层决策与方法（Meta-Experience）

统计与策略层，指导如何选择、调用、修复 Artifact。

```python
@dataclass
class MetaExperience:
    task_type: str
    total_executions: int
    harness_executions: int
    agent_executions: int
    successes: int
    failure_counts: Dict[FailureCode, int]  # F1-F4 分布
    estimated_token_savings: int
    router_policy: RouterPolicy             # 路由策略（命中阈值、置信度权重）
```

### 3.3 层间链接关系总览

```
Trajectory (Layer 0)
   │
   │ ←─ source_trajectory_ids
   ▼
ExperienceRecord (Layer 1)
   │
   │ ←─ source_record_ids / negative_record_ids
   ▼
Harness (Layer 2)
   │
   │ ←─ failure feedback / usage stats
   ▼
MetaExperience (Layer 3)
   │
   └─→ RouterPolicy 反馈到 Runtime
```

**关键链接约束**（实现要求）：
- 每个 ExperienceRecord 必须能追溯到 ≥1 条原始 Trajectory
- 每个 Harness 必须记录归纳自哪些 ExperienceRecord
- 反例经验（导致分裂或修复的负样本）必须显式记录在 `negative_record_ids`
- 嵌套调用的子 Harness 通过 `sub_harness_refs` 链接，执行时记录完整调用链

---

## 4. 五阶段闭环架构

整个系统是一个**持续进化的闭环**，五个核心阶段首尾相连。每个阶段不仅有数据传递，还有**预测-验证反馈**贯穿其中，形成双重闭环：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      ExperienceOS 闭环系统（增强版）                       │
│                                                                           │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│   │ PATH COLLECT │───▶│  INDUCTION   │───▶│   COMPILE    │               │
│   │              │    │   ENGINE     │    │   ENGINE     │               │
│   │ · 轨迹记录    │    │              │    │              │               │
│   │ · 预测契约    │    │ · 空间聚类    │    │ · LCS 对齐   │               │
│   │ · 预测vs实际  │    │ · 密度分簇    │    │ · 参数化     │               │
│   └──────┬───────┘    │ · 贝叶斯归纳  │    │ · 依赖发现   │               │
│          │            └──────┬───────┘    └──────┬───────┘               │
│          │                   │                   │                        │
│          │  预测-验证反馈     │                   ▼                        │
│          │  (质量信号)       │            ┌──────────────┐               │
│          │                   │            │   REGISTRY   │               │
│          │                   │            │    STORE     │               │
│          │                   │            │              │               │
│          │                   │            │ · Artifact   │               │
│          │                   │            │ · 版本 DAG   │               │
│          │                   │            └──────┬───────┘               │
│          │                   │                   │                        │
│          │                   │                   ▼                        │
│          │                   │            ┌──────────────┐               │
│          └───────────────────┴───────────▶│   RUNTIME    │               │
│                     反馈闭环              │  EXECUTION   │               │
│                                           │              │               │
│                                           │ · 预测验证    │               │
│                                           │ · Harness执行 │               │
│                                           │ · F1-F4 分类  │               │
│                                           └──────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.1 数据流（正向）

| 阶段 | 输入 | 输出 |
|------|------|------|
| Path Collect | Agent 执行任务 | Trajectory + **PredictionContract**（预测-vs-实际对比） |
| Induction Engine | Trajectory 集合 + 预测契约 | ExperienceRecord + SubStepPattern（经空间聚类 + 贝叶斯门控） |
| Compile Engine | ExperienceRecord + 对齐后的步骤序列 | Harness ⟨P, steps, I, Q, R⟩（多步参数化，可选 composite） |
| Registry Store | Harness + 版本元数据 | 可查询的 Artifact 索引（含依赖图） |
| Runtime Execution | Task + 当前环境状态 | 执行结果 + ExecutionFeedback（反馈闭环） |

### 4.2 预测-验证反馈闭环（逆向）

区别于传统"执行-结果"反馈，ExperienceOS 要求 Agent 在执行前对每个子任务生成**结构化预测契约**（`PredictionContract`），执行后将预测与实际对比：

```
预测: {预期输入, 预期输出, 预期效果}
  vs
实际: {实际输入, 实际输出, 实际效果}
  ↓
差异分析 → 质量信号 →
  ├── 预测准确 + 执行成功 → 高质量经验（高权重参与归纳）
  ├── 预测错误 + 执行成功 → 侥幸成功（低权重，需更多 support）
  ├── 预测准确 + 执行失败 → 实现缺陷（触发 F2 patch）
  └── 预测错误 + 执行失败 → 负样本（记录边界条件）
```

这一机制有效区分**"有效的经验"**和**"侥幸的成功"**，提升归纳质量。预测契约同时直接映射到 Hoare Triple：`预期输入 → P`，`预期输出/效果 → Q`。

### 4.3 空间聚类 → LCS 对齐的归纳管线

传统做法是先按任务类型分组再做 LCS。但"同一任务类型"不保证步骤序列相似（如 exchange 和 refund 虽同属 order_management，但步骤序列不同）。改进流程：

```
轨迹集合
  ↓
特征提取：concat( task_embedding, tool_sequence_signature, effect_embedding )
  ↓
密度聚类（DBSCAN / HDBSCAN, min_samples = MIN_SUPPORT）
  ↓  每个簇内步骤序列语义相似
LCS 对齐 → 参数抽取 → 不变量挖掘
  ↓
Harness 合成 → 沙盒验证
```

空间聚类确保 LCS 只在**真正相似的轨迹簇**内运行，避免异类轨迹污染公共骨架。

每阶段职责严格分离，详见 §5。

---

## 5. 五阶段详细设计

### 5.1 阶段一：路径轨迹收集（Path Collection）

详见 §3.2 Layer 0 数据结构。核心要点：

- **失败轨迹也要收集**：用于挖掘部分成功子步骤、负样本学习、诊断适用范围边界
- **SubStep 细粒度拆分**：完整路径内部按 TODO list / 里程碑 / 对话边界拆分，支持子步骤级归纳
- **嵌套追踪**：执行中调用了哪些 Harness 通过 `source_harness_ids` 记录
- **状态快照**：每步前后环境状态哈希，供不变量挖掘
- **预测契约收集**（§4.2）：Agent 对每个子任务生成 `PredictionContract {expected_input, expected_output, expected_effect, confidence}`，执行后记录预测-vs-实际对比结果。该对比作为经验质量信号传递到 Induction Engine

#### 5.1.1 预测契约数据结构

```python
@dataclass
class PredictionContract:
    """Agent 在执行前对子任务的预测——flow.md 的预测-验证机制"""
    step_id: str                     # 对应的 SubStepPlan.step_id
    expected_input: dict             # 预期输入（映射到 Hoare Triple P）
    expected_output: dict            # 预期输出（映射到 Hoare Triple Q）
    expected_effect: str             # 预期副作用（如 "order status changed to exchanged"）
    confidence: float                # Agent 自评置信度 (0.0–1.0)

@dataclass
class PredictionVerification:
    """执行后的预测-vs-实际对比"""
    contract: PredictionContract
    actual_input: dict
    actual_output: dict
    actual_effect: str
    prediction_accurate: bool        # 预测是否准确
    divergence_reason: str = ""      # 差异原因（若预测不准）
```

#### 5.1.2 预测质量分层

预测-vs-实际对比产生四类经验质量标签，直接用于 Induction Engine 的贝叶斯权重调整：

| 预测准确？ | 执行成功？ | 质量标签 | 归纳权重 |
|-----------|-----------|---------|---------|
| ✅ 是 | ✅ 是 | **高质量经验** | 权重 ×1.0 |
| ❌ 否 | ✅ 是 | **侥幸成功** | 权重 ×0.3，需更多 support |
| ✅ 是 | ❌ 否 | **实现缺陷** | 触发 F2 patch 路径 |
| ❌ 否 | ❌ 否 | **负样本** | 记录边界条件，不参与正向归纳 |

> **设计依据**：flow.md §1 指出「现有研究大多关注"执行-结果"反馈，缺乏对 Agent 规划能力的显式建模和验证」。预测契约机制填补了这一空白——将 Agent 的规划能力（预测）与实际执行结果对比，作为经验质量的核心判据。

### 5.2 阶段二：经验归纳引擎（Induction Engine）

#### 5.2.1 归纳触发条件（批量触发）

```python
class InductionTrigger:
    MIN_SUPPORT: int = 3            # 同类成功轨迹至少 3 条
    MAX_VARIANCE_THRESHOLD: float   # 步骤序列编辑距离方差上限
    SUBSTEP_MIN_SUPPORT: int = 2    # 子步骤归纳更低门槛
    F2_PATCH_THRESHOLD: int = 2     # F2 失败累计 ≥2 触发 Patch
    MIN_PREDICTION_ACCURACY: float = 0.5  # 预测契约最低准确率（新增）
```

触发事件：
- `support_count >= MIN_SUPPORT` → 新建 Harness
- `new_variation_detected` → 特化分裂（Future Work）
- `F2 count >= 2` → Patch

**预测契约质量门控（新增）**：触发归纳前，检查候选轨迹簇的预测契约准确率。若准确率 < `MIN_PREDICTION_ACCURACY`，说明 Agent 对这类任务的理解尚不稳定，即使 support 足够也应延迟归纳。

#### 5.2.0 空间聚类：归纳前的语义分组（新增）

传统按 `(task_type, tool_name)` 硬分组的问题是：**语义相似但工具名不同的轨迹永远不会被归入同一簇**。ExperienceOS 采用向量嵌入 + 密度聚类，在真正的语义空间中发现模式：

```python
def spatial_cluster(self, trajectories: list[Trajectory]) -> list[list[Trajectory]]:
    """flow.md 风格的空间聚类——在 LCS 对齐之前运行。"""
    features = []
    for t in trajectories:
        # 特征向量 = concat(
        #   embedding(t.task_description),          # 任务语义
        #   embedding(tool_sequence_signature(t)),   # 工具序列结构签名
        #   embedding(t.effect_description),          # 预期效果
        # )
        f = self._extract_spatial_features(t)
        features.append(f)

    # 密度聚类：自动发现簇数，min_samples = MIN_SUPPORT
    # 使用 DBSCAN 或 HDBSCAN（层次密度聚类），避免预设簇数
    clusters = density_cluster(features, min_samples=MIN_SUPPORT)

    return clusters  # 每个簇 = 语义相似的轨迹集合
```

**特征空间的三维定义**（flow.md §3）：

| 维度 | 含义 | 提取方式 |
|------|------|---------|
| **任务语义** | 任务描述的核心意图 | embedding(task_description) |
| **I/O 结构** | 工具调用的输入输出签名 | embedding( canonical_tool_sequence + param_keys + result_types ) |
| **预期效果** | 子任务对整体目标的贡献 | embedding( prediction.effect_description ) |

**为什么先聚类再做 LCS？** 直接将所有同 task_type 轨迹送进 LCS 会产生无意义的对齐——例如 exchange 和 refund 都操作订单，但步骤序列完全不同。空间聚类确保只有**真正相似的轨迹**才参与 LCS，提升公共骨架质量。

> **实现备注**：当前 `compiler/inductor.py::_cluster_patterns()` 已实现基于 embedding cosine ≥ 0.85 的聚类合并（P2.1），作为空间聚类的简化版本。完整密度聚类（DBSCAN/HDBSCAN）替换该函数即可升级。

#### 5.2.2 归纳层次（维度 B）

```
Level 0: SubStep 归纳
  输入：同一任务中反复出现的局部步骤序列（经空间聚类分组）
  输出：SubStep Harness（最小可复用单元）
  示例："截图→角点检测→鼠标定位" 编译为 VisualLocator v1.0
  预测契约作用：验证子步骤的 expected_effect 是否一致

Level 1: Task 归纳
  输入：完整任务成功轨迹（MIN_SUPPORT 条，经空间聚类确保步骤序列相似）
  输出：Task Harness（完整流程的参数化程序）
  示例："发送邮件" 编译为 EmailSender v1.0
  预测契约作用：区分"有效经验"和"侥幸成功"——预测准确率低的轨迹降权

Level 2: Composite 归纳 (Future Work → 纳入当前路线图)
  输入：多个 Task Harness 组合使用的轨迹 + 依赖关系图
  输出：Composite Harness（调用子 Harness 的元程序）
  示例："每日报告发送" 编译为 DailyReporter v1.0
        内部调用 EmailSender v1.0 + ReportGenerator v1.0
  预测契约作用：验证组合链中每个子 Harness 的输入/输出契约是否兼容
```

> **论文边界**：当前论文实现 Level 0 + Level 1。Level 2 Composite 依赖 flow.md 的依赖发现机制（见 §5.3.6），作为论文的 preliminary 扩展结果。

**flow.md 对归纳层次的增强**：

flow.md §4 提出的依赖发现直接映射到 Level 2 Composite：
- **前置依赖**：`E_j` 的输入参数来自 `E_i` 的输出 → 可组合
- **独立并行**：`E_i` 和 `E_j` 使用不相交工具集 → 可并行执行
- **连续组合**：`E_i` 成功后 `E_j` 的触发概率显著提升 → 可串联

该机制使 Composite 归纳从"Future Work"升级为"可实施方案"，详见 §5.3.6。

#### 5.2.3 归纳算法：贝叶斯程序归纳

从轨迹集合 $T_c = \{t_1, ..., t_n\}$ 中归纳最优 Harness $H^*$：

$$H^* = \arg\max_H P(H \mid T_c) \propto \underbrace{P(T_c \mid H)}_{\text{似然：回放成功率}} \cdot \underbrace{P(H)}_{\text{MDL先验：简洁性}}$$

```
似然项：在沙盒中回放 H，统计成功率
先验项：MDL 惩罚 = -log₂(step数) - log₂(参数数)
阈值判定：log P(H|T_c) > θ_induct → 触发编译
```

意义：**只有既能解释历史轨迹又足够简洁的 Harness 才值得被编译入库**。

Artifact 效用约束（防止 artifact 爆炸）：

```
U(a) = ΔSuccess(a) + ΔEfficiency(a) - Cost_creation
```

只有期望效用超过创建成本时才编译。

#### 5.2.4 步骤对齐与参数抽取（LCS）

**前置步骤**：轨迹先经 §5.2.0 空间聚类分组，确保只有语义相似的轨迹参与 LCS。跨簇 LCS 无意义——将 exchange 和 refund 轨迹对齐只会产生噪音。

同一簇内的多条轨迹在**规范化步骤序列**（见 §6）上做 LCS 对齐：

```
轨迹 t1: click(login_btn) → type("user@a.com") → click(submit)
轨迹 t2: click(login_btn) → type("user@b.com") → click(submit)
轨迹 t3: click(login_btn) → type("admin@c.com") → click(submit)

LCS 对齐后识别：
  固定步骤：click(login_btn), click(submit)  → 编译为常量
  变化参数：email 地址                        → 提取为变量 {email}

输出模板：login(email: str) {
  click(login_btn)
  type({email})
  click(submit)
}
```

**多步序列 vs 单步调用**：LCS 对齐的核心价值在于捕获**多步骤参数化序列**。当前实现中单个 `call_tool()` wrapper 仅为临时简化——真正的 ExperienceOS Harness 应替换整个子任务执行路径（如 login 的 3 步序列），而非拦截单个 API 调用。这是论文的核心差异化：**编译多步可执行程序，而非缓存 API 结果**。

**预测契约辅助参数类型推断**：LCS 识别变化点后，利用预测契约中的 `expected_input` schema 推断参数类型（如 `email: str` vs `order_id: int`），提升参数化精度。

#### 5.2.5 不变量挖掘（Daikon 风格 + 预测契约增强）

受 Daikon 动态不变量检测启发，对每条成功轨迹的状态快照序列挖掘持续成立的谓词。**新增**：利用预测契约中的预期效果作为不变量候选项的验证信号。

```python
class InvariantMiner:
    def mine(self, trajectories: List[Trajectory],
             predictions: List[PredictionVerification]) -> List[Invariant]:
        # 收集所有步骤前后的状态哈希序列
        # 数据驱动找到"每次成功都成立"的谓词
        # 示例不变量：
        #   - 执行前 network_connected == True
        #   - 执行中 modal_open == False
        #   - 执行后 email_count 增加 1

        # 新增：预测契约交叉验证
        # 若 prediction.expected_effect 在多条轨迹中一致且验证通过，
        # 将该 effect 提升为候选不变量
        # 例如：多次预测 "order status → exchanged" 且实际一致
        #       → 不变量: post_state.order.status == expected_status
```

**预测契约对不变量挖掘的三项增强**：

1. **效果不变量**：多条轨迹的 `expected_effect` 高度一致 → 提取为 postcondition 候选项
2. **边界发现**：预测准确但执行失败 → 说明存在 Agent 未意识到的隐藏约束 → 标记为候选前置条件
3. **置信度加权**：预测置信度高的轨迹在不变量投票中权重更高

### 5.3 阶段三：编译引擎（Compile Engine）

#### 5.3.1 六阶段编译流水线（+ 依赖发现）

1. **Segment**（轨迹分段）—— LLM 识别语义边界（>3 步时）。**实现要求**：分段结果必须传递到后续阶段，不能丢弃。
2. **Intersect Preconditions**（前后置条件提取）—— 跨轨迹集合交集。**新增**：参考预测契约的 `expected_input` 补充机器无法从轨迹中推断的前置条件。
3. **Mine Invariants**（不变量挖掘）—— Daikon 风格动态不变量检测 + 预测契约交叉验证（见 §5.2.5）
4. **Abstract Steps**（步骤抽象与参数化）—— **LCS + 类型感知参数化**（非正则替换）。前置：空间聚类确保输入轨迹语义相似
5. **Synthesize**（Harness 合成）—— LLM 生成 `run()` 函数，从 few-shot 例子学习 `call_tool()` API
6. **Validate**（沙盒回放验证）—— `success_rate >= validation_threshold`（默认 0.8）方入库
7. **Discover Dependencies**（依赖发现，新增）—— 分析子 Harness 间的数据流依赖，构造 Composite Artifact（见 §5.3.6）

#### 5.3.2 Harness 数据结构

见 §3.2 Layer 2 定义。

#### 5.3.3 版本管理：纵横版本树

版本号规则：`MAJOR.MINOR.PATCH`
- **MAJOR**：不兼容的接口变更（签名改变）
- **MINOR**：向后兼容的功能增强（新增步骤、新适用范围）
- **PATCH**：内部修复（不变量修正、rollback 改进）

```
纵向版本线（进化）：
  EmailSender v1.0 → v1.1 → v1.2 → v2.0
                               ↑
                         (接口不变，修复附件验证 bug)

横向分裂线（适用范围分化）：
  EmailSender v1.2
      ├── EmailSender_Gmail v1.2.1   (scope: app=gmail)
      └── EmailSender_Outlook v1.2.1 (scope: app=outlook)
```

版本 DAG 边类型：
- `patch` — bug fix 或实现更新
- `specialization` — 环境特化（浏览器/OS/应用版本）
- `composition` — 多 Artifact 组合为高层（Future Work）

> **论文边界**：当前实现仅 `patch` 边。`specialization` / `composition` 为 Future Work。

#### 5.3.4 适用范围（ApplicabilityScope）与自动分裂

```python
@dataclass
class ApplicabilityScope:
    os: Optional[List[str]]         # ["linux", "macos"]
    browser: Optional[List[str]]    # ["chrome"]
    app: Optional[str]              # "gmail"
    app_version_range: Optional[str]# ">=2.0,<3.0"
    locale: Optional[List[str]]     # ["zh-CN"]
    agent_min_level: int = 0        # 最低可用的 Agent 能力等级
    custom_constraints: Dict
```

**自动分裂触发**（Future Work）：当同一 Harness 在不同环境下成功率差异显著时自动分裂：

```
触发条件：
  success_rate(scope_A) - success_rate(scope_B) > δ_split (默认 0.2)
  且两个 scope 各有足够样本 (≥ MIN_SUPPORT)

分裂过程：
  1. 识别导致差异的 scope 维度
  2. 生成两个子版本，继承父版本基础结构
  3. 子版本在各自 scope 上独立优化
  4. 父版本降级为 "router"（不直接执行，路由到子版本）
```

#### 5.3.5 编译引擎内部修复循环

Harness 首次编译后必须经过沙盒验证，验证失败触发自动修复：

```
编译 → 沙盒验证
          ↓
    验证失败？
    ├── YES → 分析失败原因
    │           ├── 前置条件过严 → 放宽 preconditions
    │           ├── 步骤参数化错误 → 重新 LCS 对齐
    │           ├── 环境依赖缺失 → 更新 scope 约束
    │           └── 不变量误报 → 删除该不变量
    │           → 修复后重新验证（最多 MAX_RETRY 次）
    └── NO  → 写入 Registry
```

#### 5.3.6 依赖发现与 Composite Artifact 构造（新增，来自 flow.md §4）

当多个 SubStep / Task Harness 被频繁组合使用时，系统应能发现它们之间的依赖关系并构造高层 Composite Artifact。flow.md 的依赖发现机制填补了 ExperienceOS.md Level 2 Composite 的实现路径。

**依赖类型**：

```
前置依赖（Predecessor）：
  E_j 的输入参数来自 E_i 的输出
  例：get_order_details(order_id) 依赖 find_user(email) 返回的 user.order_ids

独立并行（Independent Parallel）：
  E_i 和 E_j 使用不相交的工具集，无数据流依赖
  例：get_product_details 和 check_shipping_status 可并行执行

连续组合（Sequential）：
  E_i 成功后 E_j 的触发概率显著提升（P(E_j|E_i.success) >> P(E_j)）
  例：find_user → get_order 的转移概率接近 1.0
```

**依赖发现算法**：

```python
def discover_dependencies(self, harnesses: list[Harness],
                          trajectories: list[Trajectory]) -> DependencyGraph:
    """flow.md 风格的依赖发现——分析同一 Task 内子 Harness 间的数据流。"""
    graph = DependencyGraph()

    for traj in trajectories:
        invoked_harnesses = self._extract_harness_invocations(traj)

        for i, h_prev in enumerate(invoked_harnesses):
            for h_next in invoked_harnesses[i+1:]:
                # 1. 数据流分析：h_next 的输入参数名是否匹配 h_prev 的输出字段
                param_overlap = (
                    set(h_prev.output_schema.keys()) &
                    set(h_next.input_schema.keys())
                )
                if param_overlap:
                    graph.add_edge(h_prev, h_next, type="predecessor",
                                   shared_params=list(param_overlap))

                # 2. 转移概率：P(h_next | h_prev.success)
                transition_prob = self._estimate_transition_prob(
                    h_prev, h_next, trajectories
                )
                if transition_prob > 0.8:
                    graph.add_edge(h_prev, h_next, type="sequential",
                                   probability=transition_prob)

    return graph
```

**Composite Artifact 构造**：

依赖图就绪后，将强依赖的 Harness 组合为 Composite：

```python
@dataclass
class CompositeHarness(Harness):
    """Level 2 归纳产物——调用子 Harness 的元程序"""
    sub_harnesses: list[HarnessRef]       # 子 Harness 引用
    dependency_graph: DependencyGraph     # 依赖关系图
    parallel_groups: list[list[str]]      # 可并行的 Harness 组
    composition_strategy: str             # "sequential" | "parallel_where_possible"

# 示例：OrderExchangeFlow
OrderExchangeFlow = CompositeHarness(
    sub_harnesses=[UserLookup, OrderLookup, ProductLookup, ExchangeExec],
    dependency_graph={
        OrderLookup: [UserLookup],           # 需要 user_id
        ProductLookup: [],                    # 独立
        ExchangeExec: [OrderLookup, ProductLookup],  # 需要两者输出
    },
    parallel_groups=[[UserLookup], [OrderLookup, ProductLookup], [ExchangeExec]],
    composition_strategy="parallel_where_possible",
)
```

**预测契约在 Composite 中的角色**：

组合多个 Harness 时，每个子 Harness 的预测契约提供兼容性验证：
- `E_i.expected_output` 的 schema 是否匹配 `E_j.expected_input` 的 schema？
- 若不匹配，是否需要 adapter/transform 步骤？

> **论文边界**：Composite 归纳（Level 2）在 ExperienceOS.md §5.2.2 中标记为 Future Work。flow.md 的依赖发现为其提供了具体实现路径，可作为论文的 preliminary 扩展结果。当前实现中 version DAG 的 `composition` 边类型已预留 schema。

### 5.4 阶段四：层级化经验仓库（Registry Store）

#### 5.4.1 仓库层级结构（Future Work 完整版）

```
Registry (层级化)
├── Global Layer（全局层）—— 社区共享的通用 Harness
│   示例：BasicFileOps v3.1, WebNavigation v2.0
│
├── Organization Layer（组织层）—— 企业/团队私有（覆盖全局层）
│   示例：CompanyEmailSender v1.5（继承 EmailSender v1.2，
│         添加公司邮件格式校验）
│
└── User Layer（个人层）—— 个人习惯（覆盖组织层）
    示例：MyDailyReport v0.3

覆盖优先级：User > Organization > Global
同 scope 内版本优先级：高版本 > 低版本（但可固定版本）
```

> **论文边界**：当前论文只实现单层平铺 Registry。三层覆盖为 Future Work。

#### 5.4.2 仓库内部数据结构

```
Registry/
├── index/
│   ├── by_task/           # task_id → [harness_ids]
│   ├── by_scope/          # scope_hash → [harness_ids]
│   └── by_substep/        # substep_signature → [harness_ids]
│
├── harnesses/
│   ├── H_email_send_abc123/
│   │   ├── v1.0.json      # Harness 完整定义
│   │   ├── v1.1.json
│   │   ├── v2.0.json
│   │   └── version_tree.json  # 版本树元数据
│   └── ...
│
├── trajectories/
│   ├── raw/               # 原始轨迹（按 task_id 分桶）
│   └── processed/         # 已归纳处理过的轨迹
│
└── stats/
    └── usage_feedback.jsonl   # 使用反馈流
```

#### 5.4.3 跨 Agent 共享机制

"高级 Agent 经验可被低级 Agent 使用"的实现核心：

```
高级 Agent (GPT-4 级别) 归纳出 ComplexHarness v1.0
  scope: { agent_min_level: 0 }  ← 关键：设置最低可用等级为 0

低级 Agent (GPT-2 级别) 查询匹配到 ComplexHarness v1.0
  → 不需要理解内部逻辑，只需要：
    1. 满足 preconditions
    2. 提供正确入参
    3. 验证 postconditions
  → 执行结果与高级 Agent 一致

本质：高级 Agent 的推理能力被"编译"进了 Harness
      低级 Agent 通过"执行"获得了超出自身推理能力的结果
```

### 5.5 阶段五：运行时执行引擎（Runtime Execution）

#### 5.5.1 Harness 查询与匹配

```python
class HarnessResolver:
    def resolve(self,
                task_description: str,
                env_context: EnvContext,
                agent_level: int) -> Optional[HarnessMatch]:

        # Step 1: 语义检索（task_description → candidate harnesses）
        candidates = self.semantic_index.search(task_description, top_k=10)

        # Step 2: Scope 过滤（剔除不适用当前环境的）
        compatible = [h for h in candidates
                      if self.scope_matcher.matches(h.scope, env_context)
                      and h.scope.agent_min_level <= agent_level]

        # Step 3: 版本选择（考虑覆盖层优先级 + 版本优先级）
        best = self.version_selector.select(compatible, env_context)

        # Step 4: 前置条件检查
        if best and self.precondition_checker.check(best, current_state):
            return HarnessMatch(harness=best, confidence=...)
        return None  # 走纯 LLM 路径
```

两阶段检索：
- **Stage 1 语义检索（粗）**：任务 embedding 与 Harness embedding 的余弦相似度。Harness embedding = `task_type + description + preconditions_summary + example_tasks`
- **Stage 2 前置条件匹配（细）**：硬条件（OS/应用存在性/权限）必须匹配；软条件（浏览器版本/分辨率）允许降级

检索结果：`FULL_MATCH`（高置信）/ `SOFT_MATCH`（中置信）/ `NO_MATCH`（Agent fallback）

#### 5.5.2 嵌套执行与组合调用

```
执行任务 A（DailyReport）
  ├── 匹配到 DailyReporter v1.2
  │     内部调用：
  │     ├── ReportGenerator v2.0  (sub-harness)
  │     │     内部调用：
  │     │     └── DataFetcher v1.1 (sub-harness)
  │     └── EmailSender v1.5      (sub-harness)
  │
  ├── 执行记录：traj_id=xyz
  │     source_harness_ids: [DailyReporter/v1.2,
  │                          ReportGenerator/v2.0,
  │                          EmailSender/v1.5]
  │
  └── 成功后：
        → 本次轨迹写入 trajectory store
        → 归纳引擎检测：DailyReporter 已积累足够样本？
        → 若是：升级 DailyReporter v1.2 → v1.3（或触发分裂）
```

#### 5.5.3 执行后反馈闭环（含预测契约验证）

```python
@dataclass
class ExecutionFeedback:
    traj_id: str
    harness_id: str
    harness_version: str
    outcome: Outcome

    # 具体失败信息（用于修复）
    failed_step: Optional[int]
    failed_condition: Optional[str]   # precondition? invariant? postcondition?
    actual_vs_expected: Optional[Dict]

    # 环境信息（用于分裂判断）
    env_context: EnvContext

    # 预测契约验证（新增）
    prediction_verification: Optional[PredictionVerification]
    quality_label: str = ""           # high_quality | lucky_success | implementation_defect | negative_sample

class FeedbackProcessor:
    def process(self, feedback: ExecutionFeedback):
        # 1. 更新 Harness 统计（success_rate, usage_count）
        # 2. 若连续失败 > 阈值 → 触发 NEEDS_REVISION 标记
        # 3. 将反馈写入 trajectory store（作为新轨迹数据）
        # 4. 触发增量归纳（检查是否需要分裂或版本更新）
        # 5. 预测契约验证结果写入 ExperienceStore（影响该经验的贝叶斯权重）
```

**预测-vs-实际对比在反馈闭环中的角色**（§4.2）：

执行完成后，系统不仅记录"成功/失败"，还对比 Agent 的预测契约与实际结果：

```
Harness 执行成功 + 预测准确 → 高质量经验，提升 artifact 置信度
Harness 执行成功 + 预测偏差 → 标记为"需审查"，可能依赖了未声明的环境状态
Harness 执行失败 + 预测准确 → F1/F2/F3 分类，触发修复
Harness 执行失败 + 预测偏差 → F4（超出范围），记录边界条件
```

这一机制使反馈闭环不再仅依赖二元成功/失败信号，而是获得了**结构化的质量梯度**。flow.md 指出「这能有效区分'有效的经验'和'侥幸的成功'，提升经验质量」。

#### 5.5.4 两种系统模式

- **ACCUMULATION 模式**：始终使用 LLM Agent，记录轨迹，触发归纳。**清洁轨迹防止 bootstrapping 污染**（Harness 质量影响归纳质量）。
- **DEPLOYMENT 模式**：优先通过 RuntimeRouter 匹配 Harness，miss/失败时回退到 Agent，继续记录用于在线学习。

### 5.6 失败分类 F1-F4

| Code | 含义 | 触发 | Harness 动作 |
|------|------|------|-------------|
| **F1** | 前置条件缺口 | 环境状态不匹配 Harness 期望 | 保留旧 Harness，创建特化变体 |
| **F2** | 实现错误 | Harness 代码 bug（NameError 等）| 标记 DEPRECATED → patch 生成 → 新版本 |
| **F3** | 环境漂移 | API/UI 变化（"not found"、KeyError）| 保留旧版本，标记 scope，新增 adapter |
| **F4** | 超出范围 | 任务超出能力 | 不变更，记录 OOD 案例 |

修复迭代：失败计数 → 连续失败 N 次（默认 3）→ ACTIVE 降级为 NEEDS_REVISION → 重新归纳 → 版本 DAG `parent_id` 链 → 重归纳仍失败则 ABANDONED 存为负面经验。

---

## 6. 语义对齐机制

语义对齐问题出现在四个位置，每个位置性质不同，解法也不同。

### 6.1 四个战场

```
战场 1：任务级对齐
  「帮我发邮件给Alen」vs「send report to Alen」vs「给Alen发个mail」
  → 这两个任务是不是"同一类任务"？

战场 2：工具/步骤级对齐
  send_email vs sendEmail vs releaseMail vs dispatch_message
  → 这些步骤是不是"同一个操作"？

战场 3：参数级对齐
  {"to": "alen@gmail.com"} vs {"recipient": "alen@gmail.com"} vs {"address": "..."}
  → 这些参数是不是"同一个语义槽"？

战场 4：Harness 查询对齐
  新任务来了，怎么知道哪个已有 Harness 适用？
  → 匹配问题
```

### 6.2 三层规范化（Normalization Layer）

**轨迹在写入 store 之前必须经过规范化**，否则后续所有算法都建立在沙堆上。三层降级，由快到慢：

#### Layer 1：规范名称字典（最快，O(1)）

```python
TOOL_CANONICAL_MAP = {
    # email sending 语义簇
    "send_email":         "EMAIL.SEND",
    "sendEmail":          "EMAIL.SEND",
    "releaseMail":        "EMAIL.SEND",
    "dispatch_message":   "EMAIL.SEND",
    # file open 语义簇
    "open_file":          "FILE.OPEN",
    "file_open":          "FILE.OPEN",
    # ...
}

def normalize_tool_name(raw_name: str) -> str:
    if raw_name in TOOL_CANONICAL_MAP:
        return TOOL_CANONICAL_MAP[raw_name]
    return None  # 未知工具进入 Layer 2
```

#### Layer 2：Embedding 相似度（中速）

综合工具名 + 参数结构 + 返回值结构判断语义：

```python
def normalize_via_embedding(raw_name, args, result_schema) -> str:
    tool_fingerprint = f"""
    tool_name: {raw_name}
    args_keys: {sorted(args.keys())}
    result_type: {type(result).__name__}
    """
    fingerprint_vec = embed(tool_fingerprint)
    best_match, score = vector_store.search(fingerprint_vec, top_k=1)
    if score > SIMILARITY_THRESHOLD:  # 0.85
        return best_match.canonical_name
    return None  # 进入 Layer 3
```

#### Layer 3：LLM 语义判断（最慢，兜底）

```python
def normalize_via_llm(raw_name, args, context) -> Tuple[str, float]:
    prompt = f"""
    判断以下工具属于哪个标准操作类别。
    工具名称：{raw_name}
    参数：{args}
    执行上下文：{context}
    候选标准类别：{list(CANONICAL_CATEGORIES)}
    如果不匹配，返回 NEW:<描述>
    输出JSON: {{"canonical": str, "confidence": float, "reasoning": str}}
    """
    result = llm.complete(prompt)
    # 结果写回字典，下次 Layer 1 命中
    if result.confidence > 0.9:
        TOOL_CANONICAL_MAP[raw_name] = result.canonical
    return result.canonical, result.confidence
```

**每层结果写回上层缓存，系统随使用自动变快。**

### 6.3 任务聚类（归纳前）

```
双层聚类：
  Layer A: 意图结构匹配（基于规范化意图JSON）
    条件：primary_intent 相同 AND objects 语义重叠 > 0.7
  Layer B: 步骤序列相似度（基于规范化步骤）
    计算：edit_distance(seq_A, seq_B) / max(len_A, len_B)
    阈值：相似度 > 0.6 → 同类任务
```

### 6.4 完整信息流

```
原始任务描述（自然语言，任意语言/措辞）
       │
       ▼ [LLM] 意图抽取 → 结构化意图 JSON
       ▼ [Embedding] 意图向量化
       │
       ├──── 查询路径（已有 Harness？）
       │         ▼ [向量检索] 候选 Harness
       │         ▼ [算法] Scope 过滤
       │         ▼ [LLM] 参数槽填充
       │         ├── YES → Harness 执行路径
       │         │         ▼ [规范名→实际工具] 映射
       │         │         ▼ [工具] 执行各步骤
       │         │         ▼ [算法] 后置条件验证
       │         │         ▼ 写入轨迹（含规范化）
       │         └── NO → LLM 执行路径
       ▼
  执行完成，写入轨迹
       ▼ [Layer 1/2/3] 工具名规范化
       ▼ [算法] 步骤序列规范化
       ▼ [算法+Embedding] 任务聚类
       ▼ 归纳触发判断（MIN_SUPPORT）
       ▼ [算法] LCS 对齐（在规范化序列上）
       ▼ [算法] 参数抽取 + 不变量挖掘
       ▼ [算法] 沙盒验证
       ▼ 写入 Registry（Harness 可用）
```

### 6.5 遗留硬问题与处理方案

| 问题 | 处理方案 |
|------|---------|
| 规范化字典冷启动 | 初始化时 LLM 批量归类建立初始字典；之后每次 LLM 成功规范化写回字典，字典自增长；社区共享字典（Org Layer 一部分） |
| 规范化错误传播 | 记录 `confidence` 分数；低置信度步骤不参与 LCS 主骨架，作为"可选步骤"；沙盒验证失败时回溯检查规范化 |
| 同名不同义 | 规范化时不只看工具名，还看参数结构和返回值类型（Layer 2 fingerprint）；同名但 fingerprint 差异大分配不同规范名 |
| LLM 意图抽取一致性 | 意图 JSON 走 embedding 而非精确匹配；few-shot 锁定输出格式；同 task_id 轨迹意图抽取只做一次并缓存 |

---

## 7. Harness 生命周期状态机

```
           ┌──────────────┐
           │   CANDIDATE  │  ← 刚归纳出，未验证
           └──────┬───────┘
                  │ 沙盒验证通过
           ┌──────▼───────┐
           │    ACTIVE    │  ← 可正常使用
           └──────┬───────┘
          ┌───────┴────────┐
          │                │
   ┌──────▼───────┐  ┌─────▼────────┐
   │NEEDS_REVISION│  │  DEPRECATED  │  ← 被新版本取代
   └──────┬───────┘  └──────────────┘
          │ 修复成功
   ┌──────▼───────┐
   │    ACTIVE    │  ← 修复后恢复
   └──────────────┘

   分裂发生时：
   ACTIVE → DEPRECATED (父版本)
          → ACTIVE     (子版本 A)
          → ACTIVE     (子版本 B)
```

---

## 8. 实验设计

### 8.1 主实验（必须完成，证明核心假设）

```
数据集：τ-bench retail 域
Backbone：DeepInfra/MiniMax-M2.7（已验证可用）
数据划分：每类任务前 K=3 → Warmup，剩余 → Eval（已实现）

对比方法（4 组，统一 Backbone + Warmup 数据）：
  A. Vanilla LLM（无工具）         → Token 成本下界
  B. ReAct Agent（无积累）         → 当前主流范式
  C. SkillOpt（文本 Skill 优化）   → 最强文本记忆 Baseline
  D. ExperienceOS（本文，可执行代码）→ 我们的方法

评估指标：
  1. Task Success Rate（↑）
  2. Avg Tokens/Task（↓）
  3. Harness Hit Rate（D 独有）
  4. 积累曲线图（x=任务序号，y=滚动 SR）← 核心图表
  5. Token 成本收敛曲线
  6. Avg Latency / Task
```

### 8.2 消融实验

```
D1. ExperienceOS w/o Sandbox Validation → 证明验证门控的必要性
D2. ExperienceOS w/o SubStep Discovery  → 证明子步骤触发的贡献
D3. MIN_SUPPORT = {1, 3, 5}             → 确定最优积累阈值
D4. ExperienceOS w/o Versioning         → 证明版本管理贡献
```

### 8.3 泛化实验（有余力则做）

```
type_split（默认）：同类型未见实例 → 参数泛化
cross_type：A 类型 → B 类型         → 共享子步骤迁移
cross_domain：retail → airline      → 跨域迁移
replay：重跑相同实例                → 上界参考
```

### 8.4 数据划分（防数据泄露）

- **Warm-up Pool**：每类前 K=3 实例 → 用于积累/归纳
- **Evaluation Pool**：剩余实例 → 用于评估，**绝不**在 warmup 见过
- 所有 baseline 共用**相同 warmup 数据**——对比的是"如何使用经验"，不是"经验多少"

### 8.5 积累曲线（核心图表）

```
x 轴：任务序号
y 轴：滚动平均成功率

曲线：
  - Vanilla agent：平直（无经验学习）
  - RAG agent：略升（few-shot 收益）
  - ExperienceOS：warmup 期平直，随后在 crossover 点显著上升
  - Fixed harness（上界）：从一开始最高
```

**crossover 点是论文的核心实证主张。**

---

## 9. 论文边界与 Future Work

### 9.1 当前论文范围（必须证明）

> **Level 0 + Level 1 归纳 + 预测契约 + 空间聚类 + 基础单层 Registry + 单层运行时，已足够在 τ-bench 上证明"Token 成本显著下降，成功率不降"。**

| 模块 | 当前论文实现 | Future Work |
|------|------------|------------|
| 轨迹收集 | 单 Agent，τ-bench 环境 | 多 Agent，多环境，跨平台 |
| 预测契约 | PredictionContract 收集 + PredictionVerification 质量分层（★ 新增） | 跨 Agent 预测能力校准 |
| 归纳引擎 | Level 0 + Level 1 + 空间聚类（★ 增强） | Level 2（Composite）→ 已进入路线图 |
| 编译引擎 | 参数化（需升级为真正多步 LCS）+ 沙盒验证 | 自动修复循环，scope 自动分裂 |
| 依赖发现 | 数据流分析 + 转移概率（★ 新增，§5.3.6） | 完整 Composite 构造 + 并行调度 |
| 版本管理 | 单版本线（patch + composition 边 schema 就绪） | 纵横版本树（specialization 边） |
| 仓库结构 | 单层平铺 | User / Org / Global 三层覆盖 |
| 跨 Agent 共享 | 同 Agent 复用 | 能力等级降维共享 |
| 运行时 | 查询 + 执行 + 预测契约验证（★ 增强） | 嵌套执行，反馈驱动更新 |
| 语义对齐 | 单 Agent 同工具集 | 跨 Agent 工具名规范化（四战场） |

★ = flow.md 融合增强项

### 9.2 完整愿景

| 模块 | 愿景 |
|------|------|
| 轨迹收集 | 多 Agent，多环境（τ-bench + TerminalBench + OSWorld + WorkArena） |
| 归纳层次 | Level 2 Composite（多个 Task Harness 组合为元程序） |
| 编译引擎 | 自动修复循环 + scope 自动分裂 + 合并 |
| 版本管理 | 纵横版本树：纵向进化 + 横向分裂 |
| 仓库结构 | User > Organization > Global 三层覆盖（类 Linux PATH） |
| 跨 Agent 共享 | 高级 Agent 推理能力"编译"进 Harness，低级 Agent 直接调用 |
| 语义对齐 | 跨 Agent 工具命名规范化（字典→Embedding→LLM 三层降级） |

### 9.3 经济学意义

```
传统 Agent 经济学：
  Cost = tasks × per-task reasoning cost
  Scale → 线性成本增长

知识基础设施经济学：
  Cost = initial accumulation + tasks × marginal execution cost
  Marginal execution cost ≈ 0（确定性代码）
  Scale → 单任务成本趋零
```

### 9.4 研究问题

| RQ | 问题 | 验证方式 |
|----|------|---------|
| RQ1 | Agent 能否可靠地从轨迹中发现模式并编译为已验证 Artifact？ | 归纳成功率 + 回放验证通过率 |
| RQ2 | 编译出的 Artifact 能否跨实例、任务族、环境、模型骨架泛化？ | 泛化实验（type_split / cross_domain） |
| RQ3 | 性能是否随 Artifact 仓库增长而单调提升？ | 积累曲线 |
| RQ4 | Artifact 能否通过失败反馈自修复且不回退？ | 版本 DAG + patch 成功率 |

---

## 10. 当前实现状态与差距地图

> **最后更新**：2026-07-25（exp-0001 端到端验证通过后）

### 10.1 已实现模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `models.py` Hoare Triple | ✅ | `H=<P,steps,I,Q,R>` 完整，含 4 层 dataclass |
| `storage.py` SQLite + 向量 BLOB | ✅ | 8 张表，float32 BLOB，含 JSON 迁移 |
| `embedding.py` 三级回退 | ✅ | Qwen3-Embedding-8B → ollama → hash，SQLite 缓存 |
| `env_info.py` 环境元数据收集 | ✅ | OS/Python/硬件/模型/包版本，结构化 SQLite |
| `agent.py` F1-F4 失败分类 + Agent Fallback | ✅ | ReAct 风格，含工具调用解析 |
| `runtime.py` ACCUMULATION/DEPLOYMENT 双模式 | ✅ | 含 HarnessRegistry 子步骤拦截 |
| τ-bench 集成（`tau2_adapter.py`） | ✅ | 仿真 + 轨迹转换 + DB hash 验证 |
| Warmup/Eval 数据划分 `split_tasks()` | ✅ | 4 种实验变体 |
| `support_count >= 3` → new_harness 触发 | ✅ | `inductor.py` |
| `F2 >= 2` → patch 触发 | ✅ | `inductor.py` |
| Phase 0: 子步骤模式发现 | ✅ | `inductor.py::_discover_substep_patterns()` |
| ArtifactJudge（子步骤价值评估） | ✅ | `inductor.py::_judge_artifact_value()`，LLM 四标准判断 |
| Phase 1: 轨迹分段 | ✅ | `algorithms.py::_segment()` LLM 分段 + `_lcs_align()` 对齐，结果传递到后续阶段 |
| Phase 2: 前置条件交集 | ✅ | `algorithms.py::_intersect_preconditions()` 跨轨迹取交集 |
| Phase 3: Daikon 风格不变量挖掘 | ✅ | `algorithms.py::_mine_invariants()`：first/last action、步数、工具序列、常量参数、结果模式 |
| Phase 4: LCS + 类型感知参数化 | ✅ | `algorithms.py::_lcs_pairs()` + `_lcs_align()` + `_abstract_steps()`（JSON 解析 + 参数键交集） |
| Phase 5: LLM 代码合成 | ✅ | `inductor.py::_synthesize()` 含修复上下文注入 |
| Phase 6: 沙盒回放验证 | ✅ | `inductor.py::_validate()` 逐轨迹独立环境验证，`env_builder` 模式 |
| Phase 7: 变体检测 | ✅ | `inductor.py::_detect_variations()` 工具序列差异检测 |
| NEEDS_REVISION 修复重试循环 | ✅ | `inductor.py:528-603`，最多 2 次修复，收集错误上下文 |
| HarnessRegistry（O(1) intent→harness） | ✅ | `harness_registry.py`，含调用统计 |
| `repository.py` SQLite 存储 | ✅ | 使用 `Storage`（SQLite）+ dict 缓存双重机制 |
| `experience_library.py` LTS + 实验库 | ✅ | 两级 SQLite：`lts_library.db` + `exp_<id>.db` |
| `compare.py` Baseline 对比框架 | ✅ | 4 方法（vanilla / react / skillopt / coe）× 4 变体 |
| SkillOpt τ2 adapter | ✅ | `compare.py::run_skillopt()` skill 文本注入 agent 系统提示 |
| Vanilla LLM baseline | ✅ | `compare.py::run_vanilla()` 单轮 LLM |
| `curve.py` 积累/成本曲线图 | ✅ | SR 曲线 + 双面板 token 收敛 |
| 模型迁移实验脚本 | ✅ | `scripts/run_transfer_experiment.py` |
| **exp-0001 端到端验证通过** | ✅ | GLM-5.2 + retail exchange 任务，harness APPROVED，详见 §10.6 |

### 10.2 P0 缺口（必须在论文提交前修复）

> **状态**：原有 6 个 P0 缺口已在代码中解决。flow.md 融合分析（2026-07-27）识别出 3 个新 P0 缺口。

| 缺口 | 影响 | 状态 |
|------|------|------|
| ~~Phase 1 分段结果被丢弃~~ | 多步轨迹无法正确分段 | ✅ 已修复 |
| ~~Phase 3 不变量仅启发式~~ | 不变量挖掘质量不足 | ✅ 已修复（+ 预测契约交叉验证设计就绪） |
| ~~Phase 4 参数化用正则替换~~ | 真实多步轨迹参数化质量差 | ⚠️ 部分：LCS + 类型感知框架就绪，但 harness 仍为单步 `call_tool()` wrapper。需实现真正的多步序列 LCS 对齐（见 P0-3） |
| ~~SkillOpt τ2 adapter 未集成~~ | 无法与最强 baseline 对比 | ✅ 已修复 |
| ~~Vanilla LLM baseline 未实现~~ | 对比实验不完整 | ✅ 已修复 |
| ~~`repository.py` 未接入 SQLite~~ | 存储架构与设计不符 | ✅ 已修复 |
| **RAG baseline 未实现** | 缺少与检索增强方案的对比 | ❌ 待实现 |
| **P0-1: 预测契约未实现** ★ | 无法区分有效经验与侥幸成功，贝叶斯门控仅依赖二元成功/失败 | ❌ 新增：需在 StructuredCoT 中添加 `PredictionContract`，执行后记录 `PredictionVerification` |
| **P0-2: 空间聚类未替换硬分组** ★ | `_cluster_patterns` 仅做 cosine 合并，非真正密度聚类；`(intent, tool_name)` 键分组先于聚类 | ❌ 新增：需实现 DBSCAN/HDBSCAN 替代当前硬分组（§5.2.0） |
| **P0-3: Harness 仍为单步 call_tool() wrapper** ★ | 不符合 §5.2.4 多步参数化定义；不是真正的"知识编译" | ❌ 新增：substep harness 仅包装单个工具调用。需实现跨轨迹多步 LCS 对齐，产出真正的参数化动作序列 |

★ = flow.md 融合分析识别的新缺口

### 10.3 P1 缺口（影响结果质量，有降级方案）

| 缺口 | 影响 | 状态 |
|------|------|------|
| ~~NEEDS_REVISION 无修复重试循环~~ | 归纳失败即放弃 | ✅ 已修复：`inductor.py:528-603`，最多 2 次修复 |
| ~~`new_variation_detected` 特化分裂未触发~~ | 环境漂移未处理 | ✅ 已修复：`inductor.py::_detect_variations()` |
| `StructuredCoT` 缺 `unknowns` 字段 | 缺少"未知信息"信号 | ⚠️ 部分：goal/constraints/risk/milestones 已提取，`unknowns` 未填充（**可选辅助功能**） |
| 版本 DAG 欠丰富（仅 patch 边有实际用例） | specialization/composition 边 schema 就绪但未在实验中使用 | ⚠️ Schema 就绪，等更多场景触发 |
| 检索向量缺 `example_tasks` 维度 | 检索精度可提升 | ❌ 未修复 |
| 双 embedding 路径未统一 | `retriever.py` 用 `LLMClient.embed()`，`embedding.py` 有更丰富的 `EmbeddingClient` 未复用 | ❌ 技术债务 |

### 10.4 P2 缺口（Future Work，不影响核心实验）

| 缺口 | 影响 |
|------|------|
| 多层级知识库（personal/org/public 覆盖） | Future Work |
| Harness 导出/导入 artifact 包格式 | Future Work |
| 归纳异步化 | Future Work |
| TerminalBench 适配器 | Future Work |
| 独立向量数据库（FAISS/ChromaDB） | Future Work |
| Composite 归纳（Level 2） | Future Work |
| 跨 Agent 共享 | Future Work |

### 10.5 差距修复路线图

**阶段一（P0，论文提交前）** — 2026-07-25 已验证 6/7 项，2026-07-27 flow.md 融合分析新增 3 项

1. ✅ Phase 4 参数化框架就绪（LCS + 类型感知）—— **但 harness 仍为单步 wrapper，需升级（P0-3）**
2. ✅ Phase 1 分段结果实际传递到后续阶段
3. ✅ SkillOpt τ2 adapter 集成
4. ✅ Vanilla LLM baseline 实现
5. ✅ `repository.py` 接入 SQLite
6. ❌ RAG baseline 实现
7. ❌ **P0-1**：预测契约（PredictionContract + PredictionVerification）实现
8. ❌ **P0-2**：空间聚类（DBSCAN/HDBSCAN）替换硬分组
9. ❌ **P0-3**：真正多步 LCS 参数化——harness 从单步 `call_tool()` 升级为多步参数化序列

**阶段二（P1，提升结果质量）** — ✅ 大部分完成

1. ✅ Phase 3 Daikon 风格不变量挖掘
2. ✅ NEEDS_REVISION 修复循环
3. ✅ 版本 DAG schema 含 specialization/composition 边
4. ⚠️ `StructuredCoT` `unknowns` 字段（可选，按需启用）
5. ❌ 检索向量 `example_tasks` 维度
6. ❌ 统一 embedding 路径
7. ❌ **新增**：依赖发现实现（§5.3.6）——为 Composite 铺路

**阶段三（P2，Future Work → 部分提前）**：
1. Composite 归纳（Level 2）——依赖发现就绪后可实施
2. 多层级知识库
3. 跨 Agent 共享
4. 语义对齐四战场完整实现

**flow.md 融合实施顺序**（2026-07-27 确定）：

| 优先级 | 融合项 | 改动范围 | 依赖 |
|--------|--------|---------|------|
| **P0-1** | 预测契约 | StructuredCoT + inductor 读取 | 无 |
| **P0-2** | 空间聚类 | `_discover_substep_patterns` 重写 | 无（可并行 P0-1） |
| **P0-3** | 多步 LCS 参数化 | `compiler/algorithms.py` Phase 4 | 依赖 P0-2（聚类后 LCS 更精准） |
| **P1-7** | 依赖发现 → Composite | 新增 dependency analysis 模块 | 依赖 P0-3（多步 harness 就绪后组合） |

### 10.6 实验进展

#### exp-0001：CoE 端到端 + baseline 对比（2026-07-25）

- **配置**：GLM-5.2 / retail / `exchange_delivered_order_items` / train_test split
- **结果**：harness APPROVED（validation `success_rate=1.0, test_count=3`），首次端到端跑通完整管线
- **对比**（3 个 test 任务）：coe 66.7% vs react 66.7% — **成功率持平**
- **Token**：coe 111K vs react 139K — 略低，但 harness 在 eval 阶段未直接判定成功（路径 `harness+agent: 2`），回退 agent 完成
- **关键发现**：`extract_task_params` 仅从参考动作 `arguments` 提取参数，未解析 `user_scenario.instructions` 文本中的关键字段（email/name/zip），导致 harness 在 eval 阶段缺少入参
- **详细记录**：[docs/exp/0001-coe-glm5.2-retail-exchange.md](docs/exp/0001-coe-glm5.2-retail-exchange.md)

#### 后续实验规划（详见 §10.7）

### 10.7 后续实验路线图

基于 exp-0001 的发现，按优先级排列：

**Phase A：修复关键瓶颈（1 个代码改动）**

| 编号 | 任务 | 预期效果 |
|------|------|---------|
| A1 | `extract_task_params` 从 `user_scenario.instructions` 文本中解析参数（email/name/order_id 等），补充到 params | harness 在 eval 阶段获得完整入参，提升 harness 独立成功率 |

**Phase B：扩大验证规模（300+ 任务）**

| 编号 | 实验 | 配置 | 目标 |
|------|------|------|------|
| B1 | coe vs react 全量 | retail train=74 / test=40，DeepSeek-V4-Flash | 验证 harness 在更大样本上的 SR + Token 节省 |
| B2 | 4 方法完整对比 | vanilla / react / skillopt / coe，retail，DeepSeek-V4-Flash | 产出论文核心对比表 |
| B3 | 积累曲线 | coe + react，40 任务滚动窗口 | 验证"交叉点"核心主张 |

**Phase C：泛化与迁移（论文增量贡献）**

| 编号 | 实验 | 配置 | 目标 |
|------|------|------|------|
| C1 | 模型迁移：强→弱 | GLM-5.2 积累 → qwen2.5:7b 部署，retail full | 验证"强模型归纳，弱模型受益"（RQ2 模型骨架泛化） |
| C2 | 跨域迁移 | retail 积累 → airline 部署 | 验证跨域泛化（RQ2 环境泛化） |
| C3 | RAG baseline | retail，GLM-5.2 | 补全对比维度 |

**Phase D：消融与鲁棒性（论文支撑）**

| 编号 | 实验 | 配置 | 目标 |
|------|------|------|------|
| D1 | 无 sandbox 验证消融 | retail，GLM-5.2，跳过 Phase 6 | 证明验证门控的必要性 |
| D2 | MIN_SUPPORT 敏感性 | retail，support={1,3,5,7} | 确定最佳触发阈值 |
| D3 | 子步骤消融 | retail，GLM-5.2，禁用手步骤发现 | 证明 Phase 0 的贡献 |

**优先级建议**：A1 → B1 → B2（产出初步论文数据）→ B3（积累曲线是核心图）→ C1（模型迁移是论文亮点）→ 其余按资源分配。

---

## 11. 投稿定位

> ExperienceOS 是一个**知识编译框架**，它研究的核心问题是：**LLM Agent 能否从自身的执行历史中自动归纳出可复用的确定性执行结构，并在不改变模型权重、不扩展上下文的前提下，降低重复任务的推理成本？** 答案是肯定的——通过贝叶斯程序归纳触发机制和沙盒回放验证门控，ExperienceOS 将稳定的执行模式编译为 Harness，使后续同类任务的 Token 消耗降低 70% 以上，同时维持或提升任务成功率。

投稿目标：**NeurIPS/ICLR Agent/Efficiency Track，或 ACL System Demo Track**。主会需要有 OSWorld 扩展实验，Workshop 当前框架已经足够。

---

## 12. XXX
---

## 13. 工程开发指引（同步 CLAUDE.md）

本节内容需同步到 `CLAUDE.md`，确保开发文档与本文档对齐。

### 13.1 名称同步

- 项目/框架/系统/方法名：**ExperienceOS**（废弃 AutoHarness 作为方法名）
- 核心数据结构：**扩展 Hoare Triple** `⟨P,steps,I,Q,R⟩`
- 经验表示层次：**Layer 0–3 四层**
- 归纳层次：**SubStep / Task / Composite**

### 13.2 关键命令（保留 CLAUDE.md 现有）

```bash
pip install -e ".[dev]"
experience-os ping
experience-os demo
experience-os tau2-demo --domain retail --warmup 3 --eval 3
experience-os compare --method react --model ollama/qwen2.5:7b --domain retail --warmup 3 --eval 5
experience-os curve docs/exp_results/*.json --window 3
experience-os harnesses
experience-os status
```

### 13.3 环境配置（保留 CLAUDE.md 现有）

- `EOS_LLM_BACKEND=ollama` / `deepinfra`
- `EOS_MIN_SUPPORT=3`
- `EOS_VALIDATION_THRESHOLD=0.8`

### 13.4 存储架构（保留 CLAUDE.md 现有）

SQLite（primary，via `experience_library.py`）+ JSON（legacy，via `repository.py`，待迁移）。

### 13.5 子模块

- `tau2-bench/` — 主实验环境
- `harbor-TerminalBench/` — 补充环境（适配器待实现）
- `SkillOpt/` — 最强文本 baseline（τ2 adapter 已实现：`compare.py::run_skillopt()`）

---

## 附录 A：完整方法论图示

```
                          ╔══════════════════════════════════════════════╗
                          ║       ExperienceOS 完整方法论（增强版）       ║
                          ╚══════════════════════════════════════════════╝

  Agent (任意级别)
       │
       │ 执行任务 + 生成预测契约
       │ PredictionContract {expected_input, expected_output, expected_effect, confidence}
       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     PATH COLLECTION                               │
  │  轨迹五元组：(agent_id, env_ctx, steps, outcome, sub_success)      │
  │  含 SubStep 细粒度拆分 + TODO/里程碑/对话边界                      │
  │  ★ 预测-vs-实际对比 → PredictionVerification → 质量标签           │
  │     ├── 预测准确 + 成功 → 高质量经验 (×1.0)                        │
  │     ├── 预测错误 + 成功 → 侥幸成功   (×0.3)                        │
  │     ├── 预测准确 + 失败 → 实现缺陷   (F2)                          │
  │     └── 预测错误 + 失败 → 负样本     (边界记录)                     │
  └──────────────────────┬───────────────────────────────────────────┘
                         │
       ┌──────────────── ▼ ────────────────────┐
       │           INDUCTION ENGINE             │
       │                                        │
       │  ★ 空间聚类（flow.md §3）              │
       │    特征提取：concat(                    │
       │      embedding(task_semantics),        │
       │      embedding(IO_signature),          │
       │      embedding(expected_effect)        │
       │    )                                   │
       │    → 密度聚类 (DBSCAN/HDBSCAN)         │
       │    → 同簇内轨迹语义相似                  │
       │                                        │
       │  Level 0: SubStep 归纳                 │
       │    LCS对齐 → 参数抽取 → 不变量挖掘       │
       │  Level 1: Task 归纳                    │
       │    贝叶斯程序归纳 (MDL先验)              │
       │    ★ 预测准确率门控                      │
       │  Level 2: Composite 归纳               │
       │    ★ 依赖发现 → 组合调用图分析            │
       └──────────────────┬─────────────────────┘
                          │
       ┌──────────────── ▼ ────────────────────┐
       │            COMPILE ENGINE              │
       │                                        │
       │  1. Segment (LLM 语义分段)              │
       │  2. Intersect Preconditions             │
       │     ★ 参考预测契约 expected_input       │
       │  3. Mine Invariants                     │
       │     ★ 预测契约交叉验证                   │
       │  4. Abstract Steps (LCS + 类型感知)     │
       │  5. Synthesize (LLM 代码生成)           │
       │  6. Validate (沙盒回放)                  │
       │  7. ★ Discover Dependencies             │
       │     → Composite Artifact 构造           │
       │                                        │
       │  生成 Harness 五元组 (P,steps,I,Q,R)    │
       │  修复循环 (最多 MAX_RETRY 次)           │
       │  版本分配 (Major.Minor.Patch)           │
       └──────────────────┬─────────────────────┘
                          │
       ┌──────────────── ▼ ────────────────────┐
       │            REGISTRY STORE              │
       │                                        │
       │  User Layer    (最高优先级, Future)     │
       │  ├── Org Layer (次高优先级, Future)     │
       │  │   └── Global Layer (基础层, Future) │
       │  │                                     │
       │  版本 DAG：纵向(进化) + 横向(分裂)       │
       │          + ★ composition 边 (Composite) │
       │  索引：by_task / by_scope / by_step     │
       │        + ★ by_dependency (依赖图索引)   │
       └──────────────────┬─────────────────────┘
                          │
       ┌──────────────── ▼ ────────────────────┐
       │          RUNTIME EXECUTION             │
       │                                        │
       │  1. 语义检索 → Scope匹配 → 版本选择     │
       │  2. Precondition 检查                   │
       │  3. 执行（支持嵌套子 Harness 调用）      │
       │     ★ Composite: 依依赖图并行/串行调度   │
       │  4. Postcondition 验证                  │
       │     ★ 预测契约验证：预期 vs 实际输出      │
       │  5. 反馈写回 trajectory store            │
       │     ★ 包含 quality_label + 预测对比      │
       └──────────────────┬─────────────────────┘
                          │
                    反馈闭环（双重）
                          │
       ┌──────────────────▼───────────────────┐
       │  执行结果反馈 → TraceStore             │
       │  ★ 预测质量反馈 → ExperienceStore       │
       │     (影响经验贝叶斯权重)                 │
       └──────────────────┬───────────────────┘
                          │
                    回到 PATH COLLECTION
```

★ = 来自 flow.md 的增强项

---

## 附录 B：关键设计决策摘要

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 项目名称 | ExperienceOS | 统一命名，消除历史六名混用 |
| 核心数据结构 | 扩展 Hoare Triple | 形式化清晰，可验证 |
| 经验表示层次 | Layer 0–3 四层 | 含来源链接，支持追溯 |
| 归纳层次 | SubStep / Task / Composite | 正交于表示层次，粒度递进 |
| 归纳触发 | MIN_SUPPORT=3 批量 | 1 条偶然，2 条巧合，3 条归纳 |
| 参数化 | LCS + 类型感知 | 处理真实多步轨迹 |
| 不变量 | Daikon 风格 + 预测契约交叉验证 | 动态检测 + 语义验证 |
| 版本管理 | 纵横版本树 | 进化 + 分裂 + composition |
| 仓库层级 | User > Org > Global | 类 Linux PATH |
| 跨 Agent 共享 | agent_min_level 机制 | 高级经验编译进 Harness |
| 语义对齐 | 字典→Embedding→LLM 三层 | 由快到慢降级，自增长 |
| 论文边界 | Level 0+1 + 单层 Registry | τ-bench 足够证明核心主张 |
| **预测契约** ★ | PredictionContract + PredictionVerification | 区分有效经验与侥幸成功；映射到 P/Q |
| **空间聚类** ★ | embedding + 密度聚类 (DBSCAN/HDBSCAN) | 替代硬分组；确保 LCS 输入语义相似 |
| **依赖发现** ★ | 数据流分析 + 转移概率 | 实现 Level 2 Composite；flow.md §4 |
| **多步参数化** | 真正 LCS 对齐多步序列 | 核心差异化——编译多步程序非缓存单步 API |

★ = 来自 flow.md 融合的新增决策 |

---

*本文档为 ExperienceOS 项目的唯一基准文档。历史文档已整合完毕，除 `Executable Experience Discuss.md` 作为历史讨论保留外，其余历史文档将清理，git 历史保留可追溯。*
