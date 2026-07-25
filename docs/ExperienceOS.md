# ExperienceOS：跨 Agent 跨环境的知识编译与共享运行时

> 本文档是 ExperienceOS 项目的**唯一基准文档**，整合自历史文档（OVERALL.md / STRUCTURE.md / Executable Experience RP.md / Executable Experience Discuss.md）。
> 当本文档与其他文档存在冲突时，**一律以本文档为准**。
> 历史文档（除 `Executable Experience Discuss.md` 作为历史讨论保留外）将在整合完成后清理，git 历史保留可追溯。

---

## 0. 文档定位与基准声明

| 历史文档 | 状态 | 说明 |
|---------|------|------|
| `OVERALL.md` | **已整合，待删除** | 内容已纳入本文档，git 历史存档 |
| `STRUCTURE.md` | **已整合，待删除** | 实现差距部分纳入 §10，git 历史存档 |
| `Executable Experience RP.md` | **已整合，待删除** | 研究提案部分纳入本文档，git 历史存档 |
| `Executable Experience Discuss.md` | **保留为历史讨论** | 保留早期方向推演过程，但**不作为实施依据**，仅作历史参考 |
| `CLAUDE.md` | **待同步更新** | 工程开发指引，名称/数据结构需与本文档对齐 |

**关键收敛决策**（解决历史文档冲突）：

1. **项目名称**：统一为 **ExperienceOS**。框架名、系统名、论文方法名均使用 ExperienceOS，废弃 "AutoHarness" 作为系统/方法名的历史用法（在历史文档中保留可追溯）。
2. **核心数据结构**：统一采用**扩展 Hoare Triple** `H = ⟨P, steps, I, Q, R⟩` 作为 Harness 的形式化定义。废弃并行的 Artifact Object Schema yaml 与 Experience Object `E=(C,S,A,V,P)` 两套定义。
3. **"层次"概念**：本文档明确区分**经验表示层次**（Layer 0–3 数据结构）与**归纳层次**（SubStep / Task / Composite 触发归纳的对象粒度）两个正交维度，详见 §3。
4. **论文边界**：当前论文只证明 Level 0+1 归纳 + 单层 Registry + 单层运行时在 τ-bench 上"Token 成本显著下降，成功率不降"。多 Agent、跨环境、Composite 归纳、纵横版本树、User/Org/Global 三层覆盖等作为 Future Work。

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

整个系统是一个**持续进化的闭环**，五个核心阶段首尾相连：

```
┌─────────────────────────────────────────────────────────────────┐
│                    ExperienceOS 闭环系统                         │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  PATH    │───▶│INDUCTION │───▶│COMPILE   │───▶│REGISTRY  │  │
│  │ COLLECT  │    │  ENGINE  │    │  ENGINE  │    │  STORE   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲                                                │        │
│       │              ┌──────────┐                     │        │
│       └──────────────│ RUNTIME  │◀────────────────────┘        │
│                      │EXECUTION │                               │
│                      └──────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

每阶段职责严格分离，详见 §5。

---

## 5. 五阶段详细设计

### 5.1 阶段一：路径轨迹收集（Path Collection）

详见 §3.2 Layer 0 数据结构。核心要点：

- **失败轨迹也要收集**：用于挖掘部分成功子步骤、负样本学习、诊断适用范围边界
- **SubStep 细粒度拆分**：完整路径内部按 TODO list / 里程碑 / 对话边界拆分，支持子步骤级归纳
- **嵌套追踪**：执行中调用了哪些 Harness 通过 `source_harness_ids` 记录
- **状态快照**：每步前后环境状态哈希，供不变量挖掘

### 5.2 阶段二：经验归纳引擎（Induction Engine）

#### 5.2.1 归纳触发条件（批量触发）

```python
class InductionTrigger:
    MIN_SUPPORT: int = 3            # 同类成功轨迹至少 3 条
    MAX_VARIANCE_THRESHOLD: float   # 步骤序列编辑距离方差上限
    SUBSTEP_MIN_SUPPORT: int = 2    # 子步骤归纳更低门槛
    F2_PATCH_THRESHOLD: int = 2     # F2 失败累计 ≥2 触发 Patch
```

触发事件：
- `support_count >= MIN_SUPPORT` → 新建 Harness
- `new_variation_detected` → 特化分裂（Future Work）
- `F2 count >= 2` → Patch

#### 5.2.2 归纳层次（维度 B）

```
Level 0: SubStep 归纳
  输入：同一任务中反复出现的局部步骤序列
  输出：SubStep Harness（最小可复用单元）
  示例："截图→角点检测→鼠标定位" 编译为 VisualLocator v1.0

Level 1: Task 归纳
  输入：完整任务成功轨迹（MIN_SUPPORT 条）
  输出：Task Harness（完整流程的参数化程序）
  示例："发送邮件" 编译为 EmailSender v1.0

Level 2: Composite 归纳 (Future Work)
  输入：多个 Task Harness 组合使用的轨迹
  输出：Composite Harness（调用子 Harness 的元程序）
  示例："每日报告发送" 编译为 DailyReporter v1.0
        内部调用 EmailSender v1.0 + ReportGenerator v1.0
```

> **论文边界**：当前论文只实现 Level 0 + Level 1。

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

多条轨迹在**规范化步骤序列**（见 §6）上做 LCS 对齐：

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

#### 5.2.5 不变量挖掘（Daikon 风格）

受 Daikon 动态不变量检测启发，对每条成功轨迹的状态快照序列挖掘持续成立的谓词：

```python
class InvariantMiner:
    def mine(self, trajectories: List[Trajectory]) -> List[Invariant]:
        # 收集所有步骤前后的状态哈希序列
        # 数据驱动找到"每次成功都成立"的谓词
        # 示例不变量：
        #   - 执行前 network_connected == True
        #   - 执行中 modal_open == False
        #   - 执行后 email_count 增加 1
```

### 5.3 阶段三：编译引擎（Compile Engine）

#### 5.3.1 六阶段编译流水线

1. **Segment**（轨迹分段）—— LLM 识别语义边界（>3 步时）。**实现要求**：分段结果必须传递到后续阶段，不能丢弃。
2. **Intersect Preconditions**（前后置条件提取）—— 跨轨迹集合交集
3. **Mine Invariants**（不变量挖掘）—— Daikon 风格动态不变量检测（非简单"首步一致+全成功"启发式）
4. **Abstract Steps**（步骤抽象与参数化）—— **LCS + 类型感知参数化**（非正则替换）
5. **Synthesize**（Harness 合成）—— LLM 生成 `run()` 函数，从 few-shot 例子学习 `call_tool()` API
6. **Validate**（沙盒回放验证）—— `success_rate >= validation_threshold`（默认 0.8）方入库。结果：APPROVED / NEEDS_REVISION / REJECTED

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

#### 5.5.3 执行后反馈闭环

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

class FeedbackProcessor:
    def process(self, feedback: ExecutionFeedback):
        # 1. 更新 Harness 统计（success_rate, usage_count）
        # 2. 若连续失败 > 阈值 → 触发 NEEDS_REVISION 标记
        # 3. 将反馈写入 trajectory store（作为新轨迹数据）
        # 4. 触发增量归纳（检查是否需要分裂或版本更新）
```

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

> **Level 0 + Level 1 归纳 + 基础单层 Registry + 单层运行时，已足够在 τ-bench 上证明"Token 成本显著下降，成功率不降"。**

| 模块 | 当前论文实现 | Future Work |
|------|------------|------------|
| 轨迹收集 | 单 Agent，τ-bench 环境 | 多 Agent，多环境，跨平台 |
| 归纳引擎 | Level 0 + Level 1 | Level 2（Composite） |
| 编译引擎 | 参数化 + 沙盒验证 | 自动修复循环，scope 自动分裂 |
| 版本管理 | 单版本线（patch 边） | 纵横版本树（specialization / composition 边） |
| 仓库结构 | 单层平铺 | User / Org / Global 三层覆盖 |
| 跨 Agent 共享 | 同 Agent 复用 | 能力等级降维共享 |
| 运行时 | 查询 + 执行 | 嵌套执行，反馈驱动更新 |
| 语义对齐 | 单 Agent 同工具集 | 跨 Agent 工具名规范化（四战场） |

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

### 10.1 已实现模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `models.py` Hoare Triple | ✅ | `H=<P,steps,I,Q,R>` 完整 |
| `storage.py` SQLite + 向量 BLOB | ✅ | 含 JSON 迁移 |
| `embedding.py` 三级回退 | ✅ | Qwen3-Embedding-8B → ollama → hash |
| `env_info.py` 环境元数据收集 | ✅ | OS/Python/硬件/模型/包版本 |
| `agent.py` F1-F4 失败分类 | ✅ | + Agent Fallback |
| `runtime.py` 双模式 | ✅ | ACCUMULATION / DEPLOYMENT |
| τ-bench 集成 | ✅ | 仿真 + 轨迹转换 + DB hash 验证 |
| Warmup/Eval 数据划分 `split_tasks()` | ✅ | |
| `support_count >= 3` 触发 new_harness | ✅ | |
| `F2 >= 2` 触发 patch | ✅ | |
| Phase 5 LLM 合成 + Phase 6 沙盒回放 | ✅ | |
| 子步骤提取/模式发现/ArtifactJudge | ⚠️ | 已实现但端到端验证待测试 |
| `compare.py` Baseline 对比框架 | ✅ | GLM 5.2 验证通过 |
| `curve.py` 积累曲线图 | ✅ | |
| `experience_library.py` LTS + 实验库 | ✅ | |

### 10.2 P0 缺口（必须在论文提交前修复）

| 缺口 | 影响 |
|------|------|
| Phase 1 分段结果被丢弃（compiler.py） | 多步轨迹无法正确分段 |
| Phase 3 不变量仅"首步一致+全成功"启发式 | 不变量挖掘质量不足 |
| Phase 4 参数化用正则替换，非 LCS | 真实多步轨迹参数化质量差 |
| SkillOpt τ2 adapter 未集成 | 无法与最强 baseline 对比 |
| Vanilla LLM / RAG baseline 未实现 | 对比实验不完整 |
| `repository.py` 未真正接入 SQLite（仅 JSON） | 存储架构与设计不符 |

### 10.3 P1 缺口（影响结果质量，有降级方案）

| 缺口 | 影响 |
|------|------|
| NEEDS_REVISION 无修复重试循环 | 归纳失败即放弃 |
| `new_variation_detected` 特化分裂未触发 | 环境漂移未处理 |
| `StructuredCoT` 仅填 `goal` 字段 | 缺少 constraint/unknown/risk 信号 |
| 版本 DAG 缺 `specialization` / `composition` 边 | 版本管理不完整 |
| 检索向量缺 `example_tasks` 维度 | 检索质量可提升 |

### 10.4 P2 缺口（Future Work，不影响核心实验）

| 缺口 | 影响 |
|------|------|
| 多层级知识库（personal/org/public 覆盖） | Future Work |
| Harness 导出/导入 artifact 包格式 | Future Work |
| 归纳异步化 | Future Work |
| TerminalBench 适配器 | Future Work |
| 独立向量数据库（FAISS/ChromaDB） | Future Work |

### 10.5 差距修复路线图

**阶段一（P0，论文提交前）**：
1. Phase 4 参数化：正则 → LCS + 类型感知
2. Phase 1 分段结果实际传递到后续阶段
3. SkillOpt τ2 adapter 集成
4. Vanilla LLM baseline 实现
5. `repository.py` 接入 SQLite

**阶段二（P1，提升结果质量）**：
1. Phase 3 Daikon 风格不变量挖掘
2. NEEDS_REVISION 修复循环
3. 版本 DAG 补 `specialization` 边
4. `StructuredCoT` 完整字段填充

**阶段三（P2，Future Work）**：
1. Composite 归纳
2. 多层级知识库
3. 跨 Agent 共享
4. 语义对齐四战场完整实现

---

## 11. 投稿定位

> ExperienceOS 是一个**知识编译框架**，它研究的核心问题是：**LLM Agent 能否从自身的执行历史中自动归纳出可复用的确定性执行结构，并在不改变模型权重、不扩展上下文的前提下，降低重复任务的推理成本？** 答案是肯定的——通过贝叶斯程序归纳触发机制和沙盒回放验证门控，ExperienceOS 将稳定的执行模式编译为 Harness，使后续同类任务的 Token 消耗降低 70% 以上，同时维持或提升任务成功率。

投稿目标：**NeurIPS/ICLR Agent/Efficiency Track，或 ACL System Demo Track**。主会需要有 OSWorld 扩展实验，Workshop 当前框架已经足够。

---

## 12. 历史文档冲突标记与处理记录

本节记录整合过程中识别的历史文档冲突及处理方式，供追溯。

### 12.1 项目名称冲突

| 文档 | 用名 | 处理 |
|------|------|------|
| OVERALL.md | AutoHarness | 统一为 ExperienceOS |
| STRUCTURE.md | ExperienceOS（框架）+ AutoHarness（方法）| 统一为 ExperienceOS |
| RP.md 上半 | AutoHarness | 统一为 ExperienceOS |
| RP.md 下半 | ExperienceOS | 采用 |
| Discuss.md | AutoHarness / ExperienceOS / EOS / Experience Compiler / Experience Runtime / Persistent Cognitive Runtime / Executable Experience Repository（六名混用）| 统一为 ExperienceOS |
| CLAUDE.md | ExperienceOS（框架）+ AutoHarness（方法）| 统一为 ExperienceOS |

### 12.2 核心数据结构冲突

| 文档 | 定义 | 处理 |
|------|------|------|
| OVERALL.md / RP.md 上半 / CLAUDE.md | 扩展 Hoare Triple `⟨P,steps,I,Q,R⟩` | **采用** |
| RP.md 下半 | Artifact Object Schema yaml | 废弃 |
| Discuss.md | Experience Object `E=(C,S,A,V,P)` | 废弃 |

### 12.3 "层次"概念冲突

| 文档 | "三层次"指代 | 处理 |
|------|------------|------|
| OVERALL.md | 归纳层次 SubStep / Task / Composite | **采纳为维度 B** |
| RP.md 上半 | 经验表示层次 Layer 0/1/2 | 扩展为四层（维度 A）|
| RP.md 下半 | 经验表示层次 Layer 1–4 | **采纳为维度 A** |
| Discuss.md | 三层（Embedding/Graph/Artifact）+ 四层（Raw/Graph/Artifact/Meta）并存 | 统一为四层 |

### 12.4 论文边界冲突

| 文档 | 边界 | 处理 |
|------|------|------|
| OVERALL.md | Level 0+1 + 单层 Registry + 单层运行时 | **采纳** |
| RP.md 下半 | T-Bench + SWE-Bench + OSWorld + WorkArena 多环境 | 归入 Future Work |
| Discuss.md | 多 benchmark、多层表示、多层迁移 | 归入 Future Work |

### 12.5 归纳触发条件冲突

| 文档 | 触发条件 | 处理 |
|------|---------|------|
| OVERALL.md / RP.md 上半 / Discuss.md | MIN_SUPPORT=3 批量触发 | **采纳** |
| RP.md 下半 | n=50 周期触发 + Bayesian τ/δ | 废弃 |

### 12.6 STRUCTURE.md 内部混乱（待清理）

| 混乱点 | 处理 |
|--------|------|
| §3.5a–§3.5f 与 §3.5 章节编号重复 | 整合到本文档 §5/§8，不再使用原编号 |
| §3.5a "已实现各组件但端到端验证 ❌" 自相矛盾 | 标记为 ⚠️ |
| Phase 1 三处描述不一致（退步/占位/占位未生效）| 统一为"占位未生效" |
| SQLite 主存储地位前后矛盾（§3 标 ✅ / §7 标未接入）| 统一为"storage.py 已实现，repository.py 未接入" |
| §3.5 与 §6.1 Baseline 列表不一致（含 RAG vs 含 SkillOpt）| 采纳 §6.1 四档（含 SkillOpt）|
| §4 缺口表"已解决"标记与 ❌ 混排 | 整合到本文档 §10 |
| §3.5d 与 §6.4 积累范围变体重复 | 整合到本文档 §8.3 |
| §3.5c/§3.5e/§6.1 模式代号 A/B/C 冲突 | 整合到本文档 §8，统一命名 |
| §3.4 与 §3.5a 触发机制兼容性未说明 | 整合到本文档 §5.2.1，统一为批量触发 |

### 12.7 RP.md 内部混乱（待清理）

| 混乱点 | 处理 |
|--------|------|
| 2.3 经验三层结构与 OVERALL 归纳三层次概念冲突 | 已在 §3.1 区分两个维度 |
| 4.4 泛化性分析与文末注自相矛盾 | 整合到本文档 §8.3 作为可选泛化实验 |
| RP 下半 3.1 架构图与 3.3 经验表示层次不对齐 | 统一为本文档 §3 四层结构 |
| RP 下半 4.2 五种 Split 与 §8 贡献"四种"不一致 | 整合到本文档 §8.3，保留 type_split/cross_type/cross_domain/replay 四种 |

### 12.8 Discuss.md 内部混乱（保留为历史讨论，不清理）

| 混乱点 | 处理 |
|--------|------|
| 项目名称六种混用 | 保留原文，本文档统一为 ExperienceOS |
| 经验层次三套并存 | 保留原文，本文档统一为四层 |
| "Harness 不是系统而是产物" vs "AutoHarness 才是中心" | 保留原文，本文档明确 Harness 是产物，ExperienceOS 是系统 |
| Experience Object 定义与 yaml schema 不对应 | 保留原文，本文档统一为扩展 Hoare Triple |
| F2 触发阈值 2 次 vs 3 次差异 | 保留原文，本文档统一为 F2 ≥ 2 |

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
- `SkillOpt/` — 最强文本 baseline（τ2 adapter 待实现）

---

## 附录 A：完整方法论图示

```
                          ╔═══════════════════════════════════╗
                          ║     ExperienceOS 完整方法论        ║
                          ╚═══════════════════════════════════╝

  Agent (任意级别)
       │
       │ 执行任务
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    PATH COLLECTION                          │
  │  轨迹五元组：(agent_id, env_ctx, steps, outcome, sub_success)│
  │  含 SubStep 细粒度拆分 + TODO/里程碑/对话边界                │
  └──────────────────────┬──────────────────────────────────────┘
                         │
       ┌──────────────── ▼ ──────────────────┐
       │           INDUCTION ENGINE           │
       │                                      │
       │  Level 0: SubStep 归纳               │
       │    LCS对齐 → 参数抽取 → 不变量挖掘    │
       │  Level 1: Task 归纳                  │
       │    贝叶斯程序归纳 (MDL先验)           │
       │  Level 2: Composite 归纳 (Future)    │
       │    组合调用图分析                     │
       └──────────────────┬───────────────────┘
                          │
       ┌──────────────── ▼ ──────────────────┐
       │            COMPILE ENGINE            │
       │                                      │
       │  生成 Harness 五元组 (P,steps,I,Q,R) │
       │  沙盒验证 → 自动修复循环             │
       │  版本分配 (Major.Minor.Patch)        │
       │  Scope 绑定 & 分裂决策              │
       └──────────────────┬───────────────────┘
                          │
       ┌──────────────── ▼ ──────────────────┐
       │            REGISTRY STORE            │
       │                                      │
       │  User Layer    (最高优先级, Future)  │
       │  ├── Org Layer (次高优先级, Future)  │
       │  │   └── Global Layer (基础层, Future)│
       │  │                                   │
       │  版本树：纵向(进化) + 横向(分裂)      │
       │  索引：by_task / by_scope / by_step  │
       └──────────────────┬───────────────────┘
                          │
       ┌──────────────── ▼ ──────────────────┐
       │          RUNTIME EXECUTION           │
       │                                      │
       │  1. 语义检索 → Scope匹配 → 版本选择  │
       │  2. Precondition 检查                │
       │  3. 执行（支持嵌套子 Harness 调用）  │
       │  4. Postcondition 验证               │
       │  5. 反馈写回 trajectory store        │
       └──────────────────┬───────────────────┘
                          │
                    反馈闭环
                          │
       └──────────────────▼───────────────────┘
                  回到 PATH COLLECTION
```

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
| 不变量 | Daikon 风格 | 动态检测，非启发式 |
| 版本管理 | 纵横版本树 | 进化 + 分裂 |
| 仓库层级 | User > Org > Global | 类 Linux PATH |
| 跨 Agent 共享 | agent_min_level 机制 | 高级经验编译进 Harness |
| 语义对齐 | 字典→Embedding→LLM 三层 | 由快到慢降级，自增长 |
| 论文边界 | Level 0+1 + 单层 Registry | τ-bench 足够证明核心主张 |

---

*本文档为 ExperienceOS 项目的唯一基准文档。历史文档已整合完毕，除 `Executable Experience Discuss.md` 作为历史讨论保留外，其余历史文档将清理，git 历史保留可追溯。*
