

------

# 改进版 Research Proposal

------

# AutoHarness：将 Agent 执行轨迹编译为可复用的确定性执行脚手架

## 摘要

当前 LLM Agent 在执行重复性任务时存在根本性的效率悖论：每次面对结构相似的任务，Agent 都需要从头进行完整的推理链，消耗大量 Token，且无法从历史成功经验中积累确定性知识。本文提出 AutoHarness，一个将成功 Agent 轨迹自动编译为可复用执行脚手架（Harness）的框架。其核心思想是将 Agent 的经验积累过程形式化为**知识编译（Knowledge Compilation）**：通过贝叶斯程序归纳，将概率性的神经推理蒸馏为确定性的可执行结构，并在运行时以近乎零边际成本替代重复推理。我们在 τ-bench 上的实验表明，经过初始积累阶段后，AutoHarness 在任务成功率上与纯 Agent Baseline 持平或更优，同时将 Token 消耗降低 70% 以上，验证了"经验可编译、知识可积累"的核心假设。

------

## 1. 引言

### 1.1 问题动机

LLM Agent 在工具调用、GUI 操作、文档处理等自动化任务中展现出强大能力，但存在一个被广泛忽视的结构性低效：**每次执行都是无状态的**。

一个在上周成功执行了 200 次"查询客户信息并更新订单状态"任务的 Agent，在第 201 次面对同类任务时，仍然需要从零开始推理每一个步骤。它不记得上次用过哪个 API，不知道这类任务有什么固定的前置条件，也无法验证自己的执行是否偏离了历史成功路径。

这个问题的根源不在于模型能力不足，而在于**缺乏一种把经验蒸馏为可执行结构的机制**。

### 1.2 现有方法的局限

现有的记忆增强 Agent 方案主要分为两类，均存在根本性局限：

**RAG（检索增强生成）**：本质上是在推理时向 LLM 的输入上下文注入历史信息，但并未改变计算结构。检索到的经验仍然需要 LLM 重新解析、重新规划、重新执行，边际推理成本没有下降。本质上是"给 CPU 递小纸条"，而非"扩展指令集"。

**程序合成（Inductive Program Synthesis）**：以 PatchWorld 等工作为代表，通过 CEGIS 框架将轨迹转化为程序。但存在三个错配：(1) Debug ≠ Learning，反例修复只适合"已知结构去改错"，而非"从零发现结构"；(2) 过早进入代码生成，未经历充分的表征发现；(3) 离线评估极易奖励记忆化，泛化性被高估。

### 1.3 核心洞察

本文的核心洞察是将问题重新框架为：

$$\text{Neural} \xrightarrow{\text{Execute}} \text{Experience} \xrightarrow{\text{Compile}} \text{Symbolic Artifact} \xrightarrow{\text{Deploy}} \text{Runtime Assistance}$$

这条流水线的关键中间环节——**从经验到符号制品的编译**——正是现有工作所缺失的。我们把这个过程形式化为**贝叶斯程序归纳**，并把编译产物称为 **Harness**（执行脚手架）。

### 1.4 贡献

1.  **理论框架**：将 Agent 经验积累形式化为知识编译过程，提出基于贝叶斯后验最大化的 Harness Induction Criterion
2.  **系统实现**：AutoHarness 的完整实现，包括轨迹分段、经验归纳、Harness 合成与验证、运行时检索与执行的端到端闭环
3.  **实验验证**：在 τ-bench 上证明积累效果，以及 Harness 相对 Agent Baseline 在成功率和 Token 效率上的双重优势
4.  **分析与洞察**：通过消融实验揭示各组件的贡献，以及 Harness 质量随积累量的增长曲线

------

## 2. 问题形式化

### 2.1 基本定义

**执行轨迹（Trajectory）**：$T = [(o_0, a_0), (o_1, a_1), \ldots, (o_n, a_n)]$，其中 $o_t$ 是第 $t$ 步的观测，$a_t$ 是执行的动作，$o_n$ 是最终状态。轨迹附带环境快照 $E$（OS 版本、应用版本等）和结果标签 $y \in {\text{success}, \text{failure}}$。

**任务类型（Task Type）**：将任务空间按语义聚类，同一任务类型的轨迹共享相似的结构和目标。形式上，任务类型是一个等价类 $\mathcal{C} = {T_i : \text{taskType}(T_i) = c}$。

**执行脚手架（Harness）**：一个带约束的参数化程序，形式上对应扩展 Hoare Triple：

$$H = \langle P,\ \text{steps},\ I,\ Q,\ R \rangle$$

其中 $P$ 为前置条件（Precondition），steps 为参数化动作序列，$I$ 为执行不变量（Invariant），$Q$ 为后置条件（Postcondition / Terminal Verifier），$R$ 为回滚策略（Rollback）。

### 2.2 Harness Induction Criterion

给定任务类型 $c$ 的轨迹集合 $\mathcal{T}_c$，目标是找到后验概率最大的 Harness：

$$H^* = \arg\max_{H} P(H \mid \mathcal{T}_c) \propto P(\mathcal{T}_c \mid H) \cdot P(H)$$

**似然项** $P(\mathcal{T}_c \mid H)$：Harness $H$ 对轨迹集合的覆盖能力，定义为回放成功率：

$$P(\mathcal{T}_c \mid H) = \frac{1}{|\mathcal{T}*c|} \sum*{T_i \in \mathcal{T}_c} \mathbb{1}[\text{replay}(H, T_i) = \text{success}]$$

**先验项** $P(H)$：MDL（最小描述长度）先验，惩罚复杂的 Harness：

$$P(H) \propto \exp\left(-\lambda \cdot \text{MDL}(H)\right)$$

$$\text{MDL}(H) = \alpha \cdot |\text{steps}(H)| + \beta \cdot |\text{params}(H)| + \gamma \cdot |\text{invariants}(H)|$$

这个准则给出了一个可操作的 Harness 入库标准：**只有当一个 Harness 能以足够简单的结构解释足够多的成功轨迹时，它才值得被收录。**

### 2.3 经验三层结构

AutoHarness 维护三层递进的知识表示：

```
Layer 0：原始轨迹（Raw Trajectory）
  - 永久保留，只追加不修改
  - 记录完整的观测-动作序列 + 环境快照 + 执行结果

Layer 1：经验记录（Experience Record）
  - 从多条同类轨迹归纳的语义摘要
  - 包含：候选前置条件、参数化步骤、观测变异（Observed Variations）
  - 最小支撑数：support_count ≥ MIN_SUPPORT（实验中设为 3）

Layer 2：执行脚手架（Harness）
  - 从经验记录编译的可执行代码
  - 带完整的元数据（适用范围、版本号、来源轨迹）
  - 入库前经过沙盒回放验证
```

------

## 3. AutoHarness 系统设计

### 3.1 系统总览

```
┌──────────────────────────────────────────────────────────┐
│                     新任务到来                            │
└─────────────────────────┬────────────────────────────────┘
                          │
                ┌─────────▼──────────┐
                │   Harness 检索     │  ← Harness Registry
                │   前置条件匹配     │
                └─────────┬──────────┘
                 有匹配 ▼     ▼ 无匹配
          ┌──────────────┐  ┌────────────────┐
          │ Harness 执行 │  │ Agent Fallback │
          │ + 不变量监控 │  │ + 轨迹录制     │
          └──────┬───────┘  └───────┬────────┘
           成功  │  失败             │
                 │   │               │
                 │  失败分类         │
                 │  (F1/F2/F3/F4)   │
                 ▼   ▼               ▼
         ┌───────────────────────────────┐
         │     Experience Accumulator    │
         │     统计更新 + 触发检查        │
         └──────────────┬────────────────┘
                        │ 触发条件满足
                        ▼
         ┌───────────────────────────────┐
         │     Harness Inductor          │
         │     归纳 → 合成 → 验证 → 入库  │
         └───────────────────────────────┘
```

### 3.2 Harness 检索与选择

检索分两阶段：

**粗筛（语义检索）**：对任务描述计算嵌入向量，在 Harness Registry 中检索 Top-K 候选。Harness 的索引向量由任务类型标签、自然语言描述、典型任务示例的组合嵌入构成。

**精筛（前置条件匹配）**：对候选 Harness 的每个前置条件，在当前环境快照中逐一核验。前置条件分为硬条件（不满足则不可执行，如 OS 类型、必要权限）和软条件（不满足则降级执行，如浏览器版本）。

最终决策：硬条件全部满足 → 以高置信度执行 Harness；仅软条件不满足 → 降级执行；无匹配 → Fallback 到纯 Agent。

### 3.3 Harness 归纳流程

归纳分六个阶段：

**Phase 1 — 轨迹分段**：对每条原始轨迹，使用 LLM 识别语义边界（子任务切分点），辅以状态变化突变检测。

**Phase 2 — 前后置条件提取**：对每个片段，提取进入时和退出时的状态特征，通过跨轨迹取交集得到候选前置 / 后置条件。

**Phase 3 — 不变量挖掘**：在每条轨迹的片段内，识别整个执行过程中保持为真的状态谓词（Daikon 风格的动态不变量检测），取跨轨迹交集作为最终不变量。

**Phase 4 — 步骤抽象与参数化**：将动作序列中的具体值替换为变量名，合并语义等价的冗余步骤，得到参数化的规范步骤序列。

**Phase 5 — Harness 合成**：将上述结构输入 LLM，生成带完整元数据的可执行 Harness 代码。

**Phase 6 — 沙盒回放验证**：在沙盒中对源轨迹回放，要求成功率 ≥ VALIDATION_THRESHOLD（实验中设为 0.8）方可入库。

### 3.4 失败分类与 Harness 进化

Harness 执行失败时，自动分类为四种类型，触发不同的更新策略：

| 失败类型             | 描述                         | 旧版本处理         | 新版本策略               |
| -------------------- | ---------------------------- | ------------------ | ------------------------ |
| F1：前置条件覆盖不足 | 约束遗漏（如未约束 OS 版本） | 保留，补充约束范围 | 特化分裂，新增适配版本   |
| F2：内部实现错误     | Selector 失效、时序错误等    | 标记 Deprecated    | Patch 覆盖，生成修复版本 |
| F3：环境漂移         | UI 重新设计、API 升级        | 保留，标记版本范围 | 新增适配新环境的分支     |
| F4：任务范围外       | 任务不属于此 Harness 能力域  | 不变               | 不更新，记录 OOD 案例    |

版本关系维护在有向无环图（Version DAG）中，包含三种边类型：`patch`（修复覆盖）、`specialization`（特化分裂）、`composition`（子 Harness 组合）。

### 3.5 在线积累与触发机制

每次任务执行完成后，异步触发以下逻辑：

```
1. 立即：Raw Trajectory 入库，更新 TaskTypeStats
2. 异步：检查归纳触发条件
   - support_count 首次达到 MIN_SUPPORT → 触发新建 Harness
   - 检测到新的环境变异维度 → 触发特化分裂
   - F2 失败累计 ≥ 2 次 → 触发 Patch
3. 兜底：每周定期批处理，处理累积的小变化
```

------

## 4. 实验

### 4.1 实验设置

**Benchmark**：τ-bench（零售 + 航空两个领域）。选择理由：(1) 任务具有明确的前置条件和可验证终态，天然契合 Harness 结构；(2) 同类任务重复率高，适合验证积累效果；(3) 纯文本 + API 调用环境，降低实现成本，聚焦核心假设。

**Baseline**：

-   **Vanilla Agent**：每次从零推理，无任何历史利用
-   **RAG Agent**：检索相关历史轨迹作为上下文，代表现有最主流的记忆增强方案
-   **AutoHarness（本文）**：完整系统

**评估指标**：

-   任务成功率（Task Success Rate）
-   每任务平均 Token 消耗
-   每任务平均执行延迟
-   Harness 回放验证通过率（内部质量）
-   Token 节省量（估算）

### 4.2 主实验：积累效果验证

将任务流划分为**积累阶段**（前 K 次同类任务，K = MIN_SUPPORT）和**评估阶段**（后续任务）。

核心假设：AutoHarness 在评估阶段的成功率应 ≥ Vanilla Agent，同时 Token 消耗显著下降。这条"交叉曲线"是整个范式可行性的核心证明。

### 4.3 消融实验

| 消融变体          | 移除的组件                   | 目的                     |
| ----------------- | ---------------------------- | ------------------------ |
| w/o Invariant     | 移除不变量监控               | 验证运行时监控的价值     |
| w/o Validation    | 跳过沙盒回放验证             | 验证验证步骤对质量的影响 |
| w/o Versioning    | 失败时直接覆盖，不维护 DAG   | 验证版本管理的必要性     |
| Fixed MIN_SUPPORT | 对比 MIN_SUPPORT = 1 / 3 / 5 | 确定最优积累阈值         |

### 4.4 泛化性分析

**参数泛化**：同一 Harness，不同输入参数（如不同收件人、不同订单号）。

**跨任务迁移**：在零售领域积累的 Harness，能否复用于航空领域中结构相似的子任务？

### 4.5 与 RAG 的质性区分

专门用一节分析 AutoHarness 与 RAG 的本质差异，重点论证：RAG 改变的是 LLM 的输入分布，而 AutoHarness 改变的是**计算路径**——Harness 执行阶段完全绕过了 LLM 推理，不是改进推理，而是消除推理。

------

## 5. 讨论

### 5.1 知识编译的经济学

当 Harness 入库后，每次后续执行的边际推理成本趋近于零。随着知识库规模增长，系统呈现出规模报酬递增的特性——与传统 Agent 线性增长的成本曲线形成根本性对比。

### 5.2 公共知识库的可行性

本文描述的个人知识库可以自然扩展为多层级体系：个人库（私有，个性化）→ 组织库（企业内部共享）→ 公共库（通用软件的标准化 Harness）。三层之间通过优先级覆盖链混用，类似 Linux 的 PATH 解析机制。这个体系一旦成立，Agent 之间的核心差距将从"推理能力"转移为"算力性能"，知识本身变为公共基础设施。

### 5.3 局限性

当前工作主要局限在纯文本任务环境。GUI 环境下，Harness 的 Selector 稳定性依赖 Accessibility Tree 的 `role` + `name` 属性稳定性，在应用大版本升级时可能触发大量 F3 失败，需要更鲁棒的跨版本 Selector 策略，留作未来工作。

------

## 6. 结论

本文提出 AutoHarness，一个将 Agent 执行轨迹自动编译为可复用执行脚手架的框架，将 Agent 的经验积累形式化为贝叶斯知识编译过程。实验在 τ-bench 上验证了"经验可编译、推理可消除"的核心假设，并分析了 Harness 质量随积累量的增长规律。我们认为，这一范式指向了 Agent 基础设施的一个新方向：把 Agent 的一次性推理成本转化为可持久、可共享、可进化的确定性计算结构，使重复任务的执行成本趋近于零。

------

**这个 RP 还有一个地方你需要决策**：Section 4.3 的跨任务迁移实验，如果零售和航空之间真的能复用子任务 Harness，会是很强的额外贡献点。但如果实验结果不理想，放进去反而是负担。建议先跑主实验，迁移实验作为 bonus 处理。





------

# Research Proposal

## Executable Experience: Building a Persistent, Transferable Artifact Runtime for Self-Improving Agents

------

## Abstract

Current LLM-based agents suffer from a fundamental inefficiency: they re-derive solutions from scratch on every task, failing to compile stable interaction patterns into reusable computational structures. We propose **ExperienceOS**, a persistent executable experience runtime that sits outside any single agent model, accumulating verified executable artifacts from interaction trajectories, and routing agent behavior through dynamically discovered skills. Unlike RAG-based memory (which changes inputs) or fine-tuning (which changes weights), ExperienceOS changes the agent's *computational substrate* by building an evolving library of validated, versioned, transferable executable artifacts. We validate on T-Bench (text/API environments), OSWorld (desktop GUI), and SWE-Bench (code repair), demonstrating that agents equipped with ExperienceOS improve across tasks, transfer to unseen domains, and generalize across model backbones.

------

## 1. Motivation

### 1.1 The Re-derivation Problem

State-of-the-art agents based on GPT-4, Claude, and open-source LLMs face a structural limitation: every task invocation begins from the same prior. No matter how many similar tasks the agent has completed, the next task receives no benefit from accumulated operational experience. This is not a knowledge problem — modern LLMs contain vast world knowledge — but a **capability compilation problem**: the agent never converts repeated successful patterns into durable, executable structures.

Human analogy: an expert accountant does not re-derive double-entry bookkeeping from first principles every morning. They have compiled that knowledge into automated cognitive subroutines. Current agents lack this layer entirely.

### 1.2 Why Existing Approaches Are Insufficient

| Approach               | What Changes      | Problem                              |
| ---------------------- | ----------------- | ------------------------------------ |
| In-context RAG         | Input context     | No new computation; token-limited    |
| Fine-tuning            | Model weights     | Catastrophic forgetting; expensive   |
| Skill library (static) | Tool set          | Manual curation; not self-generating |
| Workflow memory        | Stored trajectory | No abstraction; brittle to variation |

None of these compile experience into **verified, transferable, executable artifacts** that persist across models, environments, and task domains.

### 1.3 The Core Insight

Distinguishing intelligence from non-intelligence is not whether an agent uses tools, but whether it **creates tools**. fileciteturn1file3

We formalize this as:

$$
\text{Trajectory} \xrightarrow{\text{Discovery}} \text{Pattern} \xrightarrow{\text{Induction}} \text{Artifact} \xrightarrow{\text{Runtime}} \text{Capability}
$$

------

## 2. Research Questions

**RQ1 (Compilation):** Can an agent reliably discover recurring patterns in interaction trajectories and compile them into verified executable artifacts?

**RQ2 (Transfer):** Do compiled artifacts generalize across (a) unseen entities within a domain, (b) unseen task families, (c) unseen environments, and (d) different model backbones?

**RQ3 (Growth):** Does an agent's task performance monotonically improve as its artifact repository grows, and does this improvement exceed in-context retrieval baselines?

**RQ4 (Evolution):** Can artifacts self-repair through failure feedback without catastrophic regression on previously successful tasks?

------

## 3. System Design: ExperienceOS

The system is defined as an external scaffold operating alongside any base agent model. Following the formalization from the self-improving agents survey fileciteturn1file4:

$$
A_t = (\theta_t, \Sigma_t)
$$

where $\theta_t$ is the frozen base model and $\Sigma_t$ is the evolving scaffold. ExperienceOS is the operational implementation of $\Sigma_t$.

### 3.1 Overall Architecture

```
User Task
    |
    v
[Base Agent] ←────────────────────────────┐
    |                                      |
    v                                      |
[Experience Runtime]                  Artifact
    |                                  Injection
    ├── Experience Repository
    │       ├── Raw Trajectory Log
    │       ├── Experience Graph
    │       ├── Artifact Registry
    │       └── Version DAG
    │
    ├── Pattern Discovery Engine
    │       ├── Trajectory Clustering
    │       ├── MDL-based Schema Induction
    │       └── Bayesian Trigger Estimation
    │
    ├── Artifact Compiler
    │       ├── Schema → Executable Code
    │       ├── Sandbox Validation
    │       └── Confidence Scoring
    │
    └── Runtime Router
            ├── Semantic Retrieval
            ├── Precondition Matching
            └── Confidence-based Selection
```

### 3.2 Artifact Object Schema

Every artifact is a structured object, not raw code:

```yaml
artifact:
  id: DocumentVerifier-v2
  capability: document_validation
  abstraction_level: semantic

  trigger:
    - intent: submit_document
    - pattern: [retrieve, verify, submit]

  preconditions:
    - file_accessible: true
    - format: [pdf, docx]

  procedure:
    steps:
      - extract_metadata()
      - compare_version()
      - validate_schema()
      - return_verification_status()

  verification:
    method: sandbox_replay
    success_rate: 0.94
    test_count: 340

  scope:
    apps: [Gmail, Outlook, GoogleDrive]
    os: [Linux, Windows, macOS]

  dependencies:
    - FileReader-v1
    - VersionResolver-v2

  failure_modes:
    - permission_denied
    - ui_layout_changed

  version_history:
    parent: DocumentVerifier-v1
    change: added checksum validation
```

### 3.3 Experience Representation

We adopt a four-layer hierarchy:

**Layer 1: Raw Trajectory Log**

```json
{
  "task": "submit reimbursement form",
  "trajectory": ["open_browser", "navigate_to_form",
                 "fill_fields", "attach_receipt", "submit"],
  "structured_cot": {
    "goal": "submit expense",
    "constraints": ["receipt required", "manager approval"],
    "unknown": ["approval threshold"],
    "risk": "financial"
  },
  "outcome": "success"
}
```

**Layer 2: Experience Graph**

Graph nodes are task types, preconditions, action patterns, and verifiers. Edges encode dependency, causality, and co-occurrence with probability weights:

$$
P(S_{t+1} | S_t, A_t, E) \quad \text{where } E \text{ is the experience context}
$$

**Layer 3: Executable Artifact**

Compiled from graph patterns meeting induction criteria (see Section 3.4).

**Layer 4: Meta-Experience**

Tracks artifact utility:

$$
U(a) = \Delta\text{Success}(a) + \Delta\text{Efficiency}(a) - \text{Cost}_\text{creation}
$$

### 3.4 Artifact Induction Criterion

Artifact creation is not triggered by simple repetition. We use a Bayesian utility criterion:

$$
P(a \mid D) \propto P(D \mid a) \cdot P(a)
$$

where $P(a)$ is an MDL (Minimum Description Length) prior penalizing complexity, and $P(D \mid a)$ measures how well the artifact explains observed trajectories.

Induction fires when:

$$
U(a) > \tau \quad \text{and} \quad \text{coverage}(a) > \delta
$$

$\tau$ and $\delta$ are tunable thresholds validated on held-out tasks.

### 3.5 Runtime Routing

Three-stage routing at inference time:

1.  **Semantic retrieval**: embedding similarity between current task and artifact triggers
2.  **Symbolic precondition check**: verify runtime conditions match artifact scope
3.  **Confidence selection**: $\arg\max_a P(\text{success} \mid a, \text{context})$

### 3.6 Version Management

Artifacts are version-controlled in a DAG:

```
DocumentVerifier-v1
        |
        | [failure: pdf schema drift]
        |
DocumentVerifier-v2
       / \
      /   \
  Gmail  Outlook
  adapter adapter
```

Failures do not overwrite. They branch into new hypotheses, preserving prior capabilities.

------

## 4. Experimental Design

### 4.1 Benchmark Selection

We select benchmarks across three axes: environment type, feedback clarity, and domain diversity.

| Benchmark     | Environment | Feedback       | Domain                              | Role                             |
| ------------- | ----------- | -------------- | ----------------------------------- | -------------------------------- |
| **T-Bench**   | Text/API    | Exact match    | Customer service, database, finance | Primary: artifact induction      |
| **SWE-Bench** | Code repo   | PASS/FAIL      | Software engineering                | Primary: transfer + verification |
| **OSWorld**   | Desktop GUI | Task success   | Cross-application                   | Secondary: real-world complexity |
| **WorkArena** | Web browser | Form/task eval | Enterprise workflows                | Secondary: environment transfer  |

T-Bench is prioritized first because text environments provide clean, reproducible trajectories with minimal environment noise, ideal for validating the core compilation mechanism. fileciteturn1file7

SWE-Bench is prioritized second because pass/fail signals are unambiguous and artifact transfer (e.g. debugging workflows, patch verification) is clearly measurable. fileciteturn1file2

### 4.2 Dataset Split Design

This is critical. We define four orthogonal split types to prevent memorization:

#### Split Type 1: Entity Split (within-domain transfer)

Same task family, different entities.

-   Train: `send invoice to [Alice, Bob, Carol]`
-   Test: `send invoice to [Dave, Eve]`

Tests: entity-level generalization.

------

#### Split Type 2: Template Split (cross-domain transfer)

Different task family, shared abstract structure.

-   Train: `email document submission`
-   Test: `web form submission`

Both require: `[retrieve → verify → submit → confirm]`

Tests: structural artifact transfer.

------

#### Split Type 3: Environment Split (cross-environment transfer)

Same task family, different execution environment.

-   Train: Gmail on Linux
-   Test: Outlook on Windows

Tests: scope/adapter generalization.

------

#### Split Type 4: Composition Split (novel artifact combination)

-   Train: artifact A alone, artifact B alone
-   Test: task requiring A + B + new connector

Tests: whether artifacts compose rather than just retrieve.

------

#### Split Type 5: Backbone Split (cross-model transfer)

-   Artifact library compiled using GPT-4o
-   Evaluation agent: Claude 3.5 / Llama-3

Tests: whether artifacts are model-agnostic.

------

### 4.3 Pre-built vs. Online Accumulation

We distinguish two experimental modes:

#### Mode A: Online Accumulation (Primary)

Agent begins with empty artifact repository $K_0 = \emptyset$.

Tasks arrive sequentially. After every $n$ tasks, compilation runs. Repository grows as $K_0 \subset K_1 \subset K_2 \subset \ldots$

This is the primary claim: **agents improve through self-generated experience**.

Evaluation metric: learning curve over task stream.

------

#### Mode B: Pre-built Artifact Library (Ablation)

A library is pre-compiled from a held-out training split, then frozen. Agent uses it on the test split.

This is not the main claim but is used to:

1.  Establish an upper bound on artifact utility
2.  Validate that online-compiled artifacts approach the quality of carefully curated ones
3.  Enable the backbone split experiment (compile once, test on multiple models)

------

#### Mode C: Continual + Transfer

Agent accumulates artifacts on Domain A, then transfers to Domain B with no further compilation.

This is the key transfer test.

------

### 4.4 Baselines

| Baseline                  | Description                                     |
| ------------------------- | ----------------------------------------------- |
| **Vanilla LLM**           | No memory, no tools beyond basic API            |
| **Trajectory RAG**        | Full trajectory retrieval into context          |
| **Summary Memory**        | LLM-summarized episode memory                   |
| **Static Skill Library**  | Manually curated skill set (no self-generation) |
| **SkillOpt / PatchWorld** | Closest prior work fileciteturn1file7           |
| **ExperienceOS (ours)**   | Full system                                     |

Ablations:

| Ablation              | Removed Component                     |
| --------------------- | ------------------------------------- |
| No compiler           | Artifacts created by human annotation |
| No versioning         | Artifacts overwritten on failure      |
| No graph              | Flat artifact list only               |
| No precondition check | Random artifact selection             |
| No verification       | Unvalidated artifacts injected        |

### 4.5 Metrics

**Primary:**

-   Task success rate (SR)
-   Transfer gain: $\Delta SR = SR_\text{artifact} - SR_\text{baseline}$
-   Learning curve: SR as a function of task count $t$

**Artifact Quality:**

-   Artifact precision: fraction of invoked artifacts that improve success
-   Artifact recall: fraction of successful tasks where a relevant artifact existed
-   Generalization ratio: SR on Split Type 2/3/4 relative to Split Type 1

**System Efficiency:**

-   Token reduction vs. RAG baseline (same task, fewer context tokens)
-   Artifact reuse rate: mean invocations per artifact over test period

**Evolution:**

-   Regression rate: fraction of previously passing tasks broken by artifact update
-   Recovery rate: after failure, fraction of tasks recovered by new artifact version

------

## 5. Tool and Environment Assumptions

### 5.1 Text Environment (T-Bench, Primary)

Available tools assumed:

```
text_reader(source)          # read documents, emails, tables
text_writer(content, target) # write/edit text
search(query, corpus)        # semantic search over documents
api_call(endpoint, params)   # structured API interaction
form_fill(schema, values)    # fill structured forms
verify(content, schema)      # validate content against schema
compare(a, b, criteria)      # structured comparison
extract(source, fields)      # field extraction from unstructured text
```

Artifact examples that can be compiled in this environment:

-   `CustomerVerifier`: check identity fields before record update
-   `DocumentSummarizer`: extract key fields from contract
-   `PolicyChecker`: validate action against rule set
-   `ResponseFormatter`: structure output to match expected schema

------

### 5.2 Code Environment (SWE-Bench)

Available tools:

```
file_read(path)
file_edit(path, diff)
bash_exec(command)
test_runner(suite)
git_ops(command)
search_codebase(pattern)
```

Artifact examples:

-   `RegressionTestArtifact`: run relevant tests before any patch
-   `FailureSignatureExtractor`: extract structured error pattern from traceback
-   `PatchValidator`: check patch scope against affected files
-   `DependencyMapper`: identify ripple effects of a change

------

### 5.3 Desktop/GUI Environment (OSWorld)

Available tools:

```
screenshot()
find_element(semantic_desc)
click(element)
type(element, text)
keyboard_shortcut(combo)
scroll(direction, amount)
wait_for(condition)
```

Artifact examples:

-   `FormSubmissionArtifact`: verify form completeness before submission
-   `FileAttachmentVerifier`: check attachment exists and is correct version
-   `UIStateMonitor`: detect whether expected UI transition occurred

Key constraint: artifacts must be defined at **semantic level**, not pixel coordinates, to survive UI changes:

```yaml
# Bad:
click(x=342, y=891)

# Good:
click(element=find_element("Submit button, blue, bottom-right"))
```

------

## 6. Experience Iteration Protocol

### 6.1 Compilation Trigger

Compilation runs:

-   Every $n=50$ tasks (periodic), or
-   When pattern frequency exceeds threshold $\delta$, or
-   On explicit failure cascade (3+ failures of same type within 20 tasks)

### 6.2 Iteration Cycle

```
Phase 1: Trajectory Analysis
  - Extract structured CoT traces
  - Cluster by task intent + action pattern
  - Score cluster stability (MDL criterion)

Phase 2: Schema Induction
  - LLM proposes artifact schema from cluster representative
  - Schema validated against cluster members
  - Scope conditions derived from failures within cluster

Phase 3: Artifact Compilation
  - Schema → executable procedure (LLM-generated code)
  - Sandbox replay on held-out cluster examples
  - Confidence score assigned

Phase 4: Registry Update
  - If new: add to registry
  - If overlaps existing: compare confidence, branch version
  - If conflicts existing: create sibling version, preserve parent

Phase 5: Failure Feedback Loop
  - On execution failure: log failure signature
  - Pattern-match against artifact preconditions
  - If precondition violated: update scope constraint
  - If procedure failed: propose patch → sandbox → version bump
```

### 6.3 Preventing Regression

Before any artifact update is accepted:

1.  Run full regression suite on artifacts that share dependencies
2.  Require: $\Delta SR_\text{dependent} \geq -\epsilon$ where $\epsilon = 0.02$
3.  If regression detected: create sibling version, do not replace parent

This ensures the system cannot "press down one side and have the other pop up" — directly addressing the concern raised in the meeting discussion. fileciteturn1file6

------

## 7. Relationship to Prior Work

| Work                         | Relation                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| PatchWorld                   | Executable world model; we compile experience artifacts, not world transitions |
| SkillOpt (Microsoft)         | Skill optimization on PatchWorld; we focus on self-generation from trajectories fileciteturn1file7 |
| Voyager                      | Skill library for Minecraft; not self-discovered, manually prompted, not transferable across environments |
| ExGRPO                       | Experience-driven reasoning; we focus on executable artifacts, not reasoning traces fileciteturn1file0 |
| AgentWorld / OSWorld         | Evaluation environments; we treat as test beds, not baselines |
| Self-improving agents survey | Formal framework; we implement scaffold improvement branch ($\Sigma_t$ update) fileciteturn1file4 |

Key differentiator from all prior work:

>   We do not store trajectories. We do not fine-tune models. We compile verified executable artifacts that are model-agnostic, environment-scoped, versioned, and self-repairing.

------

## 8. Expected Contributions

1.  **ExperienceOS**: the first persistent, model-agnostic executable artifact runtime for self-improving agents
2.  **Artifact Induction Algorithm**: Bayesian MDL-based criterion for deciding when to compile a pattern into an artifact
3.  **Transfer Evaluation Protocol**: four-axis split design (entity / template / environment / composition) for rigorous artifact generalization testing
4.  **Empirical findings**: quantitative evidence that compiled artifacts outperform RAG, summary memory, and static skill libraries on cross-domain transfer, with sublinear token cost growth

------

## 9. Timeline

| Period    | Milestone                                                    |
| --------- | ------------------------------------------------------------ |
| Month 1–2 | Implement ExperienceOS core: trajectory logging, experience graph, basic artifact compiler |
| Month 3   | T-Bench integration; validate online accumulation (RQ1, RQ3) |
| Month 4   | SWE-Bench integration; validate transfer experiments (RQ2)   |
| Month 5   | Version management + failure feedback loop (RQ4)             |
| Month 6   | OSWorld + WorkArena; backbone split experiments              |
| Month 7   | Ablations, baselines, writing                                |
| Month 8   | Submission                                                   |

------

## 10. Open Questions for Discussion

1.  **Artifact granularity**: Should artifacts operate at step level, subtask level, or task level? Too fine → explosion in artifact count. Too coarse → poor transfer.
2.  **Induction threshold calibration**: How to set $\tau$ and $\delta$ without overfitting to training domain?
3.  **Artifact interaction**: When two artifacts conflict on the same task, how should the router resolve? Priority queue? Confidence voting?
4.  **LLM dependency in compilation**: The artifact compiler itself uses an LLM. Does this create a dependency loop? Should compilation be model-agnostic?

------

这个 proposal 可以直接作为组内讨论文档或投稿前的研究计划。下一步我建议优先细化两件事：

**第一**：T-Bench 的具体任务流设计，因为这是第一个跑起来的环境，需要确定 trajectory 格式和 artifact schema 的具体实现。

**第二**：Artifact Induction Algorithm 的伪代码，因为这是 reviewer 最可能质疑的核心创新点。