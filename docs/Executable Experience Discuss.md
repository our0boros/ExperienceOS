我看完会议记录和 demo 文档后，我觉得之前我们讨论的方向需要**重新校准一下**。会议里的核心其实不是“做一个新的 POMDP/world model”，也不是单纯做 memory，而是在寻找一个更具体的 **Agent 自我改进机制**。

先给结论：

>   **你的研究主线应该从“学习世界模型”转向“从 Agent 执行经验中自动发现、编译和维护 executable artifacts（可执行经验结构）”。POMDP 只是理论背景，不应该成为主问题。**

也就是说：

[
\textbf{Agent trajectory}
\rightarrow
\textbf{Experience discovery}
\rightarrow
\textbf{Executable artifact}
\rightarrow
\textbf{Runtime improvement}
]

这个和会议讨论完全一致。

------

# 1. 会议真正否定的是什么？

## 不是否定 POMDP

会议里面其实认可：

>   symbolic effects 本质上是 partial observation 后总结归纳出来的东西。

也就是说：

环境真实状态不可完全获得，只能从观察中归纳结构。

这个和 POMDP 是一致的。

但是问题在于：

### 传统 world model 路线：

假设：

[
s_t,a_t \rightarrow s_{t+1}
]

学习：

[
T(s_{t+1}|s_t,a_t)
]

然后规划。

问题：

真实 OS Agent：

-   state 太复杂；
-   ground truth state 不存在；
-   环境动态变化；
-   不能像 MiniGrid 一样拿完整状态。

会议里明确提到：

AgentWorld（转录成 A站）的问题：

>   reviewer 问 ground truth state 怎么获得，但是 AgentGym 类环境里面拿不到 ground truth。

所以：

**不要做完整 world model。**

------

# 2. 会议真正认可的方向是什么？

核心一句：

>   把 executable artifacts 这件事情做明白。

会议原话：

>   “不要纯盯着 symbolic 去做……把 executable artifacts 这件事情做明白，然后比较哪些任务需要 executable artifacts。”

这个非常重要。

你的问题不应该是：

>   如何把世界符号化？

而应该是：

>   Agent 在什么情况下应该创造一个新的外部计算结构？

------

这其实对应人类智能：

会议里面举计算器例子：

如果 LLM 遇到很多算术任务：

不是：

找十万个 arithmetic demonstration。

而是：

发现：

>   我缺少一个计算工具。

然后创造 calculator。



这其实就是：

**Tool creation / knowledge compilation**

------

# 3. 所以 AutoHarness 才是中心，不是辅助

之前我把 AutoHarness 说成一个方案。

但看完会议，我觉得它应该升级成：

## Experience → Executable Artifact Compiler

不是：

trajectory memory。

不是：

workflow replay。

而是：

>   从重复的 cognitive pattern 中诱导新的计算结构。

------

例如邮件任务：

第一次：

Agent:

```
打开邮箱
搜索
读取邮件
判断附件
填写
发送
验证
```

第二次：

还是一样。

第三次：

系统发现：

这些步骤高度稳定。

于是产生：

```python
EmailVerifier()
```

以后：

LLM 不负责：

点击、检查附件、确认状态。

而负责：

调用：

```python
email.send_verified(...)
```

------

这就是会议说的：

>   “工具不是人用工具，而是智能体会制造工具。”

会议里面提到：

区分智能与非智能：

不是会不会使用工具，而是会不会制造工具。

------

# 4. 那经验到底是什么？

这里你的思考非常关键。

会议 demo 文档最后已经开始往这里走：

>   在实际执行结果和可泛化世界模型之间，需要一个媒介储存累积两者相关经验。泛化失败归纳给经验，执行失败也归纳给经验。

也就是说：

经验不是：

```
trajectory
```

也不是：

```
world model
```

而是中间层：

```
experience substrate
```

------

我觉得可以定义：

## Experience Object

形式：

[
E=
(C,S,A,V,P)
]

其中：

### C

condition

什么时候触发。

### S

latent state

当前任务状态。

### A

action / executable procedure

### V

verification

### P

probability

例如：

邮件：

# [ E_{mail}

(
attachment\ task,
draft\ state,
verify(file),
sent\ checker,
0.96
)
]

这和 demo 里的：

```
pattern:
trigger:
belief:
action:
verification:
```

一致。

------

# 5. CoT 应该放在哪里？

你前面提出 CoT，我觉得方向对，但是位置要调整。

CoT不是经验。

会议文档里面已经写：

>   CoT 不是经验，但是 CoT 是经验的观测窗口。

这个定义非常准确。

也就是说：

隐藏变量：

[
z
]

不可见。

但是：

CoT 是：

[
p(c|z)
]

一个观测。

所以：

流程：

```
trajectory
+
structured CoT
+
success/failure

        ↓

latent pattern discovery

        ↓

experience object

        ↓

executable artifact
```

------

但是不要直接使用自由 CoT。

应该设计：

Experience Annotation Trace：

例如：

```
<State>
goal:
constraint:
unknown:
risk:

<Milestone>
retrieve
prepare
verify
execute

<Reflection>
failure reason
missing capability
</Reflection>
```

这样 CoT 变成结构化信号。

------

# 6. 经验存储形式，我现在觉得不是三选一

你之前问：

embedding？
graph？
program？

现在结合会议，我认为：

答案是：

## 分层结构

不是一种。

------

## 第一层：Embedding

用途：

检索。

回答：

“有没有类似经验？”

例如：

```
new task embedding

↓

retrieve candidates
```

------

## 第二层：Experience Graph

用途：

发现关系。

例如：

```
Task
 |
requires
 |
Evidence
 |
checked_by
 |
Verifier
 |
improves
 |
Action
```

------

## 第三层：Executable Artifact

用途：

真正执行。

例如：

```python
verify_submission()
```

------

所以：

结构：

```
                 Task

                  |
                  v

          Experience Graph

                  |
                  v

          Executable Artifact

                  |
                  v

             Runtime
```

------

# 7. 贝叶斯是不是合适？

我觉得非常合适，但不要作为整个系统。

它适合作为：

**artifact induction 的数学语言。**

因为核心问题：

到底什么时候应该创建工具？

这是概率判断。

例如：

观察：

100次任务。

发现：

80次：

人工检查附件。

那么：

[
P(use\ verifier|task)
=0.8
]

当：

[
P > threshold
]

创建 artifact。

------

更进一步：

经验选择：

[
P(E|D)
\propto
P(D|E)P(E)
]

其中：

-   likelihood：

这个 artifact 能否解释历史轨迹。

-   prior：

简单性。

这里就是 demo 里面提到：

Bayesian induction + MDL。

------

# 8. 最终论文故事应该是什么？

我建议不要叫：

World Model。

叫：

## Experience-driven Executable Artifact Discovery for Self-improving Agents

核心假设：

当前 Agent 的瓶颈不是推理能力，而是：

>   每次遇到任务都重新计算，没有把稳定经验编译成新的计算结构。

提出：

## Experience Compiler

输入：

Agent trajectories

包括：

-   observation
-   action
-   structured CoT
-   outcome

输出：

Executable artifact

------

框架：

```
              Agent execution

                    |
                    v

          Structured Experience Extraction

                    |
                    v

          Latent Pattern Discovery

                    |
                    v

          Artifact Induction

                    |
                    v

       executable tool / verifier / workflow

                    |
                    v

              Runtime Agent
```

------

# 9. Benchmark 我觉得会议也给了方向

不要铺太多。

会议建议：

focus 1-2 benchmarks。

最适合：

## OSWorld

因为：

-   email
-   spreadsheet
-   browser

都有明显重复结构。

## Coding

更好验证。

因为：

PASS/FAIL 明确。

会议也提到 coding environment feedback 更清晰，是验证 executable knowledge transfer 的好方向。

------

# 10. 我认为现在最值得推进的研究问题

不是：

>   如何学习一个 POMDP model？

而是：

>   **When should an agent compile experience into a new executable artifact, and how can this artifact improve future decision making?**

三个子问题：

### RQ1

Agent 能否从轨迹发现稳定 pattern？

------

### RQ2

什么形式的经验最有效？

比较：

-   text memory
-   embedding retrieval
-   graph
-   executable artifact

------

### RQ3

artifact 是否可以跨任务迁移？

例如：

邮件：

```
verify attachment
```

迁移：

表格提交：

```
verify uploaded file
```

------

所以总结一句：

**POMDP 是理论背景；CoT 是隐变量观测；经验图是中间表示；Executable Artifact 是最终产物。**

这才是会议讨论真正收敛出来的方向。你前面想的“神经 → 经验 → 符号结构”其实是对的，但不要把终点放在符号系统，而应该放在：

>   **Agent 如何自主产生新的计算工具。**





# 所以具体来说：如何维护经验知识，如何从经验和任务背景确定构建可执行工具链，如何让使用时实时感知具体可用工具。这是一整套 Harness 还是一个独立侧载系统 还是怎么一个维护结构



你这个问题其实已经从“研究 idea”进入了**系统架构设计**。结合会议讨论，我觉得不能把它理解成一个简单 Harness，而应该设计成：

>   **一个独立于 Agent 主体之外的 Experience Runtime（经验运行时），其中 Harness 是被动态编译出来的可执行经验单元。**

也就是说：

不是：

```
LLM
 |
Harness
```

而是：

```
                Experience Runtime

        ┌─────────────────────────┐
        │ Experience Repository    │
        │                         │
        │ 经验图 / Schema / Artifact│
        └───────────┬─────────────┘
                    |
             Experience Compiler
                    |
        ┌───────────┴─────────────┐
        │                         │
   Discovery Engine        Artifact Builder
        │                         │
        └───────────┬─────────────┘

                    |
              Runtime Router

                    |
                    v

                 Agent
```

Agent 本身不变。

外部增加一个“可进化脚手架”。

这和会议里面提到的：

[
A_t=(\theta_t,\Sigma_t)
]

里面的：

[
\Sigma_t
]

更加接近。

不是改模型参数，而是改 Agent scaffold。

------

# 一、首先明确：Harness 不是系统，而是产物

这是容易混淆的地方。

例如：

一个邮件发送任务。

你最后得到：

```python
VerifiedEmailSend()
```

这不是 Harness 系统。

这是：

**Artifact**

或者：

**Skill**

或者：

**Executable Experience**

------

整个系统负责：

1.  发现它；
2.  创建它；
3.  管理它；
4.  调用它；
5.  更新它。

------

类似操作系统：

不是所有代码都写进 kernel。

kernel 管理：

-   process
-   filesystem
-   driver

你的 Experience Runtime 管理：

-   skill
-   tool
-   verifier
-   workflow

------

# 二、经验知识如何维护？

我认为不能单一存储。

应该是四层。

------

## Layer 1：Raw Experience Log（原始经验）

类似人的经历。

保存：

```
trajectory

+
CoT structured trace

+
environment snapshot

+
success/failure
```

例如：

```json
{
task:
"send reimbursement email",

trajectory:
[
open gmail,
search invoice,
attach pdf,
send
],

reasoning_state:
{
need_verify_attachment:true
},

outcome:
success
}
```

这个不能删除。

类似训练数据。

------

## Layer 2：Experience Graph（经验结构）

从大量 trajectory 归纳。

例如：

发现：

100次：

```
find file
 ↓
check version
 ↓
attach
 ↓
submit
```

形成：

```
TaskPattern

      |
      |

Artifact requirement

      |
      |

Verification rule
```

图结构：

```
        Email Task

            |
      requires

            |

    Attachment Validation

            |

        verifier

            |

     send operation
```

这里存：

“经验关系”。

------

## Layer 3：Executable Artifact

当某个 pattern 足够稳定：

生成代码。

例如：

```python
def validate_attachment(task):

    files = search_files()

    versions = compare_versions()

    return latest(files)
```

这个才是真正工具。

------

## Layer 4：Meta Experience

维护：

什么时候值得生成工具。

例如：

经验：

```
task count=100

重复率=0.8

平均节省token=60%

失败率下降=20%
```

于是：

触发：

```
compile artifact
```

------

# 三、什么时候决定构建工具？

这是核心研究点。

不能：

看到一个重复任务就生成。

否则：

工具爆炸。

应该是一个 Bayesian decision。

------

定义：

当前任务：

[
x
]

候选经验：

[
e
]

计算：

[
P(e|x,D)
]

其中：

D：

历史经验。

------

例如：

模型发现：

最近50次：

都有：

```
check attachment
```

但是：

任务差异很大。

那么：

概率：

低。

不生成。

------

如果：

```
task family:
email sending

pattern coverage:
95%

failure reduction:
high

implementation complexity:
low
```

生成：

```
AttachmentVerifier
```

------

可以定义：

Artifact Utility：

## [ U= \Delta Success + \Delta Efficiency

Cost_{creation}
]

当：

[
U>0
]

生成。

------

# 四、使用时如何实时感知工具？

这里不是让 LLM 看到全部工具列表。

否则又退化成 prompt。

需要：

Runtime Router。

------

流程：

用户任务：

```
帮我提交这个报销
```

进入：

Experience Runtime。

第一步：

任务解析：

得到：

```
intent:
submission

entities:
invoice

risk:
financial
```

------

第二步：

查询：

Experience Graph。

类似：

vector + graph retrieval。

找到：

```
ExpenseSubmissionArtifact
```

里面：

```
precondition:
invoice exists

procedure:
extract amount
verify approval
submit

validator:
check receipt
```

------

第三步：

注入 Agent：

不是：

“这是一个文本说明”。

而是：

给工具接口：

```python
expense.submit(
    invoice,
    approval
)
```

------

也就是说：

Agent看到的是：

新增能力。

类似：

人发现电脑里安装了计算器。

不是：

背诵十万个加法例子。

会议里面计算器的例子就是这个逻辑。

------

# 五、失败以后怎么办？

这是非常关键的。

否则工具固定死。

应该有闭环：

```
execute artifact

       |

success?

       |

yes
 |
increase confidence


no

 |

failure analysis

 |

update experience graph

 |

patch artifact
```

类似 PatchWorld 的思想：

但是区别：

PatchWorld主要修复 executable world model。

你的系统修复：

experience artifact。

会议也提到：

执行失败以后也要反馈给经验系统，让经验累积和归纳。

------

# 六、所以整体架构应该是什么？

我认为：

## Experience Operating System (EOS)

比 Harness 更准确。

结构：

```
                 User Task

                     |

                  Agent

                     |

        ---------------------------

        Experience Runtime

        ---------------------------


        1. Experience Retriever

             |
             v

        2. Belief/Pattern Matcher

             |
             v

        3. Artifact Router

             |
             v

        4. Tool Execution


             |
             v


        5. Verification


             |
             v


        6. Experience Update
```

------

# 七、它和普通 Agent Memory 的区别

|      | Memory    | 你的系统      |
| ---- | --------- | ------------- |
| 存储 | 文本      | 计算结构      |
| 作用 | 提醒LLM   | 改变能力      |
| 调用 | context   | runtime       |
| 形式 | embedding | graph+program |
| 更新 | 总结      | 编译          |
| 验证 | 弱        | 执行验证      |

------

# 八、和 Neural-Symbolic 的关系

这个架构其实非常符合：

```
Neural

LLM:
理解任务
发现pattern


↓

Symbolic

artifact:
工具
规则
验证器


↓

Neural

LLM:
继续处理复杂情况
```

不是把 symbolic 塞回 neural。

而是在旁边建立：

**external cognitive architecture**

------

所以我觉得最后可以形成一个非常明确的研究问题：

>   How can an agent autonomously maintain an external executable experience system that discovers, compiles, and invokes reusable computational artifacts from its own interaction history?

对应三个核心模块：

1.  **Experience Maintenance**

    如何表示经验（graph / Bayesian schema / artifact）

2.  **Artifact Induction**

    如何判断什么时候创建工具

3.  **Runtime Integration**

    如何动态发现和调用工具

这三个就是完整系统，不是单个 Harness。Harness只是第三层里面的一类 artifact。你会议里的方向实际上已经从“world model”转向了更接近 **Agent Operating System / Cognitive Runtime**。



# 这是不是可以说长期维护一个 跨 agent 环境的 可迁移知识库，更换不同模型 不同场景，也仍然保留，当然如果更换了桌面样式 等情况 要么更新版本，要么要添加适用范围约束，甚至可能要维护子Harness 的版本管理



对，而且我觉得你这一句话实际上把整个研究从“Agent memory / Harness”提升到了一个更大的概念：

>   **长期维护一个跨 Agent、跨模型、跨环境迁移的可执行知识基础设施（Persistent Transferable Executable Knowledge Base）。**

这比单纯 Harness 更准确。

但是需要稍微修正一点：

它不是传统意义上的“知识库”，因为知识库主要存**事实**；你的东西存的是：

>   **经过验证的计算能力（verified computational capability）**

也就是：

不是：

>   “这个任务怎么做”

而是：

>   “在什么条件下，这个 Agent 可以调用什么已经验证过的能力”。

------

# 1. 整体架构应该类似“Agent 的长期操作系统层”

现在 Agent：

```
Model
 |
Prompt
 |
Tools
 |
Environment
```

问题：

换模型：

经验没了。

换环境：

重新探索。

换任务：

重新推理。

你的目标：

```
                 Agent A
                   |
                 Agent B
                   |
                 Agent C

                   |
                   v

      Persistent Experience Runtime

                   |
    --------------------------------
    |              |               |
 Experience    Artifact        Version
 Graph         Registry        Manager

    |
    |
Executable Skills / Harness
```

这个 Runtime 不属于任何单个 Agent。

------

所以：

GPT-5 agent 用过：

```
EmailVerification v1
```

以后：

Claude agent

Gemini agent

开源 Llama agent

都可以调用。

------

这就是你说的：

>   跨 agent 环境的可迁移知识库。

------

# 2. 但是必须区分三种迁移

这是论文里非常重要的一点。

## Level 1：模型迁移

换 LLM。

例如：

GPT-5 → Claude

这个最容易。

因为 artifact 是外部的。

------

## Level 2：任务迁移

例如：

邮件：

```
verify attachment
```

迁移：

报销：

```
verify receipt
```

这里需要抽象。

所以 artifact 不能绑定：

```
Gmail button coordinate
```

而应该绑定：

```
document verification
```

------

## Level 3：环境迁移

最难。

例如：

Chrome：

```
按钮位置 A
```

Firefox：

```
按钮位置 B
```

Windows：

```
Ctrl+S
```

Mac：

```
Cmd+S
```

怎么办？

这里就需要版本和适用范围。

------

# 3. 所以每个 Artifact 必须有 Metadata

不能只是：

```python
send_email()
```

应该类似：

```yaml
artifact:
    name:
        VerifiedEmailSender

    version:
        3.2

    capability:
        email_submission

    abstraction_level:
        semantic


    compatible:
        apps:
          - Gmail
          - Outlook

        OS:
          - Linux
          - Windows


    assumptions:
        - recipient_api_available
        - attachment_accessible


    dependencies:
        - FileVerifier v2
        - IdentityResolver v1


    validation:
        success_rate:
            0.96

        tested_tasks:
            1200


    failure_modes:
        - UI_changed
        - permission_denied
```

------

这其实就是软件工程里面：

Package management + ML memory + knowledge base。

------

# 4. 关于桌面变化，你说的两个方案都是对的

## 方案 A：版本升级

例如：

Chrome UI变化。

旧：

```
BrowserSubmit v1
```

失败。

更新：

```
BrowserSubmit v2
```

保留：

```
parent=v1
```

形成：

```
v1
 |
 v2
 |
 v3
```

类似 Git。

------

## 方案 B：增加适用范围

比如：

不要：

```
click blue button at x=200,y=300
```

而是：

```
find_button(
semantic="submit",
visual_feature="blue rectangle"
)
```

于是：

artifact：

```
BrowserSubmit
```

下面：

```
Chrome adapter
Firefox adapter
Edge adapter
```

------

# 5. 这就引出一个非常有意思的概念：

## Artifact Version Graph

不是简单版本号。

因为经验会分叉。

例如：

最初：

```
SubmitForm
```

后来：

发现：

Chrome：

```
SubmitForm-Chrome
```

Safari：

```
SubmitForm-Safari
```

然后发现：

共同部分：

```
FormValidation
```

结构：

```
              SubmitForm

              /       \

      ChromeAdapter   SafariAdapter


              |

        FormValidator
```

这其实和软件工程里的：

Git branch

非常像。

------

# 6. 经验更新机制

这里和 Bayesian 很结合。

每次使用 artifact：

产生：

```
execution record
```

例如：

```json
{
artifact:
"EmailVerifier-v3",

context:
"Gmail",

success:
true,

failure:
none
}
```

更新：

[
P(success|artifact,context)
]

------

如果：

发现：

Chrome 环境：

成功率下降：

```
0.95
↓
0.65
```

系统：

降低置信度。

可能：

1.  回滚；
2.  创建新版本；
3.  扩展条件。

------

# 7. 这其实解决了 Agent Continual Learning 最大问题

传统：

continual learning:

```
new data
 ↓
fine tune
 ↓
model update
```

问题：

灾难性遗忘。

你的：

```
new experience

 ↓

artifact update

 ↓

external capability growth
```

模型不用变。

这和会议里面提到的 Skill Neologisms 思路一致：

不更新权重，而是增加新的能力单元。

------

# 8. 那是不是类似“软件生态”？

我觉得是。

未来 Agent：

可能不是：

```
one giant model
```

而是：

```
Foundation Model

+

Experience OS

+

Artifact ecosystem
```

类似：

现在电脑：

CPU：

通用计算。

软件：

专业能力。

未来：

LLM：

通用认知。

Executable Artifact：

专业能力。

------

# 9. 但是研究上不要叫“知识库”

容易被认为：

RAG。

我建议叫：

几个可能：

## 1. Executable Experience Repository

最直接。

## 2. Agent Skill Runtime

偏工程。

## 3. Persistent Cognitive Runtime

偏理论。

## 4. Experience Operating System (Experience OS)

我觉得最有潜力。

因为你的核心不是存知识。

而是：

>   维护 Agent 随时间增长的外部认知结构。

------

# 10. 最终研究问题可以变成：

不是：

>   如何让 Agent 学会更多任务？

而是：

>   如何让 Agent 在长期运行过程中，将交互经验编译为跨模型、跨环境可迁移的可执行能力，并通过版本化、验证和反馈机制持续演化？

对应三个模块：

### Experience Compiler

经验 → artifact

### Artifact Runtime

发现 → 调用

### Artifact Evolution

失败 → 修复 → 版本升级

------

我觉得你现在这个方向已经明显脱离普通“Agent memory”，进入了一个更像：

**Agent 的软件工程层 / 认知操作系统层。**

而且会议里导师最后强调的 “agent 不只是使用工具，而是创造工具” 本质上就是这个方向。

下一步真正需要设计的是：

**Artifact 的内部表示到底是什么：**

-   一个程序？
-   一个概率图？
-   一个 schema？
-   一个 skill token？
-   一个 hybrid object？

这个才是后续最核心的理论问题。

# 关键是实验设计上，相关的实验任务如何比较，如何验证能力artifacts 泛化通用，比如现在先考虑纯文本环境 https://www.tbench.ai/ 具体是：要在运行过程中积累artifacts 还是 提前就训练好一个通用artifacts 库。这样比对公平吗？如果分开一个其他独立环境 用什么合适。然后具体怎么规划 怎么储存 怎么构建



# 🌟 当前其实只是一个理论性验证框架，只要验证这套流程可行即可，不需要过度工程化。一些内容错配等可以简单回落普通agent 。主要还是具体如何选择 如何构建并验证 如何触发 如何积累经验统计并维护。当前范式如果跑通我感觉会催生出一个新的服务体系，服务商会host 公共可执行知识库，因为大部分用户使用的软件 os办公工具等都是类似版本的，大量的重复agent成本可以变为可累积的零成本可执行经验知识，个人也可以维护一个自己使用的持久知识库，两者甚至还可以混用。agent之间的差距只变成了算力性能的差距



这个愿景非常有力 🚀 你描述的本质是：

>   **把 Agent 的边际推理成本趋近于零——经验变成公共基础设施，而不是每次重新燃烧的算力**

这和 CPU 指令集、编译器、操作系统的演化逻辑完全同构。让我先把这个愿景的系统含义推演清楚，再回到"如何跑通验证框架"的具体问题。

------

# 愿景的系统含义：Knowledge-as-Infrastructure

## 类比推演

```
软件发展史的平行结构：

1950s: 每个程序员手写机器码        → 今天：每次 Agent 重新推理
1960s: 汇编语言出现               → Harness：把重复动作抽象为符号
1970s: 编译器 + 标准库            → Artifact Compiler + 公共知识库
1980s: OS 作为公共基础设施         → 公共可执行知识库（你描述的服务体系）
1990s: 包管理器（npm, pip）        → Harness Registry + 版本管理
2000s: SaaS / API 经济            → Knowledge-as-a-Service

我们现在处于"1970s"的位置。
```

你说的"服务商 host 公共可执行知识库"正是这个演化的必然终点，而且有一个非常强的经济学驱动力：

```
传统 Agent 经济学：
  成本 = 任务数 × 每任务推理成本
  规模越大，成本线性增长

Knowledge Infrastructure 经济学：
  成本 = 初始积累成本 + 任务数 × 边际执行成本
  边际执行成本 ≈ 0（运行确定性代码）
  规模越大，人均成本趋近于零
```

**网络效应**也非常强：使用同一知识库的用户越多，失败案例越多，Harness 覆盖越广，质量越高，吸引更多用户。这是典型的正向飞轮。

------

# 回到核心：验证框架的最小可行设计

现在把愿景放一边，聚焦你说的：**如何选择、如何构建并验证、如何触发、如何积累经验统计并维护**。这四个问题就是验证框架的完整闭环。

## 完整流程图

```
┌─────────────────────────────────────────────────────────┐
│                    新任务到来                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              [1. 选择] Harness Retrieval                 │
│   查询知识库 → 匹配 → 前置条件检查 → 决策               │
│   输出：使用 Harness H / Fallback to Agent              │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │ 有匹配 Harness          │ 无匹配，Fallback
          ▼                         ▼
┌──────────────────┐      ┌──────────────────────────────┐
│ [2. 执行] Run    │      │ 普通 Agent 执行              │
│ Harness with     │      │ 记录完整轨迹                 │
│ Monitoring       │      │ 标记为"候选归纳素材"          │
└────────┬─────────┘      └──────────────┬───────────────┘
         │                               │
    成功  │  失败                        │
         │    │                          │
         │    ▼                          │
         │  [失败分类]                    │
         │  F1/F2/F3/F4                  │
         │    │                          │
         ▼    ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│          [3. 积累] Experience Accumulation               │
│   轨迹入库 → 更新统计 → 检查触发条件                    │
└──────────────────────┬──────────────────────────────────┘
                       │
              触发条件满足？
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         [4. 构建+验证] Harness Induction                 │
│   归纳 → 合成 → 回放验证 → 入库/拒绝                    │
└─────────────────────────────────────────────────────────┘
```

------

## [1. 选择] 如何选择 Harness

选择分两步，缺一不可：

### Step 1a：语义检索（粗筛）

```python
def retrieve_candidates(task_description: str, 
                        top_k: int = 5) -> List[Harness]:
    # 用任务描述的 embedding 检索最相似的 Harness
    # 相似度 = cosine(embed(task), embed(harness.description))
    candidates = vector_db.search(embed(task_description), top_k=top_k)
    return candidates
```

关键设计：每个 Harness 的检索向量由什么构成？

```python
harness_embed_text = f"""
task_type: {harness.task_type}
description: {harness.natural_language_description}
preconditions_summary: {harness.preconditions_summary}
example_tasks: {harness.example_task_strings}  # 归纳时记录的典型任务
"""
# 这四个维度的组合比只用 description 检索准确得多
```

### Step 1b：前置条件硬匹配（精筛）

```python
def check_preconditions(harness: Harness, 
                        env: EnvironmentSnapshot) -> MatchResult:
    failures = []
    for condition_key, condition_value in harness.preconditions.items():
        actual_value = env.get(condition_key)
        if not satisfies(actual_value, condition_value):
            failures.append(PreconditionFailure(
                key=condition_key,
                expected=condition_value,
                actual=actual_value
            ))
    
    if not failures:
        return MatchResult.FULL_MATCH
    elif all(f.key in SOFT_CONDITIONS for f in failures):
        # 软条件不满足 → 降级执行（带警告）
        return MatchResult.SOFT_MATCH
    else:
        # 硬条件不满足 → 不能用这个 Harness
        return MatchResult.NO_MATCH
```

**硬条件 vs 软条件的区分**：

```
硬条件（不满足则不能执行）：
  - OS 类型（MacOS vs Windows，行为根本不同）
  - 应用是否存在（没有 Outlook 不能执行邮件 Harness）
  - 必要权限（没有写权限不能保存文件）

软条件（不满足则降级，仍可尝试）：
  - 浏览器版本（可能大部分步骤仍然有效）
  - 屏幕分辨率（影响坐标，但有适配逻辑）
  - 网络延迟（影响等待时间，但不影响逻辑）
```

### 最终决策逻辑

```python
def select_harness(task, env) -> ExecutionDecision:
    candidates = retrieve_candidates(task.description)
    
    for harness in candidates:
        match = check_preconditions(harness, env)
        if match == FULL_MATCH:
            return UseHarness(harness, confidence="high")
        elif match == SOFT_MATCH:
            return UseHarness(harness, confidence="medium")
    
    # 没有匹配的 Harness
    return FallbackToAgent(reason="no_matching_harness")
```

------

## [2. 构建并验证] Harness Induction 的最小可行实现

这是核心算法，但为了验证框架可行，可以做最简版本：

### 触发条件（什么时候归纳）

```python
INDUCTION_TRIGGERS = {
    "new_harness":      lambda stats: stats.support_count == MIN_SUPPORT,    # 默认 MIN_SUPPORT=3
    "specialization":   lambda stats: stats.new_variation_detected == True,
    "patch":            lambda stats: stats.f2_failure_count >= 2,           # 连续2次F2才触发修复
}
# MIN_SUPPORT=3 的理由：
# 1条 → 可能是偶然
# 2条 → 可能巧合
# 3条 → 足够归纳共性
```

### 归纳流程（最简版）

```python
def induce_harness(experience_records: List[ExperienceRecord]) -> HarnessCandidate:
    
    # Step 1: 提取公共前置条件（交集）
    preconditions = intersect_preconditions(
        [r.observed_environment for r in experience_records]
    )
    
    # Step 2: 提取公共步骤序列（最长公共子序列 + 参数化）
    canonical_steps = extract_canonical_steps(
        [r.steps for r in experience_records]
    )
    # 具体值 → 变量名：
    # click("#user-123") → click(VAR_USER_ID)
    
    # Step 3: 用 LLM 生成可执行代码
    code = llm.generate(
        prompt=HARNESS_SYNTHESIS_PROMPT,
        context={
            "preconditions": preconditions,
            "steps": canonical_steps,
            "example_trajectories": experience_records[:2]  # 给 LLM 参考
        }
    )
    
    # Step 4: 提取不变量和终端验证器
    invariants = extract_invariants(experience_records)
    terminal_verifier = extract_terminal_state(experience_records)
    
    return HarnessCandidate(
        code=code,
        preconditions=preconditions,
        invariants=invariants,
        terminal_verifier=terminal_verifier,
        source_records=experience_records
    )
```

### 验证流程（决定 Harness 是否入库）

```python
def validate_harness(candidate: HarnessCandidate,
                     source_trajectories: List[RawTrajectory]) -> ValidationResult:
    
    results = []
    for trajectory in source_trajectories:
        # 回放验证：用 Harness 重新执行同一任务
        replay_result = sandbox.replay(
            harness=candidate,
            initial_state=trajectory.initial_state,
            task_params=trajectory.task_params
        )
        results.append(replay_result)
    
    success_rate = sum(r.success for r in results) / len(results)
    
    if success_rate >= VALIDATION_THRESHOLD:   # 默认 0.8
        return ValidationResult.APPROVED
    else:
        # 分析失败模式，决定是修复还是放弃
        failure_analysis = analyze_failures(results)
        if failure_analysis.is_fixable:
            return ValidationResult.NEEDS_REVISION(failure_analysis)
        else:
            return ValidationResult.REJECTED
```

**关键**：回放验证必须在**沙盒环境**里进行，不能在真实环境执行，否则验证过程本身会产生副作用（真的发出了邮件、真的删了文件）。

------

## [3. 触发机制] 何时触发各个环节

```
事件流：

任务执行完成
    │
    ├→ [立即] 记录 Raw Trajectory（Layer 0）
    │
    ├→ [立即] 更新 task_type 的统计计数器
    │         stats[task_type].support_count += 1
    │         stats[task_type].last_seen = now()
    │
    ├→ [异步] 检查 Induction 触发条件
    │         if support_count hits threshold → 触发归纳
    │         if new_variation detected → 触发特化分裂
    │
    └→ [异步] 如果执行失败
              → 失败分类（F1/F2/F3/F4）
              → F2 计数器 += 1
              → F2 count >= 2 → 触发 Patch
              → F1 → 记录 precondition gap，等待下次归纳时合并
```

异步触发的好处：不阻塞主执行流，归纳可以在后台批处理。

------

## [4. 积累统计] 每个 task_type 维护的统计结构

```python
@dataclass
class TaskTypeStats:
    task_type: str
    
    # === 数量统计 ===
    total_executions: int = 0
    harness_executions: int = 0      # 用 Harness 执行的次数
    agent_executions: int = 0        # Fallback 到 Agent 的次数
    
    # === 质量统计 ===
    harness_success_rate: float      # Harness 执行成功率
    agent_success_rate: float        # Agent 执行成功率
    # 这两个比较是验证 Harness 有效性的核心指标
    
    # === 失败统计 ===
    failure_counts: Dict[FailureType, int]
    # {F1: 3, F2: 1, F3: 0, F4: 2}
    
    # === 环境覆盖 ===
    observed_environments: Set[EnvironmentFingerprint]
    # 见过哪些环境配置，用于判断当前 Harness 的覆盖范围
    
    # === 归纳状态 ===
    current_harness_version: Optional[str]
    last_induction_time: Optional[datetime]
    pending_variations: List[Variation]  # 积累中但还没触发归纳的变异
    
    # === MDL 估计 ===
    estimated_token_savings: int
    # 粗略估计：agent_executions × avg_tokens_per_task
    #         - harness_executions × avg_tokens_per_harness_call
    # 这个数字是向"公共知识库"愿景论证价值的最直接指标
```

**`estimated_token_savings` 这个指标很重要**——这是你向外部证明整套范式价值的最直白的数字。每积累一次 Harness 执行，它就增长一次。

------

# 公共知识库的演化路径

现在回到你说的服务体系。基于上面的设计，这个演化有一个自然的分层：

```
Level 0：个人知识库（Personal Knowledge Base）
  - 只包含自己的使用习惯和环境配置
  - 完全私有
  - 适合高度个性化的工作流

Level 1：组织知识库（Org Knowledge Base）
  - 企业内部共享
  - 针对企业使用的特定工具版本
  - 包含企业特有的业务流程 Harness

Level 2：公共知识库（Public Knowledge Base）
  - 通用软件（Office、Chrome、Slack...）
  - 高度验证，跨平台覆盖完整
  - 类似 npm 的开放生态

混用逻辑（优先级）：
  个人 Harness > 组织 Harness > 公共 Harness
  
  个人可以"override"公共 Harness：
  比如公共库里 send_email 默认签名为空，
  个人 override 版本自动附加自己的签名
```

这个混用逻辑其实和 Linux 的 PATH 解析、Python 的包查找路径完全一样——**优先级覆盖链**，这是已经被验证过的可行设计 ✅

------

# 验证框架的最小实验设计

为了证明这套流程可行，最小实验只需要验证一件事：

>   **在固定环境下，经过 N 次执行积累，Harness 的成功率是否显著高于 Agent Baseline，且 Token 消耗显著更低？**

```
实验设置：
  环境：T-Bench 或 OSWorld 的子集（20-30 个同类任务）
  对比组：
    A：纯 Agent（每次从零推理）
    B：AutoHarness（前 5 次积累，第 6 次开始使用 Harness）
  
  评估指标：
    1. 成功率（Task Success Rate）
    2. 每任务 Token 消耗
    3. 每任务执行延迟
    4. Harness 的回放验证通过率（内部质量指标）
  
  期望结果：
    B 组在前 5 次与 A 相当（都是 Agent 执行）
    B 组从第 6 次开始成功率 >= A，Token 消耗显著下降
    这条"交叉曲线"就是整个范式可行性的核心证明
```

这个实验设计极简但足够说明问题，不需要复杂的跨域迁移实验，留给后续工作 💡

你觉得这个验证框架的闭环设计是否捕捉了你想验证的核心假设？还是有某个环节你觉得还没说清楚？





# 先回答三个具体问题

## 工作量和论证是否足够？

**足够，但需要定位准确。** 这个工作最适合投的方向是：

```
顶会：NeurIPS / ICML（系统方向 track）、ICLR
     AAAI、IJCAI（Agent/知识工程方向）
顶会 Workshop：NeurIPS Agent Workshop、ICLR LLM Reasoning Workshop
会议：EMNLP、ACL（如果强调语言任务侧）
```

论文的核心贡献点要清晰地声明为以下三个，缺一不可：

1.  **新范式定义**：Knowledge Compilation 作为 Agent 经验积累的正式框架（对应理论贡献）
2.  **系统实现**：AutoHarness 的完整 Induction-Validation-Retrieval 闭环（对应系统贡献）
3.  **实验验证**：成功率 + Token 成本的交叉曲线，证明范式有效性（对应实证贡献）

**工作量的薄弱点只有一个**：如果只在纯文本环境跑，Reviewer 会质疑"这不就是 RAG 的变体吗？"所以论文里必须有一节专门论证 AutoHarness 和 RAG 的本质区别，这个论证要在理论层面说清楚，不只是实验数字。

------

## 纯文本任务环境的 Benchmark

有几个非常合适的选择：

### 首选：τ-bench（tau-bench）

```
论文：tau-bench: A Benchmark for Tool-Agent-User Interaction
环境：零售客服、航空客服两个领域
任务：需要查询数据库、执行政策判断、多轮对话
为什么适合：
  - 任务有明确的前置条件和终态验证
  - 同类任务重复率高（完美的 Harness 积累场景）
  - 有官方 success rate 指标
  - 纯文本 + API 调用，无 GUI 成本
```

### 次选：WebArena（文本子集）

```
可以只取 Reddit、GitLab、Map 等纯文本交互的子集
排除需要图像理解的任务
已有大量 baseline 结果可以对比
```

### 也可以考虑：GAIA

```
通用 AI 助手任务，Level 1 任务主要是文本+工具调用
任务多样性高，可以验证跨任务类型的 Harness 泛化
```

### 最推荐的组合

```
主实验：τ-bench（任务重复性强，最适合验证积累效果）
补充实验：GAIA Level 1（验证泛化性）
```

τ-bench 的任务结构和你的设计几乎完美匹配——它的每个任务都有明确的 policy 规则、数据库状态前置条件、以及可验证的终态，这三个东西直接对应 Harness 的 precondition / invariant / terminal_verifier。

------

## GUI 环境的结构化输出

**有，而且非常成熟。** 主流方案：

### Accessibility Tree（无障碍树）

这是最实用的方案，几乎所有 GUI Agent 框架都在用：

```python
# 每个 UI 元素输出类似：
{
  "role": "button",
  "name": "Send Email",
  "selector": "[data-action='send']",
  "bbox": [1200, 45, 80, 32],
  "enabled": true,
  "parent": "toolbar#main"
}
# 整棵树是 XML/JSON 结构，不需要视觉模型解析
```

主流框架支持情况：

```
OSWorld：     ✅ 提供完整 Accessibility Tree
WebArena：    ✅ 提供 DOM 树（HTML 结构）
WindowsAgentArena：✅ UI Automation Tree（Windows 原生）
Android World：✅ View Hierarchy（XML）
```

**对你的意义**：GUI 环境下，Harness 的 selector 可以直接用 Accessibility Tree 里的稳定属性（`role` + `name` 组合），比 XPath 或坐标稳定得多。这也意味着 Harness 在同一应用的不同版本之间有一定的天然鲁棒性——只要按钮的 `role` 和 `name` 没变，selector 就还有效。

所以 GUI 环境完全可行，只是第一版为了节省成本，先在纯文本环境跑通，GUI 作为 Section 5 的 "Extended Evaluation" 或 Future Work。





# 所以具体 Benchmark 的选择如何确定，核心想法是先找一些更容易出现工作重复任务，成本较低的场景作为主要对比对象， OSWorld 2.0 、AgentGym2、 tao-Gym 、TerminalBench、等。 以及当前方法适合比对的baseline 怎么找， 一些 Harness？ 还是直接用 agent？ 还是直接用 GYM 本身已有的 leaderboard。对比实验 或者其他实验如何设计，分数如何评估，数据集 环境任务等 如何区分积累经验和实际环境的部分，是否要添加 积累模式 和 使用模式



------

# 一、Benchmark 选择的决策框架

在选之前，先明确你的 Benchmark **必须满足的条件**：

```
必要条件：
  C1：任务有足够的重复结构（同类任务出现频率高）
      → 不满足这个，积累就没有意义
  C2：任务有可验证的终态（success / failure 判断明确）
      → 不满足这个，实验指标没有说服力
  C3：环境可重置（同一任务可以多次执行）
      → 不满足这个，回放验证和积累实验都无法进行
  C4：成本可控（API 成本 + 执行成本）
      → 纯文本优先，GUI 次之

加分条件：
  C5：任务有自然的子任务结构（利于 Harness 分解）
  C6：已有强 baseline 结果（便于比较）
  C7：社区活跃，有持续维护
```

逐一评估你列出的选项：

------

## 各 Benchmark 评估

### τ-bench ✅✅✅ 强烈推荐

```
C1 重复结构：✅ 极强
  零售域：查询订单、退款申请、修改地址 → 高度同构
  航空域：改签、退票、升舱 → 高度同构
  同类任务在 500+ 条测试集里大量重复出现

C2 终态验证：✅ 有官方 success judge
  基于数据库状态变化的确定性判断

C3 环境可重置：✅ 每个 episode 独立数据库实例

C4 成本：✅✅ 纯 API 调用，无 GUI，极低

C5 子任务结构：✅ 每个任务自然分解为
  查询 → 验证 policy → 执行操作 → 确认

C6 已有 baseline：✅ 有 GPT-4o / Claude 等官方结果

C7 社区维护：✅ 持续更新，arxiv 2024
```

**结论：主实验首选，没有争议。**

------

### OSWorld 2.0 ⚠️ 慎选，作为扩展

```
C1 重复结构：⚠️ 中等
  任务多样性很高，重复结构不如 τ-bench 明显
  但 GUI 操作子步骤（打开文件、滚动、点击）高度重复

C2 终态验证：✅ 有截图 diff + 状态判断

C3 环境可重置：✅ 虚拟机快照

C4 成本：❌ 高
  每个 episode 需要跑虚拟机 + 视觉模型
  实验成本 10-100x τ-bench

C5 子任务结构：✅✅ 非常好
  这是 Harness 树状调用最自然的场景

C6 已有 baseline：✅✅ leaderboard 非常完整

C7 社区维护：✅✅ 最活跃的 Agent Benchmark 之一
```

**结论：如果有资源，作为 Section 5 的扩展实验，证明方法不局限于文本环境。不作为主实验。**

------

### AgentGym2 ⚠️ 观望

```
C1 重复结构：✅ 多环境聚合，可以筛选重复任务多的子集
C2 终态验证：⚠️ 各子环境质量参差不齐
C3 环境可重置：⚠️ 取决于具体子环境
C4 成本：⚠️ 混合环境，成本不确定
C6 已有 baseline：⚠️ 较新，baseline 结果不够丰富
```

**结论：不推荐作为主实验。如果有某个子环境特别契合，可以单独拿出来用。**

------

### TerminalBench ✅ 值得考虑

```
C1 重复结构：✅ Shell 操作高度重复
  文件操作、进程管理、环境配置 → 极度同构

C2 终态验证：✅ 基于文件系统状态、命令输出的确定性判断

C3 环境可重置：✅ Docker 容器，重置极快

C4 成本：✅✅ 纯 CLI，成本极低

C5 子任务结构：✅✅ Shell 命令天然是可组合的原子操作

C6 已有 baseline：⚠️ 较新，结果不够丰富

C7 社区：⚠️ 刚发布，社区规模小
```

**结论：作为补充实验非常好，特别是如果你想证明 Harness 在纯符号环境（无自然语言歧义）下效果更强。**

------

### tau-Gym / GYM 系列 ⚠️

需要搞清楚你说的是哪个具体实现。如果是基于 τ-bench 的 gym wrapper，那和 τ-bench 本质一样，直接用 τ-bench 即可。如果是其他 gym，需要单独评估。

------

## 最终 Benchmark 组合建议

```
主实验（必须）：
  τ-bench（零售 + 航空两个域）
  → 证明核心假设：积累后 Harness 优于 Agent

补充实验（强烈建议）：
  TerminalBench（子集）
  → 证明方法在纯符号环境下同样有效
  → 两个环境性质互补，增强论文说服力

扩展实验（如有资源）：
  OSWorld 2.0（子集，选重复性高的任务类别）
  → 证明方法可以扩展到 GUI 环境
  → 放在 Appendix 或 Section 6 作为 future work 的先期探索
```

------

# 二、Baseline 的选择

Baseline 的设计是论文最容易被 Reviewer 攻击的地方，必须覆盖三个层次：

## 层次 1：能力下界（证明任务本身有挑战性）

```
Baseline A：Vanilla LLM（Zero-shot）
  无工具、无历史、直接用 LLM 回答
  目的：建立任务难度的下界
  预期：成功率最低
```

## 层次 2：当前主流方案（证明你比现有方法好）

```
Baseline B：ReAct Agent
  标准的 Reasoning + Acting 框架
  无记忆、无积累
  目的：代表"当前最常用的 Agent 范式"
  预期：成功率中等，Token 消耗高

Baseline C：RAG Agent
  检索历史成功轨迹作为 few-shot 上下文
  目的：代表"最主流的记忆增强方案"
  关键：必须用和 AutoHarness 完全相同的历史数据
         才能公平比较
  预期：成功率略优于 B，但 Token 消耗同样高
```

## 层次 3：方法变体（消融用）

```
Baseline D：AutoHarness w/o Validation
  跳过沙盒回放验证直接入库
  目的：证明验证步骤的价值

Baseline E：AutoHarness w/o Versioning
  失败时直接覆盖旧 Harness，不维护 DAG
  目的：证明版本管理的必要性

Baseline F：AutoHarness w/ Fixed Harness
  预先人工编写 Harness，不做自动归纳
  目的：给出方法的"性能上界"，
         同时证明自动归纳能接近人工质量
```

## 关于 Leaderboard 的问题

**不能直接用 Leaderboard 的数字作为 Baseline**，原因：

```
1. Leaderboard 上的方法使用的模型版本可能不同
2. Leaderboard 上的方法可能用了特殊的 prompt 工程
3. 你的实验有"积累阶段"，Leaderboard 方法没有
   → 对比的不是同一个问题

正确做法：
  自己在同一模型（如 GPT-4o）上复现 Baseline B 和 C
  在论文里说明："我们使用 GPT-4o 作为 backbone，
  在相同实验设置下与 ReAct 和 RAG 比较"
  然后在 footnote 里附上 Leaderboard 的参考数字
```

------

# 三、实验设计的完整方案

## 核心设计原则：积累模式 vs 使用模式的分离

**必须要有这个分离**，而且要非常明确，否则实验存在数据泄露。

```
完整流程：

┌─────────────────────────────────────────────────────┐
│                  数据划分                            │
│                                                     │
│  全量任务集 (N 个任务实例)                           │
│       │                                             │
│       ├── Warm-up Pool（热身池）                     │
│       │   每个任务类型取 K 个实例                     │
│       │   用于 Harness 初始积累                      │
│       │   K = MIN_SUPPORT = 3（实验参数）             │
│       │                                             │
│       └── Evaluation Pool（评估池）                  │
│           剩余实例，用于正式评估                      │
│           与 Warm-up Pool 完全不重叠                 │
└─────────────────────────────────────────────────────┘
```

**关键约束**：

```
约束 1：Warm-up Pool 和 Evaluation Pool 的任务实例不重叠
        （同一任务类型可以出现在两个池里，但具体参数不同）
        例：零售域的"退款"任务：
          Warm-up：订单 #001, #002, #003 → 积累 Harness
          Evaluation：订单 #004, #005... → 评估

约束 2：所有 Baseline（包括 RAG）使用完全相同的 Warm-up Pool 数据
        RAG 把 Warm-up 轨迹作为检索库
        AutoHarness 把 Warm-up 轨迹编译成 Harness
        → 确保比较的是"如何利用历史数据"，而非"用了多少历史数据"

约束 3：Evaluation Pool 中，Vanilla Agent 和 AutoHarness 面对完全相同的任务序列
```

------

## 实验一：主实验（核心假设验证）

**问题**：经过 Warm-up 积累后，AutoHarness 是否优于 Baseline？

```
设置：
  数据集：τ-bench 零售域
  任务类型数：~10 个主要类型（退款、改地址、查询状态...）
  每类 Warm-up：3 个实例
  每类 Evaluation：20-30 个实例
  Backbone：GPT-4o（统一）

评估指标：
  1. Task Success Rate（↑）
  2. Average Tokens per Task（↓）
  3. Average Latency per Task（↓）
  4. Harness Hit Rate：评估阶段中 Harness 被成功激活的比例（↑）

预期结果表格：

| 方法                  | 成功率  | Token/任务 | 延迟/任务 |
|-----------------------|---------|------------|-----------|
| Vanilla LLM           | ~45%    | ~2000      | ~8s       |
| ReAct Agent           | ~65%    | ~3500      | ~12s      |
| RAG Agent             | ~70%    | ~4500      | ~15s      |
| AutoHarness（本文）   | ~75%    | ~800       | ~3s       |
| AutoHarness Upper Bound| ~80%   | ~600       | ~2s       |
（数字是预期量级，非真实结果）
```

**核心图表**：一张"积累曲线图"

```
x轴：任务序号（1 到 100）
y轴：滚动平均成功率

显示四条线：
  - Vanilla Agent：水平线（无积累效应）
  - RAG Agent：略微上升（few-shot 效果）
  - AutoHarness：前 K 次平平，第 K+1 次开始显著上升
                 在某个点超过所有 Baseline（交叉点）

这张图就是整篇论文最核心的一张图 📈
```

------

## 实验二：Harness 质量分析

**问题**：自动归纳的 Harness 质量如何？

```
评估维度：

1. 回放验证通过率
   Harness 在 Warm-up 轨迹上的回放成功率
   目标：≥ 80%

2. 泛化成功率
   Harness 在 Evaluation Pool 新实例上的成功率
   vs 在 Warm-up 实例上的成功率
   目标：泛化率下降 < 10%（说明没有过拟合到 Warm-up 实例）

3. Harness 精简度
   Harness 的平均步骤数 vs 原始轨迹的平均步骤数
   目标：Harness 步骤数 < 原始轨迹步骤数的 60%
   （说明成功抽象，不是简单复制）

4. 前置条件覆盖完整性
   手动检查 N=50 个 Harness，评估前置条件是否覆盖了
   执行中实际需要的所有环境依赖
   （这个需要人工评估，但样本量不大）
```

------

## 实验三：消融实验

```
对照组                    | 目的
--------------------------|------------------------------------------
w/o Invariant Monitor    | 不变量监控对成功率的贡献
w/o Sandbox Validation   | 验证步骤对 Harness 质量的影响
w/o Version DAG          | 版本管理对长期积累的影响（在长序列任务上）
MIN_SUPPORT = 1          | 过早归纳的代价（过拟合）
MIN_SUPPORT = 5          | 过晚归纳的代价（冷启动期过长）
MIN_SUPPORT = 3（默认）  | 最优点
```

MIN_SUPPORT 的消融实验会产生一个非常好看的倒 U 形曲线（太低过拟合，太高冷启动太长），这是论文里很好的分析点。

------

## 实验四：τ-bench 两域迁移（如有余力）

```
设置：
  在零售域完成完整的 Warm-up + 积累
  提取其中的子任务级 Harness（查询操作、验证操作等）
  在航空域任务上，允许复用这些子任务 Harness

评估：
  纯 AutoHarness（只用航空域 Warm-up）vs
  AutoHarness + 零售域子任务迁移

预期：迁移版本在冷启动期（前 K 次）表现更好，
      因为可以复用通用子任务 Harness
```

------

## 实验五：TerminalBench 补充（验证跨环境有效性）

```
目的：证明方法不局限于自然语言对话场景
设置：完全相同的实验流程，换一个环境
关键差异：Shell 命令 Harness 的表示和文本任务不同
          → 说明框架的环境无关性
```

------

# 四、积累模式和使用模式的形式化

你问的这个问题很关键，**必须在论文里显式定义**，否则 Reviewer 会质疑实验公平性。

```python
# 形式化定义两种模式

class SystemMode(Enum):
    ACCUMULATION = "accumulation"
    # 行为：
    # 1. 无论知识库里有没有 Harness，都先用 Agent 执行
    # 2. 执行完后录制轨迹，更新统计，检查归纳触发
    # 3. 如果触发归纳，在后台异步生成 Harness
    # 注意：积累模式下不使用 Harness 执行，
    #       避免 Harness 质量未经验证就影响数据收集

    DEPLOYMENT = "deployment"
    # 行为：
    # 1. 优先检索并使用 Harness
    # 2. 无匹配时 Fallback 到 Agent
    # 3. 继续录制轨迹（Online Learning，持续更新）
    # 4. 失败时触发 Harness 修订

# 实验中的模式切换：
# Phase 1：前 K×|task_types| 个任务 → ACCUMULATION 模式
# Phase 2：剩余任务 → DEPLOYMENT 模式
# 切换点是明确的、预先定义的，不是动态的
```

**为什么积累阶段不允许用 Harness？**

防止自举（Bootstrapping）问题：如果积累阶段就用 Harness 执行，那 Harness 的质量会影响后续 Harness 的归纳质量，形成循环依赖，实验结果难以解释。第一版系统用干净的 Agent 轨迹作为归纳素材，逻辑更清晰。

------

# 五、评分机制的设计

τ-bench 自带 success judge，但你需要在其基础上增加一层：

```python
@dataclass
class EpisodeScore:
    # τ-bench 原生指标
    task_success: bool           # 任务是否成功
    
    # AutoHarness 新增指标
    execution_path: Literal[
        "harness",               # 用 Harness 执行
        "harness_with_fallback", # Harness 部分执行后降级
        "agent_fallback",        # 完全 Fallback 到 Agent
    ]
    tokens_used: int
    latency_seconds: float
    harness_id_used: Optional[str]    # 用了哪个 Harness
    failure_type: Optional[str]       # 如果失败，是哪种类型

# 汇总指标
@dataclass
class ExperimentSummary:
    # 效果指标
    task_success_rate: float
    harness_success_rate: float          # 仅 Harness 路径的成功率
    agent_fallback_success_rate: float   # 仅 Fallback 路径的成功率
    
    # 效率指标
    avg_tokens_per_task: float
    avg_tokens_harness_path: float
    avg_tokens_agent_path: float
    token_savings_vs_baseline: float     # 相对 ReAct 的 Token 节省率
    
    # 覆盖指标
    harness_hit_rate: float              # Evaluation 中 Harness 被激活的比例
    harness_coverage_by_type: Dict[str, float]  # 按任务类型的覆盖率
    
    # 质量指标
    harness_precision: float             # Harness 执行 → 成功 的比例
    harness_recall: float                # 可用 Harness 路径 → 实际使用 的比例
```

------

# 总结：最小可行实验套餐

如果时间和资源有限，**必须做的实验**：

```
✅ 主实验（τ-bench 零售域）
   → 积累曲线图 + 成功率/Token 对比表

✅ 消融实验（w/o Validation + MIN_SUPPORT 变化）
   → 2-3 个关键组件的贡献

✅ Harness 质量分析（回放通过率 + 泛化率）
   → 证明归纳出来的 Harness 真的有质量
```

**有余力再做的**：

```
⭕ τ-bench 航空域（验证跨域）
⭕ TerminalBench（验证跨环境）
⭕ 迁移实验（子任务复用）
```

这个组合对于一篇 workshop 论文已经完全够用，对于主会论文在有 OSWorld 扩展实验的情况下也有竞争力 💪



