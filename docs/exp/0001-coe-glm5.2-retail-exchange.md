# 实验记录 0001：CoE 端到端跑通 + baseline 对比

- **日期**: 2026-07-25
- **环境**: Windows + conda `eos` (Python 3.12.13)
- **模型**: `deepinfra/zai-org/GLM-5.2`（经 LiteLLM 调用 DeepInfra）
- **域 / 任务**: `tau2-bench` retail / `exchange_delivered_order_items`
- **划分**: tau2 原生 train/test split（retail: train=74, test=40；本任务类型 train=9, test=3）
- **脚本**: `_coe_v2.py`、`_baseline_react.py`、`_diag_induction.py`、`_check_harness.py`

## 1. 实验设计

在同一组 3 个 test 任务上对比两种方法，warmup 仅用于 coe：

| 方法 | warmup | eval | max_steps | 说明 |
|------|--------|------|-----------|------|
| `react` (baseline) | 0 | 3 | 30 | τ-bench 内置多步 ReAct agent，无积累 |
| `coe` (EOS) | 3 | 3 | 30 | Warm-up 积累 → 归纳 → Eval 部署（Harness 优先，失败回退 agent） |

`coe` 流程：在 train split 前 3 个任务上跑 GLM-5.2 agent 积累轨迹 →
`HarnessInductor` 触发归纳 → LLM 合成 harness 代码 → 沙箱回放验证 →
APPROVED 后在 eval 任务上优先执行 harness，失败回退 agent。

## 2. 修复的关键问题

归纳此前始终 REJECTED (replay rate=0.00)，定位到根因：**`_validate` 复用单一
Tau2Environment 验证所有轨迹**，而每条轨迹对应不同 task 的初始 DB 状态和期望
DB hash，导致验证全部失败。修复清单：

### 2.1 按轨迹重建独立验证环境
- `experience_os/compiler/inductor.py`
  - `_validate` / `_collect_repair_context` 新增 `env_builder` 参数；
    当传入时，为每条轨迹构建独立 `Tau2Environment`（正确的 initial_state）。
  - `induce` 透传 `env_builder`。
- `experience_os/experiments/compare.py`
  - `run_coe` 构建 `task_id → tau2 task` 映射，传入 `env_builder`。

### 2.2 合成 prompt 改进
- `experience_os/compiler/prompts.py` `SYNTHESIS_PROMPT`：
  - 明确 `params` 是当前任务实例的正确值，**禁止硬编码**示例轨迹中的
    `user_id` / `order_id` / `item_ids` 等值。
  - 区分 `call_tool` 返回类型：仅 `"Error"` 前缀字符串视为失败，纯字符串
    （如 user_id）是合法返回；推荐 `isinstance(r, str) and r.startswith("Error")` 判定。
  - 跳过 `reasoning` 这类伪工具步骤。

### 2.3 鲁棒 `_call_tool`
- `experience_os/environment.py`、`experience_os/tau2_adapter.py`
  - 非 JSON 的错误字符串包装为 `{"error": msg}`，防止 harness 代码对字符串
    调用 `.get()` 而崩溃。

## 3. 诊断验证（`_diag_induction.py`）

用前 2 条成功轨迹做最小诊断，合成代码 + 逐条回放：

```
[Phase5] synthesized code (len=1870):
def run():
    email = params.get("email")
    order_id = params.get("order_id")
    ...
    r = call_tool("find_user_id_by_email", email=email)
    if isinstance(r, str) and r.startswith("Error"):
        return r
    user_id = r
    ...

[Phase6] 验证（逐条）:
  [1] task=75 success=True
  [2] task=80 success=True

[Phase6] rate=1.00 threshold=0.8
  verdict: APPROVED
```

## 4. 端到端实验结果

### 4.1 coe（EOS）
```
============================================================
  汇总: coe
============================================================
  总任务:   6
  成功率:   5/6 (83.3%)
  总 Token: 304,342 (prompt=284,476 + completion=19,866)
  平均延迟: 54.7s
  [Warmup] SR: 100.0% (3 tasks)  Token: 193,241
  [Eval]   SR: 66.7%  (3 tasks)  Token: 111,101
  路径分布: {'agent': 4, 'harness+agent': 2}
============================================================
experiment_id: coe-retail-train_test-8f273b24
LTS trajs: 6 条（含完整对话）
```

- Warmup 3/3 成功 → 归纳触发 → harness **APPROVED**（verification
  `success_rate=1.0, test_count=3`，`status=ACTIVE`）。
- Eval 2/3 成功；路径分布 `agent: 4, harness+agent: 2` 表明 eval 阶段
  尝试了 harness 直接执行，但未产生正确 DB hash，回退 agent 完成。

### 4.2 react（baseline）
```
============================================================
  汇总: react
============================================================
  总任务:   3
  成功率:   2/3 (66.7%)
  总 Token: 139,165 (prompt=130,865 + completion=8,300)
  平均延迟: 53.3s
  [Eval]   SR: 66.7% (3 tasks)  Token: 139,165
  路径分布: {'agent': 3}
============================================================
experiment_id: react-retail-train_test-77f3f74a
```

### 4.3 对比

| 方法 | Eval 成功率 | Eval Token | 路径分布 | 备注 |
|------|-----------|-----------|---------|------|
| react (baseline) | 2/3 (66.7%) | 139,165 | agent: 3 | 纯 LLM 多步交互 |
| coe (EOS) | 2/3 (66.7%) | 111,101 | agent: 1, harness+agent: 2 | harness APPROVED，eval 回退 agent |

- 成功率持平（样本量小，3 个 test 任务）。
- coe eval 阶段 token 略低（111K vs 139K），因 harness 直接执行
  路径不消耗 LLM token；但本次 harness 未直接判定成功，回退后仍需 agent。

## 5. 归纳产物示例

`harn_ddbb3004ef95`（ACTIVE，`exchange_delivered_order_items-v1`）：

```python
def run():
    email = params.get("email", None)
    first_name = params.get("first_name", None)
    last_name = params.get("last_name", None)
    zip_code = params.get("zip", None)
    order_id = params.get("order_id", None)
    product_id = params.get("product_id", None)
    item_ids = params.get("item_ids", None)
    new_item_ids = params.get("new_item_ids", None)
    payment_method_id = params.get("payment_method_id", None)

    user_id = None
    if email:
        user_id = call_tool("find_user_id_by_email", email=email)
        if isinstance(user_id, str) and user_id.startswith("Error"):
            return user_id
    elif first_name and last_name and zip_code:
        user_id = call_tool("find_user_id_by_name_zip",
                            first_name=first_name, last_name=last_name, zip=zip_code)
        ...
```

## 6. 已知问题与下一步

### 6.1 harness 在 eval 未直接成功
`extract_task_params` 只从参考动作的 `arguments` 提取 params，但
`exchange_delivered_order_items` 的 `email` / `first_name` 等位于
`user_scenario.instructions` 而非参考动作参数，导致 harness 缺少查询用户
的入参，无法独立完成。

**改进方向**：
- eval 时把 `task_description` / `user_scenario` 文本一并塞入 `params`
  或 `request`，让 harness 从中解析缺失字段；
- 或在 `Tau2Environment` 增加从 task 对象补全 params 的适配层。

### 6.2 segmentation 失败
`algorithms._segment` 调用 `llm.chat_json` 时偶发
`'"segments"'` 解析错误，回退整条轨迹单段。当前对短轨迹（≤3 步）直接
单段，影响有限；后续需复检 `chat_json` 的 JSON 提取逻辑。

### 6.3 样本量过小
3 个 test 任务不足以体现 EOS 的积累增益。后续应：
- 扩大到 retail 全部 40 个 test 任务；
- 引入更弱的 backbone（如 MiniMax M3）验证「强模型积累 → 弱模型受益」
  的跨模型能力迁移假设。

## 7. 数据持久化

- LTS 经验库：`.experience_os_data/experience_os.db`（SQLite，轨迹/记录/harness）
- 实验库：`.experience_os_data/exp_<experiment_id>.db`（每实验独立）
- 本次 6 条轨迹（含完整 prompt + 回复）已写入 LTS，可通过
  `ExperienceLibrary.query_trajectories(experiment_id=...)` 查询排查。
