"""验证 tau2 retail split 加载 + 轨迹转换。"""
import sys
sys.path.insert(0, "tau2-bench/src")

from experience_os.experiments.compare import load_train_test_split
from experience_os.tau2_adapter import infer_task_type, convert_simulation
from tau2.runner.build import build_environment, build_orchestrator
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.simulation import run_simulation

print("=== 1. tau2 retail split 加载 ===")
train, test = load_train_test_split("retail")
print(f"train: {len(train)}, test: {len(test)}")

# 统计 train 的 task_type 分布
from collections import Counter
type_counts = Counter(infer_task_type(t) for t in train)
print(f"train task_type 分布 (top 10):")
for tt, cnt in type_counts.most_common(10):
    print(f"  {tt}: {cnt}")

# 选一个简单的 task_type 做实验
simple_type = type_counts.most_common(1)[0][0]
print(f"\n选择 task_type: {simple_type}")

# 筛选该类型的任务
train_filtered = [t for t in train if infer_task_type(t) == simple_type]
test_filtered = [t for t in test if infer_task_type(t) == simple_type]
print(f"filtered: train={len(train_filtered)}, test={len(test_filtered)}")

print("\n=== 2. 单任务轨迹转换验证 ===")
task = train_filtered[0]
print(f"task_id: {task.id}")
desc = task.description
desc_text = str(desc.text if hasattr(desc, "text") else desc) if desc else ""
print(f"task_description: {desc_text[:100]}")

# 不实际调用 LLM，只验证环境构建
print("\n构建环境...")
env = build_environment("retail", solo_mode=False)
print(f"环境构建成功, tools: {len(env.get_tools())} 个")

print("\n=== 验证完成 ===")
