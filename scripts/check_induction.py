"""检查 autoharness 归纳触发情况。"""
import os
import sys
import logging

os.environ["EOS_LLM_BACKEND"] = "deepinfra"
os.environ["EOS_DEEPINFRA_MODEL"] = "zai-org/GLM-5.2"
if os.environ.get("DEEPINFRA_TOKEN"):
    os.environ["DEEPINFRA_API_KEY"] = os.environ["DEEPINFRA_TOKEN"]

sys.path.insert(0, "tau2-bench/src")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from experience_os.config import Config
from experience_os.repository import Repository
from experience_os.compiler import HarnessInductor
from experience_os.llm import LLMClient
from experience_os.environment import MockEnvironment
from experience_os.tau2_adapter import Tau2Environment, infer_task_type
from pathlib import Path
import json

cfg = Config()
cfg.data_dir = Path(".experience_os_data")
cfg.llm.backend = "deepinfra"
cfg.llm.deepinfra_model = "zai-org/GLM-5.2"
cfg.ensure_dirs()
repo = Repository(cfg)

print("=== 轨迹统计 ===")
print(f"总轨迹: {len(repo._trajectories)}")
# 重建 warmup 任务列表（从 train split 取同类型前 N 条）
from experience_os.experiments.compare import load_train_test_split
train_tasks, test_tasks = load_train_test_split("retail")
TASK_TYPE = "exchange_delivered_order_items"
train_tasks = [t for t in train_tasks if infer_task_type(t) == TASK_TYPE]
warmup_tasks = train_tasks[:5]
warmup_map = {t.id: t for t in warmup_tasks}
print(f"  warmup_tasks: {[t.id for t in warmup_tasks]}")
print(f"  traj task_ids: {[t.task_id for t in repo._trajectories.values()]}")

for tt in repo.all_task_types():
    trajs = repo.trajectories_for_type(tt, success_only=False)
    success = repo.trajectories_for_type(tt, success_only=True)
    print(f"  {tt}: total={len(trajs)} success={len(success)}")
    print(f"    support_count={repo.support_count(tt)}  min_support={cfg.induction.min_support}")

llm = LLMClient(cfg.llm)
inductor = HarnessInductor(cfg, llm, repo)
for tt in repo.all_task_types():
    trigger = inductor.check_triggers(tt)
    print(f"  check_triggers({tt}) -> {trigger}")

print(f"\n=== Harness 统计 ===")
harnesses = repo.active_harnesses()
print(f"活跃 harness: {len(harnesses)}")
for h in harnesses:
    print(f"  {h.full_name} (status={h.status})")

# env_builder：按轨迹重建独立 Tau2 环境
def _env_builder(traj):
    t = warmup_map.get(traj.task_id)
    if t is not None:
        return Tau2Environment("retail", t, solo_mode=False)
    return Tau2Environment("retail", warmup_tasks[0], solo_mode=False)

# 尝试直接 induce
print(f"\n=== 尝试 induce ===")
venv = Tau2Environment("retail", warmup_tasks[0], solo_mode=False)
for tt in repo.all_task_types():
    if repo.support_count(tt) >= cfg.induction.min_support:
        print(f"  induce({tt})...")
        try:
            h = inductor.induce(tt, venv, env_builder=_env_builder)
            if h:
                print(f"    OK: {h.full_name}")
                print(f"    pre={h.preconditions}")
                print(f"    steps={len(h.parameterized_steps)}")
                print(f"    code_len={len(h.code or '')}")
                print(f"    code preview (first 500 chars):")
                print((h.code or h.procedure_code or "")[:500])
            else:
                print(f"    None (induce 返回空)")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {exc}")
