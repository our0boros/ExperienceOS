"""检查实验后的 harness 状态和 eval 路径。"""
import os, sys, json
os.environ["EOS_LLM_BACKEND"] = "deepinfra"
os.environ["EOS_DEEPINFRA_MODEL"] = "zai-org/GLM-5.2"
sys.path.insert(0, "tau2-bench/src")
from experience_os.config import Config
from experience_os.repository import Repository
from pathlib import Path

cfg = Config()
cfg.data_dir = Path(".experience_os_data")
cfg.ensure_dirs()
repo = Repository(cfg)

print(f"=== 轨迹 ===")
for tt in repo.all_task_types():
    trajs = repo.trajectories_for_type(tt, success_only=False)
    print(f"  {tt}: total={len(trajs)} success={sum(1 for t in trajs if t.outcome=='success')}")

print(f"\n=== Harness ===")
harnesses = list(repo._harnesses.values())
print(f"总数: {len(harnesses)}")
for h in harnesses:
    print(f"  {h.full_name} status={h.status} verification={h.verification}")
    print(f"    task_type={h.task_type}")
    print(f"    code_len={len(h.procedure_code or '')}")
    print(f"    code preview:")
    print((h.procedure_code or "")[:800])
    print("    ---")
