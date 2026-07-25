"""Quick GLM 5.2 test: run 1 τ-bench task and check completion."""
import os, sys, time

# Strip proxy
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'ALL_PROXY', 'all_proxy']:
    os.environ.pop(var, None)

dt = os.environ.get('DEEPINFRA_TOKEN', '')
if dt and not os.environ.get('DEEPINFRA_API_KEY'):
    os.environ['DEEPINFRA_API_KEY'] = dt

# LiteLLM needs this
os.environ['LITELLM_LOG'] = 'WARN'

from tau2.data_model.simulation import TextRunConfig
from tau2.runner.build import build_orchestrator
from tau2.runner.simulation import run_simulation
from tau2.domains.retail.environment import get_tasks
from experience_os.tau2_adapter import infer_task_type

# Load tasks
tasks = get_tasks("base")
print(f"Total retail tasks: {len(tasks)}")

# Pick return_delivered_order_items tasks (simpler, single-action criteria)
target_type = "return_delivered_order_items"
candidates = []
for t in tasks:
    tt = infer_task_type(t)
    if tt == target_type:
        candidates.append(t)

task = candidates[0] if candidates else tasks[0]
tt = infer_task_type(task)
desc = str(getattr(task, 'description', ''))[:100]
print(f"Task: {getattr(task, 'id', '?')} — type={tt}")
print(f"Desc: '{desc}'")

# Run τ-bench with GLM 5.2 via DeepInfra
MODEL = "deepinfra/zai-org/GLM-5.2"

config = TextRunConfig(
    domain="retail",
    agent="llm_agent",
    llm_agent=MODEL,
    llm_args_agent={"temperature": 0.0, "max_tokens": 4096},
    user="user_simulator",
    llm_user=MODEL,
    llm_args_user={"temperature": 0.0, "max_tokens": 4096},
    max_steps=20,
    num_trials=1,
    seed=42,
)

t0 = time.time()
print(f"\nStarting simulation with {MODEL}...")
sys.stdout.flush()

orch = build_orchestrator(config, task, seed=42)
sim = run_simulation(orch)

elapsed = time.time() - t0
reward = sim.reward_info.reward if sim.reward_info else 0.0
cost = int(getattr(sim, "agent_cost", 0) or 0)

print(f"\n{'='*50}")
print(f"  Result: {'✅ SUCCESS' if reward >= 1.0 else '❌ FAIL'}")
print(f"  Reward: {reward}")
print(f"  Cost (agent_cost): {cost}")
print(f"  Time: {elapsed:.0f}s")

# Show message count
msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
print(f"  Messages: {len(msgs)} turns")

# Show the first user message (task description) and last few steps
for m in msgs[:3]:
    role = getattr(m, 'role', str(type(m)))
    content = str(getattr(m, 'content', ''))[:150]
    print(f"  [{role}] {content}")

print("\n  Last steps:")
for m in msgs[-4:]:
    role = getattr(m, 'role', str(type(m)))
    content = str(getattr(m, 'content', ''))[:150]
    tool_calls = getattr(m, 'tool_calls', None)
    if tool_calls:
        for tc in tool_calls:
            print(f"  [TOOL] {getattr(tc, 'name', '?')}({str(getattr(tc, 'arguments', {}))[:100]})")
    else:
        print(f"  [{role}] {content}")
