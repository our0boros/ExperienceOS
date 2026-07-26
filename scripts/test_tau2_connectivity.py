"""Quick test: τ-bench + DeepInfra (no proxy)."""
# Must strip proxy BEFORE any imports
import os
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'ALL_PROXY', 'all_proxy', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE']:
    os.environ.pop(var, None)

# litellm expects DEEPINFRA_API_KEY, we use DEEPINFRA_TOKEN
dt = os.environ.get('DEEPINFRA_TOKEN', '')
if dt and not os.environ.get('DEEPINFRA_API_KEY'):
    os.environ['DEEPINFRA_API_KEY'] = dt

from tau2.data_model.simulation import TextRunConfig
from tau2.runner.build import build_orchestrator
from tau2.runner.simulation import run_simulation
from tau2.domains.retail.environment import get_tasks

tasks = get_tasks("base")
print(f"Total retail tasks: {len(tasks)}")

# Find find_user_id_by_email tasks
task = None
for t in tasks:
    ec = getattr(t, "evaluation_criteria", None)
    if ec and hasattr(ec, "actions") and ec.actions:
        name = str(ec.actions[0].name)
        if "find_user_id_by_email" in name:
            task = t
            break
if not task:
    task = tasks[0]

desc = str(getattr(task, 'description', ''))[:80]
print(f"Task: {getattr(task, 'id', '?')} - {desc}")

config = TextRunConfig(
    domain="retail",
    agent="llm_agent",
    llm_agent="deepinfra/MiniMaxAI/MiniMax-M2.7",
    llm_args_agent={"temperature": 0.0, "max_tokens": 4096},
    user="user_simulator",
    llm_user="deepinfra/MiniMaxAI/MiniMax-M2.7",
    llm_args_user={"temperature": 0.0, "max_tokens": 4096},
    max_steps=20,
    num_trials=1,
    seed=42,
)

orch = build_orchestrator(config, task, seed=42)
sim = run_simulation(orch)
reward = sim.reward_info.reward if sim.reward_info else 0.0
print(f"Reward: {reward}")
print(f"Agent cost: {getattr(sim, 'agent_cost', 0)}")

# Show how many steps were used
msgs = sim.get_messages() if hasattr(sim, 'get_messages') else (sim.messages or [])
print(f"Total messages: {len(msgs)}")
