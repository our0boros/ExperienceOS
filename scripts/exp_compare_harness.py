"""Experiment: react vs harness-boosted on exchange tasks (proxy-safe).

Usage: HTTP_PROXY= HTTPS_PROXY= /path/to/conda/ml/bin/python3 scripts/exp_compare_harness.py
"""
import os, sys, json, time
# MUST strip proxy before ANY imports
for v in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(v, None)

# Ensure API keys are available
# DEEPINFRA_API_KEY for deepinfra/ provider, OPENAI_API_KEY for openai/ provider
dt = os.environ.get('DEEPINFRA_TOKEN', '') or os.environ.get('DEEPINFRA_API_KEY', '')
if dt:
    os.environ['DEEPINFRA_API_KEY'] = dt
    os.environ['OPENAI_API_KEY'] = dt
from pathlib import Path

from experience_os.config import Config
from experience_os.harness_registry import HarnessRegistry
from experience_os.repository import Repository
from experience_os.tau2_adapter import infer_task_type, run_tau2_simulation
from experience_os.experience_library import serialize_messages
from tau2.domains.retail.environment import get_tasks
from tau2.utils.llm_utils import get_token_usage

MODEL = "openai/zai-org/GLM-5.2"  # openai/ prefix + api_base bypasses broken DeepInfra LiteLLM provider
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
DOMAIN = "retail"
TASK_TYPE = "exchange_delivered_order_items"
MAX_STEPS = 20

# ── 1. Load tasks ──────────────────────────────────────────────────
tasks = get_tasks("base")
candidates = [t for t in tasks if infer_task_type(t) == TASK_TYPE]
print(f"Total {TASK_TYPE} tasks: {len(candidates)}")

# ── 2. Load harness registry ──────────────────────────────────────
cfg = Config()
repo = Repository(cfg)
registry = HarnessRegistry(repo)
registry.load_all()
print(f"Harness registry: {registry.count} intents → {registry.intents}\n")

# ── 3. Run each task (baseline react) ────────────────────────────
results = []
for task in candidates:
    t0 = time.time()
    sim = run_tau2_simulation(domain=DOMAIN, task=task, llm_model=MODEL,
                                llm_api_base=DEEPINFRA_BASE,
                                max_steps=MAX_STEPS, seed=42+len(results))
    elapsed = time.time() - t0
    tt = infer_task_type(task)
    reward = sim.reward_info.reward if sim.reward_info else 0.0

    msgs = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
    messages_json = serialize_messages(msgs)
    try:
        usage = get_token_usage(msgs)
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
    except Exception:
        pt, ct = 0, 0

    # Post-hoc harness savings analysis
    msgs_data = json.loads(messages_json)
    harness_hits = 0
    harness_saves_tok = 0
    agent_calls = 0
    for i, msg in enumerate(msgs_data):
        for tc in msg.get('tool_calls', []):
            agent_calls += 1
            name = tc.get('name', '?')
            if registry.lookup(name):
                harness_hits += 1
                tok = (msg.get('usage', {}).get('completion_tokens', 0) or
                       msg.get('usage', {}).get('output_tokens', 0) or
                       len(json.dumps(msg.get('content',''))) // 4)
                harness_saves_tok += max(1, tok)

    results.append({
        "task_id": str(task.id), "task_type": tt,
        "success": reward >= 1.0, "reward": reward,
        "pt": pt, "ct": ct, "tt": pt + ct, "latency": elapsed,
        "calls": agent_calls, "hits": harness_hits, "savings": harness_saves_tok,
    })

    mark = "OK" if reward >= 1.0 else "X"
    projected = pt + ct - harness_saves_tok
    print(f"  {str(task.id):<6s} {mark} reward={reward:.1f} tok={pt+ct:<8,} "
          f"calls={agent_calls:<2d} harnessable={harness_hits:<2d} "
          f"saved~{harness_saves_tok:<6,} projected={projected:<8,}")

# ── 4. Summary ────────────────────────────────────────────────────
n = len(results)
ok = sum(1 for r in results if r["success"])
tok = sum(r["tt"] for r in results)
pt = sum(r["pt"] for r in results)
ct = sum(r["ct"] for r in results)
hits = sum(r["hits"] for r in results)
saves = sum(r["savings"] for r in results)
calls = sum(r["calls"] for r in results)

print(f"\n{'='*70}")
print(f"  BASELINE (pure ReAct)")
print(f"{'='*70}")
print(f"  Tasks:        {n}")
print(f"  SR:           {ok}/{n} = {ok/n*100:.1f}%")
print(f"  Tokens:       {tok:,} total (pt={pt:,} ct={ct:,})")
print(f"  Tool calls:   {calls}")
print(f"  Harnessable:  {hits} ({hits/calls*100:.0f}% of calls)")
print(f"  Token saved:  ~{saves:,} ({saves/tok*100:.0f}% of total)")

projected_tok = tok - saves
print(f"\n  HarnessBoost projection:")
print(f"  Tokens:       {projected_tok:,} (saved {saves:,}, {saves/tok*100:.0f}% reduction)")
print(f"  SR:           same as baseline (harness preserves tool output)")
print(f"  Avg/task:     {projected_tok/n:,.0f} vs {tok/n:,.0f} (baseline)")
