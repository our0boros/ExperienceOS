#!/bin/bash
# Run all four baselines with proxy cleanup
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export DEEPINFRA_API_KEY="$DEEPINFRA_TOKEN"
cd /home/our0boros/Project/ExecutableExperience
mkdir -p docs/exp_results

echo "============================================================"
echo "  Baseline 1: react (multi-step ReAct agent)"
echo "  Task type: return_delivered_order_items (6 tasks, single action)"
echo "============================================================"
experience-os compare \
  --method react \
  --model deepinfra/MiniMaxAI/MiniMax-M2.7 \
  --domain retail \
  --warmup 2 \
  --eval 4 \
  --max-steps 30 \
  --task-type return_delivered_order_items \
  --output docs/exp_results/react_return.json 2>&1

echo "============================================================"
echo "  Baseline 2: vanilla (single-turn LLM)"
echo "  Task type: return_delivered_order_items"
echo "============================================================"
experience-os compare \
  --method vanilla \
  --model deepinfra/MiniMaxAI/MiniMax-M2.7 \
  --domain retail \
  --warmup 2 \
  --eval 4 \
  --max-steps 30 \
  --task-type return_delivered_order_items \
  --output docs/exp_results/vanilla_return.json 2>&1

echo "============================================================"
echo "  Baseline 3: skillopt (with initial seed skill)"
echo "  Task type: return_delivered_order_items"
echo "============================================================"
experience-os compare \
  --method skillopt \
  --model deepinfra/MiniMaxAI/MiniMax-M2.7 \
  --domain retail \
  --warmup 2 \
  --eval 4 \
  --max-steps 30 \
  --task-type return_delivered_order_items \
  --skill-path SkillOpt/skillopt/envs/tau2/skills/initial.md \
  --output docs/exp_results/skillopt_return.json 2>&1

echo "============================================================"
echo "  Baseline 4: autoharness (ExperienceOS DEPLOYMENT mode)"
echo "  Task type: return_delivered_order_items"
echo "============================================================"
experience-os compare \
  --method autoharness \
  --model deepinfra/MiniMaxAI/MiniMax-M2.7 \
  --domain retail \
  --warmup 2 \
  --eval 4 \
  --max-steps 30 \
  --task-type return_delivered_order_items \
  --output docs/exp_results/autoharness_return.json 2>&1

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "============================================================"
ls -la docs/exp_results/*.json 2>/dev/null
