"""Test sub-step pattern discovery from LTS data."""
import os
for v in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ.pop(v, None)

import sqlite3, json
from experience_os.models import Trajectory, Step
from experience_os.compiler import HarnessInductor
from experience_os.config import Config

conn = sqlite3.connect('.experience_os_data/lts_library.db')
cur = conn.execute('SELECT task_id, task_type, success, messages_json FROM trajectories WHERE method="react"')
trajs = []
for row in cur.fetchall():
    task_id, task_type, success, messages_json = row
    steps = []
    if messages_json:
        try:
            msgs = json.loads(messages_json)
            for msg in msgs:
                tool_calls = msg.get('tool_calls', [])
                for tc in tool_calls:
                    tc_name = tc.get('name', '?')
                    tc_args = json.dumps(tc.get('arguments', {}))
                    steps.append(Step(observation="", action=f'{tc_name}({tc_args})', action_type='tool', result="", sub_step_intent=tc_name))
        except Exception:
            pass
    if steps:
        trajs.append(Trajectory(task_id=str(task_id), task_description="", task_type=task_type or '?', steps=steps, outcome='success' if success else 'failure'))
conn.close()
print(f'Task types: {set(t.task_type for t in trajs)}')
print(f'Loaded {len(trajs)} trajectories, total steps: {sum(len(t.steps) for t in trajs)}')

if trajs:
    cfg = Config()
    inductor = HarnessInductor(cfg, None, None)
    patterns = inductor._discover_substep_patterns(trajs)
    print(f'\nDiscovered {len(patterns)} sub-step patterns:')
    for key, p in sorted(patterns.items(), key=lambda x: -x[1].support_count):
        ms = cfg.induction.min_support
        meets = '✅' if p.support_count >= ms else '  '
        print(f'  {meets} {p.action_name:<40s} support={p.support_count} sr={p.success_rate:.0%}')
