"""List τ-bench retail task types and counts."""
import os
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(var, None)

from tau2.domains.retail.environment import get_tasks
from experience_os.tau2_adapter import infer_task_type

tasks = get_tasks("base")
print(f"Total tasks: {len(tasks)}")

from collections import Counter
types = Counter()
for t in tasks:
    tt = infer_task_type(t)
    types[tt] += 1

for tt, count in types.most_common():
    print(f"  {tt:<45s} {count}")
