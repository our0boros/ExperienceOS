"""验证 LTS 完整对话保存 + 分析失败原因。"""
import sys, json
sys.path.insert(0, "tau2-bench/src")
from experience_os.experience_library import ExperienceLibrary

lts = ExperienceLibrary.persistent()
rows = lts.query_trajectories(experiment_id="vanilla-retail-train_test-11daa895", with_messages=True)
print(f"轨迹数: {len(rows)}")
for r in rows:
    print(f"\n--- task {r['task_id']} (idx={r['idx']}, phase={r['phase']}, success={r['success']}) ---")
    print(f"  tokens: {r['tokens']}, latency: {r['latency']:.1f}s")
    msgs = r.get("messages_json", "")
    print(f"  messages_json 长度: {len(msgs)} chars")
    if msgs:
        m = json.loads(msgs)
        print(f"  消息数: {len(m)}")
        # 统计工具调用
        tool_calls = sum(1 for x in m if isinstance(x, dict) and x.get("role") == "assistant" and x.get("tool_calls"))
        tool_results = sum(1 for x in m if isinstance(x, dict) and x.get("role") == "tool")
        agent_msgs = sum(1 for x in m if isinstance(x, dict) and x.get("role") == "assistant" and x.get("content"))
        user_msgs = sum(1 for x in m if isinstance(x, dict) and x.get("role") == "user")
        print(f"  agent_text={agent_msgs} tool_calls={tool_calls} tool_results={tool_results} user_msgs={user_msgs}")
        # 看最后几条消息
        for msg in m[-3:]:
            role = msg.get("role", "?")
            if role == "assistant":
                tc = msg.get("tool_calls", [])
                content = str(msg.get("content", ""))[:100]
                if tc:
                    names = [c.get("function", {}).get("name", "?") for c in tc]
                    print(f"    [last] assistant tool_calls: {names}")
                else:
                    print(f"    [last] assistant text: {content}")
            elif role == "user":
                print(f"    [last] user: {str(msg.get('content',''))[:100]}")
            elif role == "tool":
                print(f"    [last] tool result: {str(msg.get('content',''))[:100]}")
lts.close()
