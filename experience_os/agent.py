"""LLM agent fallback + harness failure classification.

When no harness matches (or harness execution fails), the runtime falls back
to a plain LLM agent that uses tool-calling to solve the task.  The agent's
interaction is recorded as a :class:`~experience_os.models.Trajectory` for
later induction.

Failure classification (§3.4):
    F1 — precondition coverage gap  (constraint missing)
    F2 — implementation error       (selector/timing/bug in harness code)
    F3 — environment drift          (API/UI changed)
    F4 — out of scope               (task outside harness capability)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from experience_os.environment import BaseEnvironment, TaskRequest
from experience_os.llm import LLMClient
from experience_os.models import (
    EnvironmentSnapshot,
    ExecutionResult,
    FailureType,
    Harness,
    Step,
    StructuredCoT,
    Trajectory,
)

log = logging.getLogger(__name__)

AGENT_SYSTEM = """\
You are a task-execution agent. You solve tasks by calling the available tools.
Always reason step-by-step, then issue a tool call. When the task is complete,
respond with "DONE: <summary>".

Available tools:
{tools}
"""


class AgentFallback:
    """A minimal ReAct-style LLM agent that uses tool calling."""

    def __init__(self, llm: LLMClient, max_steps: int = 10) -> None:
        self.llm = llm
        self.max_steps = max_steps

    def run(
        self,
        request: TaskRequest,
        env: BaseEnvironment,
        task_type: str = "",
    ) -> ExecutionResult:
        """Execute the task via LLM tool-calling and record a trajectory."""
        start = time.time()
        tools = env.get_tools()
        tools_desc = json.dumps(tools, indent=2)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM.format(tools=tools_desc)},
            {"role": "user", "content": request.task_description},
        ]
        steps: list[Step] = []
        total_tokens = 0
        output = ""

        for i in range(self.max_steps):
            # ask LLM for next action
            try:
                # try native tool-calling first
                reply = self._call_with_tools(messages, tools, request)
            except Exception:
                # fallback to plain text reasoning
                reply = self.llm.chat(messages, temperature=0.3)

            messages.append({"role": "assistant", "content": reply})
            total_tokens += len(reply) // 4  # rough estimate

            if reply.startswith("DONE:"):
                output = reply[5:].strip()
                steps.append(Step(observation=request.task_description, action="DONE", result=output))
                break

            # parse tool call from reply
            tool_name, tool_args = self._parse_tool_call(reply)
            if tool_name:
                result = env.call_tool(tool_name, tool_args)
                steps.append(
                    Step(
                        observation=f"step {i}",
                        action=f"{tool_name}({tool_args})",
                        result=result[:200],
                        action_type="write" if tool_name in ("update", "submit") else "read",
                    )
                )
                messages.append({"role": "tool", "name": tool_name, "content": result})
                output = result
            else:
                # no tool call, agent is reasoning — let it continue
                steps.append(Step(observation=f"step {i}", action=reply[:100], result="reasoning", action_type="think"))

        success = env.verify(request.expected_output, output) if request.expected_output else bool(output)
        elapsed = time.time() - start

        trajectory = Trajectory(
            task_id=request.task_id,
            task_description=request.task_description,
            task_type=task_type or request.task_type,
            steps=steps,
            structured_cot=StructuredCoT(goal=request.expected_output),
            env_snapshot=env.snapshot(),
            outcome="success" if success else "failure",
            tokens_used=total_tokens,
            latency_seconds=elapsed,
        )
        # record params in the first step's metadata for the inductor
        if steps and request.params:
            steps[0].metadata["params"] = request.params
        return ExecutionResult(
            success=success,
            path="agent_fallback",
            tokens_used=total_tokens,
            latency_seconds=elapsed,
            trajectory=trajectory,
            output=output,
        )

    # ------------------------------------------------------------------
    def _call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        request: TaskRequest,
    ) -> str:
        """Use native OpenAI tool-calling when supported by the backend."""
        client = self.llm._client  # noqa: SLF001
        resp = client.chat.completions.create(
            model=self.llm.config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = resp.choices[0].message
        # if there's a tool call, serialise it into text for our parser
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items())
            return f'{tc.function.name}({args_str})'
        return msg.content or ""

    @staticmethod
    def _parse_tool_call(text: str) -> tuple[Optional[str], dict]:
        """Parse ``tool_name(key="value", num=1)`` from agent reply."""
        m = re.match(r'\s*(\w+)\s*\((.*)\)\s*$', text)
        if not m:
            return None, {}
        name = m.group(1)
        if name in ("DONE", "say", "think"):
            return None, {}
        args_str = m.group(2).strip()
        args: dict = {}
        # parse key=value pairs
        for pair in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]+))', args_str):
            val = next(v for v in pair.groups()[1:] if v is not None)
            args[pair.group(1)] = val
        return name, args


# ======================================================================
# Failure classifier
# ======================================================================
def classify_failure(
    harness: Harness,
    error_output: str,
    env: BaseEnvironment,
) -> FailureType:
    """Classify a harness execution failure into F1–F4 (§3.4)."""

    err = error_output.lower()

    # F1 — precondition gap: harness expects env conditions that don't hold
    snap = env.snapshot()
    for key, expected in harness.preconditions.items():
        if not snap.satisfies(key, expected):
            return FailureType.F1_PRECONDITION_GAP

    # F3 — environment drift: signs of changed API/UI
    drift_markers = ["not found", "no such", "attributeerror", "keyerror",
                     "typeerror", "invalid", "unexpected", "mismatch"]
    if any(m in err for m in drift_markers):
        return FailureType.F3_ENVIRONMENT_DRIFT

    # F2 — implementation error: code bugs
    code_markers = ["nameerror", "syntaxerror", "indexerror", "traceback",
                    "run()", "function", "divisionbyzero"]
    if any(m in err for m in code_markers):
        return FailureType.F2_IMPLEMENTATION_ERROR

    # F4 — out of scope: nothing else matched
    return FailureType.F4_OUT_OF_SCOPE

