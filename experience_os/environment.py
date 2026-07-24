"""Pluggable task-environment interface.

A concrete environment (e.g. the bundled :class:`MockEnvironment`, or an adapter
around tau-bench) knows how to:

    * provide tools to an agent,
    * execute a compiled harness's Python code, and
    * verify the terminal state.

The interface is intentionally minimal so that the runtime can drive any
text/API environment without knowing its internals.
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from experience_os.models import EnvironmentSnapshot, ExecutionResult, Harness, Trajectory

log = logging.getLogger(__name__)


@dataclass
class TaskRequest:
    """A task to be executed."""

    task_id: str
    task_description: str
    task_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""  # for verification


class BaseEnvironment(ABC):
    """Abstract task environment."""

    @abstractmethod
    def snapshot(self) -> EnvironmentSnapshot:
        """Return the current environment state for precondition matching."""

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return tool schemas (OpenAI function-calling format) for the agent."""

    @abstractmethod
    def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return the result text."""

    @abstractmethod
    def verify(self, expected_output: str, actual_output: str) -> bool:
        """Check whether the task's terminal state is satisfied."""

    # ------------------------------------------------------------------
    # harness execution
    # ------------------------------------------------------------------
    def execute_harness(
        self,
        harness: Harness,
        request: TaskRequest,
    ) -> ExecutionResult:
        """Run a compiled harness's ``procedure_code`` in this environment.

        The harness code has access to ``call_tool(name, **args)`` and
        ``snapshot()`` via the ``env`` local.  Failures are caught and
        returned as :class:`ExecutionResult` with ``success=False``.
        """
        import time

        start = time.time()

        # wrapper that accepts call_tool(name, **kwargs) or call_tool(name, {dict})
        # and auto-parses JSON string results into dicts for harness convenience
        def _call_tool(name: str, *args, **kwargs):
            if args and isinstance(args[0], dict) and not kwargs:
                raw = self.call_tool(name, args[0])
            elif kwargs:
                raw = self.call_tool(name, kwargs)
            elif args:
                raw = self.call_tool(name, dict(zip(
                    ["value"] * len(args), args
                )))
            else:
                raw = self.call_tool(name, {})
            # try to parse JSON string into dict/list for harness code
            if isinstance(raw, str):
                stripped = raw.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        import json
                        return json.loads(stripped)
                    except (json.JSONDecodeError, ValueError):
                        pass
            return raw

        sandbox_globals: dict[str, Any] = {
            "env": self,
            "call_tool": _call_tool,
            "snapshot": self.snapshot,
            "params": request.params,
            "request": request,
        }
        try:
            local_ns: dict[str, Any] = {}
            exec(harness.procedure_code, sandbox_globals, local_ns)  # noqa: S102
            # the harness must define a ``run`` callable
            run_fn: Callable = local_ns.get("run") or local_ns.get("main")
            if run_fn is None:
                return ExecutionResult(
                    success=False,
                    path="harness",
                    harness_id=harness.id,
                    failure_type="F2",
                    output="harness code defines no `run()` function",
                    latency_seconds=time.time() - start,
                )
            output = run_fn()
            output_str = str(output) if output is not None else ""
            success = self.verify(request.expected_output, output_str)
            return ExecutionResult(
                success=success,
                path="harness",
                harness_id=harness.id,
                tokens_used=0,
                latency_seconds=time.time() - start,
                output=output_str,
            )
        except Exception as exc:
            log.warning("Harness %s raised: %s", harness.full_name, exc)
            return ExecutionResult(
                success=False,
                path="harness",
                harness_id=harness.id,
                failure_type="F2",
                output=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                latency_seconds=time.time() - start,
            )


class MockEnvironment(BaseEnvironment):
    """A trivial in-memory environment for testing the full loop.

    Tools:
        * ``lookup(key)``       — returns a value from an internal store
        * ``update(key, value)`` — updates the store
        * ``submit(payload)``    — records a submission
    """

    def __init__(self, store: dict[str, str] | None = None) -> None:
        self._store = store or {}
        self._submitted: list[str] = []

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            attributes={
                "env": "mock",
                "store_keys": list(self._store.keys()),
            }
        )

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value by key in the store",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update",
                    "description": "Update a value in the store",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit",
                    "description": "Submit a payload to complete the task",
                    "parameters": {
                        "type": "object",
                        "properties": {"payload": {"type": "string"}},
                        "required": ["payload"],
                    },
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        if name == "lookup":
            return self._store.get(arguments["key"], f"NOT_FOUND:{arguments['key']}")
        if name == "update":
            self._store[arguments["key"]] = arguments["value"]
            return "OK"
        if name == "submit":
            self._submitted.append(arguments["payload"])
            return "SUBMITTED"
        return f"UNKNOWN_TOOL:{name}"

    def verify(self, expected_output: str, actual_output: str) -> bool:
        if expected_output:
            return expected_output in actual_output or actual_output in expected_output
        return len(self._submitted) > 0
