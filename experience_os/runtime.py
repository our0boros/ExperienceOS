"""ExperienceOS Runtime — the central accumulate / deploy loop.

This is the orchestrator described in §3.1 of the research proposal:

    Task → Router → (Harness execution | Agent fallback)
        → Experience Accumulation → (induction trigger?)

Two execution modes (§4.3 of the Discuss doc):

    ACCUMULATION  — always use the agent, record trajectories, trigger induction.
    DEPLOYMENT   — prefer harness, fall back to agent on miss / failure,
                    continue recording for online learning.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

from experience_os.agent import AgentFallback, classify_failure
from experience_os.compiler import HarnessInductor
from experience_os.config import Config
from experience_os.environment import BaseEnvironment, TaskRequest
from experience_os.llm import LLMClient
from experience_os.models import ExecutionResult, FailureType, Harness, Trajectory
from experience_os.repository import Repository
from experience_os.retriever import RuntimeRouter

log = logging.getLogger(__name__)


class SystemMode(str, Enum):
    ACCUMULATION = "accumulation"
    DEPLOYMENT = "deployment"


class Runtime:
    """The main ExperienceOS execution loop.

    Parameters
    ----------
    config:
        :class:`~experience_os.config.Config` with LLM + induction settings.
    llm:
        :class:`~experience_os.llm.LLMClient` instance (or ``None`` to create one).
    env:
        :class:`~experience_os.environment.BaseEnvironment` subclass instance.
    """

    def __init__(
        self,
        config: Config,
        env: BaseEnvironment,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.env = env
        self.llm = llm or LLMClient(config.llm)
        self.repo = Repository(config)
        self.router = RuntimeRouter(self.repo, self.llm)
        self.agent = AgentFallback(self.llm)
        self.inductor = HarnessInductor(self.config, self.llm, self.repo)
        self.mode: SystemMode = SystemMode.ACCUMULATION

    # ==================================================================
    # public API
    # ==================================================================
    def set_mode(self, mode: SystemMode) -> None:
        log.info("Switching to %s mode", mode.value)
        self.mode = mode

    def execute(self, request: TaskRequest) -> ExecutionResult:
        """Execute a single task request.

        The routing logic depends on the current :class:`SystemMode`:

        * ``ACCUMULATION`` — always agent, record trajectory, maybe induce.
        * ``DEPLOYMENT``  — try harness first, fall back to agent on miss.
        """
        log.info("[%s] task=%s type=%s", self.mode.value, request.task_id, request.task_type)

        if self.mode == SystemMode.DEPLOYMENT:
            result = self._route_and_execute(request)
        else:
            result = self._agent_only(request)

        # always accumulate experience
        self._accumulate(request, result)
        return result

    def maybe_induce(self) -> Optional[Harness]:
        """Check all task types for induction triggers; run if any fire.

        Convenience method to call after a batch of tasks.
        """
        for task_type in self.repo.all_task_types():
            trigger = self.inductor.check_triggers(task_type)
            if trigger:
                log.info("Induction trigger '%s' fired for '%s'", trigger, task_type)
                return self.inductor.induce(task_type, self.env)
        return None

    # ==================================================================
    # internal: routing
    # ==================================================================
    def _route_and_execute(self, request: TaskRequest) -> ExecutionResult:
        env_snap = self.env.snapshot()
        retrieval = self.router.select(
            task_description=request.task_description,
            env=env_snap,
            task_type=request.task_type,
        )

        if retrieval.harness is not None:
            result = self._run_harness(retrieval.harness, request)
            if result.success:
                return result

            # harness failed — classify and maybe fall back
            ftype = classify_failure(
                retrieval.harness,
                result.output,
                self.env,
            ) if not result.failure_type else FailureType(result.failure_type)

            result.failure_type = ftype.value
            self.repo.record_failure(retrieval.harness.id, ftype)
            log.warning("Harness %s failed (%s), falling back to agent",
                        retrieval.harness.full_name, ftype.value)

            # F4 — out of scope: just fall back silently
            # F1/F2/F3 — fall back and let accumulation handle it
            agent_result = self._agent_only(request)
            agent_result.path = "harness_with_fallback"
            return agent_result

        # no harness match — agent fallback
        return self._agent_only(request)

    # ==================================================================
    # internal: harness execution
    # ==================================================================
    def _run_harness(self, harness: Harness, request: TaskRequest) -> ExecutionResult:
        result = self.env.execute_harness(harness, request)
        result.harness_id = harness.id
        log.info("Harness %s executed: success=%s", harness.full_name, result.success)
        return result

    # ==================================================================
    # internal: agent execution
    # ==================================================================
    def _agent_only(self, request: TaskRequest) -> ExecutionResult:
        result = self.agent.run(request, self.env, task_type=request.task_type)
        log.info("Agent executed: success=%s tokens=%d", result.success, result.tokens_used)
        return result

    # ==================================================================
    # internal: experience accumulation (§3.5)
    # ==================================================================
    def _accumulate(self, request: TaskRequest, result: ExecutionResult) -> None:
        # immediate: log trajectory
        if result.trajectory is None:
            # harness path — synthesize a minimal trajectory record
            result.trajectory = Trajectory(
                task_id=request.task_id,
                task_description=request.task_description,
                task_type=request.task_type,
                steps=[],
                outcome="success" if result.success else "failure",
                tokens_used=result.tokens_used,
                latency_seconds=result.latency_seconds,
            )
        self.repo.add_trajectory(result.trajectory)

        # update task-type stats
        stats = self.repo.get_stats(request.task_type)
        stats.total_executions += 1
        if result.path in ("harness", "harness_with_fallback"):
            stats.harness_executions += 1
            if result.success:
                stats.harness_successes += 1
        else:
            stats.agent_executions += 1
            if result.success:
                stats.agent_successes += 1

        # track env coverage
        env_id = f"{self.env.snapshot().attributes.get('env', 'unknown')}"
        if env_id not in stats.observed_envs:
            stats.observed_envs.append(env_id)

        # estimated token savings
        if result.path == "harness" and result.success:
            avg_agent_tokens = (
                stats.agent_successes and (1000) or 1000  # rough estimate
            )
            stats.estimated_token_savings += max(0, avg_agent_tokens - result.tokens_used)

        # record failure counts
        if result.failure_type:
            stats.failure_counts[result.failure_type] = (
                stats.failure_counts.get(result.failure_type, 0) + 1
            )
        self.repo.save_stats(request.task_type)

        # async-ish: check induction trigger for this task type
        if self.mode == SystemMode.ACCUMULATION:
            trigger = self.inductor.check_triggers(request.task_type)
            if trigger == "new_harness":
                log.info("Auto-inducing harness for '%s' (new_harness trigger)", request.task_type)
                self.inductor.induce(request.task_type, self.env)

    # ==================================================================
    # diagnostics
    # ==================================================================
    def status(self) -> dict:
        """Return a summary of the current system state."""
        return {
            "mode": self.mode.value,
            "task_types": len(self.repo.all_task_types()),
            "total_trajectories": len(self.repo._trajectories),
            "active_harnesses": len(self.repo.active_harnesses()),
            "stats": {
                tt: {
                    "total": s.total_executions,
                    "harness_sr": round(s.harness_success_rate, 2),
                    "agent_sr": round(s.agent_success_rate, 2),
                    "token_saved": s.estimated_token_savings,
                }
                for tt, s in self.repo._stats.items()
            },
        }
