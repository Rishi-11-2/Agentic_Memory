"""Six-phase self-learning Actor-Critic loop orchestration."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from core.access_scope import AccessScope, current_scope_hash
from core.memory_service import AgenticMemoryService
from core.models import ActorResult, CriticEvaluation, LoopResult, MemoryContext
from planner.retrieval_planner import HeuristicRetrievalPlanner
from runtime.actor import Actor
from runtime.critic import Critic

logger = logging.getLogger(__name__)


class SelfLearningLoop:
    """Orchestrate retrieval, Actor execution, Critic reflection, and memory consolidation."""

    def __init__(
        self,
        planner: HeuristicRetrievalPlanner,
        memory_service: AgenticMemoryService,
        actor: Actor,
        critic: Critic,
        critic_timeout_seconds: float = 8.0,
        critic_self_consistency_samples: int = 3,
        critic_self_consistency_temperature: float = 0.4,
    ) -> None:
        """Create a complete six-phase loop from explicit runtime components."""
        self._planner = planner
        self._memory_service = memory_service
        self._actor = actor
        self._critic = critic
        self._critic_timeout_seconds = critic_timeout_seconds
        self._critic_self_consistency_samples = critic_self_consistency_samples
        self._critic_self_consistency_temperature = critic_self_consistency_temperature

    async def run_turn(self, user_message: str, session_id: str, scope: AccessScope) -> LoopResult:
        """Run one full self-learning turn and return only public-safe fields."""
        loop_started = perf_counter()
        scope_hash = scope.scope_hash
        token = current_scope_hash.set(scope_hash)
        try:
            phase_timings_ms: dict[str, float] = {}

            phase_started = perf_counter()
            retrieval_plan = await self._planner.plan(user_message, scope, session_id)
            phase_timings_ms["retrieval_plan"] = _log_phase("retrieval_plan", session_id, scope_hash, phase_started)

            phase_started = perf_counter()
            memory_context = await self._memory_service.build_context(retrieval_plan)
            phase_timings_ms["context_build"] = _log_phase("context_build", session_id, scope_hash, phase_started)

            phase_started = perf_counter()
            actor_result = await self._actor.execute(user_message, memory_context, self._actor.available_tool_names())
            phase_timings_ms["actor_exec"] = _log_phase("actor_exec", session_id, scope_hash, phase_started)

            phase_started = perf_counter()
            critic_timed_out = False
            try:
                critic_evaluation = await asyncio.wait_for(
                    self._evaluate_critic(user_message, memory_context, actor_result),
                    timeout=self._critic_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("critic_timeout session=%s", session_id)
                critic_timed_out = True
                critic_evaluation = _timeout_critic_evaluation()
            phase_timings_ms["critic_eval"] = _log_phase("critic_eval", session_id, scope_hash, phase_started)

            phase_started = perf_counter()
            semantic_conflicts: list[str] = []
            if critic_timed_out:
                turn_index = await self._memory_service.next_turn_index(scope, session_id)
                memory_writes = ["Skipped consolidation because Critic timed out"]
            else:
                total_latency_ms = int((perf_counter() - loop_started) * 1000)
                turn_index, memory_writes, semantic_conflicts = await self._memory_service.consolidate(
                    prompt=user_message,
                    actor_result=actor_result,
                    critic_evaluation=critic_evaluation,
                    scope=scope,
                    session_id=session_id,
                    loop_latency_ms=total_latency_ms,
                )
                self._planner.record_feedback(session_id, retrieval_plan, critic_evaluation.passed)
            phase_timings_ms["consolidation"] = _log_phase("consolidation", session_id, scope_hash, phase_started)

            phase_started = perf_counter()
            result = LoopResult(
                final_response=actor_result.final_response,
                session_id=session_id,
                turn_index=turn_index,
                critic_score=critic_evaluation.overall_score,
                critic_pass=critic_evaluation.passed,
                memory_writes=memory_writes,
                retrieval_plan_summary=retrieval_plan.summary(),
                phase_timings_ms=phase_timings_ms,
                semantic_conflicts=semantic_conflicts,
            )
            result.phase_timings_ms["response_assembly"] = _log_phase(
                "response_assembly", session_id, scope_hash, phase_started
            )
            return result
        finally:
            current_scope_hash.reset(token)

    async def _evaluate_critic(
        self,
        user_message: str,
        memory_context: MemoryContext,
        actor_result: ActorResult,
    ) -> CriticEvaluation:
        """Run single or multi-sample Critic evaluation based on turn risk."""
        if not self._should_multi_sample(actor_result):
            return await self._critic.evaluate(user_message, memory_context, actor_result)
        samples = [
            await self._critic.evaluate(
                user_message,
                memory_context,
                actor_result,
                temperature=self._critic_self_consistency_temperature,
            )
            for _ in range(self._critic_self_consistency_samples)
        ]
        return _majority_vote(samples)

    def _should_multi_sample(self, actor_result: ActorResult) -> bool:
        """Gate Critic self-consistency to higher-risk turns."""
        low_confidence_text = f"{actor_result.reasoning} {actor_result.final_response}".lower()
        has_low_confidence_signal = any(
            phrase in low_confidence_text
            for phrase in ("not sure", "uncertain", "may be wrong", "low confidence", "i think")
        )
        return len(actor_result.tool_calls) > 1 or has_low_confidence_signal


def _log_phase(phase: str, session_id: str, scope_hash: str, started: float) -> float:
    """Emit one structured log event for a completed loop phase."""
    elapsed_ms = (perf_counter() - started) * 1000
    logger.info(
        "loop_phase_complete",
        extra={
            "phase": phase,
            "session_id": session_id,
            "scope_hash": scope_hash,
            "latency_ms": int(elapsed_ms),
        },
    )
    return round(elapsed_ms, 3)


def _timeout_critic_evaluation() -> CriticEvaluation:
    """Create a no-pass Critic result for timeout fallbacks."""
    return CriticEvaluation(
        factual_accuracy=0.0,
        preference_adherence=0.0,
        tool_efficiency=0.0,
        hallucination_risk=0.0,
        workflow_quality=0.0,
        new_semantic_facts=[],
        save_workflow=False,
        failure_summary="Critic timed out; consolidation skipped.",
    )


def _majority_vote(samples: list[CriticEvaluation]) -> CriticEvaluation:
    """Select the Critic sample matching the majority pass/fail verdict."""
    if not samples:
        return _timeout_critic_evaluation()
    pass_votes = sum(1 for sample in samples if sample.passed)
    majority_passed = pass_votes >= ((len(samples) // 2) + 1)
    candidates = [sample for sample in samples if sample.passed == majority_passed] or samples
    candidates.sort(key=lambda sample: sample.overall_score)
    return candidates[len(candidates) // 2]
