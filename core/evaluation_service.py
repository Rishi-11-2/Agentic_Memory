"""Reusable auto-evaluation service for MCP, HTTP, tests, and standalone adapters."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from core.memory_service import AgenticMemoryService
from core.models import ActorResult, CriticEvaluation, MemoryContext, NewSemanticFact, SemanticFactType
from planner.retrieval_planner import HeuristicRetrievalPlanner

logger = logging.getLogger(__name__)


class CriticProvider(Protocol):
    """Protocol for LLM-backed Critic implementations."""

    async def evaluate(
        self,
        prompt: str,
        memory_context: MemoryContext,
        actor_result: ActorResult,
        temperature: float = 0.0,
    ) -> CriticEvaluation:
        """Evaluate one completed turn."""
        ...


class AutoEvaluationService:
    """Evaluate completed turns with an optional LLM Critic and deterministic fallback."""

    def __init__(
        self,
        critic: CriticProvider | None = None,
        planner: HeuristicRetrievalPlanner | None = None,
        memory_service: AgenticMemoryService | None = None,
    ) -> None:
        """Create an evaluator that can use an LLM Critic when all dependencies are present."""
        self._critic = critic
        self._planner = planner
        self._memory_service = memory_service

    async def evaluate(
        self,
        user_message: str,
        assistant_response: str,
        actor_result: ActorResult,
        new_facts: list[str] | None = None,
        failure_summary: str | None = None,
        quality_score: float | None = None,
        critic_session_id: str = "critic-eval",
    ) -> CriticEvaluation:
        """Return a CriticEvaluation from an LLM Critic or the heuristic fallback."""
        facts_list = new_facts or []
        if self._critic is not None and self._planner is not None and self._memory_service is not None:
            try:
                plan = await self._planner.plan(user_message, critic_session_id)
                memory_context = await self._memory_service.build_context(plan)
                critic_eval = await self._critic.evaluate(user_message, memory_context, actor_result)
                if facts_list:
                    critic_eval.new_semantic_facts.extend(_agent_facts(facts_list))
                if failure_summary and not critic_eval.failure_summary:
                    critic_eval.failure_summary = failure_summary
                if quality_score is not None:
                    clamped = max(0.0, min(10.0, quality_score))
                    critic_eval.factual_accuracy = round(0.6 * critic_eval.factual_accuracy + 0.4 * clamped, 1)
                    critic_eval.preference_adherence = round(
                        0.6 * critic_eval.preference_adherence + 0.4 * clamped, 1
                    )
                    critic_eval = recompute_evaluation(critic_eval)
                return critic_eval
            except Exception as exc:
                logger.warning("critic_evaluation_failed error=%s, falling back to heuristic", exc)

        return heuristic_evaluation(
            actor_result=actor_result,
            facts=facts_list,
            failure_summary=failure_summary,
            quality_score=quality_score,
            user_message=user_message,
            assistant_response=assistant_response,
        )


_PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi\s+prefer\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\balways\s+(?:use|do|prefer|want|include)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bnever\s+(?:use|do|want|include)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bdon'?t\s+(?:ever\s+)?(?:use|do|want|include|show)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bplease\s+(?:always|never|don'?t)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:preferred|favorite|default)\s+(?:\w+\s+)?is\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
)

_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi\s+(?:use|am using|work with|develop in|code in)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(
        r"\bmy\s+(?:project|app|system|codebase|stack|setup)\s+(?:uses|is|runs)\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\bwe\s+(?:use|run|deploy|host)\s+(?:on\s+)?(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bour\s+(?:stack|infrastructure|database|backend|frontend)\s+is\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
)


def heuristic_evaluation(
    actor_result: ActorResult,
    facts: list[str],
    failure_summary: str | None,
    quality_score: float | None = None,
    user_message: str = "",
    assistant_response: str = "",
) -> CriticEvaluation:
    """Build a deterministic CriticEvaluation from self-score, tools, text, and patterns."""
    total_tools = len(actor_result.tool_calls)
    failed_tools = sum(1 for tool in actor_result.tool_calls if not tool.success)
    has_failures = failed_tools > 0 or failure_summary is not None

    if quality_score is not None:
        base = max(0.0, min(10.0, quality_score))
    elif has_failures and total_tools > 0 and failed_tools == total_tools:
        base = 3.0
    elif has_failures:
        base = 6.0
    elif total_tools > 0:
        base = 9.0
    else:
        base = 7.5

    tool_efficiency = base
    if total_tools > 0:
        success_rate = (total_tools - failed_tools) / total_tools
        tool_efficiency = max(1.0, min(10.0, round(success_rate * 10, 1)))

    factual_accuracy = base
    response_len = len(assistant_response)
    prompt_len = max(len(user_message), 1)
    if response_len < 10 and prompt_len > 20:
        factual_accuracy = max(base - 2.0, 1.0)
    elif response_len > prompt_len * 0.3:
        factual_accuracy = min(base + 0.5, 10.0)

    hallucination_risk = base
    if total_tools > 0 and failed_tools == 0:
        hallucination_risk = min(base + 1.0, 10.0)
    if failure_summary:
        hallucination_risk = max(base - 1.5, 1.0)

    workflow_quality = base
    if total_tools >= 2 and failed_tools == 0:
        workflow_quality = min(base + 1.0, 10.0)

    semantic_facts = _extract_semantic_facts(user_message)
    semantic_facts.extend(_agent_facts(facts))

    return CriticEvaluation(
        factual_accuracy=factual_accuracy,
        preference_adherence=base,
        tool_efficiency=tool_efficiency,
        hallucination_risk=hallucination_risk,
        workflow_quality=workflow_quality,
        new_semantic_facts=semantic_facts,
        save_workflow=total_tools >= 2 and failed_tools == 0,
        failure_summary=failure_summary,
    )


def recompute_evaluation(evaluation: CriticEvaluation) -> CriticEvaluation:
    """Re-run CriticEvaluation validators after post-provider score blending."""
    return CriticEvaluation.model_validate(evaluation.model_dump(mode="json", by_alias=True))


def _extract_semantic_facts(user_message: str) -> list[NewSemanticFact]:
    """Extract durable preferences and environment facts from a user message."""
    semantic_facts: list[NewSemanticFact] = []
    for pattern in _PREFERENCE_PATTERNS:
        for match in pattern.finditer(user_message):
            extracted = match.group(1).strip()
            if 5 < len(extracted) < 200:
                full_match = match.group(0).strip().rstrip(".,")
                semantic_facts.append(
                    NewSemanticFact(
                        fact_type=SemanticFactType.PREFERENCE,
                        content=f"User preference: {full_match}",
                        confidence=0.90,
                        source="user_stated",
                    )
                )

    for pattern in _FACT_PATTERNS:
        for match in pattern.finditer(user_message):
            extracted = match.group(1).strip()
            if 3 < len(extracted) < 200:
                full_match = match.group(0).strip().rstrip(".,")
                semantic_facts.append(
                    NewSemanticFact(
                        fact_type=SemanticFactType.SYSTEM_RULE,
                        content=f"Environment fact: {full_match}",
                        confidence=0.80,
                        source="user_stated",
                    )
                )
    return semantic_facts


def _agent_facts(facts: list[str]) -> list[NewSemanticFact]:
    """Convert agent-supplied fact strings into semantic fact proposals."""
    return [
        NewSemanticFact(
            fact_type=SemanticFactType.INFERRED_FACT,
            content=fact,
            confidence=0.85,
            source="llm_inferred",
        )
        for fact in facts
    ]
