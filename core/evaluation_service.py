"""Reusable auto-evaluation service for MCP, HTTP, tests, and standalone adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from core.memory_service import AgenticMemoryService
from core.models import ActorResult, CriticEvaluation, MemoryContext, NewSemanticFact, SemanticFactType
from planner.retrieval_planner import HeuristicRetrievalPlanner

ScoringSource = Literal["mcp_client_agent", "heuristic_provisional"]


@dataclass(frozen=True)
class AutoEvaluationResult:
    """Bundle scoring with its source and whether an agent rescore is needed."""

    evaluation: CriticEvaluation
    scoring_source: ScoringSource
    needs_agent_rescore: bool = False


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
    """Validate MCP-client turn evaluation and attach mined memory facts."""

    def __init__(
        self,
        critic: CriticProvider | None = None,
        planner: HeuristicRetrievalPlanner | None = None,
        memory_service: AgenticMemoryService | None = None,
    ) -> None:
        """Create an evaluator; legacy Critic dependencies are retained for compatibility."""
        self._critic = critic
        self._planner = planner
        self._memory_service = memory_service

    async def evaluate(
        self,
        user_message: str,
        assistant_response: str,
        actor_result: ActorResult,
        new_facts: list[str] | None = None,
        semantic_facts: list[NewSemanticFact] | None = None,
        agent_evaluation: CriticEvaluation | None = None,
        failure_summary: str | None = None,
        quality_score: float | None = None,
        critic_session_id: str = "critic-eval",
    ) -> CriticEvaluation:
        """Return MCP-client scoring after server-side validation."""
        result = await self.evaluate_with_metadata(
            user_message=user_message,
            assistant_response=assistant_response,
            actor_result=actor_result,
            new_facts=new_facts,
            semantic_facts=semantic_facts,
            agent_evaluation=agent_evaluation,
            failure_summary=failure_summary,
            quality_score=quality_score,
            critic_session_id=critic_session_id,
        )
        return result.evaluation

    async def evaluate_with_metadata(
        self,
        user_message: str,
        assistant_response: str,
        actor_result: ActorResult,
        new_facts: list[str] | None = None,
        semantic_facts: list[NewSemanticFact] | None = None,
        agent_evaluation: CriticEvaluation | None = None,
        failure_summary: str | None = None,
        quality_score: float | None = None,
        critic_session_id: str = "critic-eval",
    ) -> AutoEvaluationResult:
        """Return MCP-client scoring, or provisional heuristic scoring until the client rescores."""
        del quality_score, critic_session_id
        facts_list = new_facts or []
        semantic_facts_list = semantic_facts or []
        proposed_facts = _agent_semantic_facts(semantic_facts_list)
        proposed_facts.extend(_agent_facts(facts_list))
        if agent_evaluation is None:
            evaluation = _provisional_heuristic_evaluation(
                actor_result=actor_result,
                facts=proposed_facts,
                failure_summary=failure_summary,
                user_message=user_message,
                assistant_response=assistant_response,
            )
            return AutoEvaluationResult(
                evaluation=evaluation,
                scoring_source="heuristic_provisional",
                needs_agent_rescore=True,
            )

        evaluation = CriticEvaluation(
            factual_accuracy=agent_evaluation.factual_accuracy,
            preference_adherence=agent_evaluation.preference_adherence,
            tool_efficiency=agent_evaluation.tool_efficiency,
            hallucination_risk=agent_evaluation.hallucination_risk,
            workflow_quality=agent_evaluation.workflow_quality,
            save_workflow=agent_evaluation.save_workflow,
            failure_summary=failure_summary or agent_evaluation.failure_summary,
            new_semantic_facts=proposed_facts,
        )
        return AutoEvaluationResult(evaluation=evaluation, scoring_source="mcp_client_agent")


def _provisional_heuristic_evaluation(
    actor_result: ActorResult,
    facts: list[NewSemanticFact],
    failure_summary: str | None,
    user_message: str,
    assistant_response: str,
) -> CriticEvaluation:
    """Invent temporary scores so consolidation never fails before client scoring arrives."""
    total_tools = len(actor_result.tool_calls)
    failed_tools = sum(1 for tool in actor_result.tool_calls if not tool.success)
    has_failures = failed_tools > 0 or failure_summary is not None

    if has_failures and total_tools > 0 and failed_tools == total_tools:
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

    return CriticEvaluation(
        factual_accuracy=factual_accuracy,
        preference_adherence=base,
        tool_efficiency=tool_efficiency,
        hallucination_risk=hallucination_risk,
        workflow_quality=workflow_quality,
        new_semantic_facts=facts,
        save_workflow=False,
        failure_summary=failure_summary,
    )


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


def _agent_semantic_facts(facts: list[NewSemanticFact]) -> list[NewSemanticFact]:
    """Return typed semantic facts mined by the MCP client agent."""
    return [NewSemanticFact.model_validate(fact.model_dump(mode="json", by_alias=True)) for fact in facts]
