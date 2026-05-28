"""Critic reflection for Actor-Critic self-learning."""

from __future__ import annotations

from core.models import ActorResult, CriticEvaluation, LLMMessage, MemoryContext
from model import StructuredLLMClient


class Critic:
    """Evaluate Actor behavior and emit structured learning signals."""

    def __init__(self, llm_client: StructuredLLMClient, model: str) -> None:
        """Create a Critic with a model-agnostic structured LLM client."""
        self._llm_client = llm_client
        self._model = model

    async def evaluate(
        self,
        prompt: str,
        memory_context: MemoryContext,
        actor_result: ActorResult,
        temperature: float = 0.0,
    ) -> CriticEvaluation:
        """Run the hidden Critic phase and return a structured evaluation."""
        messages = [
            LLMMessage(role="system", content=self._system_prompt()),
            LLMMessage(
                role="user",
                content=(
                    "Evaluate this turn.\n\n"
                    f"USER_PROMPT:\n{prompt}\n\n"
                    f"MEMORY_CONTEXT:\n{memory_context.rendered_context}\n\n"
                    f"ACTOR_RESULT_JSON:\n{actor_result.model_dump_json()}"
                ),
            ),
        ]
        return await self._llm_client.complete_json(
            messages=messages,
            response_model=CriticEvaluation,
            model=self._model,
            temperature=temperature,
        )

    def _system_prompt(self) -> str:
        """Build the Critic prompt with scoring and consolidation instructions."""
        return (
            "You are the hidden Critic in an Actor-Critic self-learning loop. Return only valid JSON. "
            "Score each dimension from 0 to 10, where higher is better. Evaluate factual_accuracy, "
            "preference_adherence, tool_efficiency, hallucination_risk, and workflow_quality. "
            "hallucination_risk is high when the response is well grounded and low risk. "
            "Extract new_semantic_facts only when they are durable preferences, inferred facts, or system rules. "
            "Use source='user_stated' only for explicit user statements; otherwise use source='llm_inferred'. "
            "Set save_workflow true only for a successful non-trivial chain of at least two useful tools. "
            "Use this JSON shape: "
            '{"factual_accuracy": 8, "preference_adherence": 8, "tool_efficiency": 8, '
            '"hallucination_risk": 8, "workflow_quality": 8, "overall_score": 8, "pass": true, '
            '"new_semantic_facts": [{"fact_type": "preference", "content": "...", '
            '"confidence_score": 0.8, "source": "user_stated"}], '
            '"save_workflow": false, "failure_summary": null}.'
        )
