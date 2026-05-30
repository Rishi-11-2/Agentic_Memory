"""Offline evaluation harness for core Agentic Memory behavior.

Run with:
    python -m tests.evaluation_harness
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from core.evaluation_service import AutoEvaluationService
from core.memory_service import AgenticMemoryService
from core.models import (
    ActorResult,
    CriticEvaluation,
    MemoryLayer,
    NewSemanticFact,
    ProceduralToolStep,
    ProceduralWorkflow,
    RetrievalPlan,
    SemanticFactType,
    SemanticMemoryRecord,
    ToolInvocation,
)
from model.embedding_model import HashEmbeddingModel
from planner.retrieval_planner import HeuristicRetrievalPlanner
from store.sqlite_store import SQLiteMemoryStore


class AgenticMemoryHarness(unittest.IsolatedAsyncioTestCase):
    """Exercise core memory behaviors against SQLite and hash embeddings."""

    async def asyncSetUp(self) -> None:
        """Create an isolated local memory stack for each scenario."""
        fd, self.db_path = tempfile.mkstemp(prefix="agentic-memory-harness-", suffix=".db")
        os.close(fd)
        os.unlink(self.db_path)
        self.store = await SQLiteMemoryStore.create(self.db_path)
        self.embedding = HashEmbeddingModel()
        self.service = AgenticMemoryService(self.store, self.embedding, memory_window_turns=3)
        self.planner = HeuristicRetrievalPlanner(self.store, self.embedding, memory_window_turns=3)

    async def asyncTearDown(self) -> None:
        """Close and remove the temporary database."""
        await self.store.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    async def test_heuristic_evaluator_extracts_preferences_and_facts(self) -> None:
        evaluator = AutoEvaluationService()
        evaluation = await evaluator.evaluate(
            user_message="I prefer concise answers. My project uses FastAPI.",
            assistant_response="Understood. I will keep answers concise.",
            actor_result=ActorResult(final_response="Understood. I will keep answers concise."),
            quality_score=8.0,
        )
        contents = [fact.content for fact in evaluation.new_semantic_facts]
        self.assertTrue(any("I prefer concise answers" in content for content in contents))
        self.assertTrue(any("My project uses FastAPI" in content for content in contents))
        self.assertGreaterEqual(evaluation.overall_score, 7.0)

    async def test_semantic_dedup_and_pin_stale_management(self) -> None:
        evaluation = CriticEvaluation(
            factual_accuracy=8,
            preference_adherence=8,
            tool_efficiency=8,
            hallucination_risk=8,
            workflow_quality=8,
            new_semantic_facts=[
                NewSemanticFact(
                    fact_type=SemanticFactType.PREFERENCE,
                    content="User preference: I prefer concise answers",
                    confidence=0.9,
                    source="user_stated",
                ),
                NewSemanticFact(
                    fact_type=SemanticFactType.PREFERENCE,
                    content="User preference: I prefer concise answers",
                    confidence=0.8,
                    source="user_stated",
                ),
            ],
        )
        await self.service.consolidate(
            prompt="I prefer concise answers.",
            actor_result=ActorResult(final_response="Got it."),
            critic_evaluation=evaluation,
            session_id="semantic",
            loop_latency_ms=1,
        )
        self.assertEqual(await self.store.count_layer(MemoryLayer.SEMANTIC), 1)

        records = await self.store.search_semantic(
            await self.embedding.embed("concise answers"),
            limit=5,
            threshold=0.0,
            min_confidence=0.0,
        )
        self.assertEqual(len(records), 1)
        fact_id = records[0].fact_id
        stale_at = datetime.now(timezone.utc) - timedelta(days=365)
        self.assertTrue(await self.store.update_semantic_metadata(fact_id, pinned=True, last_confirmed_at=stale_at))
        pinned_records = await self.store.search_semantic(
            await self.embedding.embed("concise answers"),
            limit=5,
            threshold=0.0,
            min_confidence=0.0,
            last_confirmed_after=datetime.now(timezone.utc) - timedelta(days=30),
        )
        self.assertEqual([record.fact_id for record in pinned_records], [fact_id])
        self.assertTrue(await self.store.delete_semantic(fact_id))

    async def test_planner_feedback_persists_between_instances(self) -> None:
        record = SemanticMemoryRecord(
            fact_type=SemanticFactType.PREFERENCE,
            content="User preference: always use bullet lists",
            embedding=await self.embedding.embed("always use bullet lists"),
            confidence_score=0.9,
            source="user_stated",
        )
        await self.store.insert_semantic(record)
        plan = await self.planner.plan("I prefer bullet lists", "feedback")
        await self.planner.record_feedback("feedback", plan, critic_passed=True)

        fresh_planner = HeuristicRetrievalPlanner(self.store, self.embedding, memory_window_turns=3)
        weights = await self.store.get_retrieval_weights("feedback")
        assert weights is not None
        self.assertGreater(weights["semantic"], 1.0)
        fresh_plan = await fresh_planner.plan("I prefer bullet lists", "feedback")
        self.assertTrue(fresh_plan.query_semantic)

    async def test_context_renders_multiple_workflows_and_known_facts(self) -> None:
        workflows = [
            ProceduralWorkflow(
                workflow_signature=f"workflow-{index}",
                trigger_phrases=[f"deploy {index}"],
                success_count=index,
                tool_sequence=[
                    ProceduralToolStep(tool_name="shell_executor", expected_outcome="command succeeds"),
                    ProceduralToolStep(tool_name="document_search", expected_outcome="evidence found"),
                ],
            )
            for index in (1, 3)
        ]
        plan = RetrievalPlan(
            session_id="render",
            prompt="deploy",
            semantic_records=[
                SemanticMemoryRecord(
                    fact_type=SemanticFactType.INFERRED_FACT,
                    content="The project uses SQLite in local mode.",
                    confidence_score=0.8,
                )
            ],
            procedural_workflows=workflows,
        )
        context = await self.service.build_context(plan)
        self.assertIn("[KNOWN FACTS]", context.rendered_context)
        self.assertIn("The project uses SQLite", context.rendered_context)
        self.assertIn("Workflow 1", context.rendered_context)
        self.assertIn("Workflow 2", context.rendered_context)

    async def test_failure_and_episode_recall(self) -> None:
        tool = ToolInvocation(
            tool_name="shell_executor",
            input_summary="bad command",
            output_summary="command failed",
            success=False,
            critic_flagged=True,
            error_trace="bad command failed",
        )
        evaluation = CriticEvaluation(
            factual_accuracy=3,
            preference_adherence=3,
            tool_efficiency=1,
            hallucination_risk=3,
            workflow_quality=2,
            failure_summary="Shell command failed.",
        )
        turn_index, _, _ = await self.service.consolidate(
            prompt="run a risky command",
            actor_result=ActorResult(tool_calls=[tool], final_response="It failed."),
            critic_evaluation=evaluation,
            session_id="failure",
            loop_latency_ms=5,
        )
        self.assertEqual(turn_index, 0)
        planner = HeuristicRetrievalPlanner(self.store, self.embedding, failure_similarity_threshold=0.0)
        plan = await planner.plan("run risky command again", "failure")
        self.assertTrue(plan.query_episodic)
        self.assertTrue(plan.failure_matches)


if __name__ == "__main__":
    unittest.main()
