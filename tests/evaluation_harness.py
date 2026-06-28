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
    SemanticHierarchyNode,
    SemanticHierarchyNodeType,
    SemanticMemoryRecord,
    ToolInvocation,
)
from model.embedding_model import HashEmbeddingModel
from planner.retrieval_planner import AgenticRetrievalOrchestrator, HeuristicRetrievalPlanner
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
        self.orchestrator = AgenticRetrievalOrchestrator(self.store, self.embedding, memory_window_turns=3)
        self.planner = HeuristicRetrievalPlanner(self.store, self.embedding, memory_window_turns=3)

    async def asyncTearDown(self) -> None:
        """Close and remove the temporary database."""
        await self.store.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    async def test_evaluator_uses_agent_mined_semantic_facts(self) -> None:
        evaluator = AutoEvaluationService()
        evaluation = await evaluator.evaluate(
            user_message="I prefer concise answers. My project uses FastAPI.",
            assistant_response="Understood. I will keep answers concise.",
            actor_result=ActorResult(final_response="Understood. I will keep answers concise."),
            semantic_facts=[
                NewSemanticFact(
                    fact_type=SemanticFactType.PREFERENCE,
                    content="User preference: prefers concise answers.",
                    confidence=0.9,
                    source="user_stated",
                ),
                NewSemanticFact(
                    fact_type=SemanticFactType.SYSTEM_RULE,
                    content="Project context: the project uses FastAPI.",
                    confidence=0.85,
                    source="user_stated",
                ),
            ],
            agent_evaluation=CriticEvaluation(
                factual_accuracy=8,
                preference_adherence=8,
                tool_efficiency=8,
                hallucination_risk=8,
                workflow_quality=8,
                save_workflow=False,
            ),
            quality_score=1.0,
        )
        contents = [fact.content for fact in evaluation.new_semantic_facts]
        self.assertTrue(any("concise answers" in content for content in contents))
        self.assertTrue(any("FastAPI" in content for content in contents))
        self.assertGreaterEqual(evaluation.overall_score, 7.0)
        self.assertTrue(evaluation.passed)

        no_agent_facts = await evaluator.evaluate(
            user_message="I prefer verbose answers. My project uses Django.",
            assistant_response="Understood.",
            actor_result=ActorResult(final_response="Understood."),
            quality_score=8.0,
        )
        self.assertEqual(no_agent_facts.new_semantic_facts, [])
        provisional_result = await evaluator.evaluate_with_metadata(
            user_message="Summarize the change.",
            assistant_response="Done with a clear summary.",
            actor_result=ActorResult(final_response="Done with a clear summary."),
            quality_score=1.0,
        )
        self.assertEqual(provisional_result.scoring_source, "heuristic_provisional")
        self.assertTrue(provisional_result.needs_agent_rescore)
        self.assertTrue(provisional_result.evaluation.passed)
        self.assertFalse(provisional_result.evaluation.save_workflow)

        agent_scored = await evaluator.evaluate_with_metadata(
            user_message="Use whatever format seems best.",
            assistant_response="Done.",
            actor_result=ActorResult(final_response="Done."),
            quality_score=1.0,
            agent_evaluation=CriticEvaluation(
                factual_accuracy=8,
                preference_adherence=8,
                tool_efficiency=8,
                hallucination_risk=8,
                workflow_quality=8,
                save_workflow=False,
                new_semantic_facts=[
                    NewSemanticFact(
                        fact_type=SemanticFactType.PREFERENCE,
                        content="Embedded evaluation fact should not be saved.",
                        confidence=0.9,
                        source="llm_inferred",
                    )
                ],
            ),
            semantic_facts=[
                NewSemanticFact(
                    fact_type=SemanticFactType.PREFERENCE,
                    content="User prefers verification commands when code changes are made.",
                    confidence=0.74,
                    source="llm_inferred",
                )
            ],
        )
        self.assertEqual(agent_scored.scoring_source, "mcp_client_agent")
        self.assertFalse(agent_scored.needs_agent_rescore)
        agent_contents = [fact.content for fact in agent_scored.evaluation.new_semantic_facts]
        self.assertNotIn("Embedded evaluation fact should not be saved.", agent_contents)
        self.assertIn("User prefers verification commands when code changes are made.", agent_contents)
        self.assertEqual(agent_scored.evaluation.overall_score, 8.0)

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
        self.assertGreaterEqual(await self.store.count_layer(MemoryLayer.SEMANTIC_HIERARCHY), 4)

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

        hierarchy_records = await self.store.search_semantic_hierarchy(
            await self.embedding.embed("concise answers preference"),
            limit=5,
            threshold=0.0,
        )
        self.assertTrue(any("Concise" in record.content or "concise" in record.content for record in hierarchy_records))

        plan = await self.planner.plan("I prefer concise answers", "semantic")
        self.assertTrue(plan.semantic_hierarchy_records)
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

    async def test_mcp_client_agent_orchestrates_layer_retrieval(self) -> None:
        manifest = self.orchestrator.manifest()
        self.assertEqual(manifest["orchestrator"], "mcp_client_agent")
        self.assertTrue(manifest["agentic_contract"]["no_hidden_planner_llm"])
        self.assertEqual(manifest["scoring"]["owner"], "mcp_client_agent")
        self.assertIn("scoring_source", manifest["scoring"]["returned_as"])
        self.assertFalse(manifest["fallback"]["agentic"])
        self.assertEqual(manifest["memory_mining"]["orchestrator"], "mcp_client_agent")
        self.assertEqual(manifest["memory_mining"]["output_field"], "semantic_facts_json")
        tool_names = {tool["name"] for tool in manifest["tools"]}
        self.assertIn("rescore_episode", tool_names)
        custom_prompt = "Mine only durable coding workflow preferences."
        custom_orchestrator = AgenticRetrievalOrchestrator(
            self.store,
            self.embedding,
            memory_window_turns=3,
            memory_mining_prompt=custom_prompt,
        )
        self.assertEqual(custom_orchestrator.manifest()["memory_mining"]["prompt"], custom_prompt)

        record = SemanticMemoryRecord(
            fact_type=SemanticFactType.PREFERENCE,
            content="User preference: always use bullet lists",
            embedding=await self.embedding.embed("always use bullet lists"),
            confidence_score=0.9,
            source="user_stated",
        )
        await self.store.insert_semantic(record)

        result = await self.orchestrator.retrieve_layer(
            query="How should answers be formatted?",
            layer=MemoryLayer.SEMANTIC,
            top_k=3,
        )
        self.assertEqual(result["orchestrator"], "mcp_client_agent")
        self.assertEqual(result["layer"], MemoryLayer.SEMANTIC.value)
        self.assertTrue(any("bullet lists" in item["content"] for item in result["records"]))

    async def test_provisional_episode_can_be_rescored_by_mcp_client_agent(self) -> None:
        evaluator = AutoEvaluationService()
        tool_calls = [
            ToolInvocation(tool_name="document_search", success=True),
            ToolInvocation(tool_name="shell_executor", success=True),
        ]
        actor_result = ActorResult(tool_calls=tool_calls, final_response="Implemented and verified.")
        provisional = await evaluator.evaluate_with_metadata(
            user_message="Make the change and test it.",
            assistant_response="Implemented and verified.",
            actor_result=actor_result,
        )
        turn_index, episode_id, _, _ = await self.service.consolidate(
            prompt="Make the change and test it.",
            actor_result=actor_result,
            critic_evaluation=provisional.evaluation,
            session_id="rescore",
            loop_latency_ms=12,
            scoring_source=provisional.scoring_source,
            needs_agent_rescore=provisional.needs_agent_rescore,
        )
        self.assertEqual(turn_index, 0)
        stored = await self.store.get_episode(episode_id)
        assert stored is not None
        self.assertEqual(stored.evaluation_source, "heuristic_provisional")
        self.assertTrue(stored.needs_agent_rescore)
        self.assertEqual(await self.store.count_layer(MemoryLayer.PROCEDURAL), 0)

        final_score = CriticEvaluation(
            factual_accuracy=9,
            preference_adherence=8,
            tool_efficiency=9,
            hallucination_risk=8,
            workflow_quality=9,
            save_workflow=True,
        )
        updated, writes = await self.service.apply_episode_rescore(episode_id, final_score)
        assert updated is not None
        self.assertEqual(updated.evaluation_source, "mcp_client_agent")
        self.assertFalse(updated.needs_agent_rescore)
        self.assertEqual(updated.evaluation_score, final_score.overall_score)
        self.assertTrue(any("Updated episodic episode evaluation" in write for write in writes))
        self.assertGreaterEqual(await self.store.count_layer(MemoryLayer.PROCEDURAL), 1)

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
            semantic_hierarchy_records=[
                SemanticHierarchyNode(
                    node_key="summary:project_context",
                    node_type=SemanticHierarchyNodeType.SUMMARY,
                    facet="project_context",
                    title="Project Context Summary",
                    content="- The project uses SQLite in local mode.",
                    confidence_score=0.8,
                )
            ],
            procedural_workflows=workflows,
        )
        context = await self.service.build_context(plan)
        self.assertIn("[KNOWN FACTS]", context.rendered_context)
        self.assertIn("The project uses SQLite", context.rendered_context)
        self.assertIn("[SEMANTIC MEMORY SUMMARIES]", context.rendered_context)
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
        turn_index, _, _, _ = await self.service.consolidate(
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
