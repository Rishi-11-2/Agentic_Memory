"""FastMCP entry point exposing Agentic Memory as developer tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

from fastmcp import FastMCP

from config import Settings, load_settings
from core.memory_service import AgenticMemoryService
from core.models import ActorResult, CriticEvaluation, MemoryLayer, NewSemanticFact, SemanticFactType, ToolInvocation
from model.embedding_model import EmbeddingModel, create_embedding_model
from planner.retrieval_planner import HeuristicRetrievalPlanner
from store.base import MemoryStore
from store.factory import create_memory_store

mcp = FastMCP("agentic-memory")
_COMPONENTS: "MCPComponents | None" = None
_INIT_LOCK = asyncio.Lock()


@dataclass
class MCPComponents:
    """Hold lazily initialized MCP dependencies."""

    settings: Settings
    store: MemoryStore
    embedding_model: EmbeddingModel
    planner: HeuristicRetrievalPlanner
    memory_service: AgenticMemoryService


@mcp.tool()
async def get_session_context(
    user_message: str,
    session_id: str,
) -> str:
    """Build the memory context prompt for a session before the AI agent generates a response."""
    components = await _components()
    plan = await components.planner.plan(user_message, session_id)
    context = await components.memory_service.build_context(plan)
    return context.rendered_context


@mcp.tool()
async def consolidate_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    critic_score: float,
    critic_passed: bool,
    new_facts: list[str],
    save_workflow: bool,
    workflow_name: str,
    tool_calls_json: str,
    failure_summary: str | None,
) -> str:
    """Persist one completed turn into conversational, episodic, semantic, and procedural memory."""
    del workflow_name
    components = await _components()
    parsed_tool_calls = json.loads(tool_calls_json) if tool_calls_json.strip() else []
    if not isinstance(parsed_tool_calls, list):
        raise ValueError("tool_calls_json must decode to a JSON array")
    tool_calls = [ToolInvocation.model_validate(item) for item in parsed_tool_calls]
    semantic_facts = [
        NewSemanticFact(
            fact_type=SemanticFactType.INFERRED_FACT,
            content=fact,
            confidence=0.85,
            source="llm_inferred",
        )
        for fact in new_facts
    ]
    bounded_score = max(0.0, min(10.0, critic_score))
    critic_evaluation = CriticEvaluation(
        factual_accuracy=bounded_score,
        preference_adherence=bounded_score,
        tool_efficiency=bounded_score,
        hallucination_risk=bounded_score,
        workflow_quality=bounded_score,
        new_semantic_facts=semantic_facts,
        save_workflow=save_workflow,
        failure_summary=failure_summary,
    )
    critic_evaluation.passed = critic_passed
    actor_result = ActorResult(tool_calls=tool_calls, final_response=assistant_response)
    turn_index, writes, conflicts = await components.memory_service.consolidate(
        prompt=user_message,
        actor_result=actor_result,
        critic_evaluation=critic_evaluation,
        session_id=session_id,
        loop_latency_ms=0,
    )
    return json.dumps(
        {"turn_index": turn_index, "writes": writes, "semantic_conflicts": conflicts},
        ensure_ascii=False,
    )


@mcp.tool()
async def search_memory(
    query: str,
    layers: list[Literal["semantic", "episodic", "procedural", "failure"]] | None = None,
    top_k: int = 5,
) -> str:
    """Search selected memory layers for records relevant to a query."""
    components = await _components()
    selected = layers or ["semantic", "episodic", "procedural", "failure"]
    embedding = await components.embedding_model.embed(query)
    result: dict[str, Any] = {}
    if "semantic" in selected:
        semantic_records = await components.store.search_semantic(
            embedding,
            top_k,
            0.0,
            0.0,
            last_confirmed_after=components.memory_service.semantic_cutoff(),
        )
        result["semantic"] = [record.model_dump(mode="json") for record in semantic_records]
    if "episodic" in selected:
        episodic_records = await components.store.search_episodes(embedding, top_k, 0.0)
        result["episodic"] = [record.model_dump(mode="json") for record in episodic_records]
    if "procedural" in selected:
        procedural_records = await components.store.search_procedural(embedding, top_k, 0.0)
        result["procedural"] = [record.model_dump(mode="json") for record in procedural_records]
    if "failure" in selected:
        failure_records = await components.store.search_failures(embedding, top_k, 0.0)
        result["failure"] = [record.model_dump(mode="json") for record in failure_records]
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_conversation_history(
    session_id: str,
    last_n: int = 10,
) -> str:
    """Return recent conversation messages for a session."""
    components = await _components()
    records = await components.store.get_conversation_turns(session_id, last_n)
    return json.dumps(
        [
            {"role": record.role.value, "content": record.content, "timestamp": record.timestamp.isoformat()}
            for record in records
        ],
        ensure_ascii=False,
    )


@mcp.tool()
async def clear_session_memory(
    session_id: str,
) -> str:
    """Clear only conversational memory for a session, leaving durable memories intact."""
    components = await _components()
    await components.store.clear_conversation(session_id)
    return json.dumps({"cleared": True, "session_id": session_id}, ensure_ascii=False)


@mcp.tool()
async def inspect_memory_layers(
    limit: int = 20,
) -> str:
    """Summarize all memory layers for debugging or admin UI inspection."""
    components = await _components()
    counts = {
        layer.value: await components.store.count_layer(layer)
        for layer in (MemoryLayer.CONVERSATIONAL, MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL)
    }
    counts["failure"] = await components.store.count_layer(MemoryLayer.FAILURE)
    semantic = await components.store.inspect_layer(MemoryLayer.SEMANTIC, 3, 0)
    episodes = await components.store.inspect_layer(MemoryLayer.EPISODIC, 3, 0)
    workflows = await components.store.inspect_layer(MemoryLayer.PROCEDURAL, limit, 0)
    return json.dumps(
        {
            "counts": counts,
            "recent_semantic_facts": semantic,
            "recent_episodes": episodes,
            "active_procedural_workflows": workflows,
        },
        default=str,
        ensure_ascii=False,
    )


async def _components() -> MCPComponents:
    """Lazily initialize MCP dependencies exactly once."""
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS
    async with _INIT_LOCK:
        if _COMPONENTS is not None:
            return _COMPONENTS
        settings = load_settings()
        store = await create_memory_store(settings)
        embedding_model = create_embedding_model(settings)
        planner = HeuristicRetrievalPlanner(
            store=store,
            embedding_model=embedding_model,
            memory_window_turns=settings.memory_window_turns,
            failure_similarity_threshold=settings.failure_similarity_threshold,
            semantic_memory_ttl_days=settings.semantic_memory_ttl_days,
        )
        memory_service = AgenticMemoryService(
            store=store,
            embedding_model=embedding_model,
            memory_window_turns=settings.memory_window_turns,
            semantic_dedup_threshold=settings.semantic_dedup_threshold,
            semantic_memory_ttl_days=settings.semantic_memory_ttl_days,
        )
        _COMPONENTS = MCPComponents(
            settings=settings,
            store=store,
            embedding_model=embedding_model,
            planner=planner,
            memory_service=memory_service,
        )
        return _COMPONENTS


if __name__ == "__main__":
    settings = load_settings()
    if settings.mcp_transport == "http":
        mcp.run(transport="sse", port=settings.mcp_http_port)
    else:
        mcp.run()
