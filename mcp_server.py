"""FastMCP entry point exposing Agentic Memory as developer tools.

This is the primary interface for AI coding agents (Claude Code, Codex, etc.)
to interact with persistent multi-layer memory via the Model Context Protocol.

Workflow for AI agents:
    1. Call get_memory_tool_manifest() to inspect the retrieval contract
    2. Call retrieve_memory_layer() one or more times with your chosen layers
    3. Generate your response (the agent IS the actor and retrieval orchestrator)
    4. Call consolidate_turn() after responding with mined facts and your score
       -> if scoring was omitted, call rescore_episode() with your final score
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastmcp import FastMCP

from config import Settings, load_settings
from core.evaluation_service import AutoEvaluationService
from core.memory_service import AgenticMemoryService
from core.models import (
    ActorResult,
    CriticEvaluation,
    MemoryLayer,
    NewSemanticFact,
    SemanticMemoryRecord,
    ToolInvocation,
)
from model.embedding_model import EmbeddingModel, create_embedding_model
from planner.retrieval_planner import AgenticRetrievalOrchestrator, HeuristicRetrievalPlanner
from store.base import MemoryStore
from store.factory import create_memory_store

logger = logging.getLogger(__name__)

mcp = FastMCP("agentic-memory")
_COMPONENTS: "MCPComponents | None" = None
_INIT_LOCK = asyncio.Lock()


@dataclass
class MCPComponents:
    """Hold lazily initialized MCP dependencies."""

    settings: Settings
    store: MemoryStore
    embedding_model: EmbeddingModel
    retrieval_orchestrator: AgenticRetrievalOrchestrator
    planner: HeuristicRetrievalPlanner
    memory_service: AgenticMemoryService
    evaluation_service: AutoEvaluationService
    critic: Any = field(default=None)  # Critic instance, None if no LLM configured
    learning_loop: Any = field(default=None)  # SelfLearningLoop, None unless standalone enabled


# ── Primary Tools ────────────────────────────────────────────────────


@mcp.tool()
async def get_memory_tool_manifest() -> str:
    """Describe memory-layer tools for MCP-client agentic orchestration.

    Codex, Claude Code, Cline, or another MCP client is the orchestrator: it
    decides which layers to query, performs multi-hop retrieval, and synthesizes
    grounded context. Agentic Memory only exposes durable retrieval tools.
    """
    components = await _components()
    return json.dumps(components.retrieval_orchestrator.manifest(), ensure_ascii=False)


@mcp.tool()
async def get_session_context(
    user_message: str,
    session_id: str,
) -> str:
    """Build a fallback memory-enriched context prompt before responding.

    This is the backward-compatible convenience path. For CMA-style agentic
    retrieval, prefer get_memory_tool_manifest plus repeated retrieve_memory_layer
    calls so the MCP client agent chooses layers and multi-hop queries itself.

    Inject the returned text into your system prompt or context window to make
    your response informed by persistent memory.
    """
    components = await _components()
    plan = await components.planner.plan(user_message, session_id)
    context = await components.memory_service.build_context(plan)
    return context.rendered_context


@mcp.tool()
async def retrieve_memory_layer(
    query: str,
    layer: Literal["conversational", "semantic", "semantic_hierarchy", "episodic", "procedural", "failure"],
    session_id: str = "",
    top_k: int = 5,
) -> str:
    """Retrieve one memory layer selected by the MCP client orchestrator.

    Codex, Claude Code, Cline, or another MCP client should call this repeatedly
    with refined queries when it needs multi-hop retrieval. The server does not
    use a hidden planner LLM.
    """
    components = await _components()
    payload = await components.retrieval_orchestrator.retrieve_layer(
        query=query,
        layer=MemoryLayer(layer),
        session_id=session_id,
        top_k=top_k,
    )
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
async def consolidate_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    tool_calls_json: str = "[]",
    new_facts: list[str] | None = None,
    semantic_facts_json: str = "[]",
    agent_evaluation_json: str = "",
    failure_summary: str | None = None,
    quality_score: float | None = None,
    reasoning_summary: str | None = None,
) -> str:
    """Save a completed interaction and automatically learn from it.

    Call this AFTER responding to the user. The system will:
    - Save the conversation turn
    - Record the episode with tool usage patterns
    - Record response quality using the MCP client agent's typed score
    - Save and deduplicate semantic facts mined by the MCP client agent
    - Learn reusable tool workflows from successful multi-step patterns
    - Record failures for future avoidance

    Parameters:
    - session_id: Current conversation session identifier
    - user_message: The user's original message
    - assistant_response: Your response text
    - tool_calls_json: JSON array of tool calls made (optional, default "[]")
      Each entry: {"tool_name": "...", "input_parameters": {...}, "success": true/false, ...}
    - new_facts: Backward-compatible untyped fact strings learned by the MCP client agent.
    - semantic_facts_json: JSON array of typed facts mined by the MCP client agent.
      Each entry: {"fact_type": "preference|inferred_fact|system_rule", "content": "...",
      "confidence": 0.0-1.0, "source": "user_stated|llm_inferred|tool_derived"}.
    - agent_evaluation_json: Preferred typed scoring JSON produced by Codex,
      Cline, Claude Code, or the MCP client agent. Uses CriticEvaluation fields:
      factual_accuracy, preference_adherence, tool_efficiency,
      hallucination_risk, workflow_quality, save_workflow, failure_summary.
      If omitted, the server creates provisional scores so consolidation does
      not fail, then returns needs_agent_rescore=true and an episode_id for
      rescore_episode.
    - failure_summary: Description of any failures encountered
    - quality_score: Deprecated optional agent self-score kept for compatibility.
      Ignored when agent_evaluation_json is present.
    - reasoning_summary: Optional brief summary of the agent's internal reasoning
      or approach for this turn. Stored for episodic recall.
    """
    components = await _components()

    # Parse tool calls
    parsed_tool_calls = json.loads(tool_calls_json) if tool_calls_json.strip() else []
    if not isinstance(parsed_tool_calls, list):
        raise ValueError("tool_calls_json must decode to a JSON array")
    tool_calls = [ToolInvocation.model_validate(item) for item in parsed_tool_calls]
    parsed_semantic_facts = json.loads(semantic_facts_json) if semantic_facts_json.strip() else []
    if not isinstance(parsed_semantic_facts, list):
        raise ValueError("semantic_facts_json must decode to a JSON array")
    semantic_facts = [NewSemanticFact.model_validate(item) for item in parsed_semantic_facts]
    parsed_agent_evaluation = json.loads(agent_evaluation_json) if agent_evaluation_json.strip() else None
    if parsed_agent_evaluation is not None and not isinstance(parsed_agent_evaluation, dict):
        raise ValueError("agent_evaluation_json must decode to a JSON object")
    agent_evaluation = (
        CriticEvaluation.model_validate(parsed_agent_evaluation)
        if parsed_agent_evaluation is not None
        else None
    )

    # Build actor result (the AI agent IS the actor)
    actor_result = ActorResult(
        tool_calls=tool_calls,
        final_response=assistant_response,
        reasoning=reasoning_summary or "",
    )

    evaluation_result = await components.evaluation_service.evaluate_with_metadata(
        user_message=user_message,
        assistant_response=assistant_response,
        actor_result=actor_result,
        new_facts=new_facts,
        semantic_facts=semantic_facts,
        agent_evaluation=agent_evaluation,
        failure_summary=failure_summary,
        quality_score=quality_score,
    )
    critic_evaluation = evaluation_result.evaluation

    # Consolidate into all memory layers
    turn_index, episode_id, writes, conflicts = await components.memory_service.consolidate(
        prompt=user_message,
        actor_result=actor_result,
        critic_evaluation=critic_evaluation,
        session_id=session_id,
        loop_latency_ms=0,
        scoring_source=evaluation_result.scoring_source,
        needs_agent_rescore=evaluation_result.needs_agent_rescore,
    )

    # Feed critic result back to the fallback planner for quick-path tuning.
    plan = await components.planner.plan(user_message, session_id)
    await components.planner.record_feedback(session_id, plan, critic_evaluation.passed)

    return json.dumps(
        {
            "turn_index": turn_index,
            "episode_id": episode_id,
            "critic_score": critic_evaluation.overall_score,
            "critic_passed": critic_evaluation.passed,
            "scoring_source": evaluation_result.scoring_source,
            "needs_agent_rescore": evaluation_result.needs_agent_rescore,
            "writes": writes,
            "semantic_conflicts": conflicts,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def rescore_episode(
    episode_id: str,
    agent_evaluation_json: str,
    session_id: str = "",
) -> str:
    """Replace a provisional episode score with the MCP client agent's final score.

    Use this when consolidate_turn returned needs_agent_rescore=true. The score
    must be produced by Codex, Cline, Claude Code, or the active MCP client agent,
    not by a server-side assistant. This updates the existing episode instead of
    creating a duplicate memory turn.

    Parameters:
    - episode_id: The episode_id returned by consolidate_turn.
    - agent_evaluation_json: JSON object using CriticEvaluation fields:
      factual_accuracy, preference_adherence, tool_efficiency,
      hallucination_risk, workflow_quality, save_workflow, failure_summary.
    - session_id: Optional session identifier for retrieval feedback tuning.
    """
    components = await _components()
    parsed_agent_evaluation = json.loads(agent_evaluation_json) if agent_evaluation_json.strip() else None
    if parsed_agent_evaluation is None or not isinstance(parsed_agent_evaluation, dict):
        raise ValueError("agent_evaluation_json must decode to a JSON object")
    agent_evaluation = CriticEvaluation.model_validate(parsed_agent_evaluation)

    existing = await components.store.get_episode(episode_id)
    if existing is None:
        return json.dumps({"found": False, "episode_id": episode_id}, ensure_ascii=False)

    actor_result = ActorResult(
        reasoning=existing.reasoning_summary,
        tool_calls=existing.tool_sequence,
        final_response=existing.final_response,
    )
    evaluation_result = await components.evaluation_service.evaluate_with_metadata(
        user_message=existing.prompt_text,
        assistant_response=existing.final_response,
        actor_result=actor_result,
        agent_evaluation=agent_evaluation,
        failure_summary=agent_evaluation.failure_summary,
    )
    updated_episode, writes = await components.memory_service.apply_episode_rescore(
        episode_id=episode_id,
        critic_evaluation=evaluation_result.evaluation,
        scoring_source=evaluation_result.scoring_source,
    )
    if updated_episode is None:
        return json.dumps({"found": False, "episode_id": episode_id}, ensure_ascii=False)

    if session_id:
        plan = await components.planner.plan(existing.prompt_text, session_id)
        await components.planner.record_feedback(session_id, plan, evaluation_result.evaluation.passed)

    return json.dumps(
        {
            "found": True,
            "episode_id": episode_id,
            "critic_score": evaluation_result.evaluation.overall_score,
            "critic_passed": evaluation_result.evaluation.passed,
            "scoring_source": evaluation_result.scoring_source,
            "needs_agent_rescore": False,
            "writes": writes,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def search_memory(
    query: str,
    layers: list[Literal["semantic", "semantic_hierarchy", "episodic", "procedural", "failure"]] | None = None,
    top_k: int = 5,
) -> str:
    """Search memory layers for records relevant to a query.

    Returns matching records ranked by semantic similarity. Useful for finding
    relevant past experiences, learned facts, known workflows, or past failures.
    """
    components = await _components()
    selected = layers or ["semantic", "semantic_hierarchy", "episodic", "procedural", "failure"]
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
    if "semantic_hierarchy" in selected:
        hierarchy_records = await components.store.search_semantic_hierarchy(embedding, top_k, 0.0)
        result["semantic_hierarchy"] = [record.model_dump(mode="json") for record in hierarchy_records]
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
    """Clear conversational memory for a session, leaving durable memories intact.

    Episodic, semantic, procedural, and failure memories are preserved.
    """
    components = await _components()
    await components.store.clear_conversation(session_id)
    return json.dumps({"cleared": True, "session_id": session_id}, ensure_ascii=False)


@mcp.tool()
async def inspect_memory_layers(
    limit: int = 20,
) -> str:
    """Summarize all memory layers with record counts and recent entries."""
    components = await _components()
    counts = {
        layer.value: await components.store.count_layer(layer)
        for layer in (
            MemoryLayer.CONVERSATIONAL,
            MemoryLayer.EPISODIC,
            MemoryLayer.SEMANTIC,
            MemoryLayer.SEMANTIC_HIERARCHY,
            MemoryLayer.PROCEDURAL,
        )
    }
    counts["failure"] = await components.store.count_layer(MemoryLayer.FAILURE)
    semantic = await components.store.inspect_layer(MemoryLayer.SEMANTIC, 3, 0)
    hierarchy = await components.store.inspect_layer(MemoryLayer.SEMANTIC_HIERARCHY, 6, 0)
    episodes = await components.store.inspect_layer(MemoryLayer.EPISODIC, 3, 0)
    workflows = await components.store.inspect_layer(MemoryLayer.PROCEDURAL, limit, 0)
    return json.dumps(
        {
            "counts": counts,
            "recent_semantic_facts": semantic,
            "recent_semantic_hierarchy": hierarchy,
            "recent_episodes": episodes,
            "active_procedural_workflows": workflows,
        },
        default=str,
        ensure_ascii=False,
    )


@mcp.tool()
async def delete_semantic_fact(fact_id: str) -> str:
    """Delete one semantic fact by id."""
    components = await _components()
    deleted = await components.store.delete_semantic(fact_id)
    hierarchy_nodes = await components.memory_service.rebuild_semantic_hierarchy() if deleted else 0
    return json.dumps(
        {"deleted": deleted, "fact_id": fact_id, "rebuilt_semantic_hierarchy_nodes": hierarchy_nodes},
        ensure_ascii=False,
    )


@mcp.tool()
async def pin_semantic_fact(fact_id: str, pinned: bool = True) -> str:
    """Pin or unpin a semantic fact so TTL filtering does not hide it."""
    components = await _components()
    updated = await components.store.update_semantic_metadata(fact_id, pinned=pinned)
    hierarchy_nodes = await components.memory_service.rebuild_semantic_hierarchy() if updated else 0
    return json.dumps(
        {
            "updated": updated,
            "fact_id": fact_id,
            "pinned": pinned,
            "rebuilt_semantic_hierarchy_nodes": hierarchy_nodes,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def mark_semantic_fact_stale(fact_id: str, stale_days: int = 365) -> str:
    """Mark a semantic fact as stale by moving its confirmation timestamp into the past."""
    components = await _components()
    days = max(1, min(int(stale_days), 3650))
    stale_at = datetime.now(timezone.utc) - timedelta(days=days)
    updated = await components.store.update_semantic_metadata(
        fact_id,
        pinned=False,
        last_confirmed_at=stale_at,
    )
    hierarchy_nodes = await components.memory_service.rebuild_semantic_hierarchy() if updated else 0
    return json.dumps(
        {
            "updated": updated,
            "fact_id": fact_id,
            "last_confirmed_at": stale_at.isoformat(),
            "rebuilt_semantic_hierarchy_nodes": hierarchy_nodes,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def inspect_episode(episode_id: str) -> str:
    """Return one full episodic memory record by id."""
    components = await _components()
    episode = await components.store.get_episode(episode_id)
    if episode is None:
        return json.dumps({"found": False, "episode_id": episode_id}, ensure_ascii=False)
    return episode.model_dump_json()


@mcp.tool()
async def prune_old_episodes(older_than_days: int = 180) -> str:
    """Delete episodic memories older than the requested age, preserving detached failure records."""
    components = await _components()
    days = max(1, min(int(older_than_days), 3650))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted = await components.store.prune_episodes_before(cutoff)
    return json.dumps(
        {"deleted": deleted, "older_than_days": days, "cutoff": cutoff.isoformat()},
        ensure_ascii=False,
    )


@mcp.tool()
async def export_memory(
    layers: list[
        Literal["conversational", "episodic", "semantic", "semantic_hierarchy", "procedural", "failure"]
    ] | None = None,
    limit_per_layer: int = 100,
) -> str:
    """Export recent memory records by layer as JSON."""
    components = await _components()
    selected = layers or ["conversational", "episodic", "semantic", "semantic_hierarchy", "procedural", "failure"]
    limit = max(1, min(int(limit_per_layer), 1000))
    result: dict[str, Any] = {}
    for layer_name in selected:
        layer = MemoryLayer(layer_name)
        result[layer.value] = await components.store.inspect_layer(layer, limit, 0)
    return json.dumps(result, default=str, ensure_ascii=False)


@mcp.tool()
async def import_memory(memory_json: str, import_semantic: bool = True) -> str:
    """Import semantic memories from an export payload.

    Non-semantic layers are intentionally skipped because they are append-only audit records
    with cross-table relationships.
    """
    components = await _components()
    payload = json.loads(memory_json)
    if not isinstance(payload, dict):
        raise ValueError("memory_json must decode to a JSON object")
    imported = 0
    skipped = 0
    errors: list[str] = []
    semantic_records = payload.get("semantic", []) if import_semantic else []
    if not isinstance(semantic_records, list):
        raise ValueError("memory_json semantic field must be a list when present")
    for item in semantic_records:
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            record = SemanticMemoryRecord.model_validate(item)
            if not record.embedding:
                record.embedding = await components.embedding_model.embed(record.content)
            await components.store.insert_semantic(record)
            imported += 1
        except Exception as exc:
            skipped += 1
            if len(errors) < 5:
                errors.append(str(exc))
    hierarchy_nodes = await components.memory_service.rebuild_semantic_hierarchy() if imported else 0
    return json.dumps(
        {
            "imported_semantic": imported,
            "skipped": skipped,
            "errors": errors,
            "non_semantic_layers": "skipped",
            "rebuilt_semantic_hierarchy_nodes": hierarchy_nodes,
        },
        ensure_ascii=False,
    )


# ── Initialization ──────────────────────────────────────────────────


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
        retrieval_orchestrator = AgenticRetrievalOrchestrator(
            store=store,
            embedding_model=embedding_model,
            memory_window_turns=settings.memory_window_turns,
            semantic_memory_ttl_days=settings.semantic_memory_ttl_days,
            memory_mining_prompt=settings.memory_mining_prompt,
        )
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

        # Initialize the Critic if an LLM provider is configured
        critic = None
        learning_loop = None
        if settings.llm_provider != "deterministic":
            from model import create_llm_client, actor_model_name, critic_model_name
            from runtime.critic import Critic

            llm_client = create_llm_client(settings)
            critic = Critic(llm_client=llm_client, model=critic_model_name(settings))

            # Build the full standalone loop if enabled
            if getattr(settings, "standalone_loop_enabled", False):
                from runtime.actor import Actor
                from runtime.self_learning_loop import SelfLearningLoop
                from runtime.tools import default_tool_registry

                tool_registry = default_tool_registry(
                    brave_api_key=(
                        settings.brave_search_api_key.get_secret_value()
                        if settings.brave_search_api_key
                        else None
                    ),
                    brave_endpoint=settings.brave_search_endpoint,
                    brave_country=settings.brave_search_country,
                    brave_search_lang=settings.brave_search_lang,
                    brave_count=settings.brave_search_count,
                    brave_timeout_seconds=settings.brave_search_timeout_seconds,
                    workspace_root=settings.tool_workspace_root,
                    memory_store=store,
                    embedding_model=embedding_model,
                    semantic_ttl_cutoff=memory_service.semantic_cutoff(),
                )
                actor = Actor(
                    llm_client=llm_client,
                    model=actor_model_name(settings),
                    tool_registry=tool_registry,
                )
                learning_loop = SelfLearningLoop(
                    planner=planner,
                    memory_service=memory_service,
                    actor=actor,
                    critic=critic,
                    critic_timeout_seconds=settings.critic_timeout_seconds,
                    critic_self_consistency_samples=settings.critic_self_consistency_samples,
                    critic_self_consistency_temperature=settings.critic_self_consistency_temperature,
                )
                logger.info("standalone_loop_initialized provider=%s", settings.llm_provider)

        evaluation_service = AutoEvaluationService(
            critic=critic,
            planner=planner,
            memory_service=memory_service,
        )
        _COMPONENTS = MCPComponents(
            settings=settings,
            store=store,
            embedding_model=embedding_model,
            retrieval_orchestrator=retrieval_orchestrator,
            planner=planner,
            memory_service=memory_service,
            evaluation_service=evaluation_service,
            critic=critic,
            learning_loop=learning_loop,
        )
        return _COMPONENTS


# ── Standalone Loop Tool ────────────────────────────────────────────


@mcp.tool()
async def run_autonomous_turn(
    user_message: str,
    session_id: str,
) -> str:
    """Run a full Actor-Critic self-learning turn autonomously.

    This tool is only available when STANDALONE_LOOP_ENABLED=true and an
    LLM provider is configured. The system acts as BOTH Actor and Critic:
    it generates a response using its own LLM, evaluates it, and learns.

    Use this when you want the memory system to handle the complete
    reasoning-evaluation-learning cycle independently.

    Returns the generated response along with critic evaluation scores.
    """
    components = await _components()
    if components.learning_loop is None:
        return json.dumps(
            {
                "error": "Standalone loop is not enabled. "
                "Set STANDALONE_LOOP_ENABLED=true and configure an LLM provider "
                "(LLM_PROVIDER=anthropic/openai/groq with its API key)."
            },
            ensure_ascii=False,
        )
    result = await components.learning_loop.run_turn(user_message, session_id)
    return result.model_dump_json(indent=2)


if __name__ == "__main__":
    settings = load_settings()
    if settings.mcp_transport == "http":
        mcp.run(transport="sse", port=settings.mcp_http_port)
    else:
        mcp.run()
