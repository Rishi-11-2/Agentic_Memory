"""FastMCP entry point exposing Agentic Memory as developer tools.

This is the primary interface for AI coding agents (Claude Code, Codex, etc.)
to interact with persistent multi-layer memory via the Model Context Protocol.

Workflow for AI agents:
    1. Call get_session_context() before generating a response → get relevant memories
    2. Generate your response (the agent IS the actor)
    3. Call consolidate_turn() after responding → the system automatically evaluates
       quality, extracts facts, learns workflows, and records failures
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from fastmcp import FastMCP

from config import Settings, load_settings
from core.memory_service import AgenticMemoryService
from core.models import (
    ActorResult,
    CriticEvaluation,
    MemoryLayer,
    NewSemanticFact,
    SemanticFactType,
    ToolInvocation,
)
from model.embedding_model import EmbeddingModel, create_embedding_model
from planner.retrieval_planner import HeuristicRetrievalPlanner
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
    planner: HeuristicRetrievalPlanner
    memory_service: AgenticMemoryService
    critic: Any = field(default=None)  # Critic instance, None if no LLM configured


# ── Primary Tools ────────────────────────────────────────────────────


@mcp.tool()
async def get_session_context(
    user_message: str,
    session_id: str,
) -> str:
    """Build a memory-enriched context prompt before generating a response.

    Call this BEFORE responding to the user. Returns a structured text block
    containing relevant system rules, user preferences, suggested workflows,
    recent conversation history, and similar past episodes.

    Inject the returned text into your system prompt or context window to make
    your response informed by persistent memory.
    """
    components = await _components()
    plan = await components.planner.plan(user_message, session_id)
    context = await components.memory_service.build_context(plan)
    return context.rendered_context


@mcp.tool()
async def consolidate_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    tool_calls_json: str = "[]",
    new_facts: list[str] | None = None,
    failure_summary: str | None = None,
) -> str:
    """Save a completed interaction and automatically learn from it.

    Call this AFTER responding to the user. The system will:
    - Save the conversation turn
    - Record the episode with tool usage patterns
    - Auto-evaluate response quality using an LLM critic (if configured)
    - Extract and deduplicate semantic facts (preferences, rules, inferred knowledge)
    - Learn reusable tool workflows from successful multi-step patterns
    - Record failures for future avoidance

    Parameters:
    - session_id: Current conversation session identifier
    - user_message: The user's original message
    - assistant_response: Your response text
    - tool_calls_json: JSON array of tool calls made (optional, default "[]")
      Each entry: {"tool_name": "...", "input_parameters": {...}, "success": true/false, ...}
    - new_facts: Facts learned from this interaction (e.g. ["user prefers Python"])
    - failure_summary: Description of any failures encountered
    """
    components = await _components()

    # Parse tool calls
    parsed_tool_calls = json.loads(tool_calls_json) if tool_calls_json.strip() else []
    if not isinstance(parsed_tool_calls, list):
        raise ValueError("tool_calls_json must decode to a JSON array")
    tool_calls = [ToolInvocation.model_validate(item) for item in parsed_tool_calls]

    # Build actor result (the AI agent IS the actor)
    actor_result = ActorResult(tool_calls=tool_calls, final_response=assistant_response)

    # Auto-evaluate using the Critic LLM if available, otherwise use sensible defaults
    critic_evaluation = await _auto_evaluate(
        components, user_message, assistant_response, actor_result, new_facts, failure_summary
    )

    # Consolidate into all memory layers
    turn_index, writes, conflicts = await components.memory_service.consolidate(
        prompt=user_message,
        actor_result=actor_result,
        critic_evaluation=critic_evaluation,
        session_id=session_id,
        loop_latency_ms=0,
    )

    # Feed critic result back to planner for retrieval weight tuning
    plan = await components.planner.plan(user_message, session_id)
    components.planner.record_feedback(session_id, plan, critic_evaluation.passed)

    return json.dumps(
        {
            "turn_index": turn_index,
            "critic_score": critic_evaluation.overall_score,
            "critic_passed": critic_evaluation.passed,
            "writes": writes,
            "semantic_conflicts": conflicts,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def search_memory(
    query: str,
    layers: list[Literal["semantic", "episodic", "procedural", "failure"]] | None = None,
    top_k: int = 5,
) -> str:
    """Search memory layers for records relevant to a query.

    Returns matching records ranked by semantic similarity. Useful for finding
    relevant past experiences, learned facts, known workflows, or past failures.
    """
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


# ── Internal: Auto Critic Evaluation ────────────────────────────────


async def _auto_evaluate(
    components: MCPComponents,
    user_message: str,
    assistant_response: str,
    actor_result: ActorResult,
    new_facts: list[str] | None,
    failure_summary: str | None,
) -> CriticEvaluation:
    """Run the Critic LLM if available, otherwise build a heuristic evaluation.

    When an LLM provider is configured, the Critic evaluates the turn across
    five quality dimensions and extracts semantic facts automatically.
    When no LLM is configured, a heuristic evaluation is built from the
    agent-supplied facts and tool success/failure signals.
    """
    facts_list = new_facts or []

    if components.critic is not None:
        try:
            # Build a minimal memory context for the critic to evaluate against
            plan = await components.planner.plan(user_message, "critic-eval")
            memory_context = await components.memory_service.build_context(plan)
            critic_eval = await components.critic.evaluate(user_message, memory_context, actor_result)
            # Merge any agent-supplied facts with critic-discovered facts
            if facts_list:
                agent_facts = [
                    NewSemanticFact(
                        fact_type=SemanticFactType.INFERRED_FACT,
                        content=fact,
                        confidence=0.85,
                        source="llm_inferred",
                    )
                    for fact in facts_list
                ]
                critic_eval.new_semantic_facts.extend(agent_facts)
            if failure_summary and not critic_eval.failure_summary:
                critic_eval.failure_summary = failure_summary
            return critic_eval
        except Exception as exc:
            logger.warning("critic_evaluation_failed error=%s, falling back to heuristic", exc)

    # Heuristic fallback: build evaluation from tool signals and agent-supplied info
    return _heuristic_evaluation(actor_result, facts_list, failure_summary)


def _heuristic_evaluation(
    actor_result: ActorResult,
    facts: list[str],
    failure_summary: str | None,
) -> CriticEvaluation:
    """Build a sensible CriticEvaluation from tool success/failure signals.

    Used when no LLM provider is configured (the common case for MCP-only usage).
    """
    total_tools = len(actor_result.tool_calls)
    failed_tools = sum(1 for t in actor_result.tool_calls if not t.success)
    has_failures = failed_tools > 0 or failure_summary is not None

    if has_failures and total_tools > 0 and failed_tools == total_tools:
        score = 3.0  # All tools failed
    elif has_failures:
        score = 6.0  # Partial failure
    elif total_tools > 0:
        score = 9.0  # All tools succeeded
    else:
        score = 8.0  # No tools used, assume decent quality

    save_workflow = total_tools >= 2 and failed_tools == 0

    semantic_facts = [
        NewSemanticFact(
            fact_type=SemanticFactType.INFERRED_FACT,
            content=fact,
            confidence=0.85,
            source="llm_inferred",
        )
        for fact in facts
    ]

    return CriticEvaluation(
        factual_accuracy=score,
        preference_adherence=score,
        tool_efficiency=score,
        hallucination_risk=score,
        workflow_quality=score,
        new_semantic_facts=semantic_facts,
        save_workflow=save_workflow,
        failure_summary=failure_summary,
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
        if settings.llm_provider != "deterministic":
            from model import create_llm_client, critic_model_name
            from runtime.critic import Critic

            llm_client = create_llm_client(settings)
            critic = Critic(llm_client=llm_client, model=critic_model_name(settings))

        _COMPONENTS = MCPComponents(
            settings=settings,
            store=store,
            embedding_model=embedding_model,
            planner=planner,
            memory_service=memory_service,
            critic=critic,
        )
        return _COMPONENTS


if __name__ == "__main__":
    settings = load_settings()
    if settings.mcp_transport == "http":
        mcp.run(transport="sse", port=settings.mcp_http_port)
    else:
        mcp.run()
