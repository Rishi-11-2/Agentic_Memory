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
import re
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
    learning_loop: Any = field(default=None)  # SelfLearningLoop, None unless standalone enabled


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
    quality_score: float | None = None,
    reasoning_summary: str | None = None,
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
    - quality_score: Optional self-assessed quality score from the AI agent (0-10).
      When provided, this is used as the primary quality signal instead of heuristics.
      The agent knows best how well it handled the request.
    - reasoning_summary: Optional brief summary of the agent's internal reasoning
      or approach for this turn. Stored for episodic recall.
    """
    components = await _components()

    # Parse tool calls
    parsed_tool_calls = json.loads(tool_calls_json) if tool_calls_json.strip() else []
    if not isinstance(parsed_tool_calls, list):
        raise ValueError("tool_calls_json must decode to a JSON array")
    tool_calls = [ToolInvocation.model_validate(item) for item in parsed_tool_calls]

    # Build actor result (the AI agent IS the actor)
    actor_result = ActorResult(
        tool_calls=tool_calls,
        final_response=assistant_response,
        reasoning=reasoning_summary or "",
    )

    # Auto-evaluate using the Critic LLM if available, otherwise use enhanced heuristics
    critic_evaluation = await _auto_evaluate(
        components, user_message, assistant_response, actor_result,
        new_facts, failure_summary, quality_score,
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

# Patterns that indicate user preferences (compiled once at module level)
_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bi\s+prefer\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\balways\s+(?:use|do|prefer|want|include)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bnever\s+(?:use|do|want|include)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bdon'?t\s+(?:ever\s+)?(?:use|do|want|include|show)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bplease\s+(?:always|never|don'?t)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:preferred|favorite|default)\s+(?:\w+\s+)?is\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
]

# Patterns for extracting factual statements about the user's environment
_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bi\s+(?:use|am using|work with|develop in|code in)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:project|app|system|codebase|stack|setup)\s+(?:uses|is|runs)\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bwe\s+(?:use|run|deploy|host)\s+(?:on\s+)?(.+?)(?:\.|,|$)", re.IGNORECASE),
    re.compile(r"\bour\s+(?:stack|infrastructure|database|backend|frontend)\s+is\s+(.+?)(?:\.|,|$)", re.IGNORECASE),
]


async def _auto_evaluate(
    components: MCPComponents,
    user_message: str,
    assistant_response: str,
    actor_result: ActorResult,
    new_facts: list[str] | None,
    failure_summary: str | None,
    quality_score: float | None = None,
) -> CriticEvaluation:
    """Run the Critic LLM if available, otherwise build an enhanced heuristic evaluation.

    When an LLM provider is configured, the Critic evaluates the turn across
    five quality dimensions and extracts semantic facts automatically.
    When no LLM is configured, an enhanced heuristic evaluation is built from
    the agent self-score, tool signals, user message patterns, and response quality.
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
            # If the agent provided a self-score, blend it with the LLM critic score
            if quality_score is not None:
                clamped = max(0.0, min(10.0, quality_score))
                # Weighted blend: 60% LLM critic, 40% agent self-assessment
                critic_eval.factual_accuracy = round(0.6 * critic_eval.factual_accuracy + 0.4 * clamped, 1)
                critic_eval.preference_adherence = round(0.6 * critic_eval.preference_adherence + 0.4 * clamped, 1)
                critic_eval = _recompute_evaluation(critic_eval)
            return critic_eval
        except Exception as exc:
            logger.warning("critic_evaluation_failed error=%s, falling back to heuristic", exc)

    # Enhanced heuristic: build evaluation from agent self-score, tool signals, and patterns
    return _heuristic_evaluation(
        actor_result, facts_list, failure_summary,
        quality_score=quality_score,
        user_message=user_message,
        assistant_response=assistant_response,
    )


def _recompute_evaluation(evaluation: CriticEvaluation) -> CriticEvaluation:
    """Re-run CriticEvaluation validators after post-provider score blending."""
    return CriticEvaluation.model_validate(evaluation.model_dump(mode="json", by_alias=True))


def _heuristic_evaluation(
    actor_result: ActorResult,
    facts: list[str],
    failure_summary: str | None,
    quality_score: float | None = None,
    user_message: str = "",
    assistant_response: str = "",
) -> CriticEvaluation:
    """Build an enhanced CriticEvaluation using multi-signal analysis.

    When no LLM Critic is available (the common case for MCP usage with
    Claude/Codex), this evaluator scores quality from:
    1. Agent self-score (quality_score) — highest-priority signal
    2. Tool success/failure rates
    3. Response quality heuristics (length ratio, structure, errors)
    4. Pattern-based preference and fact extraction from the user message
    """
    total_tools = len(actor_result.tool_calls)
    failed_tools = sum(1 for t in actor_result.tool_calls if not t.success)
    has_failures = failed_tools > 0 or failure_summary is not None

    # ── 1. Compute base scores ──────────────────────────────────────
    if quality_score is not None:
        # Agent self-assessment is the primary signal — it knows best
        base = max(0.0, min(10.0, quality_score))
    elif has_failures and total_tools > 0 and failed_tools == total_tools:
        base = 3.0  # All tools failed
    elif has_failures:
        base = 6.0  # Partial failure
    elif total_tools > 0:
        base = 9.0  # All tools succeeded
    else:
        base = 7.5  # No tools used, neutral baseline

    # ── 2. Dimension-specific adjustments ───────────────────────────
    tool_efficiency = base
    if total_tools > 0:
        success_rate = (total_tools - failed_tools) / total_tools
        tool_efficiency = max(1.0, min(10.0, round(success_rate * 10, 1)))

    # Response quality heuristics
    factual_accuracy = base
    response_len = len(assistant_response)
    prompt_len = max(len(user_message), 1)
    if response_len < 10 and prompt_len > 20:
        factual_accuracy = max(base - 2.0, 1.0)  # Suspiciously short response
    elif response_len > prompt_len * 0.3:
        factual_accuracy = min(base + 0.5, 10.0)  # Reasonable length

    # Hallucination risk: higher score = lower risk, better grounded
    hallucination_risk = base
    if total_tools > 0 and failed_tools == 0:
        hallucination_risk = min(base + 1.0, 10.0)  # Tool-grounded = lower risk
    if failure_summary:
        hallucination_risk = max(base - 1.5, 1.0)

    # Workflow quality
    workflow_quality = base
    if total_tools >= 2 and failed_tools == 0:
        workflow_quality = min(base + 1.0, 10.0)  # Multi-tool success

    preference_adherence = base  # Can't assess without memory context

    # ── 3. Extract preferences from user message ────────────────────
    semantic_facts: list[NewSemanticFact] = []

    for pattern in _PREFERENCE_PATTERNS:
        for match in pattern.finditer(user_message):
            extracted = match.group(1).strip()
            if len(extracted) > 5 and len(extracted) < 200:
                # Build the full preference statement for context
                full_match = match.group(0).strip().rstrip(".,")
                semantic_facts.append(
                    NewSemanticFact(
                        fact_type=SemanticFactType.PREFERENCE,
                        content=f"User preference: {full_match}",
                        confidence=0.90,
                        source="user_stated",
                    )
                )

    # ── 4. Extract factual statements about environment ─────────────
    for pattern in _FACT_PATTERNS:
        for match in pattern.finditer(user_message):
            extracted = match.group(1).strip()
            if len(extracted) > 3 and len(extracted) < 200:
                full_match = match.group(0).strip().rstrip(".,")
                semantic_facts.append(
                    NewSemanticFact(
                        fact_type=SemanticFactType.SYSTEM_RULE,
                        content=f"Environment fact: {full_match}",
                        confidence=0.80,
                        source="user_stated",
                    )
                )

    # ── 5. Include agent-supplied facts ─────────────────────────────
    for fact in facts:
        semantic_facts.append(
            NewSemanticFact(
                fact_type=SemanticFactType.INFERRED_FACT,
                content=fact,
                confidence=0.85,
                source="llm_inferred",
            )
        )

    # ── 6. Workflow save decision ───────────────────────────────────
    save_workflow = total_tools >= 2 and failed_tools == 0

    return CriticEvaluation(
        factual_accuracy=factual_accuracy,
        preference_adherence=preference_adherence,
        tool_efficiency=tool_efficiency,
        hallucination_risk=hallucination_risk,
        workflow_quality=workflow_quality,
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

        _COMPONENTS = MCPComponents(
            settings=settings,
            store=store,
            embedding_model=embedding_model,
            planner=planner,
            memory_service=memory_service,
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
