"""Retrieval orchestration primitives for Agentic Memory.

The primary MCP path is client-orchestrated: Codex, Claude Code, Cline, or
another MCP client decides which memory layers to query and when to perform
multi-hop retrieval. The heuristic planner remains as a backward-compatible
quick context builder and deterministic fallback.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from core.models import (
    EpisodeRecord,
    FailureEpisode,
    MemoryLayer,
    ProceduralWorkflow,
    RetrievedRecord,
    RetrievalPlan,
    SemanticHierarchyNode,
    SemanticMemoryRecord,
)
from model.embedding_model import EmbeddingModel
from store.base import MemoryStore


_DEFAULT_MEMORY_MINING_PROMPT = (
    "After answering, review the user's message, the assistant response, tool outcomes, and recent "
    "retrieved memory. Propose only durable memory that should help future turns. Infer implicit "
    "preferences from repeated behavior, corrections, accepted workflows, rejected formats, and "
    "stable project context. Do not save one-off task details, secrets, unsupported guesses, or "
    "facts already contradicted by stronger memory. Prefer concise atomic facts."
)


class AgenticRetrievalOrchestrator:
    """Expose memory-layer tools for an MCP client agent to orchestrate.

    This class intentionally does not call a hidden internal LLM. The LLM in the
    MCP client is the agentic orchestrator; Agentic Memory only retrieves and
    ranks the layers the client explicitly asks for.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_model: EmbeddingModel,
        memory_window_turns: int = 10,
        semantic_memory_ttl_days: int = 180,
        memory_mining_prompt: str | None = None,
    ) -> None:
        """Create a client-orchestrated retrieval toolkit."""
        self._store = store
        self._embedding_model = embedding_model
        self._memory_window_turns = memory_window_turns
        self._semantic_memory_ttl_days = semantic_memory_ttl_days
        self._memory_mining_prompt = memory_mining_prompt.strip() if memory_mining_prompt else ""

    def manifest(self) -> dict[str, Any]:
        """Describe the MCP-client orchestration contract and retrieval tools."""
        return {
            "orchestrator": "mcp_client_agent",
            "agentic_contract": {
                "who_reasons": "Codex, Claude Code, Cline, or the MCP client currently using this server.",
                "server_role": "Expose durable memory-layer retrieval tools and rankings.",
                "no_hidden_planner_llm": True,
                "multi_hop_supported": True,
            },
            "scoring": {
                "owner": "mcp_client_agent",
                "primary": "Codex, Claude Code, Cline, or the MCP client agent scores its completed turn.",
                "server_bootstrap": (
                    "If agent_evaluation_json is omitted, the server invents provisional scores only "
                    "to keep consolidation alive."
                ),
                "rescore": (
                    "When needs_agent_rescore is true, the MCP client should score the turn and call "
                    "rescore_episode with the returned episode_id."
                ),
                "client_quality_score": "Deprecated compatibility field; use agent_evaluation_json instead.",
                "returned_as": (
                    "consolidate_turn and rescore_episode return scoring_source as mcp_client_agent "
                    "or heuristic_provisional."
                ),
                "output_field": "agent_evaluation_json",
                "output_schema": {
                    "factual_accuracy": "0-10",
                    "preference_adherence": "0-10",
                    "tool_efficiency": "0-10",
                    "hallucination_risk": "0-10, higher means better grounded and lower risk",
                    "workflow_quality": "0-10",
                    "save_workflow": "boolean",
                    "failure_summary": "string or null",
                },
            },
            "recommended_flow": [
                "Read the user query and decide which memory layers are relevant.",
                "Call retrieve_memory_layer for one layer at a time, using refined queries when a result suggests a follow-up hop.",
                "Prefer semantic_hierarchy for broad context, semantic for exact facts, procedural for reusable workflows, episodic for prior examples, and failure for known hazards.",
                "Resolve conflicts by preferring explicit user-stated, high-confidence, pinned, and recent memories.",
                "Synthesize only grounded memory into the final answer.",
                "Mine durable preferences and facts, score the completed turn, then call consolidate_turn with semantic_facts_json and agent_evaluation_json.",
                "If consolidate_turn reports needs_agent_rescore, call rescore_episode with the same episode_id and the MCP client agent score.",
            ],
            "memory_mining": {
                "orchestrator": "mcp_client_agent",
                "prompt_name": "implicit_preference_mining",
                "customizable": True,
                "prompt": self._memory_mining_prompt or _DEFAULT_MEMORY_MINING_PROMPT,
                "fact_types": {
                    "preference": "User likes, dislikes, defaults, style/format choices, workflow choices, or behavioral patterns.",
                    "system_rule": "Stable project, environment, repository, policy, or tool-use constraints.",
                    "inferred_fact": "Other durable facts that are useful but are not preferences or rules.",
                },
                "source_guidance": {
                    "user_stated": "Use only when the user explicitly stated the fact or preference.",
                    "llm_inferred": "Use for implicit preference mining from behavior across turns.",
                    "tool_derived": "Use when a tool result established the fact.",
                },
                "output_field": "semantic_facts_json",
                "output_schema": [
                    {
                        "fact_type": "preference | inferred_fact | system_rule",
                        "content": "Atomic durable memory fact.",
                        "confidence": "0.0-1.0 confidence score.",
                        "source": "user_stated | llm_inferred | tool_derived",
                    }
                ],
                "example": [
                    {
                        "fact_type": "preference",
                        "content": "User prefers concise implementation summaries with verification commands.",
                        "confidence": 0.78,
                        "source": "llm_inferred",
                    }
                ],
            },
            "layers": {
                MemoryLayer.CONVERSATIONAL.value: "Recent session messages and rolling summaries. Use for active-thread continuity.",
                MemoryLayer.SEMANTIC.value: "Flat durable facts, preferences, and system rules. Use for precise known facts.",
                MemoryLayer.SEMANTIC_HIERARCHY.value: (
                    "Aggregated semantic facets, summaries, and Q&A nodes. Use for broad preference or project-context hops."
                ),
                MemoryLayer.EPISODIC.value: "Similar completed turns with prompt, outcome, tools, and response. Use for prior examples.",
                MemoryLayer.PROCEDURAL.value: "Reusable successful tool workflows. Use for multi-step implementation or analysis tasks.",
                MemoryLayer.FAILURE.value: "Similar past tool failures. Use for cautions and known bad approaches.",
            },
            "tools": [
                {
                    "name": "retrieve_memory_layer",
                    "arguments": {
                        "query": "Natural language retrieval query chosen by the MCP client agent.",
                        "layer": "conversational | semantic | semantic_hierarchy | episodic | procedural | failure",
                        "session_id": "Required for conversational lookup; optional otherwise.",
                        "top_k": "Maximum records to return.",
                    },
                },
                {
                    "name": "consolidate_turn",
                    "arguments": {
                        "session_id": "Current conversation session identifier.",
                        "user_message": "The user's original message.",
                        "assistant_response": "The MCP client agent's final response.",
                        "tool_calls_json": "Optional JSON audit of tool calls made by the MCP client agent.",
                        "new_facts": "Backward-compatible untyped durable fact strings.",
                        "semantic_facts_json": "Typed durable facts mined by the MCP client agent using the memory_mining prompt.",
                        "agent_evaluation_json": "Typed 0-10 scoring produced by the MCP client agent.",
                        "failure_summary": "Optional summary of failures worth remembering.",
                        "quality_score": "Deprecated compatibility field; use agent_evaluation_json instead.",
                        "reasoning_summary": "Optional brief approach summary for episodic recall.",
                    },
                },
                {
                    "name": "rescore_episode",
                    "arguments": {
                        "episode_id": "Episode id returned by consolidate_turn.",
                        "agent_evaluation_json": "Typed 0-10 scoring produced by the MCP client agent.",
                        "session_id": "Optional session id for retrieval-feedback tuning.",
                    },
                },
            ],
            "fallback": {
                "name": "get_session_context",
                "role": "Backward-compatible convenience context builder.",
                "uses_heuristics": True,
                "agentic": False,
            },
        }

    async def retrieve_layer(
        self,
        query: str,
        layer: MemoryLayer,
        session_id: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve the memory layer explicitly selected by the MCP client."""
        limit = max(1, min(int(top_k), 25))
        if layer == MemoryLayer.CONVERSATIONAL:
            if not session_id:
                raise ValueError("session_id is required for conversational retrieval")
            summaries = await self._store.recent_summaries(session_id, min(2, limit))
            records = await self._store.recent_conversation(session_id, min(limit, self._memory_window_turns * 2))
            return {
                "orchestrator": "mcp_client_agent",
                "layer": layer.value,
                "query": query,
                "session_id": session_id,
                "summaries": [summary.model_dump(mode="json") for summary in summaries],
                "records": [record.model_dump(mode="json") for record in records],
            }

        embedding = await self._embedding_model.embed(query)
        if layer == MemoryLayer.SEMANTIC:
            semantic_records = _rank_semantic_records(
                await self._store.search_semantic(
                    embedding,
                    limit,
                    0.0,
                    0.0,
                    last_confirmed_after=self._semantic_cutoff(),
                )
            )[:limit]
            serialized_records = [record.model_dump(mode="json") for record in semantic_records]
        elif layer == MemoryLayer.SEMANTIC_HIERARCHY:
            hierarchy_records = _rank_semantic_hierarchy_records(
                await self._store.search_semantic_hierarchy(embedding, limit, 0.0)
            )[:limit]
            serialized_records = [record.model_dump(mode="json") for record in hierarchy_records]
        elif layer == MemoryLayer.EPISODIC:
            episodic_records = _rank_episodes(await self._store.search_episodes(embedding, limit, 0.0))[:limit]
            serialized_records = [record.model_dump(mode="json") for record in episodic_records]
        elif layer == MemoryLayer.PROCEDURAL:
            trigger_matches = await self._store.match_procedural_triggers(query, limit)
            vector_matches = await self._store.search_procedural(embedding, limit, 0.0)
            procedural_records = _rank_workflows(_dedupe_workflows(trigger_matches + vector_matches))[:limit]
            serialized_records = [record.model_dump(mode="json") for record in procedural_records]
        elif layer == MemoryLayer.FAILURE:
            failure_records = _rank_failures(await self._store.search_failures(embedding, limit, 0.0))[:limit]
            serialized_records = [record.model_dump(mode="json") for record in failure_records]
        else:
            raise ValueError(f"Unsupported memory layer: {layer.value}")

        return {
            "orchestrator": "mcp_client_agent",
            "layer": layer.value,
            "query": query,
            "records": serialized_records,
        }

    def _semantic_cutoff(self) -> datetime | None:
        """Return semantic TTL cutoff for retrieval, or None when disabled."""
        if self._semantic_memory_ttl_days <= 0:
            return None
        return datetime.now(timezone.utc) - timedelta(days=self._semantic_memory_ttl_days)


class HeuristicRetrievalPlanner:
    """Fallback context builder using explicit heuristics before actor context assembly."""

    _preference_keywords = (
        "always",
        "never",
        "i prefer",
        "prefer",
        "please don't",
        "do not",
        "don't",
        "style",
        "format",
        "tone",
    )
    _complexity_keywords = (
        "build",
        "analyze and then",
        "compare and summarize",
        "compare",
        "summarize",
        "workflow",
        "steps",
        "process",
        "tool",
        "plan",
        "implement",
    )
    _history_keywords = ("earlier", "before", "previous", "last time", "history", "what happened")

    def __init__(
        self,
        store: MemoryStore,
        embedding_model: EmbeddingModel,
        memory_window_turns: int = 10,
        failure_similarity_threshold: float = 0.80,
        semantic_memory_ttl_days: int = 180,
    ) -> None:
        """Create a planner with access to explicit stores and embeddings."""
        self._store = store
        self._embedding_model = embedding_model
        self._memory_window_turns = memory_window_turns
        self._failure_similarity_threshold = failure_similarity_threshold
        self._semantic_memory_ttl_days = semantic_memory_ttl_days
        self._session_weights: dict[str, dict[str, float]] = {}

    async def plan(self, prompt: str, session_id: str) -> RetrievalPlan:
        """Build a retrieval plan and populate it with raw retrieved records."""
        normalized = _normalize(prompt)
        tokens = _tokens(prompt)
        weights = await self._weights_for(session_id)
        semantic_density = _keyword_density(tokens, self._preference_keywords)
        procedural_density = _keyword_density(tokens, self._complexity_keywords)

        query_semantic = (
            _contains_any(normalized, self._preference_keywords)
            or semantic_density * weights["semantic"] >= 0.04
        )
        query_procedural = (
            _contains_any(normalized, self._complexity_keywords)
            or procedural_density * weights["procedural"] >= 0.05
        )

        embedding = await self._embedding_model.embed(prompt)

        conversational = await self._store.recent_conversation(session_id, self._memory_window_turns * 2)
        summaries = await self._store.recent_summaries(session_id, 2)

        trigger_workflows = await self._store.match_procedural_triggers(prompt, 3)
        if trigger_workflows:
            query_procedural = True

        semantic_records = []
        semantic_hierarchy_records = []
        if query_semantic:
            semantic_records = await self._store.search_semantic(
                embedding,
                10,
                0.35,
                0.6,
                last_confirmed_after=self._semantic_cutoff(),
            )
            semantic_records = _rank_semantic_records(semantic_records)[:5]
            semantic_hierarchy_records = _rank_semantic_hierarchy_records(
                await self._store.search_semantic_hierarchy(embedding, 8, 0.30)
            )[:4]

        procedural_workflows = trigger_workflows
        if query_procedural:
            vector_workflows = await self._store.search_procedural(embedding, 6, 0.40)
            procedural_workflows = _rank_workflows(_dedupe_workflows(trigger_workflows + vector_workflows))[:3]

        base_episode_threshold = 0.45 if _contains_any(normalized, self._history_keywords) else 0.55
        episode_threshold = max(0.25, min(0.85, base_episode_threshold / weights["episodic"]))
        episodic_records = _rank_episodes(
            await self._store.search_episodes(embedding, 6, episode_threshold)
        )[:3]
        failure_matches = await self._store.search_failures(
            embedding, 6, self._failure_similarity_threshold
        )
        failure_matches = _rank_failures(failure_matches)[:3]
        query_episodic = bool(episodic_records or failure_matches)

        rationale = {
            MemoryLayer.CONVERSATIONAL.value: "Always queried for the active session sliding window.",
            MemoryLayer.SEMANTIC.value: _semantic_rationale(query_semantic, semantic_density),
            MemoryLayer.PROCEDURAL.value: _procedural_rationale(query_procedural, procedural_density, bool(trigger_workflows)),
            MemoryLayer.EPISODIC.value: _episodic_rationale(query_episodic, failure_matches, episodic_records),
        }

        retrieved = _build_retrieved_records(
            episodic_records=episodic_records,
            failure_matches=failure_matches,
            semantic_count=len(semantic_records),
            semantic_hierarchy_count=len(semantic_hierarchy_records),
            workflow_count=len(procedural_workflows),
        )

        return RetrievalPlan(
            session_id=session_id,
            prompt=prompt,
            query_conversational=True,
            query_episodic=query_episodic,
            query_semantic=query_semantic,
            query_procedural=query_procedural,
            min_confidence=0.6,
            episodic_similarity_threshold=episode_threshold,
            rationale=rationale,
            conversational_records=conversational,
            conversation_summaries=summaries,
            episodic_records=episodic_records,
            semantic_records=semantic_records,
            semantic_hierarchy_records=semantic_hierarchy_records,
            procedural_workflows=procedural_workflows,
            failure_matches=failure_matches,
            retrieved_records=retrieved,
        )

    async def record_feedback(self, session_id: str, retrieval_plan: RetrievalPlan, critic_passed: bool) -> None:
        """Update per-session retrieval weights from observed turn quality."""
        weights = await self._weights_for(session_id)
        retrieved = {
            "semantic": bool(retrieval_plan.semantic_records),
            "procedural": bool(retrieval_plan.procedural_workflows),
            "episodic": bool(retrieval_plan.episodic_records or retrieval_plan.failure_matches),
        }
        for layer, was_retrieved in retrieved.items():
            if not was_retrieved:
                continue
            target = 1.15 if critic_passed else 0.85
            weights[layer] = max(0.5, min(1.5, (0.85 * weights[layer]) + (0.15 * target)))
        await self._store.save_retrieval_weights(session_id, weights)

    async def _weights_for(self, session_id: str) -> dict[str, float]:
        """Return mutable EMA weights for one session."""
        if session_id not in self._session_weights:
            persisted = await self._store.get_retrieval_weights(session_id)
            self._session_weights[session_id] = _normalize_weights(persisted)
        return self._session_weights[session_id]

    def _semantic_cutoff(self) -> datetime | None:
        """Return semantic TTL cutoff for retrieval, or None when disabled."""
        if self._semantic_memory_ttl_days <= 0:
            return None
        return datetime.now(timezone.utc) - timedelta(days=self._semantic_memory_ttl_days)


def _normalize(text: str) -> str:
    """Lowercase and trim text for planner heuristics."""
    return text.lower().strip()


def _tokens(text: str) -> list[str]:
    """Tokenize text for keyword-density scoring."""
    return [token for token in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split() if token]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any heuristic phrase appears in normalized text."""
    return any(needle in text for needle in needles)


def _keyword_density(tokens: list[str], phrases: tuple[str, ...]) -> float:
    """Compute prompt keyword density for a heuristic phrase set."""
    if not tokens:
        return 0.0
    phrase_tokens = {part for phrase in phrases for part in phrase.split()}
    hits = sum(1 for token in tokens if token in phrase_tokens)
    return hits / len(tokens)


def _semantic_rationale(query_semantic: bool, density: float) -> str:
    """Explain why semantic memory was queried or skipped."""
    if query_semantic:
        return f"Preference-signaling words or keyword density {density:.2f} triggered semantic lookup."
    return f"No preference trigger was detected and keyword density was only {density:.2f}."


def _procedural_rationale(query_procedural: bool, density: float, trigger_match: bool) -> str:
    """Explain why procedural memory was queried or skipped."""
    if trigger_match:
        return "A known workflow trigger phrase matched the prompt."
    if query_procedural:
        return f"Multi-step action wording or keyword density {density:.2f} triggered workflow lookup."
    return f"No multi-step workflow trigger was detected and keyword density was {density:.2f}."


def _episodic_rationale(
    query_episodic: bool, failure_matches: list[FailureEpisode], episodic_records: list[EpisodeRecord]
) -> str:
    """Explain why episodic memory was queried or skipped."""
    if failure_matches:
        best = max((match.score or 0.0) for match in failure_matches)
        return f"Past failure similarity reached {best:.2f}, so failure avoidance context is needed."
    if episodic_records:
        best = max((record.score or 0.0) for record in episodic_records)
        return f"Similar past episode similarity reached {best:.2f}."
    if query_episodic:
        return "Episodic lookup was requested by historical wording."
    return "No similar episode or failure exceeded the retrieval threshold."


def _build_retrieved_records(
    episodic_records: list[EpisodeRecord],
    failure_matches: list[FailureEpisode],
    semantic_count: int,
    semantic_hierarchy_count: int,
    workflow_count: int,
) -> list[RetrievedRecord]:
    """Build auditable retrieval pointers without exposing raw records to API users."""
    records: list[RetrievedRecord] = []
    for episode in episodic_records:
        records.append(
            RetrievedRecord(
                layer=MemoryLayer.EPISODIC,
                record_id=episode.episode_id,
                score=episode.score or 0.0,
                rationale="Similar prior prompt.",
            )
        )
    for failure in failure_matches:
        records.append(
            RetrievedRecord(
                layer=MemoryLayer.EPISODIC,
                record_id=failure.failure_id,
                score=failure.score or 0.0,
                rationale="Similar prior failure.",
            )
        )
    if semantic_count:
        records.append(
            RetrievedRecord(
                layer=MemoryLayer.SEMANTIC,
                record_id="semantic-batch",
                score=1.0,
                rationale=f"{semantic_count} semantic facts selected.",
            )
        )
    if semantic_hierarchy_count:
        records.append(
            RetrievedRecord(
                layer=MemoryLayer.SEMANTIC_HIERARCHY,
                record_id="semantic-hierarchy-batch",
                score=1.0,
                rationale=f"{semantic_hierarchy_count} semantic hierarchy nodes selected.",
            )
        )
    if workflow_count:
        records.append(
            RetrievedRecord(
                layer=MemoryLayer.PROCEDURAL,
                record_id="workflow-batch",
                score=1.0,
                rationale=f"{workflow_count} workflows selected.",
            )
        )
    return records


def _dedupe_workflows(workflows: list[ProceduralWorkflow]) -> list[ProceduralWorkflow]:
    """Deduplicate procedural workflows by workflow id while preserving rank order."""
    seen: set[str] = set()
    deduped: list[ProceduralWorkflow] = []
    for workflow in workflows:
        if workflow.workflow_id in seen:
            continue
        seen.add(workflow.workflow_id)
        deduped.append(workflow)
    return deduped


def _rank_semantic_records(records: list[SemanticMemoryRecord]) -> list[SemanticMemoryRecord]:
    """Re-rank semantic facts by similarity, confidence, freshness, and pinning."""
    for record in records:
        similarity = record.score or 0.0
        confidence = record.confidence_score
        recency = _recency_score(record.last_confirmed_at, half_life_days=45)
        pinned_bonus = 1.0 if getattr(record, "pinned", False) else 0.0
        record.score = _clip_score((0.55 * similarity) + (0.30 * confidence) + (0.10 * recency) + (0.05 * pinned_bonus))
    records.sort(key=lambda item: item.score or 0.0, reverse=True)
    return records


def _rank_semantic_hierarchy_records(records: list[SemanticHierarchyNode]) -> list[SemanticHierarchyNode]:
    """Re-rank semantic aggregates by similarity, confidence, abstraction level, and freshness."""
    level_score = {"root": 0.25, "facet": 0.45, "summary": 0.90, "qa": 1.0}
    for record in records:
        similarity = record.score or 0.0
        confidence = record.confidence_score
        recency = _recency_score(record.updated_at, half_life_days=60)
        abstraction = level_score.get(record.node_type.value, 0.5)
        record.score = _clip_score(
            (0.50 * similarity) + (0.25 * confidence) + (0.15 * abstraction) + (0.10 * recency)
        )
    records.sort(key=lambda item: item.score or 0.0, reverse=True)
    return records


def _rank_episodes(records: list[EpisodeRecord]) -> list[EpisodeRecord]:
    """Re-rank episodes by similarity, outcome quality, recency, and tool success."""
    outcome_score = {"success": 1.0, "partial": 0.55, "failure": 0.20}
    for record in records:
        similarity = record.score or 0.0
        outcome = outcome_score.get(record.outcome.value, 0.5)
        recency = _recency_score(record.timestamp, half_life_days=30)
        if record.tool_sequence:
            tool_success = sum(1 for tool in record.tool_sequence if tool.success) / len(record.tool_sequence)
        else:
            tool_success = 0.7
        record.score = _clip_score((0.60 * similarity) + (0.20 * outcome) + (0.15 * recency) + (0.05 * tool_success))
    records.sort(key=lambda item: item.score or 0.0, reverse=True)
    return records


def _rank_failures(records: list[FailureEpisode]) -> list[FailureEpisode]:
    """Re-rank failures by similarity and recency so fresh hazards surface quickly."""
    for record in records:
        similarity = record.score or 0.0
        recency = _recency_score(record.timestamp, half_life_days=21)
        record.score = _clip_score((0.70 * similarity) + (0.30 * recency))
    records.sort(key=lambda item: item.score or 0.0, reverse=True)
    return records


def _rank_workflows(workflows: list[ProceduralWorkflow]) -> list[ProceduralWorkflow]:
    """Re-rank workflows by similarity, success maturity, canonical status, and recency."""
    for workflow in workflows:
        similarity = workflow.score or 0.0
        maturity = min(1.0, workflow.success_count / 5)
        status = 1.0 if workflow.status.value == "canonical" else 0.45
        recency = _recency_score(workflow.updated_at, half_life_days=60)
        trigger_score = 1.0 if similarity == 0.0 else similarity
        workflow.score = _clip_score((0.35 * trigger_score) + (0.30 * maturity) + (0.20 * status) + (0.15 * recency))
    workflows.sort(key=lambda item: item.score or 0.0, reverse=True)
    return workflows


def _recency_score(value: datetime, half_life_days: int) -> float:
    """Score recency as a smooth 0..1 decay."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - value.astimezone(timezone.utc)).total_seconds() / 86400)
    return 1.0 / (1.0 + (age_days / max(1, half_life_days)))


def _clip_score(value: float) -> float:
    """Clip retrieval score into the Pydantic-safe range."""
    return max(0.0, min(1.0, round(value, 4)))


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Return all planner weights with safe defaults and bounds."""
    source = weights or {}
    return {
        "semantic": _clip_weight(source.get("semantic", 1.0)),
        "procedural": _clip_weight(source.get("procedural", 1.0)),
        "episodic": _clip_weight(source.get("episodic", 1.0)),
    }


def _clip_weight(value: object) -> float:
    """Clip one adaptive planner weight."""
    parsed = float(value) if isinstance(value, (int, float, str)) else 1.0
    return max(0.5, min(1.5, parsed))
