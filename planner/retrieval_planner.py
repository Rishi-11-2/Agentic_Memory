"""Heuristic retrieval planner ported from Java and extended with density scoring."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from core.access_scope import AccessScope
from core.models import (
    EpisodeRecord,
    FailureEpisode,
    MemoryLayer,
    ProceduralWorkflow,
    RetrievedRecord,
    RetrievalPlan,
)
from model.embedding_model import EmbeddingModel
from store.base import MemoryStore


class HeuristicRetrievalPlanner:
    """Choose memory layers with explicit heuristics before assembling actor context."""

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

    async def plan(self, prompt: str, scope: AccessScope, session_id: str) -> RetrievalPlan:
        """Build a retrieval plan and populate it with raw retrieved records."""
        normalized = _normalize(prompt)
        tokens = _tokens(prompt)
        weights = self._weights_for(session_id)
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
        scope_hash = scope.scope_hash

        conversational = await self._store.recent_conversation(scope_hash, session_id, self._memory_window_turns * 2)
        summaries = await self._store.recent_summaries(scope_hash, session_id, 2)

        trigger_workflows = await self._store.match_procedural_triggers(scope_hash, prompt, 3)
        if trigger_workflows:
            query_procedural = True

        semantic_records = []
        if query_semantic:
            semantic_records = await self._store.search_semantic(
                scope_hash,
                embedding,
                5,
                0.35,
                0.6,
                last_confirmed_after=self._semantic_cutoff(),
            )

        procedural_workflows = trigger_workflows
        if query_procedural:
            vector_workflows = await self._store.search_procedural(scope_hash, embedding, 3, 0.40)
            procedural_workflows = _dedupe_workflows(trigger_workflows + vector_workflows)

        base_episode_threshold = 0.45 if _contains_any(normalized, self._history_keywords) else 0.55
        episode_threshold = max(0.25, min(0.85, base_episode_threshold / weights["episodic"]))
        episodic_records = await self._store.search_episodes(scope_hash, embedding, 3, episode_threshold)
        failure_matches = await self._store.search_failures(
            scope_hash, embedding, 3, self._failure_similarity_threshold
        )
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
            workflow_count=len(procedural_workflows),
        )

        return RetrievalPlan(
            scope_hash=scope_hash,
            session_id=session_id,
            prompt=prompt,
            query_conversational=True,
            query_episodic=query_episodic,
            query_semantic=query_semantic,
            query_procedural=query_procedural,
            min_confidence=0.6,
            episodic_similarity_threshold=self._failure_similarity_threshold,
            rationale=rationale,
            conversational_records=conversational,
            conversation_summaries=summaries,
            episodic_records=episodic_records,
            semantic_records=semantic_records,
            procedural_workflows=procedural_workflows,
            failure_matches=failure_matches,
            retrieved_records=retrieved,
        )

    def record_feedback(self, session_id: str, retrieval_plan: RetrievalPlan, critic_passed: bool) -> None:
        """Update per-session retrieval weights from observed turn quality."""
        weights = self._weights_for(session_id)
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

    def _weights_for(self, session_id: str) -> dict[str, float]:
        """Return mutable EMA weights for one session."""
        if session_id not in self._session_weights:
            self._session_weights[session_id] = {"semantic": 1.0, "procedural": 1.0, "episodic": 1.0}
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
