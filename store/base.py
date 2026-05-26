"""Abstract asynchronous memory-store protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.models import (
    ConversationalSummary,
    ConversationalTurnRecord,
    EpisodeRecord,
    FailureEpisode,
    MemoryLayer,
    ProceduralWorkflow,
    SemanticMemoryRecord,
)


class MemoryStore(Protocol):
    """Define explicit async persistence operations for every memory layer."""

    async def next_turn_index(self, scope_hash: str, session_id: str) -> int:
        """Return the next turn index for the scoped session."""
        ...

    async def append_conversation_message(self, record: ConversationalTurnRecord) -> ConversationalTurnRecord:
        """Append one user or assistant message to conversational memory."""
        ...

    async def recent_conversation(self, scope_hash: str, session_id: str, limit: int) -> list[ConversationalTurnRecord]:
        """Return recent conversational messages for the active session."""
        ...

    async def get_conversation_turns(
        self, scope_hash: str, session_id: str, limit: int
    ) -> list[ConversationalTurnRecord]:
        """Return the most recent conversation messages for a session in chronological order."""
        ...

    async def conversation_turn(
        self, scope_hash: str, session_id: str, turn_index: int
    ) -> list[ConversationalTurnRecord]:
        """Return all role records for one session turn index."""
        ...

    async def clear_conversation(self, scope_hash: str, session_id: str) -> int:
        """Delete conversational messages and summaries for one scoped session only."""
        ...

    async def conversation_before_turn(
        self, scope_hash: str, session_id: str, before_turn_index: int
    ) -> list[ConversationalTurnRecord]:
        """Return conversation messages old enough to be summarized and pruned."""
        ...

    async def delete_conversation_before_turn(self, scope_hash: str, session_id: str, before_turn_index: int) -> int:
        """Delete raw conversation messages that have been safely summarized."""
        ...

    async def save_conversation_summary(self, summary: ConversationalSummary) -> ConversationalSummary:
        """Persist a rolling conversational summary."""
        ...

    async def recent_summaries(self, scope_hash: str, session_id: str, limit: int) -> list[ConversationalSummary]:
        """Return recent summaries for context assembly."""
        ...

    async def save_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Persist an append-only episodic memory record."""
        ...

    async def search_episodes(
        self, scope_hash: str, embedding: list[float], limit: int, threshold: float
    ) -> list[EpisodeRecord]:
        """Search episodic memory using cosine similarity."""
        ...

    async def save_failure_episode(self, failure: FailureEpisode) -> FailureEpisode:
        """Persist a detailed failure episode for future avoidance."""
        ...

    async def search_failures(
        self, scope_hash: str, embedding: list[float], limit: int, threshold: float
    ) -> list[FailureEpisode]:
        """Search past failures using cosine similarity."""
        ...

    async def insert_semantic(self, record: SemanticMemoryRecord) -> SemanticMemoryRecord:
        """Insert a new deduplicated semantic memory fact."""
        ...

    async def reinforce_semantic(
        self, fact_id: str, confidence_score: float | None = None, source: str | None = None
    ) -> None:
        """Update reinforcement metadata for a duplicate semantic fact."""
        ...

    async def replace_semantic(self, fact_id: str, record: SemanticMemoryRecord) -> SemanticMemoryRecord:
        """Replace an existing semantic fact after conflict resolution."""
        ...

    async def search_semantic(
        self,
        scope_hash: str,
        embedding: list[float],
        limit: int,
        threshold: float,
        min_confidence: float,
        last_confirmed_after: datetime | None = None,
    ) -> list[SemanticMemoryRecord]:
        """Search semantic memory using cosine similarity and confidence filtering."""
        ...

    async def search_procedural(
        self, scope_hash: str, embedding: list[float], limit: int, threshold: float
    ) -> list[ProceduralWorkflow]:
        """Search procedural workflows using vector similarity."""
        ...

    async def match_procedural_triggers(
        self, scope_hash: str, prompt: str, limit: int
    ) -> list[ProceduralWorkflow]:
        """Find workflows whose trigger phrases appear in the current prompt."""
        ...

    async def upsert_procedural_workflow(self, workflow: ProceduralWorkflow) -> ProceduralWorkflow:
        """Insert or reinforce a procedural workflow by signature."""
        ...

    async def inspect_layer(self, scope_hash: str, layer: MemoryLayer, limit: int, offset: int) -> list[dict[str, object]]:
        """Return raw scoped records for a single memory layer."""
        ...

    async def count_layer(self, scope_hash: str, layer: MemoryLayer) -> int:
        """Return the number of scoped records in one memory layer."""
        ...
