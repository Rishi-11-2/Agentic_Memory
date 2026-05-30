"""Local PostgreSQL + pgvector implementation of the memory-store protocol.

Connects directly to a self-hosted PostgreSQL instance with the pgvector
extension using asyncpg. Reuses schema.sql and server-side cosine search.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast

import asyncpg  # type: ignore[import-untyped]

from core.models import (
    ConversationalSummary,
    ConversationalTurnRecord,
    ConversationRole,
    EpisodeOutcome,
    EpisodeRecord,
    FailureEpisode,
    MemoryLayer,
    ProceduralToolStep,
    ProceduralWorkflow,
    SemanticFactType,
    SemanticMemoryRecord,
    ToolInvocation,
    WorkflowStatus,
    utc_now,
)
from store.base import MemoryStore


class PostgresMemoryStore(MemoryStore):
    """Persist all four memory layers in a local PostgreSQL database with pgvector.

    Uses the same tables, indexes, and RPC functions defined in schema.sql.
    Server-side cosine similarity via the pgvector ``<=>`` operator provides
    production-grade vector search without external memory services.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize the store with an asyncpg connection pool."""
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str, min_size: int = 2, max_size: int = 10) -> "PostgresMemoryStore":
        """Create a connection pool and validate pgvector availability."""
        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        async with pool.acquire() as conn:
            pgvector_installed = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            if not pgvector_installed:
                raise RuntimeError(
                    "pgvector extension is not installed. Run: CREATE EXTENSION IF NOT EXISTS vector;"
                )
            await _migrate_schema(conn)
        return cls(pool)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()

    # ── Conversational ──────────────────────────────────────────────

    async def next_turn_index(self, session_id: str) -> int:
        """Return the next turn index for the session."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(turn_index) AS max_idx FROM am_conversation_turns "
                "WHERE session_id = $1",
                session_id,
            )
        if row is None or row["max_idx"] is None:
            return 0
        return int(row["max_idx"]) + 1

    async def append_conversation_message(self, record: ConversationalTurnRecord) -> ConversationalTurnRecord:
        """Append one user or assistant message to conversational memory."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO am_conversation_turns (id, session_id, turn_index, role, content, token_count, timestamp) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                record.id, record.session_id, record.turn_index,
                record.role.value, record.content, record.token_count, record.timestamp,
            )
        return record

    async def recent_conversation(self, session_id: str, limit: int) -> list[ConversationalTurnRecord]:
        """Return recent conversational messages for the active session."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_conversation_turns WHERE session_id = $1 "
                "ORDER BY turn_index DESC, timestamp DESC LIMIT $2",
                session_id, limit,
            )
        return list(reversed([self._conv_from_row(row) for row in rows]))

    async def get_conversation_turns(
        self, session_id: str, limit: int
    ) -> list[ConversationalTurnRecord]:
        """Return recent conversational messages in chronological order."""
        return await self.recent_conversation(session_id, limit)

    async def conversation_turn(
        self, session_id: str, turn_index: int
    ) -> list[ConversationalTurnRecord]:
        """Return all role records for one session turn index."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_conversation_turns WHERE session_id = $1 AND turn_index = $2 "
                "ORDER BY timestamp",
                session_id,
                turn_index,
            )
        return [self._conv_from_row(row) for row in rows]

    async def clear_conversation(self, session_id: str) -> int:
        """Delete raw conversation and summaries for one session."""
        async with self._pool.acquire() as conn:
            turns = await conn.execute(
                "DELETE FROM am_conversation_turns WHERE session_id = $1",
                session_id,
            )
            summaries = await conn.execute(
                "DELETE FROM am_conversation_summaries WHERE session_id = $1",
                session_id,
            )
        return _row_count(turns) + _row_count(summaries)

    async def conversation_before_turn(
        self, session_id: str, before_turn_index: int
    ) -> list[ConversationalTurnRecord]:
        """Return conversation messages old enough to be summarized and pruned."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_conversation_turns WHERE session_id = $1 AND turn_index < $2 "
                "ORDER BY turn_index, timestamp",
                session_id, before_turn_index,
            )
        return [self._conv_from_row(row) for row in rows]

    async def delete_conversation_before_turn(self, session_id: str, before_turn_index: int) -> int:
        """Delete raw conversation messages that have been safely summarized."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM am_conversation_turns WHERE session_id = $1 AND turn_index < $2",
                session_id, before_turn_index,
            )
        return int(result.split()[-1]) if result else 0

    async def save_conversation_summary(self, summary: ConversationalSummary) -> ConversationalSummary:
        """Persist a rolling conversational summary."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO am_conversation_summaries (summary_id, session_id, start_turn_index, end_turn_index, summary, token_count, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                summary.summary_id, summary.session_id,
                summary.start_turn_index, summary.end_turn_index,
                summary.summary, summary.token_count, summary.created_at,
            )
        return summary

    async def recent_summaries(self, session_id: str, limit: int) -> list[ConversationalSummary]:
        """Return recent summaries for context assembly."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_conversation_summaries WHERE session_id = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                session_id, limit,
            )
        return [self._summary_from_row(row) for row in rows]

    # ── Episodic ────────────────────────────────────────────────────

    async def save_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Persist an append-only episodic memory record."""
        tool_seq = json.dumps([tool.model_dump(mode="json") for tool in episode.tool_sequence])
        embedding = episode.prompt_embedding if episode.prompt_embedding else None
        vec_col, vec_val = _embedding_columns(embedding)

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO am_episodic_memory (episode_id, prompt_text, reasoning_summary, {vec_col}, tool_sequence, "
                "final_response, outcome, error_trace, latency_ms, timestamp) "
                f"VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6, $7, $8, $9, $10)",
                episode.episode_id, episode.prompt_text,
                episode.reasoning_summary, _vec_literal(vec_val), tool_seq,
                episode.final_response, episode.outcome.value, episode.error_trace,
                episode.latency_ms, episode.timestamp,
            )
        return episode

    async def search_episodes(
        self, embedding: list[float], limit: int, threshold: float
    ) -> list[EpisodeRecord]:
        """Search episodic memory using server-side pgvector cosine similarity."""
        if len(embedding) == 384:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT *, 1 - (prompt_embedding <=> $1::vector) AS similarity "
                    "FROM am_episodic_memory WHERE prompt_embedding IS NOT NULL "
                    "AND 1 - (prompt_embedding <=> $1::vector) >= $2 "
                    "ORDER BY prompt_embedding <=> $1::vector LIMIT $3",
                    _vec_literal(embedding), threshold, limit,
                )
            return [self._episode_from_row(row) for row in rows]
        # Fallback: hash embeddings use client-side ranking
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_episodic_memory LIMIT 200"
            )
        return self._rank_episodes(rows, embedding, threshold, limit)

    # ── Failure ─────────────────────────────────────────────────────

    async def save_failure_episode(self, failure: FailureEpisode) -> FailureEpisode:
        """Persist a detailed failure episode for future avoidance."""
        embedding = failure.prompt_embedding if failure.prompt_embedding else None
        vec_col, vec_val = _embedding_columns(embedding)
        tool_input = json.dumps(failure.tool_input)

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO am_failure_episodes (failure_id, episode_id, prompt_text, {vec_col}, "
                "tool_name, tool_input, exception_message, error_trace, timestamp) "
                f"VALUES ($1, $2, $3, $4::vector, $5, $6::jsonb, $7, $8, $9)",
                failure.failure_id, failure.episode_id, failure.prompt_text,
                _vec_literal(vec_val), failure.tool_name, tool_input,
                failure.exception_message, failure.error_trace, failure.timestamp,
            )
        return failure

    async def search_failures(
        self, embedding: list[float], limit: int, threshold: float
    ) -> list[FailureEpisode]:
        """Search past failures using server-side pgvector cosine similarity."""
        if len(embedding) == 384:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT *, 1 - (prompt_embedding <=> $1::vector) AS similarity "
                    "FROM am_failure_episodes WHERE prompt_embedding IS NOT NULL "
                    "AND 1 - (prompt_embedding <=> $1::vector) >= $2 "
                    "ORDER BY prompt_embedding <=> $1::vector LIMIT $3",
                    _vec_literal(embedding), threshold, limit,
                )
            return [self._failure_from_row(row) for row in rows]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_failure_episodes LIMIT 200"
            )
        return self._rank_failures(rows, embedding, threshold, limit)

    # ── Semantic ────────────────────────────────────────────────────

    async def insert_semantic(self, record: SemanticMemoryRecord) -> SemanticMemoryRecord:
        """Insert a new deduplicated semantic memory fact."""
        embedding = record.embedding if record.embedding else None
        vec_col, vec_val = _semantic_embedding_columns(embedding)

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO am_semantic_memory (fact_id, fact_type, content, {vec_col}, "
                "confidence_score, source, source_episode_id, created_at, last_reinforced_at, last_confirmed_at) "
                f"VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10)",
                record.fact_id, record.fact_type.value, record.content,
                _vec_literal(vec_val), record.confidence_score, record.source, record.source_episode_id,
                record.created_at, record.last_reinforced_at, record.last_confirmed_at,
            )
        return record

    async def reinforce_semantic(
        self, fact_id: str, confidence_score: float | None = None, source: str | None = None
    ) -> None:
        """Update reinforcement metadata for a duplicate semantic fact."""
        now = utc_now()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT confidence_score, source FROM am_semantic_memory WHERE fact_id = $1", fact_id)
            current_confidence = float(row["confidence_score"]) if row is not None else 0.0
            current_source = str(row["source"]) if row is not None and row["source"] else "llm_inferred"
            await conn.execute(
                "UPDATE am_semantic_memory SET confidence_score = $1, source = $2, last_reinforced_at = $3, "
                "last_confirmed_at = $4 WHERE fact_id = $5",
                max(current_confidence, confidence_score if confidence_score is not None else 0.0),
                _stronger_source(current_source, source),
                now,
                now,
                fact_id,
            )

    async def replace_semantic(self, fact_id: str, record: SemanticMemoryRecord) -> SemanticMemoryRecord:
        """Replace an existing semantic fact after conflict resolution."""
        embedding = record.embedding if record.embedding else None
        vec_col, vec_val = _semantic_embedding_columns(embedding)
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE am_semantic_memory SET fact_type = $1, content = $2, {vec_col} = $3::vector, "
                "confidence_score = $4, source = $5, source_episode_id = $6, last_reinforced_at = $7, "
                "last_confirmed_at = $8 WHERE fact_id = $9",
                record.fact_type.value,
                record.content,
                _vec_literal(vec_val),
                record.confidence_score,
                record.source,
                record.source_episode_id,
                record.last_reinforced_at,
                record.last_confirmed_at,
                fact_id,
            )
        record.fact_id = fact_id
        return record

    async def search_semantic(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        min_confidence: float,
        last_confirmed_after: datetime | None = None,
    ) -> list[SemanticMemoryRecord]:
        """Search semantic memory using server-side pgvector cosine similarity."""
        if len(embedding) == 384:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT *, 1 - (embedding <=> $1::vector) AS similarity "
                    "FROM am_semantic_memory WHERE embedding IS NOT NULL "
                    "AND confidence_score >= $4 "
                    "AND ($5::timestamptz IS NULL OR last_confirmed_at >= $5::timestamptz) "
                    "AND 1 - (embedding <=> $1::vector) >= $2 "
                    "ORDER BY embedding <=> $1::vector LIMIT $3",
                    _vec_literal(embedding), threshold, limit, min_confidence, last_confirmed_after,
                )
            return [self._semantic_from_row(row) for row in rows]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_semantic_memory WHERE confidence_score >= $1 "
                "AND ($2::timestamptz IS NULL OR last_confirmed_at >= $2::timestamptz) LIMIT 200",
                min_confidence, last_confirmed_after,
            )
        return self._rank_semantic(rows, embedding, threshold, limit)

    # ── Procedural ──────────────────────────────────────────────────

    async def search_procedural(
        self, embedding: list[float], limit: int, threshold: float
    ) -> list[ProceduralWorkflow]:
        """Search procedural workflows using server-side pgvector similarity."""
        if len(embedding) == 384:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT *, 1 - (embedding <=> $1::vector) AS similarity "
                    "FROM am_procedural_workflows WHERE embedding IS NOT NULL "
                    "AND 1 - (embedding <=> $1::vector) >= $2 "
                    "ORDER BY embedding <=> $1::vector LIMIT $3",
                    _vec_literal(embedding), threshold, limit,
                )
            return [self._workflow_from_row(row) for row in rows]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_procedural_workflows LIMIT 200"
            )
        return self._rank_workflows(rows, embedding, threshold, limit)

    async def match_procedural_triggers(
        self, prompt: str, limit: int
    ) -> list[ProceduralWorkflow]:
        """Find workflows whose trigger phrases appear in the current prompt."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM am_procedural_workflows ORDER BY success_count DESC LIMIT 100",
            )
        prompt_normalized = prompt.lower()
        matches: list[ProceduralWorkflow] = []
        for row in rows:
            phrases = row.get("trigger_phrases") or []
            if any(phrase.lower() in prompt_normalized for phrase in phrases if phrase):
                matches.append(self._workflow_from_row(row))
            if len(matches) >= limit:
                break
        return matches

    async def upsert_procedural_workflow(self, workflow: ProceduralWorkflow) -> ProceduralWorkflow:
        """Insert or reinforce a procedural workflow by signature."""
        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT * FROM am_procedural_workflows WHERE workflow_signature = $1",
                workflow.workflow_signature,
            )
        if existing:
            current = self._workflow_from_row(existing)
            new_count = current.success_count + 1
            avg_latency = ((current.avg_latency_ms * current.success_count) + workflow.avg_latency_ms) / new_count
            status = WorkflowStatus.CANONICAL.value if new_count >= 3 else current.status.value
            new_triggers = sorted(set(current.trigger_phrases + workflow.trigger_phrases))
            now = utc_now()
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE am_procedural_workflows SET trigger_phrases = $1, success_count = $2, status = $3, "
                    "avg_latency_ms = $4, updated_at = $5 WHERE workflow_id = $6",
                    new_triggers, new_count, status, avg_latency, now, current.workflow_id,
                )
            current.success_count = new_count
            current.status = WorkflowStatus(status)
            current.avg_latency_ms = avg_latency
            current.trigger_phrases = new_triggers
            return current

        embedding = workflow.embedding if workflow.embedding else None
        vec_col, vec_val = _semantic_embedding_columns(embedding)
        tool_seq = json.dumps([step.model_dump(mode="json") for step in workflow.tool_sequence])

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO am_procedural_workflows (workflow_id, workflow_signature, trigger_phrases, "
                f"tool_sequence, success_count, status, avg_latency_ms, {vec_col}, created_at, updated_at) "
                f"VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8::vector, $9, $10)",
                workflow.workflow_id, workflow.workflow_signature,
                workflow.trigger_phrases, tool_seq,
                workflow.success_count, workflow.status.value, workflow.avg_latency_ms,
                _vec_literal(vec_val), workflow.created_at, workflow.updated_at,
            )
        return workflow

    async def inspect_layer(self, layer: MemoryLayer, limit: int, offset: int) -> list[dict[str, object]]:
        """Return raw records for a single memory layer."""
        table_by_layer = {
            MemoryLayer.CONVERSATIONAL: "am_conversation_turns",
            MemoryLayer.EPISODIC: "am_episodic_memory",
            MemoryLayer.SEMANTIC: "am_semantic_memory",
            MemoryLayer.PROCEDURAL: "am_procedural_workflows",
            MemoryLayer.FAILURE: "am_failure_episodes",
        }
        order_by_layer = {
            MemoryLayer.CONVERSATIONAL: "timestamp",
            MemoryLayer.EPISODIC: "timestamp",
            MemoryLayer.SEMANTIC: "last_confirmed_at",
            MemoryLayer.PROCEDURAL: "updated_at",
            MemoryLayer.FAILURE: "timestamp",
        }
        table = table_by_layer[layer]
        order_column = order_by_layer[layer]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {table} ORDER BY {order_column} DESC LIMIT $1 OFFSET $2",  # noqa: S608
                limit, offset,
            )
        return [dict(row) for row in rows]

    async def count_layer(self, layer: MemoryLayer) -> int:
        """Return the number of records in one memory layer."""
        table_by_layer = {
            MemoryLayer.CONVERSATIONAL: "am_conversation_turns",
            MemoryLayer.EPISODIC: "am_episodic_memory",
            MemoryLayer.SEMANTIC: "am_semantic_memory",
            MemoryLayer.PROCEDURAL: "am_procedural_workflows",
            MemoryLayer.FAILURE: "am_failure_episodes",
        }
        table = table_by_layer[layer]
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(
                f"SELECT COUNT(*) FROM {table}",  # noqa: S608
            )
        return int(value or 0)

    # ── Row Parsers ─────────────────────────────────────────────────

    def _conv_from_row(self, row: Any) -> ConversationalTurnRecord:
        """Convert a PostgreSQL row into a conversational turn record."""
        return ConversationalTurnRecord(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            turn_index=int(row["turn_index"]),
            role=ConversationRole(str(row["role"])),
            content=str(row["content"]),
            token_count=int(row["token_count"]),
            timestamp=row["timestamp"],
        )

    def _summary_from_row(self, row: Any) -> ConversationalSummary:
        """Convert a PostgreSQL row into a conversational summary."""
        return ConversationalSummary(
            summary_id=str(row["summary_id"]),
            session_id=str(row["session_id"]),
            start_turn_index=int(row["start_turn_index"]),
            end_turn_index=int(row["end_turn_index"]),
            summary=str(row["summary"]),
            token_count=int(row["token_count"]),
            created_at=row["created_at"],
        )

    def _episode_from_row(self, row: Any) -> EpisodeRecord:
        """Convert a PostgreSQL row into an episodic memory model."""
        tool_seq = row.get("tool_sequence") or []
        if isinstance(tool_seq, str):
            tool_seq = json.loads(tool_seq)
        tools = [ToolInvocation.model_validate(tool) for tool in tool_seq]
        embedding = _parse_pg_vector(row.get("prompt_embedding") or row.get("prompt_embedding_hash"))
        return EpisodeRecord(
            episode_id=str(row["episode_id"]),
            prompt_text=str(row["prompt_text"]),
            reasoning_summary=str(row.get("reasoning_summary") or ""),
            prompt_embedding=embedding,
            tool_sequence=tools,
            final_response=str(row.get("final_response") or ""),
            outcome=EpisodeOutcome(str(row["outcome"])),
            error_trace=cast(str | None, row.get("error_trace")),
            latency_ms=int(row.get("latency_ms") or 0),
            timestamp=row.get("timestamp") or utc_now(),
            score=_clip_similarity(row.get("similarity")),
        )

    def _failure_from_row(self, row: Any) -> FailureEpisode:
        """Convert a PostgreSQL row into a failure episode model."""
        embedding = _parse_pg_vector(row.get("prompt_embedding") or row.get("prompt_embedding_hash"))
        tool_input = row.get("tool_input") or {}
        if isinstance(tool_input, str):
            tool_input = json.loads(tool_input)
        return FailureEpisode(
            failure_id=str(row["failure_id"]),
            episode_id=cast(str | None, row.get("episode_id")),
            prompt_text=str(row["prompt_text"]),
            prompt_embedding=embedding,
            tool_name=str(row["tool_name"]),
            tool_input=cast(dict[str, Any], tool_input),
            exception_message=str(row["exception_message"]),
            error_trace=str(row["error_trace"]),
            timestamp=row.get("timestamp") or utc_now(),
            score=_clip_similarity(row.get("similarity")),
        )

    def _semantic_from_row(self, row: Any) -> SemanticMemoryRecord:
        """Convert a PostgreSQL row into a semantic memory model."""
        embedding = _parse_pg_vector(row.get("embedding") or row.get("hash_embedding"))
        return SemanticMemoryRecord(
            fact_id=str(row["fact_id"]),
            fact_type=SemanticFactType(str(row["fact_type"])),
            content=str(row["content"]),
            embedding=embedding,
            confidence_score=float(row["confidence_score"]),
            source=str(row.get("source") or "llm_inferred"),
            source_episode_id=cast(str | None, row.get("source_episode_id")),
            created_at=row.get("created_at") or utc_now(),
            last_reinforced_at=row.get("last_reinforced_at") or utc_now(),
            last_confirmed_at=row.get("last_confirmed_at") or row.get("last_reinforced_at") or utc_now(),
            score=_clip_similarity(row.get("similarity")),
        )

    def _workflow_from_row(self, row: Any) -> ProceduralWorkflow:
        """Convert a PostgreSQL row into a procedural workflow model."""
        tool_seq = row.get("tool_sequence") or []
        if isinstance(tool_seq, str):
            tool_seq = json.loads(tool_seq)
        steps = [ProceduralToolStep.model_validate(step) for step in tool_seq]
        embedding = _parse_pg_vector(row.get("embedding") or row.get("hash_embedding"))
        return ProceduralWorkflow(
            workflow_id=str(row["workflow_id"]),
            workflow_signature=str(row["workflow_signature"]),
            trigger_phrases=list(row.get("trigger_phrases") or []),
            tool_sequence=steps,
            success_count=int(row.get("success_count") or 1),
            status=WorkflowStatus(str(row.get("status") or WorkflowStatus.CANDIDATE.value)),
            avg_latency_ms=float(row.get("avg_latency_ms") or 0.0),
            embedding=embedding,
            created_at=row.get("created_at") or utc_now(),
            updated_at=row.get("updated_at") or utc_now(),
            score=_clip_similarity(row.get("similarity")),
        )

    # ── Client-side fallback ranking for hash embeddings ────────────

    def _rank_episodes(self, rows: list[Any], embedding: list[float], threshold: float, limit: int) -> list[EpisodeRecord]:
        """Rank episode rows by client-side cosine similarity."""
        scored = _client_rank(rows, embedding, "prompt_embedding_hash", threshold, limit)
        return [self._episode_from_row(row) for row in scored]

    def _rank_failures(self, rows: list[Any], embedding: list[float], threshold: float, limit: int) -> list[FailureEpisode]:
        """Rank failure rows by client-side cosine similarity."""
        scored = _client_rank(rows, embedding, "prompt_embedding_hash", threshold, limit)
        return [self._failure_from_row(row) for row in scored]

    def _rank_semantic(self, rows: list[Any], embedding: list[float], threshold: float, limit: int) -> list[SemanticMemoryRecord]:
        """Rank semantic rows by client-side cosine similarity."""
        scored = _client_rank(rows, embedding, "hash_embedding", threshold, limit)
        return [self._semantic_from_row(row) for row in scored]

    def _rank_workflows(self, rows: list[Any], embedding: list[float], threshold: float, limit: int) -> list[ProceduralWorkflow]:
        """Rank workflow rows by client-side cosine similarity."""
        scored = _client_rank(rows, embedding, "hash_embedding", threshold, limit)
        return [self._workflow_from_row(row) for row in scored]


# ── Helpers ─────────────────────────────────────────────────────────


async def _migrate_schema(conn: asyncpg.Connection) -> None:
    """Apply additive PostgreSQL migrations for databases created by older releases."""
    await conn.execute(
        "ALTER TABLE IF EXISTS am_episodic_memory "
        "ADD COLUMN IF NOT EXISTS reasoning_summary text NOT NULL DEFAULT ''"
    )
    await conn.execute(
        "ALTER TABLE IF EXISTS am_semantic_memory "
        "ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'llm_inferred'"
    )
    await conn.execute(
        "ALTER TABLE IF EXISTS am_semantic_memory "
        "ADD COLUMN IF NOT EXISTS last_confirmed_at timestamptz"
    )
    semantic_table_exists = await conn.fetchval("SELECT to_regclass('public.am_semantic_memory') IS NOT NULL")
    if semantic_table_exists:
        await conn.execute(
            "UPDATE am_semantic_memory "
            "SET last_confirmed_at = coalesce(last_confirmed_at, last_reinforced_at, created_at, now()) "
            "WHERE last_confirmed_at IS NULL"
        )
        await conn.execute(
            "ALTER TABLE am_semantic_memory "
            "ALTER COLUMN last_confirmed_at SET NOT NULL"
        )


def _row_count(command_tag: str) -> int:
    """Extract affected-row count from an asyncpg command tag."""
    try:
        return int(command_tag.split()[-1])
    except (IndexError, ValueError):
        return 0


def _stronger_source(current: str, candidate: str | None) -> str:
    """Return the higher-authority semantic source label."""
    if not candidate:
        return current
    ranks = {"llm_inferred": 1, "tool_derived": 2, "user_stated": 3}
    return candidate if ranks.get(candidate, 1) > ranks.get(current, 1) else current


def _vec_literal(embedding: list[float] | None) -> str | None:
    """Format a Python list as a pgvector-compatible string literal."""
    if embedding is None:
        return None
    return "[" + ",".join(str(v) for v in embedding) + "]"


def _embedding_columns(embedding: list[float] | None) -> tuple[str, list[float] | None]:
    """Choose the correct vector column based on embedding dimensionality."""
    if embedding is None:
        return "prompt_embedding", None
    if len(embedding) == 384:
        return "prompt_embedding", embedding
    if len(embedding) == 256:
        return "prompt_embedding_hash", embedding
    raise ValueError(f"Unsupported embedding dimension {len(embedding)}")


def _semantic_embedding_columns(embedding: list[float] | None) -> tuple[str, list[float] | None]:
    """Choose the correct vector column for semantic/procedural tables."""
    if embedding is None:
        return "embedding", None
    if len(embedding) == 384:
        return "embedding", embedding
    if len(embedding) == 256:
        return "hash_embedding", embedding
    raise ValueError(f"Unsupported embedding dimension {len(embedding)}")


def _parse_pg_vector(value: Any) -> list[float]:
    """Parse a pgvector value returned as string, list, or None."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        cleaned = value.strip("[]")
        if not cleaned:
            return []
        return [float(v) for v in cleaned.split(",")]
    return []


def _clip_similarity(val: Any) -> float | None:
    """Clip a similarity score to the valid [0.0, 1.0] range to prevent floating-point schema validation errors."""
    if val is None:
        return None
    try:
        return max(0.0, min(1.0, float(val)))
    except (ValueError, TypeError):
        return None


def _client_rank(rows: list[Any], embedding: list[float], col: str, threshold: float, limit: int) -> list[Any]:
    """Rank rows client-side using cosine similarity for hash-embedding fallback."""
    import math

    scored: list[tuple[float, Any]] = []
    for row in rows:
        candidate = _parse_pg_vector(row.get(col))
        if not candidate:
            continue
        size = min(len(embedding), len(candidate))
        dot = sum(embedding[i] * candidate[i] for i in range(size))
        l_norm = math.sqrt(sum(embedding[i] ** 2 for i in range(size)))
        r_norm = math.sqrt(sum(candidate[i] ** 2 for i in range(size)))
        sim = max(0.0, min(1.0, dot / (l_norm * r_norm))) if l_norm > 0 and r_norm > 0 else 0.0
        if sim >= threshold:
            enriched = dict(row)
            enriched["similarity"] = sim
            scored.append((sim, enriched))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:limit]]
