"""SQLite implementation of the explicit memory-store protocol.

Provides persistent local storage without any cloud dependency. Vector search
is performed client-side using cosine similarity over JSON-serialized embeddings.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, cast, Iterable

import aiosqlite

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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS am_conversation_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL CHECK (turn_index >= 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    UNIQUE (session_id, turn_index, role)
);

CREATE TABLE IF NOT EXISTS am_conversation_summaries (
    summary_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    start_turn_index INTEGER NOT NULL,
    end_turn_index INTEGER NOT NULL,
    summary TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS am_episodic_memory (
    episode_id TEXT PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    prompt_embedding TEXT,
    tool_sequence TEXT NOT NULL DEFAULT '[]',
    final_response TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'partial', 'failure')),
    error_trace TEXT,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS am_failure_episodes (
    failure_id TEXT PRIMARY KEY,
    episode_id TEXT,
    prompt_text TEXT NOT NULL,
    prompt_embedding TEXT,
    tool_name TEXT NOT NULL,
    tool_input TEXT NOT NULL DEFAULT '{}',
    exception_message TEXT NOT NULL,
    error_trace TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS am_semantic_memory (
    fact_id TEXT PRIMARY KEY,
    fact_type TEXT NOT NULL CHECK (fact_type IN ('preference', 'inferred_fact', 'system_rule')),
    content TEXT NOT NULL,
    embedding TEXT,
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    source TEXT NOT NULL DEFAULT 'llm_inferred',
    source_episode_id TEXT,
    created_at TEXT NOT NULL,
    last_reinforced_at TEXT NOT NULL,
    last_confirmed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS am_procedural_workflows (
    workflow_id TEXT PRIMARY KEY,
    workflow_signature TEXT NOT NULL UNIQUE,
    trigger_phrases TEXT NOT NULL DEFAULT '[]',
    tool_sequence TEXT NOT NULL DEFAULT '[]',
    success_count INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'canonical')),
    avg_latency_ms REAL NOT NULL DEFAULT 0.0,
    embedding TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON am_conversation_turns (session_id, turn_index DESC);
CREATE INDEX IF NOT EXISTS idx_summ_session ON am_conversation_summaries (session_id, created_at DESC);
"""


class SQLiteMemoryStore(MemoryStore):
    """Persist all four memory layers in a local SQLite database.

    Vector search is performed client-side using cosine similarity over
    JSON-serialized embedding arrays. This avoids any dependency on pgvector
    while providing durable local storage.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Initialize with an open aiosqlite connection."""
        self._db = db

    @classmethod
    async def create(cls, db_path: str) -> "SQLiteMemoryStore":
        """Open or create a SQLite database and initialize the schema."""
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        await db.executescript(_SCHEMA)
        await _migrate_schema(db)
        await db.commit()
        return cls(db)

    async def close(self) -> None:
        """Close the underlying database connection."""
        await self._db.close()

    # ── Conversational ──────────────────────────────────────────────

    async def next_turn_index(self, session_id: str) -> int:
        """Return the next turn index for the session."""
        cursor = await self._db.execute(
            "SELECT MAX(turn_index) FROM am_conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0]) + 1

    async def append_conversation_message(self, record: ConversationalTurnRecord) -> ConversationalTurnRecord:
        """Append one user or assistant message to conversational memory."""
        await self._db.execute(
            "INSERT INTO am_conversation_turns (id, session_id, turn_index, role, content, token_count, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.id, record.session_id, record.turn_index,
             record.role.value, record.content, record.token_count, record.timestamp.isoformat()),
        )
        await self._db.commit()
        return record

    async def recent_conversation(self, session_id: str, limit: int) -> list[ConversationalTurnRecord]:
        """Return recent conversational messages for the active session."""
        cursor = await self._db.execute(
            "SELECT * FROM am_conversation_turns WHERE session_id = ? "
            "ORDER BY turn_index DESC, timestamp DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
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
        cursor = await self._db.execute(
            "SELECT * FROM am_conversation_turns WHERE session_id = ? AND turn_index = ? "
            "ORDER BY timestamp",
            (session_id, turn_index),
        )
        rows = await cursor.fetchall()
        return [self._conv_from_row(row) for row in rows]

    async def clear_conversation(self, session_id: str) -> int:
        """Delete raw conversation and summaries for one session."""
        turns_cursor = await self._db.execute(
            "DELETE FROM am_conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        summaries_cursor = await self._db.execute(
            "DELETE FROM am_conversation_summaries WHERE session_id = ?",
            (session_id,),
        )
        await self._db.commit()
        return (turns_cursor.rowcount or 0) + (summaries_cursor.rowcount or 0)

    async def conversation_before_turn(
        self, session_id: str, before_turn_index: int
    ) -> list[ConversationalTurnRecord]:
        """Return conversation messages old enough to be summarized and pruned."""
        cursor = await self._db.execute(
            "SELECT * FROM am_conversation_turns WHERE session_id = ? AND turn_index < ? "
            "ORDER BY turn_index, timestamp",
            (session_id, before_turn_index),
        )
        rows = await cursor.fetchall()
        return [self._conv_from_row(row) for row in rows]

    async def delete_conversation_before_turn(self, session_id: str, before_turn_index: int) -> int:
        """Delete raw conversation messages that have been safely summarized."""
        cursor = await self._db.execute(
            "DELETE FROM am_conversation_turns WHERE session_id = ? AND turn_index < ?",
            (session_id, before_turn_index),
        )
        await self._db.commit()
        return cursor.rowcount or 0

    async def save_conversation_summary(self, summary: ConversationalSummary) -> ConversationalSummary:
        """Persist a rolling conversational summary."""
        await self._db.execute(
            "INSERT INTO am_conversation_summaries (summary_id, session_id, start_turn_index, end_turn_index, summary, token_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (summary.summary_id, summary.session_id, summary.start_turn_index,
             summary.end_turn_index, summary.summary, summary.token_count, summary.created_at.isoformat()),
        )
        await self._db.commit()
        return summary

    async def recent_summaries(self, session_id: str, limit: int) -> list[ConversationalSummary]:
        """Return recent summaries for context assembly."""
        cursor = await self._db.execute(
            "SELECT * FROM am_conversation_summaries WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._summary_from_row(row) for row in rows]

    # ── Episodic ────────────────────────────────────────────────────

    async def save_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Persist an append-only episodic memory record."""
        await self._db.execute(
            "INSERT INTO am_episodic_memory (episode_id, prompt_text, prompt_embedding, tool_sequence, "
            "final_response, outcome, error_trace, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (episode.episode_id, episode.prompt_text,
             json.dumps(episode.prompt_embedding) if episode.prompt_embedding else None,
             json.dumps([tool.model_dump(mode="json") for tool in episode.tool_sequence]),
             episode.final_response, episode.outcome.value, episode.error_trace,
             episode.latency_ms, episode.timestamp.isoformat()),
        )
        await self._db.commit()
        return episode

    async def search_episodes(
        self, embedding: list[float], limit: int, threshold: float
    ) -> list[EpisodeRecord]:
        """Search episodic memory using client-side cosine similarity."""
        cursor = await self._db.execute("SELECT * FROM am_episodic_memory")
        rows = await cursor.fetchall()
        scored = _rank_rows(rows, embedding, "prompt_embedding", threshold, limit)
        return [self._episode_from_row(row, sim) for row, sim in scored]

    # ── Failure ─────────────────────────────────────────────────────

    async def save_failure_episode(self, failure: FailureEpisode) -> FailureEpisode:
        """Persist a detailed failure episode for future avoidance."""
        await self._db.execute(
            "INSERT INTO am_failure_episodes (failure_id, episode_id, prompt_text, prompt_embedding, "
            "tool_name, tool_input, exception_message, error_trace, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (failure.failure_id, failure.episode_id, failure.prompt_text,
             json.dumps(failure.prompt_embedding) if failure.prompt_embedding else None,
             failure.tool_name, json.dumps(failure.tool_input),
             failure.exception_message, failure.error_trace, failure.timestamp.isoformat()),
        )
        await self._db.commit()
        return failure

    async def search_failures(
        self, embedding: list[float], limit: int, threshold: float
    ) -> list[FailureEpisode]:
        """Search past failures using client-side cosine similarity."""
        cursor = await self._db.execute("SELECT * FROM am_failure_episodes")
        rows = await cursor.fetchall()
        scored = _rank_rows(rows, embedding, "prompt_embedding", threshold, limit)
        return [self._failure_from_row(row, sim) for row, sim in scored]

    # ── Semantic ────────────────────────────────────────────────────

    async def insert_semantic(self, record: SemanticMemoryRecord) -> SemanticMemoryRecord:
        """Insert a new deduplicated semantic memory fact."""
        await self._db.execute(
            "INSERT INTO am_semantic_memory (fact_id, fact_type, content, embedding, confidence_score, "
            "source, source_episode_id, created_at, last_reinforced_at, last_confirmed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record.fact_id, record.fact_type.value, record.content,
             json.dumps(record.embedding) if record.embedding else None,
             record.confidence_score, record.source, record.source_episode_id,
             record.created_at.isoformat(), record.last_reinforced_at.isoformat(),
             record.last_confirmed_at.isoformat()),
        )
        await self._db.commit()
        return record

    async def reinforce_semantic(
        self, fact_id: str, confidence_score: float | None = None, source: str | None = None
    ) -> None:
        """Update reinforcement metadata for a duplicate semantic fact."""
        now = utc_now().isoformat()
        if confidence_score is None and source is None:
            await self._db.execute(
                "UPDATE am_semantic_memory SET last_reinforced_at = ?, last_confirmed_at = ? WHERE fact_id = ?",
                (now, now, fact_id),
            )
            await self._db.commit()
            return

        existing = await self._db.execute("SELECT confidence_score, source FROM am_semantic_memory WHERE fact_id = ?", (fact_id,))
        row = await existing.fetchone()
        current_confidence = float(row["confidence_score"]) if row is not None else 0.0
        current_source = str(row["source"]) if row is not None and row["source"] else "llm_inferred"
        merged_confidence = max(current_confidence, confidence_score if confidence_score is not None else 0.0)
        merged_source = _stronger_source(current_source, source)
        await self._db.execute(
            "UPDATE am_semantic_memory SET confidence_score = ?, source = ?, last_reinforced_at = ?, "
            "last_confirmed_at = ? WHERE fact_id = ?",
            (merged_confidence, merged_source, now, now, fact_id),
        )
        await self._db.commit()

    async def replace_semantic(self, fact_id: str, record: SemanticMemoryRecord) -> SemanticMemoryRecord:
        """Replace an existing semantic fact after conflict resolution."""
        await self._db.execute(
            "UPDATE am_semantic_memory SET fact_type = ?, content = ?, embedding = ?, confidence_score = ?, source = ?, "
            "source_episode_id = ?, last_reinforced_at = ?, last_confirmed_at = ? WHERE fact_id = ?",
            (
                record.fact_type.value,
                record.content,
                json.dumps(record.embedding) if record.embedding else None,
                record.confidence_score,
                record.source,
                record.source_episode_id,
                record.last_reinforced_at.isoformat(),
                record.last_confirmed_at.isoformat(),
                fact_id,
            ),
        )
        await self._db.commit()
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
        """Search semantic memory using client-side cosine similarity and confidence filtering."""
        sql = "SELECT * FROM am_semantic_memory WHERE confidence_score >= ?"
        params: list[object] = [min_confidence]
        if last_confirmed_after is not None:
            sql += " AND last_confirmed_at >= ?"
            params.append(last_confirmed_after.isoformat())
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        scored = _rank_rows(rows, embedding, "embedding", threshold, limit)
        return [self._semantic_from_row(row, sim) for row, sim in scored]

    # ── Procedural ──────────────────────────────────────────────────

    async def search_procedural(
        self, embedding: list[float], limit: int, threshold: float
    ) -> list[ProceduralWorkflow]:
        """Search procedural workflows using client-side vector similarity."""
        cursor = await self._db.execute("SELECT * FROM am_procedural_workflows")
        rows = await cursor.fetchall()
        scored = _rank_rows(rows, embedding, "embedding", threshold, limit)
        return [self._workflow_from_row(row, sim) for row, sim in scored]

    async def match_procedural_triggers(
        self, prompt: str, limit: int
    ) -> list[ProceduralWorkflow]:
        """Find workflows whose trigger phrases appear in the current prompt."""
        cursor = await self._db.execute(
            "SELECT * FROM am_procedural_workflows ORDER BY success_count DESC LIMIT 100"
        )
        rows = await cursor.fetchall()
        prompt_normalized = prompt.lower()
        matches: list[ProceduralWorkflow] = []
        for row in rows:
            phrases = json.loads(row["trigger_phrases"]) if row["trigger_phrases"] else []
            if any(phrase.lower() in prompt_normalized for phrase in phrases if phrase):
                matches.append(self._workflow_from_row(row))
            if len(matches) >= limit:
                break
        return matches

    async def upsert_procedural_workflow(self, workflow: ProceduralWorkflow) -> ProceduralWorkflow:
        """Insert or reinforce a procedural workflow by signature."""
        cursor = await self._db.execute(
            "SELECT * FROM am_procedural_workflows WHERE workflow_signature = ?",
            (workflow.workflow_signature,),
        )
        existing_row = await cursor.fetchone()
        if existing_row:
            current = self._workflow_from_row(existing_row)
            new_count = current.success_count + 1
            avg_latency = ((current.avg_latency_ms * current.success_count) + workflow.avg_latency_ms) / new_count
            status = WorkflowStatus.CANONICAL.value if new_count >= 3 else current.status.value
            now = utc_now().isoformat()
            new_triggers = sorted(set(current.trigger_phrases + workflow.trigger_phrases))
            await self._db.execute(
                "UPDATE am_procedural_workflows SET trigger_phrases = ?, success_count = ?, status = ?, "
                "avg_latency_ms = ?, updated_at = ? WHERE workflow_id = ?",
                (json.dumps(new_triggers), new_count, status, avg_latency, now, current.workflow_id),
            )
            await self._db.commit()
            current.success_count = new_count
            current.status = WorkflowStatus(status)
            current.avg_latency_ms = avg_latency
            current.trigger_phrases = new_triggers
            return current

        await self._db.execute(
            "INSERT INTO am_procedural_workflows (workflow_id, workflow_signature, trigger_phrases, "
            "tool_sequence, success_count, status, avg_latency_ms, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (workflow.workflow_id, workflow.workflow_signature,
             json.dumps(workflow.trigger_phrases),
             json.dumps([step.model_dump(mode="json") for step in workflow.tool_sequence]),
             workflow.success_count, workflow.status.value, workflow.avg_latency_ms,
             json.dumps(workflow.embedding) if workflow.embedding else None,
             workflow.created_at.isoformat(), workflow.updated_at.isoformat()),
        )
        await self._db.commit()
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
        cursor = await self._db.execute(
            f"SELECT * FROM {table} ORDER BY {order_column} DESC LIMIT ? OFFSET ?",  # noqa: S608
            (limit, offset),
        )
        rows = await cursor.fetchall()
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
        cursor = await self._db.execute(
            f"SELECT COUNT(*) FROM {table}",  # noqa: S608
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    # ── Row Parsers ─────────────────────────────────────────────────

    def _conv_from_row(self, row: Any) -> ConversationalTurnRecord:
        """Convert a SQLite row into a conversational turn record."""
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
        """Convert a SQLite row into a conversational summary."""
        return ConversationalSummary(
            summary_id=str(row["summary_id"]),
            session_id=str(row["session_id"]),
            start_turn_index=int(row["start_turn_index"]),
            end_turn_index=int(row["end_turn_index"]),
            summary=str(row["summary"]),
            token_count=int(row["token_count"]),
            created_at=row["created_at"],
        )

    def _episode_from_row(self, row: Any, similarity: float | None = None) -> EpisodeRecord:
        """Convert a SQLite row into an episodic memory model."""
        tool_seq = json.loads(row["tool_sequence"]) if row["tool_sequence"] else []
        tools = [ToolInvocation.model_validate(tool) for tool in tool_seq]
        embedding = json.loads(row["prompt_embedding"]) if row["prompt_embedding"] else []
        return EpisodeRecord(
            episode_id=str(row["episode_id"]),
            prompt_text=str(row["prompt_text"]),
            prompt_embedding=embedding,
            tool_sequence=tools,
            final_response=str(row["final_response"] or ""),
            outcome=EpisodeOutcome(str(row["outcome"])),
            error_trace=cast(str | None, row["error_trace"]),
            latency_ms=int(row["latency_ms"] or 0),
            timestamp=row["timestamp"] or utc_now(),
            score=similarity,
        )

    def _failure_from_row(self, row: Any, similarity: float | None = None) -> FailureEpisode:
        """Convert a SQLite row into a failure episode model."""
        embedding = json.loads(row["prompt_embedding"]) if row["prompt_embedding"] else []
        return FailureEpisode(
            failure_id=str(row["failure_id"]),
            episode_id=cast(str | None, row["episode_id"]),
            prompt_text=str(row["prompt_text"]),
            prompt_embedding=embedding,
            tool_name=str(row["tool_name"]),
            tool_input=json.loads(row["tool_input"]) if row["tool_input"] else {},
            exception_message=str(row["exception_message"]),
            error_trace=str(row["error_trace"]),
            timestamp=row["timestamp"] or utc_now(),
            score=similarity,
        )

    def _semantic_from_row(self, row: Any, similarity: float | None = None) -> SemanticMemoryRecord:
        """Convert a SQLite row into a semantic memory model."""
        embedding = json.loads(row["embedding"]) if row["embedding"] else []
        return SemanticMemoryRecord(
            fact_id=str(row["fact_id"]),
            fact_type=SemanticFactType(str(row["fact_type"])),
            content=str(row["content"]),
            embedding=embedding,
            confidence_score=float(row["confidence_score"]),
            source=str(row["source"] or "llm_inferred"),
            source_episode_id=cast(str | None, row["source_episode_id"]),
            created_at=row["created_at"] or utc_now(),
            last_reinforced_at=row["last_reinforced_at"] or utc_now(),
            last_confirmed_at=row["last_confirmed_at"] or row["last_reinforced_at"] or utc_now(),
            score=similarity,
        )

    def _workflow_from_row(self, row: Any, similarity: float | None = None) -> ProceduralWorkflow:
        """Convert a SQLite row into a procedural workflow model."""
        tool_seq = json.loads(row["tool_sequence"]) if row["tool_sequence"] else []
        steps = [ProceduralToolStep.model_validate(step) for step in tool_seq]
        embedding = json.loads(row["embedding"]) if row["embedding"] else []
        triggers = json.loads(row["trigger_phrases"]) if row["trigger_phrases"] else []
        return ProceduralWorkflow(
            workflow_id=str(row["workflow_id"]),
            workflow_signature=str(row["workflow_signature"]),
            trigger_phrases=triggers,
            tool_sequence=steps,
            success_count=int(row["success_count"] or 1),
            status=WorkflowStatus(str(row["status"] or WorkflowStatus.CANDIDATE.value)),
            avg_latency_ms=float(row["avg_latency_ms"] or 0.0),
            embedding=embedding,
            created_at=row["created_at"] or utc_now(),
            updated_at=row["updated_at"] or utc_now(),
            score=similarity,
        )


# ── Schema migration helpers ───────────────────────────────────────

async def _migrate_schema(db: aiosqlite.Connection) -> None:
    """Apply additive SQLite migrations for databases created by older releases."""
    cursor = await db.execute("PRAGMA table_info(am_semantic_memory)")
    columns = {str(row["name"]) for row in await cursor.fetchall()}
    if "source" not in columns:
        await db.execute("ALTER TABLE am_semantic_memory ADD COLUMN source TEXT NOT NULL DEFAULT 'llm_inferred'")
    if "last_confirmed_at" not in columns:
        await db.execute("ALTER TABLE am_semantic_memory ADD COLUMN last_confirmed_at TEXT")
        await db.execute(
            "UPDATE am_semantic_memory SET last_confirmed_at = COALESCE(last_reinforced_at, created_at, datetime('now')) "
            "WHERE last_confirmed_at IS NULL"
        )


def _stronger_source(current: str, candidate: str | None) -> str:
    """Return the higher-authority semantic source label."""
    if not candidate:
        return current
    ranks = {"llm_inferred": 1, "tool_derived": 2, "user_stated": 3}
    return candidate if ranks.get(candidate, 1) > ranks.get(current, 1) else current


# ── Client-side vector ranking ──────────────────────────────────────

def _rank_rows(
    rows: Iterable[Any], embedding: list[float], vector_column: str, threshold: float, limit: int
) -> list[tuple[Any, float]]:
    """Rank SQLite rows by cosine similarity against a JSON-serialized embedding column."""
    scored: list[tuple[float, Any]] = []
    for row in rows:
        raw = row[vector_column]
        if not raw:
            continue
        candidate = json.loads(raw) if isinstance(raw, str) else raw
        score = _cosine(embedding, candidate)
        if score >= threshold:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(row, score) for score, row in scored[:limit]]


def _cosine(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for client-side vector ranking."""
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot: float = sum(left[i] * right[i] for i in range(size))
    left_norm: float = math.sqrt(sum(left[i] * left[i] for i in range(size)))
    right_norm: float = math.sqrt(sum(right[i] * right[i] for i in range(size)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    val = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, val))
