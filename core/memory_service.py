"""AgenticMemoryService orchestration for context assembly and consolidation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Protocol

from core.access_scope import AccessScope
from core.models import (
    ActorResult,
    ConversationRole,
    ConversationalSummary,
    ConversationalTurnRecord,
    CriticEvaluation,
    EpisodeOutcome,
    EpisodeRecord,
    FailureEpisode,
    MemoryContext,
    ProceduralToolStep,
    ProceduralWorkflow,
    RetrievalPlan,
    SemanticFactType,
    SemanticMemoryRecord,
    ToolInvocation,
    WorkflowStatus,
    utc_now,
)
from model.embedding_model import EmbeddingModel
from store.base import MemoryStore

logger = logging.getLogger(__name__)


class SummaryProvider(Protocol):
    """Protocol for a lightweight summarizer used when the conversation window overflows."""

    async def summarize(self, records: list[ConversationalTurnRecord]) -> str:
        """Summarize pruned conversation records into one durable memory."""
        ...


class ExtractiveSummaryProvider:
    """Dependency-free fallback summarizer for rolling conversational summaries."""

    async def summarize(self, records: list[ConversationalTurnRecord]) -> str:
        """Create a compact extractive summary of pruned conversation messages."""
        if not records:
            return ""
        start = min(record.turn_index for record in records)
        end = max(record.turn_index for record in records)
        chunks = []
        for record in records:
            chunks.append(f"{record.role.value}: {_excerpt(record.content, 120)}")
        return f"Summary of turns {start} to {end}: " + " ".join(chunks)


class AgenticMemoryService:
    """Own context rendering and all four consolidation write paths."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_model: EmbeddingModel,
        memory_window_turns: int = 10,
        semantic_dedup_threshold: float = 0.92,
        semantic_memory_ttl_days: int = 180,
        summary_provider: SummaryProvider | None = None,
    ) -> None:
        """Create the memory service with explicit store and embedding dependencies."""
        self._store = store
        self._embedding_model = embedding_model
        self._memory_window_turns = memory_window_turns
        self._semantic_dedup_threshold = semantic_dedup_threshold
        self._semantic_memory_ttl_days = semantic_memory_ttl_days
        self._summary_provider = summary_provider or ExtractiveSummaryProvider()

    async def build_context(self, retrieval_plan: RetrievalPlan) -> MemoryContext:
        """Render a MemoryContext string in the exact layer order required by the Actor."""
        logger.info(
            "memory_hit_rates session=%s semantic=%s episodic=%s procedural=%s failure=%s",
            retrieval_plan.session_id,
            bool(retrieval_plan.semantic_records),
            bool(retrieval_plan.episodic_records),
            bool(retrieval_plan.procedural_workflows),
            bool(retrieval_plan.failure_matches),
        )
        lines: list[str] = []
        system_rules = [
            fact for fact in retrieval_plan.semantic_records if fact.fact_type == SemanticFactType.SYSTEM_RULE
        ]
        preferences = [
            fact for fact in retrieval_plan.semantic_records if fact.fact_type == SemanticFactType.PREFERENCE
        ]

        lines.append("[SYSTEM RULES]")
        lines.extend(_render_semantic_lines(system_rules))
        lines.append("")
        lines.append("[USER PREFERENCES]")
        lines.extend(_render_semantic_lines(preferences))

        if retrieval_plan.procedural_workflows:
            lines.append("")
            lines.append("[SUGGESTED WORKFLOW]")
            lines.extend(_render_workflow_lines(retrieval_plan.procedural_workflows[0]))

        lines.append("")
        lines.append("[RECENT CONVERSATION]")
        if retrieval_plan.conversation_summaries:
            for summary in retrieval_plan.conversation_summaries:
                lines.append(f"summary: {_excerpt(summary.summary, 500)}")
        for record in retrieval_plan.conversational_records:
            lines.append(f"{record.role.value}: {record.content}")
        if not retrieval_plan.conversation_summaries and not retrieval_plan.conversational_records:
            lines.append("(none)")

        episode_lines = _render_episode_lines(retrieval_plan)
        if episode_lines:
            lines.append("")
            lines.append("[PAST SIMILAR EPISODE]")
            lines.extend(episode_lines)

        return MemoryContext(retrieval_plan=retrieval_plan, rendered_context="\n".join(lines).strip())

    async def record_turn(
        self,
        scope: AccessScope,
        session_id: str,
        turn_index: int,
        user_message: str,
        assistant_message: str,
    ) -> list[ConversationalTurnRecord]:
        """Append the current user and assistant messages to conversational memory."""
        scope_hash = scope.scope_hash
        records = [
            ConversationalTurnRecord(
                scope_hash=scope_hash,
                session_id=session_id,
                turn_index=turn_index,
                role=ConversationRole.USER,
                content=user_message,
                token_count=_estimate_tokens(user_message),
            ),
            ConversationalTurnRecord(
                scope_hash=scope_hash,
                session_id=session_id,
                turn_index=turn_index,
                role=ConversationRole.ASSISTANT,
                content=assistant_message,
                token_count=_estimate_tokens(assistant_message),
            ),
        ]
        saved: list[ConversationalTurnRecord] = []
        for record in records:
            saved.append(await self._store.append_conversation_message(record))
        return saved

    async def next_turn_index(self, scope: AccessScope, session_id: str) -> int:
        """Return the next turn index without writing memory."""
        return await self._store.next_turn_index(scope.scope_hash, session_id)

    async def consolidate(
        self,
        prompt: str,
        actor_result: ActorResult,
        critic_evaluation: CriticEvaluation,
        scope: AccessScope,
        session_id: str,
        loop_latency_ms: int,
    ) -> tuple[int, list[str], list[str]]:
        """Write lessons from one loop turn into conversational, episodic, semantic, and procedural memory."""
        scope_hash = scope.scope_hash
        turn_index = await self._store.next_turn_index(scope_hash, session_id)
        writes: list[str] = []

        await self.record_turn(scope, session_id, turn_index, prompt, actor_result.final_response)
        writes.append("Saved 2 conversational messages")
        summary = await self._maybe_roll_summary(scope_hash, session_id, turn_index)
        if summary is not None:
            writes.append(f"Saved rolling summary for turns {summary.start_turn_index}-{summary.end_turn_index}")

        prompt_embedding = await self._embedding_model.embed(prompt)
        outcome = _episode_outcome(actor_result.tool_calls, critic_evaluation)
        episode = EpisodeRecord(
            scope_hash=scope_hash,
            prompt_text=prompt,
            prompt_embedding=prompt_embedding,
            tool_sequence=actor_result.tool_calls,
            final_response=actor_result.final_response,
            outcome=outcome,
            error_trace=_combined_error_trace(actor_result.tool_calls, critic_evaluation.failure_summary),
            latency_ms=loop_latency_ms,
        )
        saved_episode = await self._store.save_episode(episode)
        writes.append("Saved 1 episodic episode")

        saved_facts, semantic_conflicts = await self._consolidate_semantic_facts(
            critic_evaluation, scope_hash, saved_episode.episode_id
        )
        if saved_facts:
            writes.append(f"Saved {saved_facts} semantic fact{'s' if saved_facts != 1 else ''}")
        if semantic_conflicts:
            writes.append(f"Resolved {len(semantic_conflicts)} semantic conflict{'s' if len(semantic_conflicts) != 1 else ''}")

        workflow = await self._consolidate_workflow(prompt, actor_result, critic_evaluation, scope_hash)
        if workflow is not None:
            writes.append(
                f"Saved procedural workflow ({workflow.status.value}, success_count={workflow.success_count})"
            )

        failures = await self._consolidate_failures(prompt, prompt_embedding, actor_result, scope_hash, saved_episode.episode_id)
        if failures:
            writes.append(f"Saved {failures} failure episode{'s' if failures != 1 else ''}")

        return turn_index, writes, semantic_conflicts

    async def _maybe_roll_summary(
        self, scope_hash: str, session_id: str, turn_index: int
    ) -> ConversationalSummary | None:
        """Summarize and prune raw turns that fall outside the sliding window."""
        first_kept_turn = turn_index - self._memory_window_turns + 1
        if first_kept_turn <= 0:
            return None
        old_records = await self._store.conversation_before_turn(scope_hash, session_id, first_kept_turn)
        if not old_records:
            return None
        summary_text = await self._summary_provider.summarize(old_records)
        summary = ConversationalSummary(
            scope_hash=scope_hash,
            session_id=session_id,
            start_turn_index=min(record.turn_index for record in old_records),
            end_turn_index=max(record.turn_index for record in old_records),
            summary=summary_text,
            token_count=_estimate_tokens(summary_text),
        )
        saved = await self._store.save_conversation_summary(summary)
        await self._store.delete_conversation_before_turn(scope_hash, session_id, first_kept_turn)
        return saved

    async def _consolidate_semantic_facts(
        self, critic_evaluation: CriticEvaluation, scope_hash: str, episode_id: str
    ) -> tuple[int, list[str]]:
        """Deduplicate and save Critic-proposed semantic facts."""
        saved = 0
        conflicts: list[str] = []
        facts = critic_evaluation.new_semantic_facts
        embeddings = await self._embedding_model.embed_batch([fact.content for fact in facts])
        for fact, embedding in zip(facts, embeddings, strict=True):
            duplicates = await self._store.search_semantic(
                scope_hash,
                embedding,
                limit=1,
                threshold=self._semantic_dedup_threshold,
                min_confidence=0.0,
                last_confirmed_after=None,
            )
            source = _normalise_source(fact.source)
            if duplicates:
                existing = duplicates[0]
                if _semantic_contradiction(existing.content, fact.content):
                    record = SemanticMemoryRecord(
                        scope_hash=scope_hash,
                        fact_type=fact.fact_type,
                        content=fact.content,
                        embedding=embedding,
                        confidence_score=fact.confidence_score,
                        source=source,
                        source_episode_id=episode_id,
                    )
                    if _new_fact_wins(existing, record):
                        await self._store.replace_semantic(existing.fact_id, record)
                        conflicts.append(
                            f"semantic_conflict_replaced fact_id={existing.fact_id} old={_excerpt(existing.content, 80)} "
                            f"new={_excerpt(record.content, 80)}"
                        )
                    else:
                        await self._store.reinforce_semantic(
                            existing.fact_id,
                            confidence_score=fact.confidence_score,
                            source=source,
                        )
                        conflicts.append(
                            f"semantic_conflict_kept fact_id={existing.fact_id} old={_excerpt(existing.content, 80)} "
                            f"new={_excerpt(fact.content, 80)}"
                        )
                    continue
                await self._store.reinforce_semantic(
                    existing.fact_id,
                    confidence_score=fact.confidence_score,
                    source=source,
                )
                continue
            record = SemanticMemoryRecord(
                scope_hash=scope_hash,
                fact_type=fact.fact_type,
                content=fact.content,
                embedding=embedding,
                confidence_score=fact.confidence_score,
                source=source,
                source_episode_id=episode_id,
            )
            await self._store.insert_semantic(record)
            saved += 1
        return saved, conflicts

    def semantic_cutoff(self) -> datetime | None:
        """Return the active semantic TTL cutoff, or None when TTL is disabled."""
        if self._semantic_memory_ttl_days <= 0:
            return None
        return datetime.now(timezone.utc) - timedelta(days=self._semantic_memory_ttl_days)

    async def _consolidate_workflow(
        self,
        prompt: str,
        actor_result: ActorResult,
        critic_evaluation: CriticEvaluation,
        scope_hash: str,
    ) -> ProceduralWorkflow | None:
        """Save successful non-trivial tool chains as reusable procedural memory."""
        successful_tools = [tool for tool in actor_result.tool_calls if tool.success]
        if not critic_evaluation.save_workflow or len(successful_tools) < 2:
            return None
        steps = [
            ProceduralToolStep(
                tool_name=tool.tool_name,
                param_schema={key: type(value).__name__ for key, value in tool.input_parameters.items()},
                expected_outcome=_excerpt(tool.output_summary, 180),
            )
            for tool in successful_tools
        ]
        signature = _workflow_signature(successful_tools)
        workflow_text = f"{prompt} " + " ".join(step.tool_name for step in steps)
        workflow = ProceduralWorkflow(
            scope_hash=scope_hash,
            workflow_signature=signature,
            trigger_phrases=_trigger_phrases(prompt),
            tool_sequence=steps,
            success_count=1,
            status=WorkflowStatus.CANDIDATE,
            avg_latency_ms=float(sum(tool.latency_ms for tool in successful_tools)),
            embedding=await self._embedding_model.embed(workflow_text),
        )
        return await self._store.upsert_procedural_workflow(workflow)

    async def _consolidate_failures(
        self,
        prompt: str,
        prompt_embedding: list[float],
        actor_result: ActorResult,
        scope_hash: str,
        episode_id: str,
    ) -> int:
        """Save critic-flagged tool failures for future caution prompts."""
        count = 0
        for tool in actor_result.tool_calls:
            if not tool.critic_flagged:
                continue
            failure = FailureEpisode(
                scope_hash=scope_hash,
                episode_id=episode_id,
                prompt_text=prompt,
                prompt_embedding=prompt_embedding,
                tool_name=tool.tool_name,
                tool_input=tool.input_parameters,
                exception_message=tool.output_summary or "Tool failure",
                error_trace=tool.error_trace or tool.output_summary or "No error trace captured.",
            )
            await self._store.save_failure_episode(failure)
            count += 1
        return count


def _render_semantic_lines(records: list[SemanticMemoryRecord]) -> list[str]:
    """Render semantic facts for the Actor prompt."""
    if not records:
        return ["(none)"]
    return [f"- {record.content}" for record in records]


def _render_workflow_lines(workflow: ProceduralWorkflow) -> list[str]:
    """Render a suggested procedural workflow as an ordered tool list."""
    lines = []
    for index, step in enumerate(workflow.tool_sequence, start=1):
        schema = ", ".join(f"{key}: {value}" for key, value in step.param_schema.items()) or "no parameters"
        lines.append(f"{index}. {step.tool_name}({schema}) -> {step.expected_outcome}")
    return lines or ["(none)"]


def _render_episode_lines(retrieval_plan: RetrievalPlan) -> list[str]:
    """Render the most relevant prior episode or failure caution block."""
    if retrieval_plan.failure_matches:
        failure = retrieval_plan.failure_matches[0]
        return [
            f"Outcome: failure | Tools used: {failure.tool_name}",
            f"CAUTION: {failure.error_trace}",
        ]
    if not retrieval_plan.episodic_records:
        return []
    episode = retrieval_plan.episodic_records[0]
    tools = ", ".join(episode.tool_names) or "none"
    lines = [f"Outcome: {episode.outcome.value} | Tools used: {tools}"]
    if episode.outcome == EpisodeOutcome.FAILURE and episode.error_trace:
        lines.append(f"CAUTION: {episode.error_trace}")
    else:
        lines.append(_excerpt(episode.final_response, 400))
    return lines


def _episode_outcome(tool_calls: list[ToolInvocation], critic_evaluation: CriticEvaluation) -> EpisodeOutcome:
    """Derive an episodic outcome from tool results and Critic pass/fail state."""
    if tool_calls and all(not tool.success for tool in tool_calls):
        return EpisodeOutcome.FAILURE
    if any(tool.critic_flagged for tool in tool_calls):
        return EpisodeOutcome.FAILURE
    if not critic_evaluation.passed or any(not tool.success for tool in tool_calls):
        return EpisodeOutcome.PARTIAL
    return EpisodeOutcome.SUCCESS


def _combined_error_trace(tool_calls: list[ToolInvocation], failure_summary: str | None) -> str | None:
    """Combine tool error traces and Critic failure summary for episodic memory."""
    traces = [tool.error_trace for tool in tool_calls if tool.error_trace]
    if failure_summary:
        traces.append(failure_summary)
    return "\n".join(traces) if traces else None


def _workflow_signature(tools: list[ToolInvocation]) -> str:
    """Hash ordered tool names and parameter types into a workflow signature."""
    material = [
        {
            "tool_name": tool.tool_name,
            "param_types": sorted((key, type(value).__name__) for key, value in tool.input_parameters.items()),
        }
        for tool in tools
    ]
    encoded = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trigger_phrases(prompt: str) -> list[str]:
    """Extract compact trigger phrases that can recall a successful workflow later."""
    normalized = " ".join(prompt.lower().split())
    phrases = [_excerpt(normalized, 90)]
    words = [word for word in normalized.split() if len(word) > 3]
    if words:
        phrases.append(" ".join(words[:6]))
    return sorted({phrase for phrase in phrases if phrase})


def _estimate_tokens(text: str) -> int:
    """Estimate token count cheaply for memory-window accounting."""
    return max(1, len(text.split())) if text else 0


def _normalise_source(source: str) -> str:
    """Normalize semantic fact source labels for conflict resolution."""
    normalized = source.strip().lower()
    if normalized in {"user", "user_stated", "explicit_user"}:
        return "user_stated"
    if normalized in {"tool", "tool_derived"}:
        return "tool_derived"
    return "llm_inferred"


def _semantic_contradiction(existing: str, candidate: str) -> bool:
    """Detect simple opposite-meaning conflicts among near-duplicate semantic facts."""
    existing_norm = _normalise_fact_text(existing)
    candidate_norm = _normalise_fact_text(candidate)
    if existing_norm == candidate_norm:
        return False
    return _polarity(existing_norm) != _polarity(candidate_norm)


def _normalise_fact_text(text: str) -> str:
    """Normalize semantic fact text for lightweight contradiction checks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _polarity(text: str) -> int:
    """Return a coarse semantic polarity for preference/fact conflicts."""
    negative_patterns = (
        r"\bdo not\b",
        r"\bdon't\b",
        r"\bnever\b",
        r"\bno longer\b",
        r"\bnot\b",
        r"\bdislikes?\b",
        r"\bhates?\b",
        r"\bavoid(?:s|ed)?\b",
        r"\bdoes not prefer\b",
    )
    return -1 if any(re.search(pattern, text) for pattern in negative_patterns) else 1


def _new_fact_wins(existing: SemanticMemoryRecord, candidate: SemanticMemoryRecord) -> bool:
    """Resolve semantic conflicts by source authority, confidence, and recency."""
    source_delta = _source_rank(candidate.source) - _source_rank(existing.source)
    if source_delta != 0:
        return source_delta > 0
    confidence_delta = candidate.confidence_score - existing.confidence_score
    if abs(confidence_delta) >= 0.05:
        return confidence_delta > 0
    return candidate.last_confirmed_at >= existing.last_confirmed_at


def _source_rank(source: str) -> int:
    """Return semantic source authority rank."""
    return {"llm_inferred": 1, "tool_derived": 2, "user_stated": 3}.get(source, 1)


def _excerpt(text: str, max_length: int) -> str:
    """Return a single-line excerpt capped to the requested length."""
    single_line = " ".join(text.split())
    if len(single_line) <= max_length:
        return single_line
    return single_line[: max(0, max_length - 3)] + "..."
