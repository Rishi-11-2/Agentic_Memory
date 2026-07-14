"""Typed Pydantic schemas shared by the memory store and self-learning loop."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonDict = dict[str, Any]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for durable memory records."""
    return datetime.now(timezone.utc)


class MemoryLayer(str, Enum):
    """Enumerate explicit memory layers plus failure episodes for inspection/search."""

    CONVERSATIONAL = "conversational"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SEMANTIC_HIERARCHY = "semantic_hierarchy"
    PROCEDURAL = "procedural"
    FAILURE = "failure"


class ConversationRole(str, Enum):
    """Constrain conversational records to user and assistant turns."""

    USER = "user"
    ASSISTANT = "assistant"


class EpisodeOutcome(str, Enum):
    """Capture whether a full turn succeeded, partially succeeded, or failed."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class SemanticFactType(str, Enum):
    """Classify durable semantic facts for prompt rendering and governance."""

    PREFERENCE = "preference"
    INFERRED_FACT = "inferred_fact"
    SYSTEM_RULE = "system_rule"


class SemanticHierarchyNodeType(str, Enum):
    """Classify semantic hierarchy nodes by abstraction level."""

    ROOT = "root"
    FACET = "facet"
    SUMMARY = "summary"
    QA = "qa"


class WorkflowStatus(str, Enum):
    """Track whether a learned tool workflow is tentative or canonical."""

    CANDIDATE = "candidate"
    CANONICAL = "canonical"


class LLMMessage(BaseModel):
    """Represent a model-agnostic chat message for structured LLM adapters."""

    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ConversationalTurnRecord(BaseModel):
    """Store one role-specific conversational message in the sliding window."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    turn_index: int = Field(ge=0)
    role: ConversationRole
    content: str
    token_count: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=utc_now)


class ConversationalSummary(BaseModel):
    """Store a compressed summary of conversation turns pruned from the window."""

    summary_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    start_turn_index: int = Field(ge=0)
    end_turn_index: int = Field(ge=0)
    summary: str
    token_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class ToolInvocation(BaseModel):
    """Audit one attempted tool call, including failures flagged for learning."""

    tool_name: str
    input_parameters: JsonDict = Field(default_factory=dict)
    input_summary: str = ""
    output_summary: str = ""
    success: bool = False
    latency_ms: int = Field(default=0, ge=0)
    error_trace: str | None = None
    critic_flagged: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    """Persist an append-only record of a complete actor turn."""

    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    prompt_text: str
    reasoning_summary: str = ""
    prompt_embedding: list[float] = Field(default_factory=list)
    tool_sequence: list[ToolInvocation] = Field(default_factory=list)
    final_response: str = ""
    outcome: EpisodeOutcome = EpisodeOutcome.SUCCESS
    error_trace: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=utc_now)
    evaluation_score: float | None = Field(default=None, ge=0.0, le=10.0)
    evaluation_source: str | None = None
    needs_agent_rescore: bool = False
    evaluated_at: datetime | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def tool_names(self) -> list[str]:
        """Return ordered tool names for prompt rendering and workflow signatures."""
        return [tool.tool_name for tool in self.tool_sequence]


class FailureEpisode(BaseModel):
    """Persist detailed tool failure context for future avoidance."""

    failure_id: str = Field(default_factory=lambda: str(uuid4()))
    episode_id: str | None = None
    prompt_text: str
    prompt_embedding: list[float] = Field(default_factory=list)
    tool_name: str
    tool_input: JsonDict = Field(default_factory=dict)
    exception_message: str
    error_trace: str
    timestamp: datetime = Field(default_factory=utc_now)
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class NewSemanticFact(BaseModel):
    """Represent one Critic-proposed semantic fact to deduplicate and consolidate."""

    model_config = ConfigDict(populate_by_name=True)

    fact_type: SemanticFactType
    content: str
    confidence_score: float = Field(ge=0.0, le=1.0, alias="confidence")
    source: str = "llm_inferred"


class SemanticMemoryRecord(BaseModel):
    """Store a durable preference, inferred fact, or system rule with an embedding."""

    model_config = ConfigDict(populate_by_name=True)

    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    fact_type: SemanticFactType
    content: str
    embedding: list[float] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    source: str = "llm_inferred"
    source_episode_id: str | None = None
    pinned: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    last_reinforced_at: datetime = Field(default_factory=utc_now)
    last_confirmed_at: datetime = Field(default_factory=utc_now)
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class SemanticHierarchyNode(BaseModel):
    """Store a hierarchical aggregate derived from flat semantic facts."""

    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_key: str
    parent_id: str | None = None
    node_type: SemanticHierarchyNodeType
    facet: str = "general"
    title: str
    content: str = ""
    question: str | None = None
    answer: str | None = None
    source_fact_ids: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class ProceduralToolStep(BaseModel):
    """Describe one reusable tool step inside a learned workflow."""

    tool_name: str
    param_schema: dict[str, str] = Field(default_factory=dict)
    expected_outcome: str


class ProceduralWorkflow(BaseModel):
    """Store an ordered reusable tool workflow learned from successful episodes."""

    workflow_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_signature: str
    trigger_phrases: list[str] = Field(default_factory=list)
    tool_sequence: list[ProceduralToolStep] = Field(default_factory=list)
    success_count: int = Field(default=1, ge=1)
    status: WorkflowStatus = WorkflowStatus.CANDIDATE
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def tool_names(self) -> list[str]:
        """Return ordered tool names for rendering suggested workflows."""
        return [step.tool_name for step in self.tool_sequence]


class RetrievedRecord(BaseModel):
    """Provide a lightweight auditable pointer to any retrieved memory item."""

    layer: MemoryLayer
    record_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class RetrievalPlan(BaseModel):
    """Capture layer flags, rationales, and raw records selected by the planner."""

    session_id: str
    prompt: str
    query_conversational: bool = True
    query_episodic: bool = False
    query_semantic: bool = False
    query_procedural: bool = False
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    episodic_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    rationale: dict[str, str] = Field(default_factory=dict)
    conversational_records: list[ConversationalTurnRecord] = Field(default_factory=list)
    conversation_summaries: list[ConversationalSummary] = Field(default_factory=list)
    episodic_records: list[EpisodeRecord] = Field(default_factory=list)
    semantic_records: list[SemanticMemoryRecord] = Field(default_factory=list)
    semantic_hierarchy_records: list[SemanticHierarchyNode] = Field(default_factory=list)
    procedural_workflows: list[ProceduralWorkflow] = Field(default_factory=list)
    failure_matches: list[FailureEpisode] = Field(default_factory=list)
    retrieved_records: list[RetrievedRecord] = Field(default_factory=list)

    def summary(self) -> str:
        """Render a user-safe summary of which memory layers were queried."""
        queried: list[str] = ["conversational"]
        skipped: list[str] = []
        if self.query_semantic:
            queried.append(f"semantic ({len(self.semantic_records)} facts)")
            if self.semantic_hierarchy_records:
                queried.append(f"semantic_hierarchy ({len(self.semantic_hierarchy_records)} nodes)")
        else:
            skipped.append("semantic (no trigger)")
        if self.query_procedural:
            queried.append(f"procedural ({len(self.procedural_workflows)} workflows)")
        else:
            skipped.append("procedural (no trigger)")
        if self.query_episodic:
            queried.append(f"episodic ({len(self.episodic_records)} matches)")
        else:
            skipped.append("episodic (low similarity)")
        return f"Queried: {', '.join(queried)}. Skipped: {', '.join(skipped)}."


class MemoryContext(BaseModel):
    """Bundle the rendered actor prompt context with its retrieval plan."""

    retrieval_plan: RetrievalPlan
    rendered_context: str


class ActorLLMOutput(BaseModel):
    """Parse the Actor model's structured response before tool execution."""

    reasoning: str = ""
    tool_calls: list[ToolInvocation] = Field(default_factory=list)
    final_response: str


class ActorResult(BaseModel):
    """Return the Actor response, hidden reasoning, tool traces, and latency."""

    reasoning: str = ""
    tool_calls: list[ToolInvocation] = Field(default_factory=list)
    final_response: str
    latency_ms: int = Field(default=0, ge=0)


class CriticEvaluation(BaseModel):
    """Score the Actor and propose memory writes for consolidation."""

    model_config = ConfigDict(populate_by_name=True)

    factual_accuracy: float = Field(ge=0.0, le=10.0)
    preference_adherence: float = Field(ge=0.0, le=10.0)
    tool_efficiency: float = Field(ge=0.0, le=10.0)
    hallucination_risk: float = Field(ge=0.0, le=10.0)
    workflow_quality: float = Field(ge=0.0, le=10.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=10.0)
    passed: bool = Field(default=False, alias="pass")
    new_semantic_facts: list[NewSemanticFact] = Field(default_factory=list)
    save_workflow: bool = False
    failure_summary: str | None = None

    @model_validator(mode="after")
    def compute_overall(self) -> "CriticEvaluation":
        """Recompute weighted score and pass state to keep provider output honest."""
        weighted = (
            self.factual_accuracy * 0.25
            + self.preference_adherence * 0.20
            + self.tool_efficiency * 0.20
            + self.hallucination_risk * 0.15
            + self.workflow_quality * 0.20
        )
        self.overall_score = round(weighted, 2)
        self.passed = self.overall_score >= 7.0
        return self


class LoopResult(BaseModel):
    """Return the public response produced by one six-phase loop iteration."""

    final_response: str
    session_id: str
    turn_index: int
    critic_score: float
    critic_pass: bool
    memory_writes: list[str]
    retrieval_plan_summary: str
    phase_timings_ms: dict[str, float] = Field(default_factory=dict)
    semantic_conflicts: list[str] = Field(default_factory=list)


class MemoryInspectResponse(BaseModel):
    """Return paginated raw memory records for admin/debug inspection."""

    layer: MemoryLayer
    limit: int
    offset: int
    records: list[JsonDict]
