# Architecture Notes

Agentic Memory is split into adapters, core services, planning, persistence, model clients, and optional standalone runtime. The default MCP mode treats the external assistant as the Actor: Claude, Codex, Cline, or another MCP-compatible agent does the user-facing reasoning and calls memory tools before and after its response.

## Layer Responsibilities

| Layer | Modules | Responsibility |
|---|---|---|
| Adapters | `mcp_server.py`, `main.py` | Expose MCP and HTTP interfaces, parse inputs, serialize outputs. |
| Core | `core/memory_service.py`, `core/evaluation_service.py`, `core/models.py` | Build memory context, consolidate turns, evaluate completed turns, define shared schemas. |
| Planner | `planner/retrieval_planner.py` | Decide which memory layers to query and rank retrieved records. |
| Persistence | `store/base.py`, `store/sqlite_store.py`, `store/postgres_store.py` | Implement the memory-store contract for SQLite and PostgreSQL/pgvector. |
| Model | `model/*` | Provide structured JSON LLM clients and embedding adapters. |
| Runtime | `runtime/*` | Optional standalone Actor-Critic loop and local tools. |

## MCP Mode

MCP mode is the primary product path. The agent calls `get_session_context` before responding, creates the user-facing answer, then calls `consolidate_turn` after responding.

`consolidate_turn` wraps the external response as an `ActorResult`, sends it through `AutoEvaluationService`, then writes conversational, episodic, semantic, procedural, and failure memories through `AgenticMemoryService`.

For agentic retrieval orchestration, the external MCP client is the LLM
orchestrator. Codex, Claude Code, Cline, or another assistant can call
`get_memory_tool_manifest`, then invoke `retrieve_memory_layer` repeatedly to
plan and synthesize retrieval across conversational, semantic, semantic
hierarchy, episodic, procedural, and failure memory. This avoids adding a
second hidden LLM provider just for retrieval planning.

## Standalone Mode

Standalone mode is optional and requires an LLM provider. In this mode the project creates its own `Actor`, `Critic`, tool registry, planner, and memory service, then runs the six-phase loop in `runtime/self_learning_loop.py`.

## Evaluation

`core/evaluation_service.py` owns turn evaluation. It can call an LLM Critic when one is configured, or use deterministic heuristics when no provider exists. Keeping this outside `mcp_server.py` lets MCP, HTTP, tests, and standalone flows share evaluation behavior.

## Retrieval

The quick-path planner combines keyword triggers with vector search and adaptive per-session weights. Feedback weights are persisted through the store contract, so retrieval tuning survives process restarts. Retrieved records are re-ranked before context rendering using similarity, recency, confidence, outcome quality, workflow maturity, hierarchy-node abstraction level, and pinning.

## Semantic Hierarchy

Semantic hierarchy is a derived layer built from flat semantic facts. It gives
retrieval a higher-level view of preferences and project context without
requiring a hidden internal planner LLM or an asynchronous batch pipeline.

The hierarchy is stored in `am_semantic_hierarchy_nodes` and represented by
`SemanticHierarchyNode`. Nodes have deterministic keys, parent links, a facet,
source fact IDs, embeddings, confidence, and timestamps.

| Node type | Purpose | Key shape |
|---|---|---|
| `root` | Top-level semantic index. | `root` |
| `facet` | Groups related facts into stable facets such as `communication_preferences`, `workflow_preferences`, `project_context`, `environment`, and `general_knowledge`. | `facet:{facet}` |
| `summary` | Compact bottom-up summary for the facet. | `summary:{facet}` |
| `qa` | Q&A-style retrieval node whose answer is grounded in one or more semantic facts. | `qa:{facet}:{question_hash}` |

Build flow:

```text
semantic fact
  -> classify facet
  -> upsert root node
  -> upsert facet node
  -> merge fact into facet summary
  -> merge fact into facet Q&A node
```

This means the canonical semantic record remains the flat fact. The hierarchy
does not own truth; it is a retrieval index over active semantic records. The
quick-path planner searches semantic hierarchy nodes when semantic memory is
triggered, and `retrieve_memory_layer(..., layer="semantic_hierarchy")` lets
Codex, Claude Code, or another MCP client explicitly use it during external
agentic retrieval orchestration.

## Memory Governance

Semantic facts can be deleted, pinned, marked stale, exported, and imported. Pinned facts bypass semantic TTL filtering. Episodic memories remain append-only by default, but old episodes can be pruned while failure records are detached and preserved.

Semantic hierarchy nodes are derived data. When semantic facts are deleted,
pinned, marked stale, or imported through the MCP management tools, the
hierarchy is rebuilt from the currently active semantic facts so stale derived
content is not retained.

## Testing

The offline evaluation harness lives at `tests/evaluation_harness.py` and uses SQLite plus hash embeddings. It covers heuristic evaluation, semantic deduplication, semantic hierarchy creation and retrieval, pin/stale behavior, persisted planner feedback, context rendering, and failure recall.

Run it with:

```bash
python -m tests.evaluation_harness
```
