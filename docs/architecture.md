# Architecture Notes

Agentic Memory is split into adapters, core services, planning, persistence, model clients, and optional standalone runtime. The default MCP mode treats the MCP client agent as both Actor and retrieval orchestrator: Claude Code, Codex, Cline, or another MCP-compatible agent reasons over memory layers, calls retrieval tools, answers the user, and then consolidates the turn.

## Layer Responsibilities

| Layer | Modules | Responsibility |
|---|---|---|
| Adapters | `mcp_server.py`, `main.py` | Expose MCP and HTTP interfaces, parse inputs, serialize outputs. |
| Core | `core/memory_service.py`, `core/evaluation_service.py`, `core/models.py` | Build memory context, consolidate turns, evaluate completed turns, define shared schemas. |
| Planner | `planner/retrieval_planner.py` | Expose MCP-client orchestrated retrieval primitives, shared ranking, and the fallback heuristic context builder. |
| Persistence | `store/base.py`, `store/sqlite_store.py`, `store/postgres_store.py` | Implement the memory-store contract for SQLite and PostgreSQL/pgvector. |
| Model | `model/*` | Provide structured JSON LLM clients and embedding adapters. |
| Runtime | `runtime/*` | Optional standalone Actor-Critic loop and local tools. |

## MCP Mode

MCP mode is the primary product path. The agent calls `get_memory_tool_manifest`, invokes `retrieve_memory_layer` one or more times with the layer and query choices it reasons are useful, creates the user-facing answer, then calls `consolidate_turn` after responding.

`consolidate_turn` wraps the MCP client agent's response as an `ActorResult`, sends it through `AutoEvaluationService`, then writes conversational, episodic, semantic, procedural, and failure memories through `AgenticMemoryService`.

For agentic retrieval orchestration, the MCP client agent is the LLM
orchestrator. Codex, Claude Code, Cline, or another MCP-compatible client agent can call
`get_memory_tool_manifest`, then invoke `retrieve_memory_layer` repeatedly to
plan and synthesize retrieval across conversational, semantic, semantic
hierarchy, episodic, procedural, and failure memory. The server does not add a
second hidden LLM provider for retrieval planning.

## Standalone Mode

Standalone mode is optional and requires an LLM provider. In this mode the project creates its own `Actor`, `Critic`, tool registry, planner, and memory service, then runs the six-phase loop in `runtime/self_learning_loop.py`.

## Evaluation

`core/evaluation_service.py` owns turn evaluation plumbing. In normal MCP mode,
the MCP client agent scores its completed answer and passes that typed score as
`agent_evaluation_json`. If the client omits the score, the server creates
temporary provisional scores so consolidation does not fail, returns
`needs_agent_rescore=true` plus `episode_id`, and expects the client to call
`rescore_episode(episode_id, agent_evaluation_json)` with the proper agent
score. The callback updates the same episodic record instead of creating a
duplicate turn. `consolidate_turn` and `rescore_episode` return
`scoring_source` so callers can tell whether the score came from
`mcp_client_agent` or `heuristic_provisional`.

In normal MCP mode, implicit preference mining belongs to the MCP client agent,
not to local English regexes in the server. `get_memory_tool_manifest` includes
a customizable `memory_mining` prompt that Codex, Claude Code, Cline, or another
client agent can use after a turn. The client reviews user behavior, corrections,
accepted workflows, rejected formats, tool outcomes, and retrieved memory, then
passes typed facts through `consolidate_turn(..., semantic_facts_json=...)`.
The deterministic evaluator packages those agent-mined facts into
`CriticEvaluation.new_semantic_facts`; it does not infer preferences from
hardcoded text patterns. `MEMORY_MINING_PROMPT` can override the default
manifest prompt without moving mining back into the server.

Provisional scoring is deliberately not authoritative. When it is used, the
episode is marked with `evaluation_source="heuristic_provisional"` and
`needs_agent_rescore=true`; deferred learning that depends on a trusted score,
such as saving a successful procedural workflow, is applied only after the MCP
client agent calls `rescore_episode`.

## Retrieval

The agentic retrieval path uses `AgenticRetrievalOrchestrator` as a server-side toolkit, not as a hidden reasoner. It retrieves exactly the layer requested by the MCP client agent and re-ranks records using similarity, recency, confidence, outcome quality, workflow maturity, hierarchy-node abstraction level, and pinning.

`get_session_context` remains as a backward-compatible quick path. Its fallback planner combines keyword triggers with vector search and adaptive per-session weights. Feedback weights are persisted through the store contract, so fallback retrieval tuning survives process restarts.

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
  -> upsert root node parent_id=null
  -> upsert facet node parent_id=root.node_id
  -> merge fact into summary child parent_id=facet.node_id
  -> merge fact into Q&A child parent_id=facet.node_id
```

Parent and child nodes are created deterministically from `node_key`, not from
random insertion order. `_stable_node_id(node_key)` turns each key into the same
UUID-shaped id every time, so rebuilding the hierarchy recreates the same graph.
When a fact is consolidated, `_consolidate_semantic_hierarchy` first ensures the
single `root` node exists. It then classifies the fact into one facet with
`_semantic_facet`, creates or updates `facet:{facet}` under the root, and creates
or updates two children under that facet: `summary:{facet}` and
`qa:{facet}:{question_hash}`.

The hierarchy is aggregate-oriented rather than one-node-per-fact. Existing
nodes are upserted by `node_key`: `source_fact_ids` gains the new fact id once,
summary content appends a bounded bullet line, Q&A answers merge into a compact
semicolon-separated answer list, confidence becomes the max of existing and new
fact confidence, and the node embedding is regenerated from title, content, and
answer.

Example:

```text
Semantic fact:
  fact_id = fact_123
  fact_type = preference
  content = "User preference: I prefer concise bullet-list answers."
  confidence_score = 0.90

Facet classification:
  communication_preferences

Nodes upserted:
  root
    node_key = "root"
    parent_id = null
    source_fact_ids includes fact_123

  facet
    node_key = "facet:communication_preferences"
    parent_id = root.node_id
    title = "Communication Preferences"
    source_fact_ids includes fact_123

  summary child
    node_key = "summary:communication_preferences"
    parent_id = facet.node_id
    title = "Communication Preferences Summary"
    content includes "- User preference: I prefer concise bullet-list answers."
    source_fact_ids includes fact_123

  Q&A child
    node_key = "qa:communication_preferences:{stable_hash}"
    parent_id = facet.node_id
    question = "What user preference is known for communication preferences?"
    answer includes "User preference: I prefer concise bullet-list answers."
    source_fact_ids includes fact_123
```

This means the canonical semantic record remains the flat fact. The hierarchy
does not own truth; it is a retrieval index over active semantic records.
`retrieve_memory_layer(..., layer="semantic_hierarchy")` lets Codex, Claude
Code, Cline, or another MCP client explicitly use broad semantic context during
agentic retrieval orchestration. The fallback quick-path planner also searches
semantic hierarchy nodes when semantic memory is triggered.

## Memory Governance

Semantic facts can be deleted, pinned, marked stale, exported, and imported. Pinned facts bypass semantic TTL filtering. Episodic memories remain append-only by default, but old episodes can be pruned while failure records are detached and preserved.

Semantic hierarchy nodes are derived data. When semantic facts are deleted,
pinned, marked stale, or imported through the MCP management tools, the
hierarchy is rebuilt from the currently active semantic facts so stale derived
content is not retained.

## Testing

The offline evaluation harness lives at `tests/evaluation_harness.py` and uses SQLite plus hash embeddings. It covers heuristic evaluation, MCP-client orchestrated retrieval, semantic deduplication, semantic hierarchy creation and retrieval, pin/stale behavior, persisted fallback planner feedback, context rendering, and failure recall.

Run it with:

```bash
python -m tests.evaluation_harness
```
