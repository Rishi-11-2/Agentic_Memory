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

## Standalone Mode

Standalone mode is optional and requires an LLM provider. In this mode the project creates its own `Actor`, `Critic`, tool registry, planner, and memory service, then runs the six-phase loop in `runtime/self_learning_loop.py`.

## Evaluation

`core/evaluation_service.py` owns turn evaluation. It can call an LLM Critic when one is configured, or use deterministic heuristics when no provider exists. Keeping this outside `mcp_server.py` lets MCP, HTTP, tests, and standalone flows share evaluation behavior.

## Retrieval

The planner combines keyword triggers with vector search and adaptive per-session weights. Feedback weights are persisted through the store contract, so retrieval tuning survives process restarts. Retrieved records are re-ranked before context rendering using similarity, recency, confidence, outcome quality, workflow maturity, and pinning.

## Memory Governance

Semantic facts can be deleted, pinned, marked stale, exported, and imported. Pinned facts bypass semantic TTL filtering. Episodic memories remain append-only by default, but old episodes can be pruned while failure records are detached and preserved.

## Testing

The offline evaluation harness lives at `tests/evaluation_harness.py` and uses SQLite plus hash embeddings. It covers heuristic evaluation, semantic deduplication, pin/stale behavior, persisted planner feedback, context rendering, and failure recall.

Run it with:

```bash
python -m tests.evaluation_harness
```
