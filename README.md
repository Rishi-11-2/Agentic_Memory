# Agentic Memory

Persistent, self-learning memory for AI coding agents. Agentic Memory runs as an
MCP server for Claude Code, Cline, Codex, and other MCP-compatible assistants.

The default mode needs no LLM API key. Your assistant remains the Actor and the
retrieval orchestrator: it reasons over which memory layers to query, performs
multi-hop retrieval through MCP tools, calls other tools, and writes the
user-facing response. The same MCP client agent scores the completed turn and
mines durable memory facts; Agentic Memory validates and consolidates those
signals into long-term memory. You can optionally add an LLM provider for
standalone Actor-Critic mode.

## What It Provides

- Multi-layer long-term memory for conversations, episodes, facts, semantic
  hierarchy summaries, workflows, and failures.
- MCP-client orchestrated retrieval where Codex, Claude Code, Cline, or another
  client plans memory-layer calls directly.
- Backward-compatible quick context tool for clients that do not need multi-hop
  retrieval.
- Deterministic zero-key mode for local use.
- Optional Anthropic, OpenAI, or Groq provider for standalone Actor-Critic mode.
- SQLite by default, with PostgreSQL/pgvector support for larger deployments.
- Hash embeddings for fast zero-dependency startup, or sentence-transformer
  embeddings for higher-quality retrieval.
- Semantic fact governance: delete, pin, stale, export, and import memories.

## Architecture

Agentic Memory is an MCP-native memory substrate. In the default path, the
assistant using the MCP server remains the Actor and retrieval orchestrator:
Claude Code, Codex, Cline, or another client decides what context it needs,
answers the user, and then asks Agentic Memory to consolidate the completed
turn.

```text
MCP client (Codex / Claude Code / Cline)
  -> get_memory_tool_manifest
  -> retrieve_memory_layer one or more times with refined queries
  -> user-facing reasoning and tool use
  -> consolidate_turn
  -> rescore_episode if consolidate_turn reports needs_agent_rescore
  -> Agentic Memory validates, stores, aggregates, and tunes retrieval
```

The codebase is split into these parts:

| Area | Modules | Role |
|---|---|---|
| Adapter | `mcp_server.py`, `main.py` | Expose MCP tools and a small REST/debug API. |
| Core | `core/memory_service.py`, `core/evaluation_service.py`, `core/models.py` | Build context, evaluate turns, consolidate memory, and define schemas. |
| Planner | `planner/retrieval_planner.py` | Expose MCP-client orchestrated retrieval tools, ranking, and a fallback heuristic context builder. |
| Store | `store/*` | Persist memory layers in SQLite or PostgreSQL/pgvector. |
| Model | `model/*` | Provide embeddings and optional structured LLM clients for Critic/standalone mode. |
| Runtime | `runtime/*` | Optional provider-backed standalone Actor-Critic loop and local tools. |

No internal LLM provider is required for normal MCP retrieval orchestration.
Provider-backed LLMs are only used when you enable standalone Actor-Critic mode.

## Memory Loop

Agents use Agentic Memory with an agentic retrieval loop:

```text
1. get_memory_tool_manifest()
   Inspect the available memory layers and retrieval contract.

2. retrieve_memory_layer(query, layer, session_id?, top_k?)
   Call this one or more times. The MCP client agent decides the layers,
   query refinements, and whether another hop is needed.

3. The agent answers the user.

4. consolidate_turn(session_id, user_message, assistant_response, ...)
   Saves the turn, records the quality evaluation, stores agent-mined durable
   facts, learns successful workflows, records failures, and tunes retrieval
   weights.

5. rescore_episode(episode_id, agent_evaluation_json), only when requested
   Replaces a provisional score with the MCP client agent's final score without
   creating a duplicate memory turn.
```

`get_session_context(user_message, session_id)` remains as a convenience fallback
for clients that want a single assembled context block.

Quality scoring is owned by the MCP client agent. Codex, Claude Code, Cline, or
another MCP client scores its completed answer and passes `agent_evaluation_json`
to `consolidate_turn`. If that score is missing, the server invents provisional
scores so consolidation does not fail, returns `needs_agent_rescore=true` plus
`episode_id`, and expects the client to call `rescore_episode` with the final
agent score. Provisional scores never update the fallback planner's persisted
retrieval-feedback weights; those are applied only after the final agent score.

The system does not move one memory record from one layer into another. It
derives higher-level records from a completed turn, keeps the source episode,
and links derived records back to that episode where useful.

## Memory Layers

| Layer | Stores | Created From | Notes |
|---|---|---|---|
| Conversational | Recent user and assistant messages | Every `consolidate_turn` call | Kept as a sliding window; older turns can be rolled into summaries. |
| Episodic | Full completed turns with prompt, response, tools, outcome, latency, and errors | Every completed turn | Append-only audit trail used for similar-past-episode recall. |
| Semantic | Durable preferences, environment facts, and system rules | MCP-client mined `semantic_facts_json`, legacy `new_facts`, and standalone loop Critic output | Deduplicated by embedding similarity; conflicts are resolved by source, confidence, and recency. |
| Semantic hierarchy | Facets, summaries, and Q&A nodes derived from semantic facts | Every saved or reinforced semantic fact | Deterministic hierarchical aggregates for broader preference/context retrieval. |
| Procedural | Reusable multi-step tool workflows | Successful turns with at least two successful tool calls | Upserted by workflow signature and promoted by repeated success. |
| Failure | Tool failures worth avoiding later | Critic-flagged failed tool calls | Linked to the source episode; preserved when old episodes are pruned. |

Typical derivation:

```text
completed turn
  -> conversational messages
  -> episodic record
  -> semantic facts, if the MCP client agent proposes durable facts/preferences/rules
  -> semantic hierarchy nodes, if semantic facts are saved or reinforced
  -> procedural workflow, if a successful repeatable tool chain is found
  -> failure record, if a tool failure should be remembered
```

## Retrieval Orchestration

The primary retrieval path is MCP-client orchestrated:

1. `get_memory_tool_manifest` describes the retrieval contract and available
   layers.
2. `retrieve_memory_layer` retrieves exactly the layer requested by the MCP
   client agent.
3. Codex, Claude Code, Cline, or another MCP client decides the query order,
   performs any multi-hop refinements, reconciles conflicts, and synthesizes the
   answer.

`get_session_context` is a backward-compatible quick path. It uses the local
heuristic fallback planner plus vector search to assemble one context block, but
it is not the CMA-style agentic path.

For a CMA-style retrieval loop, Codex or Claude Code can do this:

```text
user query
  -> inspect available memory tools with get_memory_tool_manifest
  -> query semantic_hierarchy for broad preferences and project context
  -> query semantic for precise durable facts
  -> query episodic for similar prior turns
  -> query procedural for reusable workflows
  -> query failure for known hazards
  -> synthesize the final answer using only grounded memory
```

Simple queries can use one layer. More complex or longitudinal questions can
trigger several `retrieve_memory_layer` calls with refined queries. Agentic
Memory provides the memory tools and ranking; the MCP client provides the
reasoning loop. There is no hidden planner LLM inside the MCP server for normal
retrieval orchestration.

## Semantic Hierarchy

Flat semantic facts remain the canonical source of truth. The semantic
hierarchy is a derived retrieval index built from those facts.

Each saved or reinforced semantic fact updates:

```text
root
  -> facet
  -> facet summary
  -> Q&A node
```

Hierarchy nodes are stored in `am_semantic_hierarchy_nodes` and represented by
`SemanticHierarchyNode`. They include deterministic keys, parent links, facet
names, source fact IDs, embeddings, confidence, and timestamps.

| Node type | Purpose |
|---|---|
| `root` | Top-level index for semantic memory. |
| `facet` | Groups facts into facets such as communication preferences, workflow preferences, project context, environment, and general knowledge. |
| `summary` | Compact bottom-up summary for a facet. |
| `qa` | Q&A-style retrieval node grounded in one or more semantic facts. |

When semantic facts are deleted, pinned, marked stale, or imported through MCP
management tools, the hierarchy is rebuilt from active semantic facts so stale
derived content does not linger.

## Install

Use Python 3.13 for the virtual environment. Python 3.14 can fail while
building older native dependencies such as `pydantic-core`.

```bash
git clone https://github.com/Rishi-11-2/Agentic_Memory.git
cd Agentic_Memory
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the fastest local setup, use hash embeddings:

```bash
export MEMORY_BACKEND=sqlite
export SQLITE_DB_PATH=agentic_memory.db
export EMBEDDING_BACKEND=hash
export LLM_PROVIDER=deterministic
```

## MCP Client Setup

Replace `/absolute/path/to/Agentic_Memory` with your local checkout path.

### Claude Code

Add this to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "/absolute/path/to/Agentic_Memory/.venv/bin/python",
      "args": ["/absolute/path/to/Agentic_Memory/mcp_server.py"],
      "cwd": "/absolute/path/to/Agentic_Memory",
      "env": {
        "MEMORY_BACKEND": "sqlite",
        "SQLITE_DB_PATH": "agentic_memory.db",
        "EMBEDDING_BACKEND": "hash",
        "LLM_PROVIDER": "deterministic"
      }
    }
  }
}
```

### Codex

```bash
codex mcp add agentic-memory \
  --env MEMORY_BACKEND=sqlite \
  --env SQLITE_DB_PATH=agentic_memory.db \
  --env EMBEDDING_BACKEND=hash \
  --env LLM_PROVIDER=deterministic \
  -- /absolute/path/to/Agentic_Memory/.venv/bin/python /absolute/path/to/Agentic_Memory/mcp_server.py
```

Verify registration:

```bash
codex mcp
```

You can also edit `~/.codex/config.toml` directly:

```toml
[mcp_servers.agentic-memory]
command = "/absolute/path/to/Agentic_Memory/.venv/bin/python"
args = ["/absolute/path/to/Agentic_Memory/mcp_server.py"]

[mcp_servers.agentic-memory.env]
MEMORY_BACKEND = "sqlite"
SQLITE_DB_PATH = "agentic_memory.db"
EMBEDDING_BACKEND = "hash"
LLM_PROVIDER = "deterministic"
```

### Cline

Open the Cline MCP settings file and add the stdio server:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "/absolute/path/to/Agentic_Memory/.venv/bin/python",
      "args": ["/absolute/path/to/Agentic_Memory/mcp_server.py"],
      "cwd": "/absolute/path/to/Agentic_Memory",
      "env": {
        "MEMORY_BACKEND": "sqlite",
        "SQLITE_DB_PATH": "agentic_memory.db",
        "EMBEDDING_BACKEND": "hash",
        "LLM_PROVIDER": "deterministic"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

To skip approval prompts for the common memory calls:

```json
"autoApprove": ["get_session_context", "get_memory_tool_manifest", "retrieve_memory_layer", "consolidate_turn", "search_memory"]
```

For clients that need HTTP/SSE, start the MCP server with HTTP transport:

```bash
MCP_TRANSPORT=http MCP_HTTP_PORT=8001 .venv/bin/python mcp_server.py
```

Then configure the client URL as:

```text
http://localhost:8001/sse
```

## Operating Modes

### MCP Mode, Zero Key

This is the default mode. The MCP client agent is the Actor, retrieval
orchestrator, implicit preference miner, and turn scorer. Agentic Memory uses
client-supplied evaluation when provided; if it is missing, it writes temporary
provisional scores and returns `needs_agent_rescore=true`.

```bash
.venv/bin/python mcp_server.py
```

### Standalone Actor-Critic Mode

In standalone mode, Agentic Memory generates the response, evaluates it, and
learns from it. This requires a real LLM provider.

```bash
LLM_PROVIDER=anthropic \
ANTHROPIC_API_KEY=sk-ant-... \
STANDALONE_LOOP_ENABLED=true \
.venv/bin/python mcp_server.py
```

This enables the extra MCP tool `run_autonomous_turn`.

## MCP Tools

| Tool | Parameters | Purpose |
|---|---|---|
| `get_memory_tool_manifest` | none | Describe the MCP-client agentic retrieval contract and available memory layers. |
| `get_session_context` | `user_message`, `session_id` | Retrieve a fallback memory-enriched context block before the agent answers. |
| `retrieve_memory_layer` | `query`, `layer`, `session_id?`, `top_k?` | Query one memory layer selected by the MCP client agent for multi-step retrieval. |
| `consolidate_turn` | `session_id`, `user_message`, `assistant_response`, `tool_calls_json?`, `semantic_facts_json?`, `agent_evaluation_json?`, `new_facts?`, `failure_summary?`, `quality_score?`, `reasoning_summary?` | Save and learn from a completed turn. |
| `rescore_episode` | `episode_id`, `agent_evaluation_json`, `session_id?` | Replace a provisional score with the MCP client agent's final score. |
| `search_memory` | `query`, `layers?`, `top_k?` | Search semantic, semantic hierarchy, episodic, procedural, and failure memory by similarity. |
| `get_conversation_history` | `session_id`, `last_n?` | Fetch recent conversation messages. |
| `clear_session_memory` | `session_id` | Clear only the session's conversational window. |
| `inspect_memory_layers` | `limit?` | Show counts and recent records by layer. |
| `inspect_episode` | `episode_id` | Fetch one full episodic record. |
| `delete_semantic_fact` | `fact_id` | Delete one semantic fact. |
| `pin_semantic_fact` | `fact_id`, `pinned?` | Keep or unkeep a fact outside TTL filtering. |
| `mark_semantic_fact_stale` | `fact_id`, `stale_days?` | Age a semantic fact for TTL testing or cleanup. |
| `prune_old_episodes` | `older_than_days?` | Delete old episodic records while preserving detached failures. |
| `export_memory` | `layers?`, `limit_per_layer?` | Export recent memory records as JSON. |
| `import_memory` | `memory_json`, `import_semantic?` | Import semantic facts from an export payload. |
| `run_autonomous_turn` | `user_message`, `session_id` | Run the full standalone Actor-Critic loop when enabled. |

### `consolidate_turn` Details

Required fields:

| Parameter | Description |
|---|---|
| `session_id` | Stable conversation/session identifier. |
| `user_message` | Original user request. |
| `assistant_response` | Final response produced by the agent. |

Optional fields:

| Parameter | Description |
|---|---|
| `tool_calls_json` | JSON array of tool calls. Each item can include `tool_name`, `input_parameters`, `output_summary`, `success`, `latency_ms`, `error_trace`, and `critic_flagged`. |
| `semantic_facts_json` | Preferred typed JSON array of durable facts mined by the MCP client agent. Each item uses `fact_type`, `content`, `confidence`, and `source`. |
| `agent_evaluation_json` | Preferred typed 0-10 scoring JSON produced by the MCP client agent. If omitted, server uses temporary provisional scores and returns `needs_agent_rescore=true`. |
| `new_facts` | Legacy untyped fact strings the agent wants to propose for semantic memory. |
| `failure_summary` | Summary of failures encountered during the turn. |
| `quality_score` | Deprecated compatibility field; use `agent_evaluation_json`. |
| `reasoning_summary` | Brief approach summary stored in episodic memory. |

The response includes `episode_id`, `critic_score`, `critic_passed`,
`scoring_source`, and `needs_agent_rescore`. `scoring_source` is
`mcp_client_agent` when the client provided scoring and
`heuristic_provisional` when the server had to invent temporary scores.

Example:

```json
{
  "session_id": "session-1",
  "user_message": "Fix the auth bug",
  "assistant_response": "I found and fixed the token validation issue.",
  "semantic_facts_json": "[{\"fact_type\":\"preference\",\"content\":\"User prefers concise implementation summaries with verification commands.\",\"confidence\":0.78,\"source\":\"llm_inferred\"}]",
  "agent_evaluation_json": "{\"factual_accuracy\":8,\"preference_adherence\":8,\"tool_efficiency\":8,\"hallucination_risk\":8,\"workflow_quality\":8,\"save_workflow\":false,\"failure_summary\":null}",
  "reasoning_summary": "Reproduced the failure, patched token validation, and verified tests."
}
```

### `rescore_episode` Details

Call this when `consolidate_turn` returns `needs_agent_rescore=true`. The MCP
client agent supplies `agent_evaluation_json`, and Agentic Memory updates the
existing episode's evaluation metadata instead of writing another turn.

```json
{
  "episode_id": "episode-id-from-consolidate-turn",
  "agent_evaluation_json": "{\"factual_accuracy\":9,\"preference_adherence\":8,\"tool_efficiency\":9,\"hallucination_risk\":8,\"workflow_quality\":9,\"save_workflow\":true}",
  "session_id": "session-1"
}
```

## Configuration

Settings come from environment variables or `.env`. See `.env.example` for the
full list.

| Variable | Default | Description |
|---|---|---|
| `MEMORY_BACKEND` | `sqlite` | `sqlite` or `postgres`. |
| `SQLITE_DB_PATH` | `agentic_memory.db` | SQLite database path. |
| `POSTGRES_DSN` | `postgresql://postgres:postgres@localhost:5432/agentic_memory` | PostgreSQL connection string. |
| `LLM_PROVIDER` | `deterministic` | `deterministic`, `groq`, `anthropic`, or `openai`. |
| `EMBEDDING_BACKEND` | `sentence-transformer` | `sentence-transformer` or `hash`. |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `http`. |
| `MCP_HTTP_PORT` | `8001` | HTTP/SSE port when `MCP_TRANSPORT=http`. |
| `STANDALONE_LOOP_ENABLED` | `false` | Enables `run_autonomous_turn`; requires an LLM provider. |
| `MEMORY_WINDOW_TURNS` | `10` | Number of recent turns kept in conversational memory. |
| `SEMANTIC_DEDUP_THRESHOLD` | `0.92` | Similarity threshold for semantic deduplication. |
| `SEMANTIC_MEMORY_TTL_DAYS` | `180` | Semantic freshness window; pinned facts bypass TTL filtering. |
| `FAILURE_SIMILARITY_THRESHOLD` | `0.80` | Similarity threshold for failure recall. |
| `MEMORY_MINING_PROMPT` | empty | Optional custom prompt exposed in `get_memory_tool_manifest` to guide MCP-client preference mining. |
| `TOOL_WORKSPACE_ROOT` | `.` | Workspace root for standalone runtime tools. |

## Storage Backends

SQLite is the default and works well for local agents:

```bash
MEMORY_BACKEND=sqlite SQLITE_DB_PATH=agentic_memory.db .venv/bin/python mcp_server.py
```

PostgreSQL uses `schema.sql` and pgvector indexes:

```bash
MEMORY_BACKEND=postgres \
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/agentic_memory \
.venv/bin/python mcp_server.py
```

## REST API And Docker

`main.py` exposes a small FastAPI admin/debug API alongside the MCP server.

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Docker Compose starts the REST API by default:

```bash
docker compose up agentic-memory
```

Start PostgreSQL with pgvector:

```bash
docker compose --profile postgres up
```

## Demo And Tests

Run the offline demo with SQLite and deterministic evaluation:

```bash
python -m demo.demo_main
```

Run the behavioral harness:

```bash
python -m tests.evaluation_harness
```

The harness covers heuristic evaluation, MCP-client orchestrated retrieval,
semantic deduplication, semantic hierarchy creation and retrieval, pin/stale
behavior, persisted fallback planner feedback, context rendering, and failure
recall.

## Project Structure

```text
+-- mcp_server.py              # Primary MCP server
+-- main.py                    # FastAPI admin/debug API
+-- config.py                  # Environment settings and JSON logging
+-- core/                      # Evaluation, consolidation, and Pydantic models
+-- planner/                   # Client-orchestrated retrieval and fallback planning
+-- store/                     # SQLite and PostgreSQL memory stores
+-- model/                     # Embeddings and LLM provider clients
+-- runtime/                   # Optional standalone Actor-Critic loop and tools
+-- demo/                      # Offline/LLM demo
+-- docs/architecture.md       # Contributor-facing architecture notes
+-- tests/evaluation_harness.py
+-- schema.sql                 # PostgreSQL + pgvector schema
+-- .env.example               # Full configuration reference
+-- .mcp.json                  # Example MCP client config
```

For deeper implementation notes, see `docs/architecture.md`.

## License

MIT
