# Agentic Memory

Persistent, self-learning memory for AI coding agents. Agentic Memory runs as an
MCP server for Claude Code, Cline, Codex, and other MCP-compatible assistants.

The default mode needs no LLM API key. Your assistant remains the Actor: it
reasons, calls tools, and writes the user-facing response. Agentic Memory stores
the turn, evaluates it with deterministic heuristics, and consolidates useful
signals into long-term memory. You can optionally add an LLM provider for a
stronger Critic or enable standalone Actor-Critic mode.

## What It Provides

- Multi-layer long-term memory for conversations, episodes, facts, workflows,
  and failures.
- Two-tool MCP workflow: retrieve context before answering, consolidate after
  answering.
- Optional external-orchestrator workflow where Codex, Claude Code, or another
  MCP client plans memory-layer tool calls directly.
- Deterministic zero-key mode for local use.
- Optional Anthropic, OpenAI, or Groq Critic for deeper evaluation.
- SQLite by default, with PostgreSQL/pgvector support for larger deployments.
- Hash embeddings for fast zero-dependency startup, or sentence-transformer
  embeddings for higher-quality retrieval.
- Semantic fact governance: delete, pin, stale, export, and import memories.

## Memory Loop

Agents use Agentic Memory with two calls:

```text
1. get_session_context(user_message, session_id)
   Returns relevant system rules, preferences, facts, workflows, recent
   conversation, similar episodes, and past failures.

2. The agent answers the user.

3. consolidate_turn(session_id, user_message, assistant_response, ...)
   Saves the turn, evaluates quality, extracts durable facts, learns successful
   workflows, records failures, and tunes retrieval weights.
```

The system does not move one memory record from one layer into another. It
derives higher-level records from a completed turn, keeps the source episode,
and links derived records back to that episode where useful.

## Memory Layers

| Layer | Stores | Created From | Notes |
|---|---|---|---|
| Conversational | Recent user and assistant messages | Every `consolidate_turn` call | Kept as a sliding window; older turns can be rolled into summaries. |
| Episodic | Full completed turns with prompt, response, tools, outcome, latency, and errors | Every completed turn | Append-only audit trail used for similar-past-episode recall. |
| Semantic | Durable preferences, environment facts, and system rules | Critic/heuristic fact extraction and `new_facts` | Deduplicated by embedding similarity; conflicts are resolved by source, confidence, and recency. |
| Semantic hierarchy | Facets, summaries, and Q&A nodes derived from semantic facts | Every saved or reinforced semantic fact | Deterministic hierarchical aggregates for broader preference/context retrieval. |
| Procedural | Reusable multi-step tool workflows | Successful turns with at least two successful tool calls | Upserted by workflow signature and promoted by repeated success. |
| Failure | Tool failures worth avoiding later | Critic-flagged failed tool calls | Linked to the source episode; preserved when old episodes are pruned. |

Typical derivation:

```text
completed turn
  -> conversational messages
  -> episodic record
  -> semantic facts, if durable facts/preferences/rules are found
  -> semantic hierarchy nodes, if semantic facts are saved or reinforced
  -> procedural workflow, if a successful repeatable tool chain is found
  -> failure record, if a tool failure should be remembered
```

## Retrieval Orchestration

The default quick path is still `get_session_context`, which uses the local
heuristic planner plus vector search.

For a CMA-style agentic retrieval loop, the LLM orchestrator is the MCP client
itself: Codex, Claude Code, Cline, or another assistant. Call
`get_memory_tool_manifest` to see the available memory tools, then call
`retrieve_memory_layer` one or more times with refined queries across
conversational, semantic, semantic hierarchy, episodic, procedural, and failure
memory. No additional internal LLM provider is required for retrieval planning.

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

This is the default mode. The external assistant is the Actor, and Agentic
Memory uses deterministic evaluation for fact extraction, quality scoring,
workflow detection, and failure recording.

```bash
.venv/bin/python mcp_server.py
```

### MCP Mode With LLM Critic

Set one provider to let an LLM Critic score the turn and extract richer
semantic facts:

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python mcp_server.py
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... .venv/bin/python mcp_server.py
LLM_PROVIDER=groq GROQ_API_KEY=gsk_... .venv/bin/python mcp_server.py
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
| `get_memory_tool_manifest` | none | Describe memory layers and retrieval-tool strategy for Codex/Claude-style external orchestration. |
| `get_session_context` | `user_message`, `session_id` | Retrieve memory-enriched context before the agent answers. |
| `retrieve_memory_layer` | `query`, `layer`, `session_id?`, `top_k?` | Query one memory layer so the external MCP client can plan multi-step retrieval. |
| `consolidate_turn` | `session_id`, `user_message`, `assistant_response`, `tool_calls_json?`, `new_facts?`, `failure_summary?`, `quality_score?`, `reasoning_summary?` | Save and learn from a completed turn. |
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
| `new_facts` | Explicit facts the agent wants to propose for semantic memory. |
| `failure_summary` | Summary of failures encountered during the turn. |
| `quality_score` | Agent self-score from 0 to 10; used as a strong quality signal. |
| `reasoning_summary` | Brief approach summary stored in episodic memory. |

Example:

```json
{
  "session_id": "session-1",
  "user_message": "Fix the auth bug",
  "assistant_response": "I found and fixed the token validation issue.",
  "quality_score": 8.5,
  "reasoning_summary": "Reproduced the failure, patched token validation, and verified tests."
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

The harness covers heuristic evaluation, semantic deduplication, pin/stale
behavior, persisted planner feedback, context rendering, and failure recall.

## Project Structure

```text
+-- mcp_server.py              # Primary MCP server
+-- main.py                    # FastAPI admin/debug API
+-- config.py                  # Environment settings and JSON logging
+-- core/                      # Evaluation, consolidation, and Pydantic models
+-- planner/                   # Retrieval planning and feedback weights
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
