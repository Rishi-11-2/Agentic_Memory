# Agentic Memory

Persistent, multi-layer memory for AI coding agents. Designed as an **MCP server** for Claude Code, Codex, and other AI assistants.

## What It Does

Agentic Memory gives AI agents long-term memory across sessions. It stores, retrieves, and learns from past interactions using five specialized memory layers:

| Layer | Purpose | Example |
|---|---|---|
| **Conversational** | Sliding window of recent messages | Current chat history |
| **Episodic** | Full turn records with outcomes | "Last time I ran tests, 3 failed" |
| **Semantic** | Durable facts, preferences, rules | "User prefers TypeScript over JavaScript" |
| **Procedural** | Learned tool workflows | "Deploy = build → test → push → deploy" |
| **Failure** | Past tool failures for caution | "npm install failed due to permissions" |

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/agentic-memory.git
cd agentic-memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure for Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/agentic-memory",
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

### 3. Configure for Codex / OpenAI

Run the MCP server over HTTP:

```bash
MCP_TRANSPORT=http MCP_HTTP_PORT=8001 python mcp_server.py
```

## MCP Tools

| Tool | Parameters | Description |
|---|---|---|
| `get_session_context` | `user_message`, `session_id` | Build memory context for the current session |
| `consolidate_turn` | `session_id`, `user_message`, `assistant_response`, ... | Save a completed turn into all memory layers |
| `search_memory` | `query`, `layers?`, `top_k?` | Search across memory layers by similarity |
| `get_conversation_history` | `session_id`, `last_n?` | Get recent conversation messages |
| `clear_session_memory` | `session_id` | Clear conversation memory for a session |
| `inspect_memory_layers` | `limit?` | View memory layer counts and recent records |

## Architecture

```
┌─────────────────────────────────────────┐
│           MCP Server (FastMCP)          │
│  get_session_context, consolidate_turn  │
│  search_memory, inspect_memory_layers   │
├─────────────────────────────────────────┤
│         AgenticMemoryService            │
│  Context rendering, consolidation,      │
│  semantic dedup, conflict resolution    │
├─────────────────────────────────────────┤
│       HeuristicRetrievalPlanner         │
│  Multi-layer query planning             │
├─────────────────────────────────────────┤
│     MemoryStore (SQLite / PostgreSQL)   │
│  5 memory layer tables + vector search  │
├─────────────────────────────────────────┤
│         EmbeddingModel                  │
│  Sentence Transformers / Hash fallback  │
└─────────────────────────────────────────┘
```

## Self-Learning Loop

When configured with an LLM provider (Groq, Anthropic, OpenAI), the system runs an Actor-Critic loop:

1. **Retrieval Planning** — query all relevant memory layers
2. **Context Building** — render memory into a structured prompt
3. **Actor Execution** — LLM generates response + tool calls
4. **Critic Evaluation** — LLM scores the response across 5 dimensions
5. **Memory Consolidation** — save lessons into episodic, semantic, procedural, and failure memory
6. **Response Assembly** — return the final result

## REST API

A FastAPI server is available as a secondary interface:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Endpoints: `POST /chat`, `GET /memory/inspect`, `GET /debug/turn/{session_id}/{turn_index}`

## Configuration

All settings are loaded from environment variables or a `.env` file. See `.env.example` for the full reference.

### Key Settings

| Variable | Default | Description |
|---|---|---|
| `MEMORY_BACKEND` | `sqlite` | Storage backend: `sqlite` or `postgres` |
| `SQLITE_DB_PATH` | `agentic_memory.db` | Path to SQLite database file |
| `LLM_PROVIDER` | `deterministic` | LLM: `deterministic`, `groq`, `anthropic`, `openai` |
| `EMBEDDING_BACKEND` | `sentence-transformer` | Embeddings: `sentence-transformer` or `hash` |
| `MEMORY_WINDOW_TURNS` | `10` | Conversation window size |
| `SEMANTIC_DEDUP_THRESHOLD` | `0.92` | Similarity threshold for semantic dedup |
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` or `http` |

## Docker

```bash
docker compose up agentic-memory
```

For PostgreSQL with pgvector:

```bash
docker compose --profile postgres up
```

## Project Structure

```
├── mcp_server.py          # MCP server entry point
├── main.py                # FastAPI REST API
├── config.py              # Settings and logging
├── core/
│   ├── memory_service.py  # Context assembly and consolidation
│   └── models.py          # Pydantic schemas
├── store/
│   ├── base.py            # MemoryStore protocol
│   ├── sqlite_store.py    # SQLite implementation
│   └── postgres_store.py  # PostgreSQL + pgvector
├── model/
│   ├── embedding_model.py # Text embeddings
│   ├── anthropic_client.py
│   ├── groq_client.py
│   └── openai_client.py
├── planner/
│   └── retrieval_planner.py
├── runtime/
│   ├── actor.py           # Actor (tool execution)
│   ├── critic.py          # Critic (evaluation)
│   ├── self_learning_loop.py
│   └── tools.py           # Tool registry
└── demo/
    └── demo_main.py       # Offline demo
```

## License

MIT
