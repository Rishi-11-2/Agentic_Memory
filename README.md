# Agentic Memory

Persistent, self-learning memory for AI coding agents. Ships as an **MCP server** for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Codex](https://openai.com/index/codex/), and other MCP-compatible assistants.

---

## What It Does

Agentic Memory gives AI agents **long-term memory that learns**. Every interaction is stored, evaluated, and consolidated into five specialized memory layers:

| Layer | What It Stores | Example |
|---|---|---|
| **Conversational** | Sliding window of recent messages | Current chat context |
| **Episodic** | Complete turn records with outcomes | "Last time I ran tests, 3 failed on auth" |
| **Semantic** | Durable facts, preferences, rules | "User prefers TypeScript over JavaScript" |
| **Procedural** | Learned multi-step tool workflows | "Deploy = build → test → push → deploy" |
| **Failure** | Past tool failures for avoidance | "pip install failed — use pip3 instead" |

---

## How Agents Use It

The agent calls **two MCP tools** — the self-learning loop runs automatically inside `consolidate_turn`:

```
1. get_session_context(message, session_id)
   → Returns relevant memories as context

2. Agent generates its response using the memory context

3. consolidate_turn(session_id, message, response, ...)
   → Auto-evaluates quality (Critic LLM if configured)
   → Extracts and deduplicates semantic facts
   → Learns reusable tool workflows
   → Records failures for future avoidance
```

No special prompting needed. The tools are self-describing — Claude Code and Codex will discover and use them automatically.

---

## Quick Start

### Install

```bash
git clone https://github.com/Rishi-11-2/Agentic_Memory.git
cd Agentic_Memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/absolute/path/to/Agentic_Memory",
      "env": {
        "MEMORY_BACKEND": "sqlite",
        "SQLITE_DB_PATH": "memory.db",
        "EMBEDDING_BACKEND": "hash"
      }
    }
  }
}
```

### Codex

Start the MCP server over HTTP:

```bash
MCP_TRANSPORT=http MCP_HTTP_PORT=8001 python mcp_server.py
```

Connect Codex to the MCP endpoint at `http://localhost:8001`.

### Enable Auto-Critic (optional)

Set an LLM provider so the system automatically scores responses and extracts facts:

```bash
# Any one of:
LLM_PROVIDER=groq     GROQ_API_KEY=gsk_...       python mcp_server.py
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python mcp_server.py
LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...       python mcp_server.py
```

Without an LLM, the system still learns from tool success/failure signals and any facts the agent supplies.

---

## MCP Tools

| Tool | Parameters | What It Does |
|---|---|---|
| `get_session_context` | `user_message`, `session_id` | Retrieve relevant memories before responding |
| `consolidate_turn` | `session_id`, `user_message`, `assistant_response`, `tool_calls_json?`, `new_facts?`, `failure_summary?` | Save a turn and auto-learn from it |
| `search_memory` | `query`, `layers?`, `top_k?` | Search across memory layers by similarity |
| `get_conversation_history` | `session_id`, `last_n?` | Fetch recent conversation messages |
| `clear_session_memory` | `session_id` | Clear a session's conversation history |
| `inspect_memory_layers` | `limit?` | View memory layer counts and recent records |

---

## Architecture

```
┌──────────────────────────────────────────┐
│          MCP Server  (FastMCP)           │
│                                          │
│  get_session_context   consolidate_turn  │
│  search_memory         clear_session     │
│  get_conversation_history                │
│  inspect_memory_layers                   │
├──────────────────────────────────────────┤
│        Auto-Critic  (optional LLM)       │
│  Quality scoring · fact extraction       │
├──────────────────────────────────────────┤
│         AgenticMemoryService             │
│  Context rendering · consolidation       │
│  semantic dedup · conflict resolution    │
├──────────────────────────────────────────┤
│       HeuristicRetrievalPlanner          │
│  Keyword + density triggers              │
│  per-session weight tuning               │
├──────────────────────────────────────────┤
│    MemoryStore  (SQLite / PostgreSQL)    │
│  5 memory tables · vector similarity     │
├──────────────────────────────────────────┤
│          EmbeddingModel                  │
│  Sentence Transformers / hash fallback   │
└──────────────────────────────────────────┘
```

---

## Configuration

All settings come from environment variables or a `.env` file. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `MEMORY_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `SQLITE_DB_PATH` | `agentic_memory.db` | SQLite database path |
| `LLM_PROVIDER` | `deterministic` | Auto-critic LLM: `deterministic` / `groq` / `anthropic` / `openai` |
| `EMBEDDING_BACKEND` | `sentence-transformer` | `sentence-transformer` (384d) or `hash` (256d, zero deps) |
| `MCP_TRANSPORT` | `stdio` | `stdio` (Claude Code) or `http` (Codex) |
| `MCP_HTTP_PORT` | `8001` | Port when using HTTP transport |
| `MEMORY_WINDOW_TURNS` | `10` | Conversation sliding window size |
| `SEMANTIC_DEDUP_THRESHOLD` | `0.92` | Cosine threshold for fact deduplication |

---

## Docker

```bash
# SQLite (default)
docker compose up agentic-memory

# PostgreSQL + pgvector
docker compose --profile postgres up
```

---

## Project Structure

```
├── mcp_server.py              # Primary entry point (MCP server)
├── main.py                    # Secondary REST API (admin/debug)
├── config.py                  # Settings and structured logging
├── core/
│   ├── memory_service.py      # Context assembly + memory consolidation
│   └── models.py              # Pydantic schemas for all memory layers
├── store/
│   ├── base.py                # MemoryStore protocol
│   ├── factory.py             # Backend factory (sqlite/postgres)
│   ├── sqlite_store.py        # SQLite + client-side vector search
│   └── postgres_store.py      # PostgreSQL + pgvector
├── model/
│   ├── embedding_model.py     # Sentence Transformers / hash embeddings
│   ├── anthropic_client.py    # Anthropic structured LLM adapter
│   ├── groq_client.py         # Groq structured LLM adapter
│   └── openai_client.py       # OpenAI structured LLM adapter
├── planner/
│   └── retrieval_planner.py   # Heuristic multi-layer query planner
├── runtime/
│   ├── actor.py               # Actor (standalone loop mode)
│   ├── critic.py              # Critic (auto-evaluation in MCP + standalone)
│   ├── self_learning_loop.py  # Full Actor-Critic loop orchestration
│   └── tools.py               # Tool registry (calculator, web search, etc.)
├── schema.sql                 # PostgreSQL schema with pgvector
├── demo/
│   └── demo_main.py           # Offline 3-turn demo
├── .env.example               # All configuration options
└── .mcp.json                  # Example MCP server configuration
```

---

## License

MIT
