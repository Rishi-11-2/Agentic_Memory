# Agentic Memory

Persistent, self-learning memory for AI coding agents. Ships as an **MCP server** for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Cline](https://github.com/cline/cline), [Codex](https://openai.com/index/codex/), and other MCP-compatible assistants.

**No LLM API keys required.** The AI agent (Claude, Codex, etc.) IS the Actor — it reasons, picks tools, and generates responses. The memory system learns from every interaction using enhanced heuristics. Optionally add an LLM provider for deeper auto-evaluation or standalone mode.

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
   → Auto-evaluates quality (enhanced heuristics or LLM Critic)
   → Extracts preferences and facts from user messages
   → Learns reusable tool workflows
   → Records failures for future avoidance
```

No special prompting needed. The tools are self-describing — Claude Code, Cline, and Codex will discover and use them automatically.

### Agent Self-Score

The agent can optionally rate its own response quality via `quality_score` (0–10). This gives the system a much richer learning signal than heuristics alone:

```json
consolidate_turn(
  session_id="session-1",
  user_message="Fix the auth bug",
  assistant_response="I found and fixed the issue in auth.py...",
  quality_score=8.5,
  reasoning_summary="Found root cause in token validation, applied fix and verified tests pass"
)
```

---

## Quick Start

### Install

```bash
git clone https://github.com/Rishi-11-2/Agentic_Memory.git
cd Agentic_Memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Claude Code (Zero Config)

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

That's it. No API keys needed — the enhanced heuristic Critic automatically extracts preferences, learns facts, scores quality, and saves workflows.

### Cline (VS Code)

1. Open VS Code → Cline panel → click the **MCP Servers** icon → **Configure** tab → **Configure MCP Servers**.  
   This opens `cline_mcp_settings.json`.

2. Add the agentic-memory server:

**Option A — stdio (recommended, runs locally):**

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
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

**Option B — HTTP/SSE (connect to a running server):**

First start the server:
```bash
MCP_TRANSPORT=http MCP_HTTP_PORT=8001 python mcp_server.py
```

Then in `cline_mcp_settings.json`:
```json
{
  "mcpServers": {
    "agentic-memory": {
      "url": "http://localhost:8001/sse",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

3. Save the file and restart Cline (click **Restart** in the MCP Servers panel).  
   The memory tools will appear in Cline's tool list.

> **Tip:** To auto-approve memory tools without confirmation prompts, add their names to the `autoApprove` array:
> ```json
> "autoApprove": ["get_session_context", "consolidate_turn", "search_memory"]
> ```

### Codex

**Option A — CLI command:**

```bash
codex mcp add agentic-memory \
  --env MEMORY_BACKEND=sqlite \
  --env SQLITE_DB_PATH=memory.db \
  --env EMBEDDING_BACKEND=hash \
  -- python /absolute/path/to/Agentic_Memory/mcp_server.py
```

**Option B — Edit `~/.codex/config.toml` directly:**

```toml
[mcp_servers.agentic-memory]
command = "python"
args = ["/absolute/path/to/Agentic_Memory/mcp_server.py"]

[mcp_servers.agentic-memory.env]
MEMORY_BACKEND = "sqlite"
SQLITE_DB_PATH = "memory.db"
EMBEDDING_BACKEND = "hash"
```

For project-scoped config, create `.codex/config.toml` in your project root instead.

Verify the server is registered:

```bash
codex mcp
```

---

## Operating Modes

### 1. MCP Mode (Default — No API Key)

The AI agent IS the Actor. The system uses enhanced heuristic evaluation:

- **Preference detection** — Automatically extracts "I prefer...", "always use...", "never do..." patterns
- **Fact extraction** — Learns about your environment ("I use Python 3.12", "our stack runs on AWS")
- **Tool efficiency scoring** — Scores based on tool success/failure rates
- **Response quality analysis** — Detects suspiciously short responses, structured content, error patterns
- **Agent self-score** — When the agent provides `quality_score`, it's used as the primary signal

```bash
# Zero config — works immediately
python mcp_server.py
```

### 2. MCP Mode + LLM Critic (One API Key)

Add an LLM provider for deeper auto-evaluation. The Critic LLM scores responses across 5 quality dimensions and extracts semantic facts automatically:

```bash
# Any one of:
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python mcp_server.py
LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...       python mcp_server.py
LLM_PROVIDER=groq     GROQ_API_KEY=gsk_...        python mcp_server.py
```

### 3. Standalone Mode (One API Key + Flag)

The system acts as BOTH Actor and Critic — it generates responses using its own LLM, evaluates them, and learns. Enable with one API key and a flag:

```bash
LLM_PROVIDER=anthropic \
ANTHROPIC_API_KEY=sk-ant-... \
STANDALONE_LOOP_ENABLED=true \
python mcp_server.py
```

This exposes an additional MCP tool `run_autonomous_turn(user_message, session_id)` that runs the full 6-phase self-learning loop.

---

## MCP Tools

| Tool | Parameters | What It Does |
|---|---|---|
| `get_session_context` | `user_message`, `session_id` | Retrieve relevant memories before responding |
| `consolidate_turn` | `session_id`, `user_message`, `assistant_response`, `tool_calls_json?`, `new_facts?`, `failure_summary?`, `quality_score?`, `reasoning_summary?` | Save a turn and auto-learn from it |
| `search_memory` | `query`, `layers?`, `top_k?` | Search across memory layers by similarity |
| `get_conversation_history` | `session_id`, `last_n?` | Fetch recent conversation messages |
| `clear_session_memory` | `session_id` | Clear a session's conversation history |
| `inspect_memory_layers` | `limit?` | View memory layer counts and recent records |
| `delete_semantic_fact` | `fact_id` | Delete one semantic fact |
| `pin_semantic_fact` | `fact_id`, `pinned?` | Pin/unpin a fact so TTL filtering does not hide it |
| `mark_semantic_fact_stale` | `fact_id`, `stale_days?` | Move a fact's confirmation timestamp into the past |
| `inspect_episode` | `episode_id` | Fetch one full episodic record |
| `prune_old_episodes` | `older_than_days?` | Delete old episodic records while preserving detached failures |
| `export_memory` | `layers?`, `limit_per_layer?` | Export recent memory records as JSON |
| `import_memory` | `memory_json`, `import_semantic?` | Import semantic facts from an export payload |
| `run_autonomous_turn` | `user_message`, `session_id` | *(Standalone only)* Run a full Actor-Critic loop |

### `consolidate_turn` Parameters

| Parameter | Required | Description |
|---|---|---|
| `session_id` | ✅ | Current conversation session identifier |
| `user_message` | ✅ | The user's original message |
| `assistant_response` | ✅ | Your response text |
| `tool_calls_json` | ❌ | JSON array of tool calls made |
| `new_facts` | ❌ | Facts learned from this interaction |
| `failure_summary` | ❌ | Description of any failures encountered |
| `quality_score` | ❌ | Agent self-assessed quality score (0–10) |
| `reasoning_summary` | ❌ | Brief summary of the agent's reasoning approach |

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
│  run_autonomous_turn (standalone only)   │
├──────────────────────────────────────────┤
│   Enhanced Heuristic / LLM Auto-Critic  │
│  Preference detection · fact extraction  │
│  Agent self-score · quality analysis     │
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

For contributor-facing architecture notes, see [`docs/architecture.md`](docs/architecture.md).

## Evaluation Harness

Run the offline harness with SQLite and hash embeddings:

```bash
python -m tests.evaluation_harness
```

It checks preference/fact extraction, semantic deduplication, pin/stale management, persisted planner feedback, context rendering, and failure recall.

---

## Configuration

All settings come from environment variables or a `.env` file. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `MEMORY_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `SQLITE_DB_PATH` | `agentic_memory.db` | SQLite database path |
| `LLM_PROVIDER` | `deterministic` | `deterministic` (no key) / `groq` / `anthropic` / `openai` |
| `STANDALONE_LOOP_ENABLED` | `false` | Enable full Actor-Critic standalone loop |
| `EMBEDDING_BACKEND` | `sentence-transformer` | `sentence-transformer` (384d) or `hash` (256d, zero deps) |
| `MCP_TRANSPORT` | `stdio` | `stdio` (Claude Code) or `http` (Codex) |
| `MCP_HTTP_PORT` | `8001` | Port when using HTTP transport |
| `MEMORY_WINDOW_TURNS` | `10` | Conversation sliding window size |
| `SEMANTIC_DEDUP_THRESHOLD` | `0.92` | Cosine threshold for fact deduplication |

---

## Demo

```bash
# Offline demo (zero config, no API key)
python -m demo.demo_main

# LLM-powered demo (one API key)
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python -m demo.demo_main
```

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
│   ├── evaluation_service.py  # Shared auto-evaluation service
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
│   ├── groq_client.py         # Groq structured LLM adapter + Protocol
│   └── openai_client.py       # OpenAI structured LLM adapter
├── planner/
│   └── retrieval_planner.py   # Heuristic multi-layer query planner
├── runtime/
│   ├── actor.py               # Actor (standalone loop mode)
│   ├── critic.py              # Critic (auto-evaluation)
│   ├── self_learning_loop.py  # Full Actor-Critic loop orchestration
│   └── tools.py               # Tool registry (calculator, web search, etc.)
├── schema.sql                 # PostgreSQL schema with pgvector
├── demo/
│   └── demo_main.py           # 3-turn demo (offline or LLM-powered)
├── docs/
│   └── architecture.md        # Contributor-facing architecture notes
├── tests/
│   └── evaluation_harness.py  # Offline behavioral harness
├── .env.example               # All configuration options
└── .mcp.json                  # Example MCP server configuration
```

---

## License

MIT
