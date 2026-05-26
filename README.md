# Agentic Memory Self-Learning Loop

Production-ready Python backend for a standalone self-learning agent with an Actor-Critic reflection loop and a four-layer Agentic Memory system. The implementation is intentionally explicit: no LangChain, no LlamaIndex, and no hidden memory abstractions. Every memory read and write is typed, scoped, auditable, and owned by the application.

## Architecture

The backend sits between a user-facing API and model providers. Each chat turn flows through retrieval, context assembly, actor execution, critic reflection, consolidation, and response assembly.

```mermaid
flowchart TD
    API["FastAPI /chat"] --> Scope["AccessScope -> scope_hash"]
    Scope --> Planner["HeuristicRetrievalPlanner"]
    Planner --> StoreRead["MemoryStore reads"]
    StoreRead --> Context["AgenticMemoryService.build_context"]
    Context --> Actor["Actor LLM adapter"]
    Actor --> Tools["Explicit tool registry"]
    Tools --> ActorResult["ActorResult"]
    ActorResult --> Critic["Critic LLM adapter"]
    Critic --> Eval["CriticEvaluation"]
    Eval --> Consolidation["AgenticMemoryService.consolidate"]
    Consolidation --> StoreWrite["MemoryStore writes"]
    StoreWrite --> Result["LoopResult response"]
```

## Memory Architecture

The agent maintains four distinct memory layers plus a dedicated failure episode store. Each layer has a clear purpose, independent retrieval strategy, and explicit consolidation path.

```mermaid
flowchart LR
    subgraph Read["Retrieval (Phase 1-2)"]
        P["Prompt"] --> RP["RetrievalPlanner"]
        RP -->|always| C["Conversational"]
        RP -->|preference keywords + density| S["Semantic"]
        RP -->|complexity keywords + triggers| PR["Procedural"]
        RP -->|vector similarity ≥ threshold| E["Episodic"]
        RP -->|failure similarity ≥ 0.80| F["Failure Episodes"]
    end

    subgraph Write["Consolidation (Phase 5)"]
        CR["CriticEvaluation"] --> CW["Save conv turns + rolling summary"]
        CR --> EW["Save episode record"]
        CR --> SW["Dedupe & save/reinforce facts"]
        CR --> PW["Upsert workflow by signature"]
        CR --> FW["Save critic-flagged tool failures"]
    end

    Read --> Actor --> Critic --> Write
```

### Four Memory Layers

| Layer | Purpose | Retrieval | Consolidation |
|---|---|---|---|
| **Conversational** | Active session sliding window plus rolling summaries | Always loaded by `scope_hash` + `session_id` | Saves user/assistant messages every turn; summarizes overflow via `SummaryProvider` |
| **Episodic** | Append-only turn history with prompt, tools, outcome, latency, errors | Vector similarity of prompt embedding with adaptive threshold (0.25–0.85) | Saves every turn as `EpisodeRecord` with outcome classification |
| **Semantic** | Durable preferences, inferred facts, and system rules with source authority and TTL | Vector similarity with confidence and TTL filtering | Critic-proposed facts deduped, reinforced, or conflict-resolved by authority, confidence, and recency |
| **Procedural** | Reusable multi-tool workflows with SHA-256 signatures | Trigger phrase prefix lookup + vector similarity | Successful non-trivial tool chains (≥2 tools) saved as `CANDIDATE` and promoted to `CANONICAL` |

---

### In-Depth Layer Specifications

#### 1. Conversational Layer (Session & Sliding Window)
* **Purpose**: Houses active session state, representing the direct sliding window of human-to-agent interactions.
* **Storage Schema**: Table `am_conversation_turns` (storing `ConversationalTurnRecord` messages per `scope_hash`, `session_id`, and sequential `turn_index`) and `am_conversation_summaries` (storing `ConversationalSummary` token-aware summaries of pruned history).
* **Key Model Fields**:
  - `ConversationalTurnRecord`: `id`, `scope_hash`, `session_id`, `turn_index`, `role` (`ConversationRole.USER` | `ASSISTANT`), `content`, `token_count`, `timestamp`.
  - `ConversationalSummary`: `summary_id`, `scope_hash`, `session_id`, `start_turn_index`, `end_turn_index`, `summary`, `token_count`, `created_at`.
* **Sliding Window Limits**: Controlled by `MEMORY_WINDOW_TURNS` (default: 10 turns). Each turn produces 2 messages (user + assistant), so the window holds up to 20 raw messages.
* **Context Compaction**:
  - Every turn, `AgenticMemoryService._maybe_roll_summary()` calculates the first kept turn index as `turn_index - MEMORY_WINDOW_TURNS + 1`.
  - When turns exist before the first kept index, they are passed to the `SummaryProvider` (implemented by default as `ExtractiveSummaryProvider` or replaceable with an LLM-based adapter).
  - `ExtractiveSummaryProvider` concatenates turn content into `"Summary of turns {start} to {end}: ..."` with per-turn excerpts capped at 120 characters each.
  - The summary is persisted as a `ConversationalSummary` and the raw turns are deleted from the database.
  - Context assembly injects the latest 2 rolling summaries alongside active window messages.

#### 2. Episodic Layer (Append-Only Interaction Traces)
* **Purpose**: Captures trace logs of complete execution cycles, enabling the agent to learn from historical experiences and avoid past mistakes.
* **Storage Schema**: Table `am_episodic_memory` for all execution cycles, and `am_failure_episodes` for critic-flagged tool failures.
* **Key Model Fields**:
  - `EpisodeRecord`: `episode_id`, `scope_hash`, `prompt_text`, `prompt_embedding`, `tool_sequence` (`list[ToolInvocation]`), `final_response`, `outcome` (`EpisodeOutcome`), `error_trace`, `latency_ms`, `timestamp`, `score`.
  - `ToolInvocation`: `tool_name`, `input_parameters`, `input_summary`, `output_summary`, `success`, `latency_ms`, `error_trace`, `critic_flagged`, `metadata`.
  - `FailureEpisode`: `failure_id`, `scope_hash`, `episode_id`, `prompt_text`, `prompt_embedding`, `tool_name`, `tool_input`, `exception_message`, `error_trace`, `timestamp`, `score`.
* **Outcome Classification**: The `_episode_outcome()` function classifies results into three states:
  - `EpisodeOutcome.FAILURE` — All tool calls failed, or any tool call was critic-flagged.
  - `EpisodeOutcome.PARTIAL` — Critic did not pass, or some (but not all) tool calls failed.
  - `EpisodeOutcome.SUCCESS` — All tool calls succeeded and the Critic passed.
* **Failure Consolidation**: Individual tool failures are saved as separate `FailureEpisode` records when `tool.critic_flagged == True`, each linked back to the parent `episode_id`.
* **Retrieval Strategy**: Vector similarity of the current prompt embedding against historical `prompt_embedding` values. Top-$K$ similar episodes are retrieved with an adaptive threshold (see Retrieval Planner below).

#### 3. Semantic Layer (Durable Preferences & Conflict Resolution)
* **Purpose**: Durable long-term facts, including user preferences, inferred traits, and system rules.
* **Storage Schema**: Table `am_semantic_memory` featuring `SemanticMemoryRecord` with `fact_id`, `scope_hash`, `fact_type` (`SemanticFactType`), `content`, `embedding`, `confidence_score`, `source`, `source_episode_id`, `created_at`, `last_reinforced_at`, and `last_confirmed_at` metadata.
* **Fact Types** (`SemanticFactType` enum):
  - `PREFERENCE` — User preferences (rendered under `[USER PREFERENCES]` in Actor context).
  - `INFERRED_FACT` — Facts inferred by the Critic LLM.
  - `SYSTEM_RULE` — Durable system rules (rendered under `[SYSTEM RULES]` in Actor context).
* **Source Authority Hierarchy**:
  Authority ranks control conflict resolution when competing facts exist:
  1. `user_stated` (Authority Rank 3) – Direct, explicit instructions provided by the user.
  2. `tool_derived` (Authority Rank 2) – Facts validated by external tools or databases.
  3. `llm_inferred` (Authority Rank 1) – Inferences made by the Critic LLM during consolidation.
  Source labels are normalized via `_normalise_source()`, which maps aliases like `"user"`, `"explicit_user"` → `"user_stated"` and `"tool"` → `"tool_derived"`.
* **Lightweight Contradiction Engine**:
  - During consolidation, candidate facts are evaluated against existing near-duplicates (similarity ≥ `SEMANTIC_DEDUP_THRESHOLD = 0.92`).
  - The `_semantic_contradiction()` function normalizes both fact strings and evaluates their polarity using `_polarity()`, which detects 9 negative regex patterns:
    `\bdo not\b`, `\bdon't\b`, `\bnever\b`, `\bno longer\b`, `\bnot\b`, `\bdislikes?\b`, `\bhates?\b`, `\bavoid(s|ed)?\b`, `\bdoes not prefer\b`.
  - A fact returns polarity `-1` if any negative pattern matches, else `+1`. If the existing and candidate fact have opposite polarity, a contradiction is raised.
* **Conflict Resolution Strategy** (`_new_fact_wins()`):
  - **Authority Wins**: If a contradiction exists, the fact with the higher source authority rank overwrites the lower rank. For example, `user_stated` (3) immediately evicts an existing `llm_inferred` (1) fact.
  - **Confidence Resolution**: If authority ranks are equal, the fact with higher `confidence_score` wins (retaining a $\ge 0.05$ difference threshold).
  - **Recency Resolution**: If both authority and confidence are equal, the newer observation (latest `last_confirmed_at`) overwrites the older fact.
  - **Reinforcement**: If a candidate matches an existing fact with vector similarity $\ge \text{SEMANTIC\_DEDUP\_THRESHOLD}=0.92$ and has no contradiction, the existing record is *reinforced*. The `last_confirmed_at`, `last_reinforced_at`, confidence score, and source are updated.
* **Temporal Eviction (TTL)**:
  - Retrieval filters out semantic records whose `last_confirmed_at` is older than `SEMANTIC_MEMORY_TTL_DAYS` (default: 180 days).
  - A TTL of `0` completely disables eviction, retaining facts indefinitely.

#### 4. Procedural Layer (Workflow Signatures & Promotion)
* **Purpose**: Automatically learns, promotes, and optimizes multi-tool sequences into reusable composite workflows.
* **Storage Schema**: Table `am_procedural_workflows` containing `ProceduralWorkflow` with ordered `tool_sequence` (`list[ProceduralToolStep]`), `workflow_signature`, `trigger_phrases`, `success_count`, `avg_latency_ms`, `status`, and `embedding`.
* **Key Model Fields**:
  - `ProceduralToolStep`: `tool_name`, `param_schema` (dict mapping param names to type names), `expected_outcome`.
  - `ProceduralWorkflow`: `workflow_id`, `scope_hash`, `workflow_signature`, `trigger_phrases`, `tool_sequence`, `success_count`, `status` (`WorkflowStatus`), `avg_latency_ms`, `embedding`, `created_at`, `updated_at`, `score`.
* **Workflow Signature**:
  Multi-tool chains are hashed deterministically based on tool names *and* parameter type signatures using SHA-256:
  ```python
  material = [
      {"tool_name": tool.tool_name,
       "param_types": sorted((k, type(v).__name__) for k, v in tool.input_parameters.items())}
      for tool in successful_tools
  ]
  signature = sha256(json.dumps(material, sort_keys=True))
  ```
  This means the same tools called with structurally different parameter types produce different signatures.
* **Trigger Phrase Extraction** (`_trigger_phrases()`):
  - The prompt is normalized and truncated to 90 characters as the first phrase.
  - A second phrase is built from the first 6 words longer than 3 characters.
  - Both are deduplicated and sorted.
* **Promotion Pipeline**:
  - A tool chain with $\ge 2$ successful tools executed during a `save_workflow = True` Critic evaluation is registered as a candidate workflow with `status = WorkflowStatus.CANDIDATE`.
  - If the same signature already exists, `upsert_procedural_workflow()` increments `success_count` and updates `avg_latency_ms`.
  - Workflows graduate to `status = WorkflowStatus.CANONICAL` after repeated successful executions.
* **Retrieval & Suggestion**:
  - **Trigger Phrase Lookup**: `match_procedural_triggers()` finds workflows whose stored trigger phrases appear in the current prompt.
  - **Vector Matching**: Cosine similarity is computed against prompt embeddings to recommend workflows with similarity ≥ 0.40.
  - Retrieved workflows are rendered as structured recipes (`[SUGGESTED WORKFLOW]`) listing each step with its tool name, parameter schema, and expected outcome.

---

### Vector Search & Distance Metrics

The Agentic Memory system leverages Cosine Similarity as its foundational vector metric for episodic, semantic, procedural, and failure matching.

#### Mathematical Definition
For a query vector $\mathbf{q}$ and a candidate memory vector $\mathbf{m}$:
$$\text{Similarity}(\mathbf{q}, \mathbf{m}) = \cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{m}}{\|\mathbf{q}\| \|\mathbf{m}\|} = \frac{\sum_{i=1}^{D} q_i m_i}{\sqrt{\sum_{i=1}^{D} q_i^2} \sqrt{\sum_{i=1}^{D} m_i^2}}$$

To guarantee memory safety and ensure full compliance with the Pydantic schema constraints (which strictly enforce $0.0 \le \text{score} \le 1.0$), both the SQLite and PostgreSQL store implementations apply explicit clamping to the computed cosine value:
$$\text{Score}_{\text{clamped}} = \max(0.0, \min(1.0, \text{Similarity}))$$
This prevents floating-point precision inaccuracies (such as $1.0000000000000002$ or $-1.0 \times 10^{-16}$) from triggering validation errors during Pydantic model instantiation.

#### Pluggable Vector Engine
* **PostgreSQL (pgvector)**: Executes similarity calculations server-side using the fast cosine distance operator `<=>` inside raw SQL queries:
  ```sql
  SELECT *, 1 - (embedding <=> $1::vector) AS similarity 
  FROM am_semantic_memory 
  WHERE scope_hash = $2 AND 1 - (embedding <=> $1::vector) >= $3
  ORDER BY embedding <=> $1::vector LIMIT $4
  ```
* **SQLite (Client-Side Fallback)**: Pulls candidate rows within the active scope, parses the serialized JSON embeddings, and calculates the similarity using an optimized client-side `_cosine_similarity()` dot-product implementation in standard Python.

### Storage Backends

The memory store is pluggable. Choose the backend that fits your deployment:

| Backend | `MEMORY_BACKEND` | Storage | Vector Search | Dependencies | Best For |
|---|---|---|---|---|---|
| **SQLite** | `sqlite` | Local `.db` file | Client-side cosine similarity | `aiosqlite` | Local dev, single-node, offline |
| **PostgreSQL + pgvector** | `postgres` | Local/remote PostgreSQL | Server-side `<=>` cosine operator | `asyncpg`, pgvector extension | Self-hosted production |

Both backends implement the same `MemoryStore` protocol — the loop, planner, and memory service are backend-agnostic.

### Embedding Strategy

| Backend | Model | Dimensions | Dependencies |
|---|---|---|---|
| `sentence-transformer` | `all-MiniLM-L6-v2` | 384 | `sentence-transformers` |
| `hash` | Deterministic token hashing | 256 | None |

The `HashEmbeddingModel` tokenizes text, hashes each token with MD5, distributes hash bytes across a normalized vector of configurable dimensions. It is deterministic: the same text always produces the same embedding, making it suitable for offline testing without any model downloads.

Set `EMBEDDING_BACKEND=hash` for zero-dependency local operation.

---

## Retrieval Planner

The `HeuristicRetrievalPlanner` determines which memory layers to query before each turn using keyword heuristics, density scoring, and per-session adaptive weights.

### Layer Activation Rules

| Layer | Trigger Logic | Default Threshold |
|---|---|---|
| **Conversational** | Always queried | — |
| **Semantic** | Keyword match (`always`, `never`, `prefer`, `please don't`, `do not`, `don't`, `style`, `format`, `tone`) OR keyword density × session weight ≥ 0.04 | density ≥ 0.04 |
| **Procedural** | Keyword match (`build`, `analyze and then`, `compare and summarize`, `compare`, `summarize`, `workflow`, `steps`, `process`, `tool`, `plan`, `implement`) OR trigger phrase match OR density × weight ≥ 0.05 | density ≥ 0.05 |
| **Episodic** | Vector similarity of prompt against stored episodes exceeds adaptive threshold | 0.25–0.85 adaptive |
| **Failure** | Vector similarity ≥ `FAILURE_SIMILARITY_THRESHOLD` (default 0.80) | 0.80 |

### Adaptive Retrieval Weights

The planner maintains per-session **exponential moving average (EMA)** weights for semantic, procedural, and episodic layers:

```
weights[layer] = max(0.5, min(1.5, 0.85 × current_weight + 0.15 × target))
```

After each turn, `record_feedback()` updates weights based on Critic pass/fail:
- **Critic passed** → target = 1.15 (boost layers that were retrieved)
- **Critic failed** → target = 0.85 (penalize layers that were retrieved)

This creates a session-local learning effect where frequently unhelpful memory layers are gradually down-weighted.

### Episodic Threshold Adaptation

The base episode similarity threshold adjusts based on history-related keywords (`earlier`, `before`, `previous`, `last time`, `history`, `what happened`):
- **History keywords present** → base = 0.45 (more permissive)
- **No history keywords** → base = 0.55 (more selective)
- Final threshold: `max(0.25, min(0.85, base / episodic_weight))`

---

## Six-Phase Loop

```mermaid
flowchart LR
    P1["Phase 1\nRetrieval Planning"] --> P2["Phase 2\nContext Assembly"]
    P2 --> P3["Phase 3\nActor Execution"]
    P3 --> P4["Phase 4\nCritic Reflection"]
    P4 --> P5["Phase 5\nConsolidation"]
    P5 --> P6["Phase 6\nResponse Assembly"]
```

1. **Retrieval Planning**
   `HeuristicRetrievalPlanner.plan()` embeds the prompt, scores keyword density against preference and complexity phrase sets, queries all relevant stores, and populates a `RetrievalPlan` with raw records and layer-specific rationales.

2. **Context Assembly**
   `AgenticMemoryService.build_context()` renders memory into the Actor prompt in this fixed order:
   `[SYSTEM RULES]`, `[USER PREFERENCES]`, `[SUGGESTED WORKFLOW]`, `[RECENT CONVERSATION]`, `[PAST SIMILAR EPISODE]`.
   Failure warnings are rendered as `CAUTION:` blocks inside `[PAST SIMILAR EPISODE]` when failure matches take priority over normal episodes.

3. **Actor Execution**
   `Actor.execute()` calls a structured LLM client, parses JSON into `ActorLLMOutput` (containing `reasoning`, `tool_calls`, `final_response`), dispatches real tools via `ToolRegistry.execute()`, and returns `ActorResult` with latency measurement. Tool exceptions are caught and the invocation is marked with `critic_flagged = True`.

4. **Critic Reflection**
   `Critic.evaluate()` performs a hidden secondary evaluation and returns `CriticEvaluation`. The Critic has a configurable timeout (`CRITIC_TIMEOUT_SECONDS`, default 8s) enforced via `asyncio.wait_for`. On timeout, a zero-score no-pass evaluation is returned and consolidation is skipped entirely — the user response is never blocked.

5. **Consolidation**
   `AgenticMemoryService.consolidate()` writes lessons back into memory: conversational turns, rolling summaries, episodic records, deduplicated semantic facts, procedural workflows, and failure episodes. After consolidation, `record_feedback()` updates planner weights.

6. **Response Assembly**
   `SelfLearningLoop.run_turn()` returns only safe public fields via `LoopResult`: final response, session id, turn index, critic score/pass, write log, retrieval plan summary, phase timings, and semantic conflict notices.

---

### Critic Evaluation Details

The Critic scores each turn across five weighted dimensions on a 0–10 scale:

| Dimension | Weight | Description |
|---|---|---|
| `factual_accuracy` | 25% | Is the response factually correct? |
| `preference_adherence` | 20% | Does it respect known user preferences? |
| `tool_efficiency` | 20% | Were tools used appropriately and efficiently? |
| `hallucination_risk` | 15% | Is the response well-grounded (higher = less risk)? |
| `workflow_quality` | 20% | Was the tool workflow well-structured? |

**Overall score computation** (recomputed by Pydantic `model_validator`):
$$\text{overall} = 0.25 \times \text{accuracy} + 0.20 \times \text{preference} + 0.20 \times \text{efficiency} + 0.15 \times \text{hallucination} + 0.20 \times \text{workflow}$$

**Pass threshold**: $\text{overall\_score} \ge 7.0$

**Self-Consistency Voting**: Higher-risk turns trigger multiple Critic evaluations. Multi-sampling is activated when:
- The Actor used more than 1 tool call, OR
- Low-confidence signals appear in the Actor's reasoning or response (`"not sure"`, `"uncertain"`, `"may be wrong"`, `"low confidence"`, `"i think"`)

When activated, `CRITIC_SELF_CONSISTENCY_SAMPLES` (default 3) evaluations run at `CRITIC_SELF_CONSISTENCY_TEMPERATURE` (default 0.4). The final result is selected by majority vote on `passed`, then the median-scored sample from the majority group.

**Critic Learning Signals**: Each `CriticEvaluation` proposes:
- `new_semantic_facts` — Durable facts to consolidate (typed as `NewSemanticFact` with `fact_type`, `content`, `confidence_score`, `source`).
- `save_workflow` — Whether the tool chain should be saved as a procedural workflow.
- `failure_summary` — Description of what went wrong (used for episodic error traces).

---

## Model Agnosticism

The runtime depends on `StructuredLLMClient`, not a specific provider. Supported providers:

| Provider | `LLM_PROVIDER` | Actor default | Critic default |
|---|---|---|---|
| Deterministic local | `deterministic` | `deterministic-actor` | `deterministic-critic` |
| Groq | `groq` | `llama3-8b-8192` | `llama3-70b-8192` |
| Anthropic | `anthropic` | `claude-sonnet-4-5` | `claude-opus-4-5` |
| OpenAI-compatible | `openai` | `gpt-4o-mini` | `gpt-4o` |

For offline demos and tests, `DeterministicStructuredClient` implements the same protocol without network calls.

Provider clients include a small circuit breaker: after five consecutive provider failures, the circuit opens and new calls fail fast. After `reset_timeout` (default 60s), the circuit transitions to `half_open` to test one request. A successful call resets the breaker to `closed`.

Embeddings are similarly swappable:

- `SentenceTransformerEmbeddingModel` uses `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional vectors.
- `HashEmbeddingModel` is a zero-dependency fallback using deterministic normalized token hashing with 256-dimensional vectors.

## Tool Registry

The Actor in the self-learning loop has access to an explicit, auditable tool registry (`runtime/tools.py`). There are 7 production-ready tools registered by default, complete with strict safety boundaries and PII-safe sanitization via `InjectionGuard`:

1. **`calculator`** — Evaluates safe mathematical and arithmetic expressions.
   - **Allowed**: `abs`, `round`, `sqrt`, `sin`, `cos`, `tan`, `log`, `log10`, `log2`, `ceil`, `floor`, and constants `pi`, `e`, `inf`.
   - **Safety**: Rejects arbitrary code/syntax via strict AST parsing.
2. **`web_search`** — Searches the live web using the Brave Search API.
   - **Configuration**: Optional `BRAVE_SEARCH_API_KEY`. If missing, supports simulated web search.
3. **`file_search`** — Find files by glob pattern or substring in a workspace.
   - **Configuration**: Defaults to `TOOL_WORKSPACE_ROOT` (configurable in `.env`).
   - **Safety**: Fully resolves paths and protects against path traversal escapes. Skips hidden/virtual env directories.
4. **`document_search`** — Search for regex/text matches within file contents (similar to grep).
   - **Safety**: Resolves paths in the workspace root, skips binary files, and caps individual read sizes at 2 MB.
5. **`memory_search`** — Performs direct vector lookup against the agent's own memory store.
   - **Safety**: Multi-tenant aware. Dynamic request-scoped `scope_hash` is resolved automatically using the async-safe `current_scope_hash` context variable.
6. **`python_executor`** — Runs sandboxed Python code for math and data processing.
   - **Allowed modules**: `math`, `statistics`, `json`, `datetime`, `re`, `collections`, `itertools`, `functools`, `string`, `textwrap`, `decimal`, `fractions`, `random`, `typing`, `operator`, `dataclasses`, `enum`, `abc`, `copy`, `pprint`, `numbers`, `cmath`.
   - **Safety**: AST-restricted `exec()` blocks imports of unlisted modules, filesystem access (e.g. `open`), process forks (e.g. `os.system`, `subprocess`), and dangerous builtins. Subprocess runs with minimal environment variables, a 10s timeout, and truncated output.
7. **`shell_executor`** — Executes safe, read-only commands for local inspection.
   - **Whitelisted commands**: `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `echo`, `date`, `pwd`, `env`, `du`, `df`, `uname`, `whoami`, `sort`, `uniq`, `cut`, `tr`, `sed`, `awk`, `file`, `stat`, `which`, `dirname`, `basename`, `realpath`, `readlink`, `diff`, `comm`, `tee`, `xargs`, `printf`.
   - **Safety**: Subprocess execution with minimal environment, 15s timeout, and truncated output. Rejects shell operators like pipes (`|`), redirects (`>`, `<`), and command chaining (`;`, `&&`, `||`).

## Security & Isolation

### Scope Isolation
All memory reads and writes are scoped by a SHA-256 hash:
```text
sha256(application_id || tenant_id || user_id)
```
Raw PII (`application_id`, `tenant_id`, `user_id`) is never stored in memory tables. Only the resulting `scope_hash` is persisted. PostgreSQL enforces this via Row-Level Security (RLS) policies.

### Input Sanitization
`InjectionGuard` (in `core/injection_guard.py`) provides three static defenses:
- `sanitize_input()` — Strips system prompt override patterns (`<<SYS>>`, `[INST]`, `### Instruction:`, etc.) via regex before any text reaches the LLM.
- `validate_scope()` — Validates that all scope fields are non-empty, alphanumeric-safe (allows `-_`), and ≤ 128 characters.
- `check_prompt_length()` — Rough word-count-based token check against a configurable max (default 4096).

### Circuit Breaker
All LLM provider calls pass through a `CircuitBreaker` with three states:
- **Closed** (normal operation) → counts consecutive failures.
- **Open** (fail-fast) → triggered after 5 consecutive failures, raises `CircuitBreakerOpenError` without calling the provider.
- **Half-Open** (recovery probe) → after 60s, allows one test call through. Success resets to Closed; failure reopens.

## Project Layout

```text
Self_Learning_Agent/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── schema.sql
├── .mcp.json
├── mcp_server.py
├── main.py
├── config.py
├── README.md
├── core/
│   ├── models.py            # All Pydantic schemas, enums, typed models
│   ├── access_scope.py      # AccessScope and scope_hash computation
│   ├── injection_guard.py   # Input sanitization and validation
│   └── memory_service.py    # AgenticMemoryService orchestration
├── store/
│   ├── base.py              # MemoryStore protocol (28 async methods)
│   ├── sqlite_store.py      # SQLite + client-side cosine similarity
│   ├── postgres_store.py    # PostgreSQL + pgvector server-side search
│   └── factory.py           # Backend factory from settings
├── planner/
│   └── retrieval_planner.py # Heuristic + density + adaptive planner
├── model/
│   ├── embedding_model.py   # SentenceTransformer and Hash embeddings
│   ├── groq_client.py       # Groq LLM client + StructuredLLMClient protocol
│   ├── anthropic_client.py  # Anthropic LLM client
│   ├── openai_client.py     # OpenAI-compatible LLM client
│   └── circuit_breaker.py   # Three-state circuit breaker
├── runtime/
│   ├── tools.py             # ToolRegistry, ToolDefinition, built-in tools
│   ├── actor.py             # Actor LLM execution and tool dispatch
│   ├── critic.py            # Critic evaluation with JSON schema
│   ├── openai_adapter.py    # OpenAI chat.completion adapter
│   └── self_learning_loop.py # Six-phase loop orchestration
└── demo/
    └── demo_main.py         # Offline demo with deterministic client
```

## Quick Start

The main use case is running this as a local memory layer for Claude Code or Codex. Start with local SQLite and hash embeddings:

```bash
git clone <repo-url>
cd Self_Learning_Agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Recommended local `.env` values:

```bash
MEMORY_BACKEND=sqlite
SQLITE_DB_PATH=agentic_memory.db
EMBEDDING_BACKEND=hash
LLM_PROVIDER=deterministic
```

Memory is persisted to `agentic_memory.db`.

## Claude Code Integration

Claude Code should be the Actor. This repo runs as an MCP tool server that provides memory context before a response and consolidation after a response.

1. Keep `.mcp.json` in the repository root. It registers the server as `agentic-memory`:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": ".",
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

2. Start Claude Code from this repository or point Claude Code at this project so it can read `.mcp.json`.

```bash
claude
```

3. In Claude Code, use the `agentic-memory` tools around your normal coding workflow:

- Call `get_session_context` before drafting a response or code plan.
- Call `consolidate_turn` after the answer is delivered.
- Use `search_memory`, `get_conversation_history`, `clear_session_memory`, and `inspect_memory_layers` for debugging or manual memory work.

4. Optional: run MCP over HTTP instead of stdio for sidecar setups:

```bash
MCP_TRANSPORT=http MCP_HTTP_PORT=8001 python mcp_server.py
```

In the default stdio path, you normally do not run `python mcp_server.py` yourself; Claude Code launches it from `.mcp.json`.

## Codex Integration

Codex and any OpenAI-compatible client can use the FastAPI server through `/v1/chat/completions`.

1. Start the local memory-augmented OpenAI-compatible server:

```bash
uvicorn main:app --host 127.0.0.1 --port 7860 --reload
```

2. Point the client at the local `/v1` base URL. Any non-empty API key works for local use:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7860/v1", api_key="local-dev")

response = client.chat.completions.create(
    model="memory-agent",
    messages=[
        {"role": "user", "content": "I prefer short answers. What is 12 * 9?"}
    ],
    extra_headers={
        "X-Session-Id": "session-1",
        "X-Application-Id": "codex",
        "X-Tenant-Id": "local",
        "X-User-Id": "rishi",
    },
)

print(response.choices[0].message.content)
```

3. Equivalent curl smoke test:

```bash
curl http://127.0.0.1:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev" \
  -H "X-Session-Id: session-1" \
  -H "X-Application-Id: codex" \
  -H "X-Tenant-Id: local" \
  -H "X-User-Id: rishi" \
  -d '{
    "model": "memory-agent",
    "messages": [
      {"role": "user", "content": "I prefer concise answers. Calculate 21 * 2."}
    ]
  }'
```

The adapter returns a normal `chat.completion` object while the backend runs retrieval, Actor execution, Critic evaluation, and consolidation.

## Quick Start — Local PostgreSQL + pgvector

For server-side vector search with a local PostgreSQL instance:

```bash
# 1. Start PostgreSQL with pgvector
docker run -d --name pgvector \
  -e POSTGRES_DB=agentic_memory \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 2. Initialize the schema
psql postgresql://postgres:postgres@localhost:5432/agentic_memory < schema.sql

# 3. Configure
cp .env.example .env
# Edit .env:
#   MEMORY_BACKEND=postgres
#   POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/agentic_memory

# 4. Run
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

## Database Schema

Run [`schema.sql`](schema.sql) against local PostgreSQL when using `MEMORY_BACKEND=postgres`. It creates:

- `pgvector` and `pgcrypto` extensions
- `am_conversation_turns`
- `am_conversation_summaries`
- `am_episodic_memory`
- `am_failure_episodes`
- `am_semantic_memory`
- `am_procedural_workflows`
- Vector RPC functions for semantic, episodic, failure, and procedural search
- GIN indexes for JSONB columns
- ivfflat indexes for vector columns (`lists = 100`, cosine ops)
- RLS policies scoped by `scope_hash`

The SQLite backend creates equivalent tables automatically on first run (without pgvector extensions or RLS).

## Docker

### SQLite (default, no sidecar)

```bash
docker compose up --build
```

### With local PostgreSQL + pgvector

```bash
docker compose --profile postgres up --build
```

The Docker image runs on port `7860`.

## Offline Demo

The demo runs fully locally without an external LLM by using hash embeddings, SQLite storage, and the deterministic structured client:

```bash
python demo/demo_main.py
```

It executes three turns and prints `LoopResult` objects plus structured JSON logs for each phase. Memory is persisted to `demo_memory.db`.

## Configuration Reference

| Variable | Default | Required When | Description |
|---|---|---|---|
| `MEMORY_BACKEND` | `sqlite` | Always | Storage backend: `sqlite`, `postgres` |
| `SQLITE_DB_PATH` | `agentic_memory.db` | `MEMORY_BACKEND=sqlite` | Path to SQLite database file |
| `POSTGRES_DSN` | `postgresql://postgres:postgres@localhost:5432/agentic_memory` | `MEMORY_BACKEND=postgres` | PostgreSQL connection string |
| `LLM_PROVIDER` | `deterministic` | Always | LLM backend: `deterministic`, `groq`, `anthropic`, `openai` |
| `GROQ_API_KEY` | — | `LLM_PROVIDER=groq` | Groq Cloud API key |
| `GROQ_BASE_URL` | — | Never | Custom Groq endpoint override |
| `ACTOR_MODEL` | `llama3-8b-8192` | `LLM_PROVIDER=groq` | Groq model for Actor |
| `CRITIC_MODEL` | `llama3-70b-8192` | `LLM_PROVIDER=groq` | Groq model for Critic |
| `ANTHROPIC_API_KEY` | — | `LLM_PROVIDER=anthropic` | Anthropic API key |
| `ANTHROPIC_ACTOR_MODEL` | `claude-sonnet-4-5` | `LLM_PROVIDER=anthropic` | Anthropic model for Actor |
| `ANTHROPIC_CRITIC_MODEL` | `claude-opus-4-5` | `LLM_PROVIDER=anthropic` | Anthropic model for Critic |
| `OPENAI_API_KEY` | — | `LLM_PROVIDER=openai` | OpenAI or compatible endpoint API key |
| `OPENAI_BASE_URL` | — | OpenAI-compatible endpoints | Base URL for Ollama, LM Studio, Azure-compatible APIs, or proxies |
| `OPENAI_ACTOR_MODEL` | `gpt-4o-mini` | `LLM_PROVIDER=openai` | OpenAI-compatible model for Actor |
| `OPENAI_CRITIC_MODEL` | `gpt-4o` | `LLM_PROVIDER=openai` | OpenAI-compatible model for Critic |
| `MCP_TRANSPORT` | `stdio` | MCP server | `stdio` for Claude Code or `http` for sidecar mode |
| `MCP_HTTP_PORT` | `8001` | `MCP_TRANSPORT=http` | HTTP MCP sidecar port |
| `TOOL_WORKSPACE_ROOT` | `.` | Never | Root directory for file_search and document_search |
| `BRAVE_SEARCH_API_KEY` | — | For live web search | Brave Search API key |
| `BRAVE_SEARCH_ENDPOINT` | `https://api.search.brave.com/res/v1/web/search` | Never | Brave API endpoint |
| `BRAVE_SEARCH_COUNTRY` | `us` | Never | Search country code |
| `BRAVE_SEARCH_LANG` | `en` | Never | Search language code |
| `BRAVE_SEARCH_COUNT` | `5` | Never | Results per search |
| `BRAVE_SEARCH_TIMEOUT_SECONDS` | `10` | Never | Search request timeout |
| `EMBEDDING_BACKEND` | `sentence-transformer` | Always | Embedding model: `sentence-transformer`, `hash` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | `EMBEDDING_BACKEND=sentence-transformer` | HuggingFace model name |
| `HASH_EMBEDDING_DIMENSIONS` | `256` | `EMBEDDING_BACKEND=hash` | Hash vector dimensionality |
| `MEMORY_WINDOW_TURNS` | `10` | Never | Conversational sliding window size |
| `SEMANTIC_DEDUP_THRESHOLD` | `0.92` | Never | Cosine threshold for semantic dedup |
| `SEMANTIC_MEMORY_TTL_DAYS` | `180` | Never | Exclude semantic facts not confirmed within this many days; `0` disables TTL |
| `FAILURE_SIMILARITY_THRESHOLD` | `0.80` | Never | Cosine threshold for failure matching |
| `CRITIC_TIMEOUT_SECONDS` | `8` | Never | Critic timeout after which the actor response still returns and consolidation is skipped |
| `CRITIC_SELF_CONSISTENCY_SAMPLES` | `3` | Never | Number of Critic samples for higher-risk turns |
| `CRITIC_SELF_CONSISTENCY_TEMPERATURE` | `0.4` | Never | Temperature used for multi-sample Critic evaluation |
| `RATE_LIMIT_RPM` | `60` | Never | In-memory per-scope request limit per minute; `0` disables |
| `DEBUG_KEY` | — | Debug endpoint | Enables `/debug/turn/{session_id}/{turn_index}` when matched by `X-Debug-Key` |
| `LOG_LEVEL` | `INFO` | Never | Logging level |

## Validation

```bash
# Compile check
python -m compileall .

# Type check
mypy --strict .

# Run demo
python demo/demo_main.py
```

All runtime code uses typed Pydantic v2 schemas, async methods, structured logging, retrying LLM calls, explicit tool dispatch, and direct memory-store operations.

## Key Design Principles

- Memory is explicit and auditable.
- Scope isolation uses hashed identifiers, not raw tenant/user PII.
- The Actor never writes memory directly.
- The Critic never appears in public API responses.
- Consolidation is the only place where durable learning occurs.
- Provider adapters are replaceable; the loop is not tied to any single provider.
- Tool calls are real, traceable, and failure-aware.
- The entire agent can run locally with zero cloud dependencies.
- The primary integrations are Claude Code via MCP and Codex/OpenAI-compatible clients via `/v1/chat/completions`.
