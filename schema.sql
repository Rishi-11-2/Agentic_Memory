-- Agentic Memory PostgreSQL schema.
-- Run this file against a PostgreSQL database with pgvector enabled.

create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists am_conversation_turns (
    id uuid primary key default gen_random_uuid(),
    session_id text not null,
    turn_index integer not null check (turn_index >= 0),
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    token_count integer not null default 0 check (token_count >= 0),
    timestamp timestamptz not null default now(),
    unique (session_id, turn_index, role)
);

create table if not exists am_conversation_summaries (
    summary_id uuid primary key default gen_random_uuid(),
    session_id text not null,
    start_turn_index integer not null,
    end_turn_index integer not null,
    summary text not null,
    token_count integer not null default 0 check (token_count >= 0),
    created_at timestamptz not null default now(),
    check (end_turn_index >= start_turn_index)
);

create table if not exists am_episodic_memory (
    episode_id uuid primary key default gen_random_uuid(),
    prompt_text text not null,
    reasoning_summary text not null default '',
    prompt_embedding vector(384),
    prompt_embedding_hash vector(256),
    tool_sequence jsonb not null default '[]'::jsonb,
    final_response text not null default '',
    outcome text not null check (outcome in ('success', 'partial', 'failure')),
    error_trace text,
    latency_ms integer not null default 0 check (latency_ms >= 0),
    timestamp timestamptz not null default now()
);

alter table am_episodic_memory
    add column if not exists reasoning_summary text not null default '';

create table if not exists am_failure_episodes (
    failure_id uuid primary key default gen_random_uuid(),
    episode_id uuid references am_episodic_memory(episode_id) on delete set null,
    prompt_text text not null,
    prompt_embedding vector(384),
    prompt_embedding_hash vector(256),
    tool_name text not null,
    tool_input jsonb not null default '{}'::jsonb,
    exception_message text not null,
    error_trace text not null,
    timestamp timestamptz not null default now()
);

create table if not exists am_semantic_memory (
    fact_id uuid primary key default gen_random_uuid(),
    fact_type text not null check (fact_type in ('preference', 'inferred_fact', 'system_rule')),
    content text not null,
    embedding vector(384),
    hash_embedding vector(256),
    confidence_score double precision not null check (confidence_score >= 0.0 and confidence_score <= 1.0),
    source text not null default 'llm_inferred',
    source_episode_id uuid references am_episodic_memory(episode_id) on delete set null,
    pinned boolean not null default false,
    created_at timestamptz not null default now(),
    last_reinforced_at timestamptz not null default now(),
    last_confirmed_at timestamptz not null default now()
);

create table if not exists am_semantic_hierarchy_nodes (
    node_id uuid primary key,
    node_key text not null unique,
    parent_id uuid references am_semantic_hierarchy_nodes(node_id) on delete set null,
    node_type text not null check (node_type in ('root', 'facet', 'summary', 'qa')),
    facet text not null default 'general',
    title text not null,
    content text not null default '',
    question text,
    answer text,
    source_fact_ids jsonb not null default '[]'::jsonb,
    embedding vector(384),
    hash_embedding vector(256),
    confidence_score double precision not null default 0.0
        check (confidence_score >= 0.0 and confidence_score <= 1.0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table am_semantic_memory
    add column if not exists source text not null default 'llm_inferred';

alter table am_semantic_memory
    add column if not exists last_confirmed_at timestamptz;

alter table am_semantic_memory
    add column if not exists pinned boolean not null default false;

update am_semantic_memory
set last_confirmed_at = coalesce(last_confirmed_at, last_reinforced_at, created_at, now())
where last_confirmed_at is null;

alter table am_semantic_memory
    alter column last_confirmed_at set not null;

create table if not exists am_procedural_workflows (
    workflow_id uuid primary key default gen_random_uuid(),
    workflow_signature text not null unique,
    trigger_phrases text[] not null default '{}',
    tool_sequence jsonb not null default '[]'::jsonb,
    success_count integer not null default 1 check (success_count >= 1),
    status text not null default 'candidate' check (status in ('candidate', 'canonical')),
    avg_latency_ms double precision not null default 0.0 check (avg_latency_ms >= 0.0),
    embedding vector(384),
    hash_embedding vector(256),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists am_retrieval_feedback (
    session_id text primary key,
    semantic_weight double precision not null default 1.0,
    procedural_weight double precision not null default 1.0,
    episodic_weight double precision not null default 1.0,
    updated_at timestamptz not null default now()
);

create index if not exists am_conversation_session_idx
    on am_conversation_turns (session_id, turn_index desc, timestamp desc);

create index if not exists am_summaries_session_idx
    on am_conversation_summaries (session_id, created_at desc);

create index if not exists am_episodic_tool_sequence_gin_idx
    on am_episodic_memory using gin (tool_sequence);

create index if not exists am_failure_tool_input_gin_idx
    on am_failure_episodes using gin (tool_input);

create index if not exists am_procedural_tool_sequence_gin_idx
    on am_procedural_workflows using gin (tool_sequence);

create index if not exists am_procedural_trigger_phrases_gin_idx
    on am_procedural_workflows using gin (trigger_phrases);

create index if not exists am_semantic_embedding_ivfflat_idx
    on am_semantic_memory using ivfflat (embedding vector_cosine_ops) with (lists = 100)
    where embedding is not null;

create index if not exists am_semantic_hash_embedding_ivfflat_idx
    on am_semantic_memory using ivfflat (hash_embedding vector_cosine_ops) with (lists = 100)
    where hash_embedding is not null;

create index if not exists am_semantic_hierarchy_facet_idx
    on am_semantic_hierarchy_nodes (facet, node_type);

create index if not exists am_semantic_hierarchy_embedding_ivfflat_idx
    on am_semantic_hierarchy_nodes using ivfflat (embedding vector_cosine_ops) with (lists = 100)
    where embedding is not null;

create index if not exists am_semantic_hierarchy_hash_embedding_ivfflat_idx
    on am_semantic_hierarchy_nodes using ivfflat (hash_embedding vector_cosine_ops) with (lists = 100)
    where hash_embedding is not null;

create index if not exists am_episodic_embedding_ivfflat_idx
    on am_episodic_memory using ivfflat (prompt_embedding vector_cosine_ops) with (lists = 100)
    where prompt_embedding is not null;

create index if not exists am_episodic_hash_embedding_ivfflat_idx
    on am_episodic_memory using ivfflat (prompt_embedding_hash vector_cosine_ops) with (lists = 100)
    where prompt_embedding_hash is not null;

create index if not exists am_failure_embedding_ivfflat_idx
    on am_failure_episodes using ivfflat (prompt_embedding vector_cosine_ops) with (lists = 100)
    where prompt_embedding is not null;

create index if not exists am_failure_hash_embedding_ivfflat_idx
    on am_failure_episodes using ivfflat (prompt_embedding_hash vector_cosine_ops) with (lists = 100)
    where prompt_embedding_hash is not null;

create index if not exists am_procedural_embedding_ivfflat_idx
    on am_procedural_workflows using ivfflat (embedding vector_cosine_ops) with (lists = 100)
    where embedding is not null;

create index if not exists am_procedural_hash_embedding_ivfflat_idx
    on am_procedural_workflows using ivfflat (hash_embedding vector_cosine_ops) with (lists = 100)
    where hash_embedding is not null;

-- Match functions for server-side vector search

create or replace function match_semantic_memory(
    query_embedding vector,
    threshold float,
    "limit" int
)
returns table (
    fact_id uuid,
    fact_type text,
    content text,
    confidence_score double precision,
    source text,
    source_episode_id uuid,
    pinned boolean,
    created_at timestamptz,
    last_reinforced_at timestamptz,
    last_confirmed_at timestamptz,
    similarity double precision
)
language sql stable
as $$
    select
        m.fact_id,
        m.fact_type,
        m.content,
        m.confidence_score,
        m.source,
        m.source_episode_id,
        m.pinned,
        m.created_at,
        m.last_reinforced_at,
        m.last_confirmed_at,
        1 - (m.embedding <=> $1) as similarity
    from am_semantic_memory m
    where m.embedding is not null
      and 1 - (m.embedding <=> $1) >= $2
    order by m.embedding <=> $1
    limit $3;
$$;

create or replace function match_episodic_memory(
    query_embedding vector,
    threshold float,
    "limit" int
)
returns table (
    episode_id uuid,
    prompt_text text,
    reasoning_summary text,
    tool_sequence jsonb,
    final_response text,
    outcome text,
    error_trace text,
    latency_ms integer,
    timestamp timestamptz,
    similarity double precision
)
language sql stable
as $$
    select
        e.episode_id,
        e.prompt_text,
        e.reasoning_summary,
        e.tool_sequence,
        e.final_response,
        e.outcome,
        e.error_trace,
        e.latency_ms,
        e.timestamp,
        1 - (e.prompt_embedding <=> $1) as similarity
    from am_episodic_memory e
    where e.prompt_embedding is not null
      and 1 - (e.prompt_embedding <=> $1) >= $2
    order by e.prompt_embedding <=> $1
    limit $3;
$$;

create or replace function match_failure_episodes(
    query_embedding vector,
    threshold float,
    "limit" int
)
returns table (
    failure_id uuid,
    episode_id uuid,
    prompt_text text,
    tool_name text,
    tool_input jsonb,
    exception_message text,
    error_trace text,
    timestamp timestamptz,
    similarity double precision
)
language sql stable
as $$
    select
        f.failure_id,
        f.episode_id,
        f.prompt_text,
        f.tool_name,
        f.tool_input,
        f.exception_message,
        f.error_trace,
        f.timestamp,
        1 - (f.prompt_embedding <=> $1) as similarity
    from am_failure_episodes f
    where f.prompt_embedding is not null
      and 1 - (f.prompt_embedding <=> $1) >= $2
    order by f.prompt_embedding <=> $1
    limit $3;
$$;

create or replace function match_procedural_workflows(
    query_embedding vector,
    threshold float,
    "limit" int
)
returns table (
    workflow_id uuid,
    workflow_signature text,
    trigger_phrases text[],
    tool_sequence jsonb,
    success_count integer,
    status text,
    avg_latency_ms double precision,
    created_at timestamptz,
    updated_at timestamptz,
    similarity double precision
)
language sql stable
as $$
    select
        p.workflow_id,
        p.workflow_signature,
        p.trigger_phrases,
        p.tool_sequence,
        p.success_count,
        p.status,
        p.avg_latency_ms,
        p.created_at,
        p.updated_at,
        1 - (p.embedding <=> $1) as similarity
    from am_procedural_workflows p
    where p.embedding is not null
      and 1 - (p.embedding <=> $1) >= $2
    order by p.embedding <=> $1
    limit $3;
$$;
