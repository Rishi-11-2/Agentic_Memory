-- Agentic Memory PostgreSQL schema.
-- Run this file against a PostgreSQL database with pgvector enabled.

create extension if not exists vector;
create extension if not exists pgcrypto;

create or replace function am_scope_hash(application_id text, tenant_id text, user_id text)
returns text
language sql immutable
as $$
    select encode(digest(application_id || tenant_id || user_id, 'sha256'), 'hex');
$$;

create table if not exists am_conversation_turns (
    id uuid primary key default gen_random_uuid(),
    scope_hash text not null,
    session_id text not null,
    turn_index integer not null check (turn_index >= 0),
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    token_count integer not null default 0 check (token_count >= 0),
    timestamp timestamptz not null default now(),
    unique (scope_hash, session_id, turn_index, role)
);

create table if not exists am_conversation_summaries (
    summary_id uuid primary key default gen_random_uuid(),
    scope_hash text not null,
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
    scope_hash text not null,
    prompt_text text not null,
    prompt_embedding vector(384),
    prompt_embedding_hash vector(256),
    tool_sequence jsonb not null default '[]'::jsonb,
    final_response text not null default '',
    outcome text not null check (outcome in ('success', 'partial', 'failure')),
    error_trace text,
    latency_ms integer not null default 0 check (latency_ms >= 0),
    timestamp timestamptz not null default now()
);

create table if not exists am_failure_episodes (
    failure_id uuid primary key default gen_random_uuid(),
    scope_hash text not null,
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
    scope_hash text not null,
    fact_type text not null check (fact_type in ('preference', 'inferred_fact', 'system_rule')),
    content text not null,
    embedding vector(384),
    hash_embedding vector(256),
    confidence_score double precision not null check (confidence_score >= 0.0 and confidence_score <= 1.0),
    source text not null default 'llm_inferred',
    source_episode_id uuid references am_episodic_memory(episode_id) on delete set null,
    created_at timestamptz not null default now(),
    last_reinforced_at timestamptz not null default now(),
    last_confirmed_at timestamptz not null default now()
);

alter table am_semantic_memory
    add column if not exists source text not null default 'llm_inferred';

alter table am_semantic_memory
    add column if not exists last_confirmed_at timestamptz;

update am_semantic_memory
set last_confirmed_at = coalesce(last_confirmed_at, last_reinforced_at, created_at, now())
where last_confirmed_at is null;

alter table am_semantic_memory
    alter column last_confirmed_at set not null;

create table if not exists am_procedural_workflows (
    workflow_id uuid primary key default gen_random_uuid(),
    scope_hash text not null,
    workflow_signature text not null,
    trigger_phrases text[] not null default '{}',
    tool_sequence jsonb not null default '[]'::jsonb,
    success_count integer not null default 1 check (success_count >= 1),
    status text not null default 'candidate' check (status in ('candidate', 'canonical')),
    avg_latency_ms double precision not null default 0.0 check (avg_latency_ms >= 0.0),
    embedding vector(384),
    hash_embedding vector(256),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (scope_hash, workflow_signature)
);

create index if not exists am_conversation_scope_session_idx
    on am_conversation_turns (scope_hash, session_id, turn_index desc, timestamp desc);

create index if not exists am_summaries_scope_session_idx
    on am_conversation_summaries (scope_hash, session_id, created_at desc);

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

create or replace function match_semantic_memory(
    query_embedding vector,
    scope_hash text,
    threshold float,
    "limit" int
)
returns table (
    fact_id uuid,
    scope_hash text,
    fact_type text,
    content text,
    confidence_score double precision,
    source text,
    source_episode_id uuid,
    created_at timestamptz,
    last_reinforced_at timestamptz,
    last_confirmed_at timestamptz,
    similarity double precision
)
language sql stable
as $$
    select
        m.fact_id,
        m.scope_hash,
        m.fact_type,
        m.content,
        m.confidence_score,
        m.source,
        m.source_episode_id,
        m.created_at,
        m.last_reinforced_at,
        m.last_confirmed_at,
        1 - (m.embedding <=> $1) as similarity
    from am_semantic_memory m
    where m.scope_hash = $2
      and m.embedding is not null
      and 1 - (m.embedding <=> $1) >= $3
    order by m.embedding <=> $1
    limit $4;
$$;

create or replace function match_episodic_memory(
    query_embedding vector,
    scope_hash text,
    threshold float,
    "limit" int
)
returns table (
    episode_id uuid,
    scope_hash text,
    prompt_text text,
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
        e.scope_hash,
        e.prompt_text,
        e.tool_sequence,
        e.final_response,
        e.outcome,
        e.error_trace,
        e.latency_ms,
        e.timestamp,
        1 - (e.prompt_embedding <=> $1) as similarity
    from am_episodic_memory e
    where e.scope_hash = $2
      and e.prompt_embedding is not null
      and 1 - (e.prompt_embedding <=> $1) >= $3
    order by e.prompt_embedding <=> $1
    limit $4;
$$;

create or replace function match_failure_episodes(
    query_embedding vector,
    scope_hash text,
    threshold float,
    "limit" int
)
returns table (
    failure_id uuid,
    scope_hash text,
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
        f.scope_hash,
        f.episode_id,
        f.prompt_text,
        f.tool_name,
        f.tool_input,
        f.exception_message,
        f.error_trace,
        f.timestamp,
        1 - (f.prompt_embedding <=> $1) as similarity
    from am_failure_episodes f
    where f.scope_hash = $2
      and f.prompt_embedding is not null
      and 1 - (f.prompt_embedding <=> $1) >= $3
    order by f.prompt_embedding <=> $1
    limit $4;
$$;

create or replace function match_procedural_workflows(
    query_embedding vector,
    scope_hash text,
    threshold float,
    "limit" int
)
returns table (
    workflow_id uuid,
    scope_hash text,
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
        p.scope_hash,
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
    where p.scope_hash = $2
      and p.embedding is not null
      and 1 - (p.embedding <=> $1) >= $3
    order by p.embedding <=> $1
    limit $4;
$$;

alter table am_conversation_turns enable row level security;
alter table am_conversation_summaries enable row level security;
alter table am_episodic_memory enable row level security;
alter table am_failure_episodes enable row level security;
alter table am_semantic_memory enable row level security;
alter table am_procedural_workflows enable row level security;

create or replace function am_jwt_scope_hash()
returns text
language sql stable
as $$
    select coalesce(
        auth.jwt() ->> 'scope_hash',
        nullif(current_setting('request.jwt.claim.scope_hash', true), '')
    );
$$;

drop policy if exists "scope read conversation" on am_conversation_turns;
drop policy if exists "scope write conversation" on am_conversation_turns;
drop policy if exists "scope read summaries" on am_conversation_summaries;
drop policy if exists "scope write summaries" on am_conversation_summaries;
drop policy if exists "scope read episodic" on am_episodic_memory;
drop policy if exists "scope write episodic" on am_episodic_memory;
drop policy if exists "scope read failures" on am_failure_episodes;
drop policy if exists "scope write failures" on am_failure_episodes;
drop policy if exists "scope read semantic" on am_semantic_memory;
drop policy if exists "scope write semantic" on am_semantic_memory;
drop policy if exists "scope read procedural" on am_procedural_workflows;
drop policy if exists "scope write procedural" on am_procedural_workflows;

create policy "scope read conversation" on am_conversation_turns
    for select using (scope_hash = am_jwt_scope_hash());
create policy "scope write conversation" on am_conversation_turns
    for all using (scope_hash = am_jwt_scope_hash()) with check (scope_hash = am_jwt_scope_hash());

create policy "scope read summaries" on am_conversation_summaries
    for select using (scope_hash = am_jwt_scope_hash());
create policy "scope write summaries" on am_conversation_summaries
    for all using (scope_hash = am_jwt_scope_hash()) with check (scope_hash = am_jwt_scope_hash());

create policy "scope read episodic" on am_episodic_memory
    for select using (scope_hash = am_jwt_scope_hash());
create policy "scope write episodic" on am_episodic_memory
    for all using (scope_hash = am_jwt_scope_hash()) with check (scope_hash = am_jwt_scope_hash());

create policy "scope read failures" on am_failure_episodes
    for select using (scope_hash = am_jwt_scope_hash());
create policy "scope write failures" on am_failure_episodes
    for all using (scope_hash = am_jwt_scope_hash()) with check (scope_hash = am_jwt_scope_hash());

create policy "scope read semantic" on am_semantic_memory
    for select using (scope_hash = am_jwt_scope_hash());
create policy "scope write semantic" on am_semantic_memory
    for all using (scope_hash = am_jwt_scope_hash()) with check (scope_hash = am_jwt_scope_hash());

create policy "scope read procedural" on am_procedural_workflows
    for select using (scope_hash = am_jwt_scope_hash());
create policy "scope write procedural" on am_procedural_workflows
    for all using (scope_hash = am_jwt_scope_hash()) with check (scope_hash = am_jwt_scope_hash());
