"""FastAPI service for the standalone self-learning Agentic Memory backend."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from config import Settings, load_settings
from core.access_scope import AccessScope
from core.memory_service import AgenticMemoryService
from core.models import ChatRequest, LoopResult, MemoryInspectResponse, MemoryLayer
from model import actor_model_name, create_llm_client, critic_model_name
from model.embedding_model import create_embedding_model
from planner.retrieval_planner import HeuristicRetrievalPlanner
from runtime.actor import Actor
from runtime.critic import Critic
from runtime.openai_adapter import router as openai_router
from runtime.self_learning_loop import SelfLearningLoop
from runtime.tools import default_tool_registry
from store.base import MemoryStore
from store.factory import create_memory_store

_RATE_LIMIT_WINDOWS: defaultdict[str, deque[float]] = defaultdict(deque)


class AppComponents:
    """Hold initialized backend components for FastAPI request handlers."""

    def __init__(self, settings: Settings, store: MemoryStore, loop: SelfLearningLoop) -> None:
        """Create an immutable component bundle for the app lifespan."""
        self.settings = settings
        self.store = store
        self.loop = loop


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize settings, store, models, and loop components at startup."""
    settings = load_settings()
    store = await create_memory_store(settings)
    embedding_model = create_embedding_model(settings)
    llm_client = create_llm_client(settings)
    tool_registry = default_tool_registry(
        brave_api_key=(
            settings.brave_search_api_key.get_secret_value() if settings.brave_search_api_key is not None else None
        ),
        brave_endpoint=settings.brave_search_endpoint,
        brave_country=settings.brave_search_country,
        brave_search_lang=settings.brave_search_lang,
        brave_count=settings.brave_search_count,
        brave_timeout_seconds=settings.brave_search_timeout_seconds,
        workspace_root=settings.tool_workspace_root,
        memory_store=store,
        embedding_model=embedding_model,
    )
    planner = HeuristicRetrievalPlanner(
        store=store,
        embedding_model=embedding_model,
        memory_window_turns=settings.memory_window_turns,
        failure_similarity_threshold=settings.failure_similarity_threshold,
        semantic_memory_ttl_days=settings.semantic_memory_ttl_days,
    )
    memory_service = AgenticMemoryService(
        store=store,
        embedding_model=embedding_model,
        memory_window_turns=settings.memory_window_turns,
        semantic_dedup_threshold=settings.semantic_dedup_threshold,
        semantic_memory_ttl_days=settings.semantic_memory_ttl_days,
    )
    actor = Actor(llm_client=llm_client, model=actor_model_name(settings), tool_registry=tool_registry)
    critic = Critic(llm_client=llm_client, model=critic_model_name(settings))
    app.state.components = AppComponents(
        settings=settings,
        store=store,
        loop=SelfLearningLoop(
            planner=planner,
            memory_service=memory_service,
            actor=actor,
            critic=critic,
            critic_timeout_seconds=settings.critic_timeout_seconds,
            critic_self_consistency_samples=settings.critic_self_consistency_samples,
            critic_self_consistency_temperature=settings.critic_self_consistency_temperature,
        ),
    )
    yield


class RateLimitExceeded(Exception):
    """Raised when a scope exceeds the in-memory request budget."""


async def check_rate_limit(request: Request) -> None:
    """Apply a per-scope-hash sliding-window request limit."""
    components = getattr(request.app.state, "components", None)
    settings = components.settings if isinstance(components, AppComponents) else load_settings()
    if settings.rate_limit_rpm <= 0:
        return
    scope_hash = await _scope_hash_from_request(request)
    now = time.monotonic()
    window = _RATE_LIMIT_WINDOWS[scope_hash]
    while window and now - window[0] >= 60.0:
        window.popleft()
    if len(window) >= settings.rate_limit_rpm:
        raise RateLimitExceeded()
    window.append(now)


async def _scope_hash_from_request(request: Request) -> str:
    """Derive a scope hash from headers, query parameters, or JSON body."""
    if all(request.headers.get(name) for name in ("X-Application-Id", "X-Tenant-Id", "X-User-Id")):
        return AccessScope(
            application_id=request.headers["X-Application-Id"],
            tenant_id=request.headers["X-Tenant-Id"],
            user_id=request.headers["X-User-Id"],
        ).scope_hash

    query = request.query_params
    if all(query.get(name) for name in ("application_id", "tenant_id", "user_id")):
        return AccessScope(
            application_id=str(query["application_id"]),
            tenant_id=str(query["tenant_id"]),
            user_id=str(query["user_id"]),
        ).scope_hash

    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict) and all(body.get(name) for name in ("application_id", "tenant_id", "user_id")):
            return AccessScope(
                application_id=str(body["application_id"]),
                tenant_id=str(body["tenant_id"]),
                user_id=str(body["user_id"]),
            ).scope_hash
    return "anonymous"


app = FastAPI(
    title="Agentic Memory Self-Learning Loop",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(check_rate_limit)],
)
app.include_router(openai_router, prefix="/v1")


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return the exact rate-limit response body expected by clients."""
    del request, exc
    return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})


@app.post("/chat", response_model=LoopResult)
async def chat(request: ChatRequest) -> LoopResult:
    """Run one self-learning chat turn without exposing Critic internals."""
    components = _components()
    scope = AccessScope(
        application_id=request.application_id,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
    )
    return await components.loop.run_turn(request.user_message, request.session_id, scope)


@app.get("/memory/inspect", response_model=MemoryInspectResponse)
async def inspect_memory(
    application_id: str,
    tenant_id: str,
    user_id: str,
    layer: MemoryLayer,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MemoryInspectResponse:
    """Return paginated scoped memory records for debugging and admin dashboards."""
    components = _components()
    scope = AccessScope(application_id=application_id, tenant_id=tenant_id, user_id=user_id)
    records = await components.store.inspect_layer(scope.scope_hash, layer, limit, offset)
    return MemoryInspectResponse(
        layer=layer,
        scope_hash=scope.scope_hash,
        limit=limit,
        offset=offset,
        records=records,
    )


@app.get("/debug/turn/{session_id}/{turn_index}")
async def debug_turn(
    session_id: str,
    turn_index: int,
    x_debug_key: str = Header(default="", alias="X-Debug-Key"),
    x_application_id: str = Header(default="default-application", alias="X-Application-Id"),
    x_tenant_id: str = Header(default="default-tenant", alias="X-Tenant-Id"),
    x_user_id: str = Header(default="default-user", alias="X-User-Id"),
) -> dict[str, object]:
    """Return the raw conversation records for one session turn when debug access is enabled."""
    components = _components()
    if not components.settings.debug_key or x_debug_key != components.settings.debug_key:
        raise HTTPException(status_code=404, detail="Debug endpoint is disabled")
    scope = AccessScope(application_id=x_application_id, tenant_id=x_tenant_id, user_id=x_user_id)
    records = await components.store.conversation_turn(scope.scope_hash, session_id, turn_index)
    if not records:
        raise HTTPException(status_code=404, detail="Turn not found")
    return {
        "session_id": session_id,
        "turn_index": turn_index,
        "records": [record.model_dump(mode="json") for record in records],
    }


def _components() -> AppComponents:
    """Return initialized components or raise a clear service initialization error."""
    components = getattr(app.state, "components", None)
    if not isinstance(components, AppComponents):
        raise HTTPException(status_code=503, detail="Application components are not initialized")
    return components
