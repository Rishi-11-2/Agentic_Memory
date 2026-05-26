"""FastAPI service for the standalone self-learning Agentic Memory backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from config import Settings, load_settings
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


app = FastAPI(
    title="Agentic Memory Self-Learning Loop",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(openai_router, prefix="/v1")


@app.post("/chat", response_model=LoopResult)
async def chat(request: ChatRequest) -> LoopResult:
    """Run one self-learning chat turn."""
    components = _components()
    return await components.loop.run_turn(request.user_message, request.session_id)


@app.get("/memory/inspect", response_model=MemoryInspectResponse)
async def inspect_memory(
    layer: MemoryLayer,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MemoryInspectResponse:
    """Return paginated memory records for debugging and admin dashboards."""
    components = _components()
    records = await components.store.inspect_layer(layer, limit, offset)
    return MemoryInspectResponse(
        layer=layer,
        limit=limit,
        offset=offset,
        records=records,
    )


@app.get("/debug/turn/{session_id}/{turn_index}")
async def debug_turn(
    session_id: str,
    turn_index: int,
) -> dict[str, object]:
    """Return the raw conversation records for one session turn."""
    components = _components()
    records = await components.store.conversation_turn(session_id, turn_index)
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
