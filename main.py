"""FastAPI REST interface for Agentic Memory.

Secondary interface alongside the MCP server. Provides the same memory
operations over HTTP for services that can't use MCP directly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query

from config import load_settings
from core.memory_service import AgenticMemoryService
from core.models import MemoryInspectResponse, MemoryLayer
from model.embedding_model import create_embedding_model
from planner.retrieval_planner import HeuristicRetrievalPlanner
from store.base import MemoryStore
from store.factory import create_memory_store


class AppComponents:
    """Hold initialized backend components for FastAPI request handlers."""

    def __init__(self, store: MemoryStore, planner: HeuristicRetrievalPlanner, memory_service: AgenticMemoryService) -> None:
        """Create a component bundle for the app lifespan."""
        self.store = store
        self.planner = planner
        self.memory_service = memory_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize store, embedding model, planner, and memory service at startup."""
    settings = load_settings()
    store = await create_memory_store(settings)
    embedding_model = create_embedding_model(settings)
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
    app.state.components = AppComponents(store=store, planner=planner, memory_service=memory_service)
    yield


app = FastAPI(
    title="Agentic Memory",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/memory/inspect", response_model=MemoryInspectResponse)
async def inspect_memory(
    layer: MemoryLayer,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MemoryInspectResponse:
    """Return paginated memory records for debugging."""
    components = _components()
    records = await components.store.inspect_layer(layer, limit, offset)
    return MemoryInspectResponse(
        layer=layer,
        limit=limit,
        offset=offset,
        records=records,
    )


@app.get("/debug/turn/{session_id}/{turn_index}")
async def debug_turn(session_id: str, turn_index: int) -> dict[str, object]:
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
