"""Offline three-turn demo for the self-learning loop.

Runs entirely locally using SQLite storage, hash embeddings, and
the deterministic structured client. No cloud dependencies required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import configure_logging
from core.memory_service import AgenticMemoryService
from model.embedding_model import HashEmbeddingModel
from model.groq_client import DeterministicStructuredClient
from planner.retrieval_planner import HeuristicRetrievalPlanner
from runtime.actor import Actor
from runtime.critic import Critic
from runtime.self_learning_loop import SelfLearningLoop
from runtime.tools import default_tool_registry
from store.sqlite_store import SQLiteMemoryStore


async def run_demo() -> None:
    """Run three loop turns with hash embeddings and deterministic local LLM behavior."""
    configure_logging("INFO")

    db_path = os.environ.get("SQLITE_DB_PATH", "demo_memory.db")
    store = await SQLiteMemoryStore.create(db_path)
    embedding_model = HashEmbeddingModel()
    llm_client = DeterministicStructuredClient()
    registry = default_tool_registry(simulate_web_search=True)
    planner = HeuristicRetrievalPlanner(store, embedding_model, memory_window_turns=3)
    memory_service = AgenticMemoryService(store, embedding_model, memory_window_turns=3)
    loop = SelfLearningLoop(
        planner=planner,
        memory_service=memory_service,
        actor=Actor(llm_client, "deterministic-actor", registry),
        critic=Critic(llm_client, "deterministic-critic"),
    )
    prompts = [
        "I prefer concise bullet-style answers. Calculate 21 * 2.",
        "Search the web for agentic memory patterns.",
        "Calculate 10 + 5 and search the web for the latest memory workflow.",
    ]
    for prompt in prompts:
        result = await loop.run_turn(prompt, "demo-session")
        print(result.model_dump_json(indent=2))

    await store.close()
    print(f"\nDemo complete. Memory persisted to {db_path}")


if __name__ == "__main__":
    asyncio.run(run_demo())
