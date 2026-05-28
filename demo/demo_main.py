"""Three-turn demo for the self-learning loop.

Supports two modes:
1. **Offline** (default): Uses SQLite, hash embeddings, and the deterministic
   structured client. No cloud dependencies required.
2. **LLM-powered**: Set LLM_PROVIDER and its API key (e.g. ANTHROPIC_API_KEY)
   to use a real LLM for both Actor and Critic. One key is all you need.

Examples:
    # Offline demo (zero config)
    python -m demo.demo_main

    # LLM-powered demo (one API key)
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... python -m demo.demo_main
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import configure_logging, load_settings
from core.memory_service import AgenticMemoryService
from model import create_llm_client, actor_model_name, critic_model_name
from model.embedding_model import HashEmbeddingModel, create_embedding_model
from model.groq_client import DeterministicStructuredClient
from planner.retrieval_planner import HeuristicRetrievalPlanner
from runtime.actor import Actor
from runtime.critic import Critic
from runtime.self_learning_loop import SelfLearningLoop
from runtime.tools import default_tool_registry
from store.sqlite_store import SQLiteMemoryStore


async def run_demo() -> None:
    """Run three loop turns, auto-detecting whether to use a real LLM or deterministic mode."""
    settings = load_settings()
    provider = settings.llm_provider
    use_real_llm = provider != "deterministic"

    print(f"🧠 Agentic Memory Demo — LLM provider: {provider}")

    db_path = os.environ.get("SQLITE_DB_PATH", "demo_memory.db")
    store = await SQLiteMemoryStore.create(db_path)

    # Use real embeddings when an LLM is configured, hash embeddings otherwise
    if use_real_llm:
        embedding_model = create_embedding_model(settings)
    else:
        embedding_model = HashEmbeddingModel()

    # Create the LLM client — one key is all you need
    if use_real_llm:
        llm_client = create_llm_client(settings)
        actor_model = actor_model_name(settings)
        critic_model = critic_model_name(settings)
        print(f"   Actor model: {actor_model}")
        print(f"   Critic model: {critic_model}")
    else:
        llm_client = DeterministicStructuredClient()
        actor_model = "deterministic-actor"
        critic_model = "deterministic-critic"

    registry = default_tool_registry(
        simulate_web_search=not use_real_llm,
        workspace_root=settings.tool_workspace_root,
        memory_store=store,
        embedding_model=embedding_model,
    )
    planner = HeuristicRetrievalPlanner(store, embedding_model, memory_window_turns=3)
    memory_service = AgenticMemoryService(store, embedding_model, memory_window_turns=3)
    loop = SelfLearningLoop(
        planner=planner,
        memory_service=memory_service,
        actor=Actor(llm_client, actor_model, registry),
        critic=Critic(llm_client, critic_model),
    )
    prompts = [
        "I prefer concise bullet-style answers. Calculate 21 * 2.",
        "Search the web for agentic memory patterns.",
        "Calculate 10 + 5 and search the web for the latest memory workflow.",
    ]
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}: {prompt}")
        print(f"{'='*60}")
        result = await loop.run_turn(prompt, "demo-session")
        print(result.model_dump_json(indent=2))

    await store.close()
    print(f"\n✅ Demo complete. Memory persisted to {db_path}")


if __name__ == "__main__":
    asyncio.run(run_demo())
