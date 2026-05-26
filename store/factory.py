"""Factory for creating the configured memory-store backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Settings
    from store.base import MemoryStore


async def create_memory_store(settings: "Settings") -> "MemoryStore":
    """Instantiate the memory store selected by MEMORY_BACKEND.

    Supported backends:
        sqlite   — Local persistent SQLite file with client-side vector search.
        postgres — Local or remote PostgreSQL with pgvector server-side search.
    """
    backend = settings.memory_backend

    if backend == "sqlite":
        from store.sqlite_store import SQLiteMemoryStore

        return await SQLiteMemoryStore.create(settings.sqlite_db_path)

    if backend == "postgres":
        from store.postgres_store import PostgresMemoryStore

        return await PostgresMemoryStore.create(settings.postgres_dsn)

    raise ValueError(f"Unknown MEMORY_BACKEND: {backend!r}. Must be 'sqlite' or 'postgres'.")
