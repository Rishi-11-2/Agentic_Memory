"""Embedding models used by vectorized Agentic Memory retrieval."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from config import Settings

CallableBatchEmbedder = Callable[[list[str]], Awaitable[list[list[float]]]]


class EmbeddingModel(Protocol):
    """Protocol for explicit, swappable text embedding implementations."""

    @property
    def dimensions(self) -> int:
        """Return the vector dimensionality written to the memory store."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed text asynchronously without blocking the event loop."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts asynchronously."""
        ...


class HashEmbeddingModel:
    """Zero-dependency fallback embedding model ported from the Java implementation."""

    def __init__(self, dimensions: int = 256) -> None:
        """Create a deterministic normalized hashing embedding model."""
        self._dimensions = max(32, dimensions)

    @property
    def dimensions(self) -> int:
        """Return the configured hash-vector dimensionality."""
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Embed text with Java-style token hashing and L2 normalization."""
        return self._embed_sync(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with deterministic hashing."""
        return [self._embed_sync(text) for text in texts]

    def _embed_sync(self, text: str) -> list[float]:
        """Embed text synchronously for cheap hash-vector batches."""
        vector = [0.0 for _ in range(self._dimensions)]
        tokens = _normalized_tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            index = _java_remainder(_java_string_hash(token), self._dimensions)
            vector[index] += 1.0
        return _l2_normalize(vector)


class SentenceTransformerEmbeddingModel:
    """Primary embedding model backed by sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Create a lazy-loading sentence-transformer embedding adapter."""
        self._model_name = model_name
        self._model: Any | None = None
        self._lock = asyncio.Lock()
        self._batcher = _BatchWindow(self._embed_batch_now, max_batch_size=16, window_seconds=0.05)

    @property
    def dimensions(self) -> int:
        """Return the all-MiniLM-L6-v2 vector dimensionality."""
        return 384

    async def embed(self, text: str) -> list[float]:
        """Embed text using the shared batch window."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using sentence-transformers batched inference."""
        if not texts:
            return []
        return await self._batcher.submit(texts)

    async def _embed_batch_now(self, texts: list[str]) -> list[list[float]]:
        """Run one concrete sentence-transformers batch in a worker thread."""
        model = await self._get_model()
        result = await asyncio.to_thread(lambda: model.encode(texts, normalize_embeddings=True))
        raw = result.tolist() if hasattr(result, "tolist") else list(result)
        return [[float(value) for value in row] for row in raw]

    async def _get_model(self) -> Any:
        """Load the transformer model once without blocking the event loop."""
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self) -> Any:
        """Import and instantiate SentenceTransformer lazily for hash-only deployments."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name)


def create_embedding_model(settings: Settings) -> EmbeddingModel:
    """Create the configured embedding model while keeping callers provider-agnostic."""
    if settings.embedding_backend == "hash":
        return HashEmbeddingModel(settings.hash_embedding_dimensions)
    return SentenceTransformerEmbeddingModel(settings.embedding_model_name)


@dataclass
class _BatchRequest:
    """Hold one embedding request inside a short batch window."""

    texts: list[str]
    future: asyncio.Future[list[list[float]]]


class _BatchWindow:
    """Accumulate embedding requests for a short window before running inference."""

    def __init__(
        self,
        embedder: CallableBatchEmbedder,
        max_batch_size: int,
        window_seconds: float,
    ) -> None:
        """Create a bounded async batcher for embedding inference."""
        self._embedder = embedder
        self._max_batch_size = max_batch_size
        self._window_seconds = window_seconds
        self._pending: list[_BatchRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    async def submit(self, texts: list[str]) -> list[list[float]]:
        """Queue texts and return their embeddings after the current batch flushes."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[list[float]]] = loop.create_future()
        async with self._lock:
            self._pending.append(_BatchRequest(texts=texts, future=future))
            total_items = sum(len(request.texts) for request in self._pending)
            if total_items >= self._max_batch_size:
                await self._flush_locked()
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._delayed_flush())
        return await future

    async def _delayed_flush(self) -> None:
        """Flush after the configured batch window expires."""
        await asyncio.sleep(self._window_seconds)
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Flush pending requests. Caller must hold the lock."""
        if not self._pending:
            return
        requests = self._pending
        self._pending = []
        all_texts = [text for request in requests for text in request.texts]
        try:
            embeddings = await self._embedder(all_texts)
        except Exception as exc:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(exc)
            return
        offset = 0
        for request in requests:
            count = len(request.texts)
            if not request.future.done():
                request.future.set_result(embeddings[offset : offset + count])
            offset += count


def _normalized_tokens(text: str) -> list[str]:
    """Tokenize text with the same normalization semantics as the Java reference."""
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.lower().strip())
    return [part for part in normalized.split() if part]


def _java_string_hash(value: str) -> int:
    """Compute Java's String.hashCode value for deterministic cross-language hashes."""
    result = 0
    for char in value:
        result = (31 * result + ord(char)) & 0xFFFFFFFF
    if result & 0x80000000:
        result -= 0x100000000
    return result


def _java_remainder(value: int, modulus: int) -> int:
    """Return Java Math.floorMod-compatible indices for hashed tokens."""
    return ((value % modulus) + modulus) % modulus


def _l2_normalize(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length while preserving all-zero vectors."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
