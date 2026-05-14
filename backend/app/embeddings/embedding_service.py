"""
embedding_service.py — Async Batch Embedding Service
=====================================================
Wraps BAAI/bge-base-en-v1.5 (768-dim) for async use via thread pool.

Key design choices:
  - Singleton pattern — model loads once per process
  - asyncio.get_event_loop().run_in_executor() — never blocks the event loop
  - BGE-specific instruction prefix for query embeddings (improves retrieval)
  - Batch size 32 — balances memory vs throughput
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# BGE models perform best when query text is prefixed with an instruction
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_DOC_INSTRUCTION = ""  # No prefix needed for document/chunk embeddings

BATCH_SIZE = 32


class EmbeddingService:
    """
    Async singleton embedding service using BAAI/bge-base-en-v1.5.

    Usage:
        svc = EmbeddingService()
        vectors = await svc.embed_texts(["hello world", "another sentence"])
        query_vec = await svc.embed_query("show my 2017 electricity bill")
    """

    _instance: "EmbeddingService | None" = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _initialize(self) -> None:
        if self._initialized:
            return
        model_name = settings.BGE_MODEL_NAME
        logger.info("[EMBED] Loading embedding model: %s …", model_name)
        self._model = SentenceTransformer(model_name)
        self._vector_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "[EMBED] Model loaded. Vector dimension: %d", self._vector_dim
        )
        self._initialized = True

    # ── Synchronous helpers (called inside executor) ───────────────────────────

    def _encode_sync(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Synchronous encoding — called from thread pool to avoid blocking.
        BGE query embeddings benefit from a task-specific instruction prefix.
        """
        self._initialize()
        if is_query:
            texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
        embeddings = self._model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,   # cosine similarity — unit norm required
            show_progress_bar=False,
        )
        return embeddings.tolist()

    # ── Public async API ───────────────────────────────────────────────────────

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts (chunks, summaries, titles).
        No instruction prefix applied — pure document representation.
        """
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts, False)

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single document text string."""
        results = await self.embed_texts([text])
        return results[0] if results else []

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a natural language query with BGE instruction prefix.
        Use this for query-time encoding; use embed_texts() for indexing.
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._encode_sync, [query], True)
        return results[0] if results else []

    async def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Batch embed multiple queries (e.g. for batch search)."""
        if not queries:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, queries, True)

    @property
    def vector_dim(self) -> int:
        """Return embedding dimension (768 for bge-base-en-v1.5)."""
        self._initialize()
        return self._vector_dim


# ── Module-level singleton ─────────────────────────────────────────────────────
embedding_service = EmbeddingService()
