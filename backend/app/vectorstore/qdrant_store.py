"""
qdrant_store.py — Qdrant Vector Store for Document Chunks
=========================================================
Manages the `document_chunks` collection (768-dim, Cosine).

Separation from the legacy `vector_db.py`:
  - `vector_db.py` → `user_documents` (384-dim, all-MiniLM-L6-v2) — unchanged
  - `qdrant_store.py` → `document_chunks` (768-dim, BAAI/bge-base-en-v1.5) — new

Payload schema per chunk:
  user_id, document_id, chunk_id, document_type, filename,
  summary, text, date, year, month, entities, tags, metadata,
  chunk_index, page_number
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import settings
from app.core.qdrant_client_singleton import shared_qdrant_client

logger = logging.getLogger(__name__)

VECTOR_DIM = 768          # BAAI/bge-base-en-v1.5
COLLECTION  = "document_chunks"


class QdrantStore:
    """
    Async-friendly Qdrant wrapper for the document_chunks collection.

    The QdrantClient itself is synchronous; async methods use
    asyncio.get_event_loop().run_in_executor() to avoid blocking.
    """

    _instance: "QdrantStore | None" = None

    def __new__(cls) -> "QdrantStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # ── Initialization ─────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        if self._initialized:
            return

        # Use the shared client — prevents storage lock conflict with VectorDBService
        self._client = shared_qdrant_client
        self._collection = COLLECTION
        self._ensure_collection()
        self._initialized = True

    @property
    def client(self) -> QdrantClient:
        self._initialize()
        return self._client

    @property
    def collection(self) -> str:
        return COLLECTION

    def _ensure_collection(self) -> None:
        """Create document_chunks collection + payload indexes if missing."""
        existing = [c.name for c in self._client.get_collections().collections]

        if COLLECTION not in existing:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=qm.VectorParams(
                    size=VECTOR_DIM,
                    distance=qm.Distance.COSINE,
                ),
            )
            logger.info("[QDRANT] Created collection '%s' (dim=%d)", COLLECTION, VECTOR_DIM)

            # ── Payload indexes for filtered search performance ─────────────
            index_specs = [
                ("user_id",       qm.PayloadSchemaType.KEYWORD),
                ("document_id",   qm.PayloadSchemaType.KEYWORD),
                ("chunk_id",      qm.PayloadSchemaType.KEYWORD),
                ("document_type", qm.PayloadSchemaType.KEYWORD),
                ("filename",      qm.PayloadSchemaType.KEYWORD),
                ("month",         qm.PayloadSchemaType.KEYWORD),
                ("year",          qm.PayloadSchemaType.INTEGER),
            ]
            for field_name, schema_type in index_specs:
                try:
                    self._client.create_payload_index(
                        collection_name=COLLECTION,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                    logger.info("[QDRANT] Created payload index: %s (%s)", field_name, schema_type)
                except Exception as e:
                    logger.warning("[QDRANT] Could not create index %s: %s", field_name, e)
        else:
            logger.info("[QDRANT] Collection '%s' already exists — skipping creation", COLLECTION)

    # ── Async wrappers ─────────────────────────────────────────────────────────

    async def _run(self, fn, *args, **kwargs):
        """Execute a synchronous Qdrant call in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ── Core Operations ────────────────────────────────────────────────────────

    async def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Batch upsert chunk vectors and payloads into Qdrant.

        Each chunk dict must have:
          - 'vector': List[float]  (768-dim embedding)
          - 'chunk_id': str        (used as deterministic point ID via UUID5)
          - All payload fields defined in the schema
        """
        self._initialize()
        if not chunks:
            return

        points = []
        for chunk in chunks:
            vector = chunk.get("vector")
            if not vector or len(vector) != VECTOR_DIM:
                logger.warning("[QDRANT] Skipping chunk with invalid vector: %s", chunk.get("chunk_id"))
                continue

            # Deterministic UUID from chunk_id string
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))

            payload = {k: v for k, v in chunk.items() if k != "vector"}
            points.append(qm.PointStruct(id=point_id, vector=vector, payload=payload))

        if not points:
            logger.warning("[QDRANT] No valid points to upsert")
            return

        await self._run(
            self.client.upsert,
            collection_name=COLLECTION,
            points=points,
        )
        logger.info("[QDRANT] Upserted %d chunk(s)", len(points))

    async def search_chunks(
        self,
        query_vector: List[float],
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        score_threshold: float = 0.20,
    ) -> List[qm.ScoredPoint]:
        """
        Filtered vector search returning top-k chunks.

        Always applies user_id filter — mandatory for data isolation.
        Additional filters (year, month, document_type) are AND-combined.
        """
        self._initialize()

        # Build filter conditions — user_id is always required
        must_conditions = [
            qm.FieldCondition(
                key="user_id",
                match=qm.MatchValue(value=user_id),
            )
        ]

        if filters:
            if filters.get("document_type"):
                must_conditions.append(
                    qm.FieldCondition(
                        key="document_type",
                        match=qm.MatchValue(value=filters["document_type"]),
                    )
                )
            if filters.get("year") is not None:
                must_conditions.append(
                    qm.FieldCondition(
                        key="year",
                        range=qm.Range(gte=filters["year"], lte=filters["year"]),
                    )
                )
            if filters.get("month"):
                must_conditions.append(
                    qm.FieldCondition(
                        key="month",
                        match=qm.MatchValue(value=filters["month"]),
                    )
                )

        qdrant_filter = qm.Filter(must=must_conditions)

        response = await self._run(
            self.client.query_points,
            collection_name=COLLECTION,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return response.points

    async def get_chunks_by_document(
        self, document_id: str, user_id: str
    ) -> List[qm.Record]:
        """Retrieve all chunks for a specific document (for detail view)."""
        self._initialize()
        points, _ = await self._run(
            self.client.scroll,
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
                    qm.FieldCondition(key="user_id",     match=qm.MatchValue(value=user_id)),
                ]
            ),
            limit=500,
            with_payload=True,
        )
        return points

    async def chunk_exists(self, chunk_id: str) -> bool:
        """Return True if a chunk with this chunk_id already exists (dedup)."""
        self._initialize()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        try:
            results = await self._run(
                self.client.retrieve,
                collection_name=COLLECTION,
                ids=[point_id],
                with_payload=False,
            )
            return len(results) > 0
        except Exception:
            return False

    async def delete_document_chunks(self, document_id: str, user_id: str) -> None:
        """Delete all chunks for a document (used on document deletion)."""
        self._initialize()
        await self._run(
            self.client.delete,
            collection_name=COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
                        qm.FieldCondition(key="user_id",     match=qm.MatchValue(value=user_id)),
                    ]
                )
            ),
        )
        logger.info("[QDRANT] Deleted all chunks for document_id=%s", document_id)

    async def get_document_metadata(
        self, document_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve document-level metadata by fetching the first chunk.
        All chunks carry the full document metadata in their payload.
        """
        self._initialize()
        points, _ = await self._run(
            self.client.scroll,
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
                    qm.FieldCondition(key="user_id",     match=qm.MatchValue(value=user_id)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        return points[0].payload


# ── Module-level singleton ─────────────────────────────────────────────────────
qdrant_store = QdrantStore()
