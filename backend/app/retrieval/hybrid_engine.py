"""
hybrid_engine.py — Hybrid Retrieval Engine
===========================================
Orchestrates the full retrieval pipeline:

  NL Query
    → QueryParser (structured filters)
    → EmbeddingService (query vector)
    → QdrantStore (filtered vector search, top-50 chunks)
    → Collapse chunks → CandidateDocuments
    → Ranker (boost + sort)
    → Return top-k SearchResults
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.embeddings.embedding_service import embedding_service
from app.retrieval.query_parser import query_parser
from app.retrieval.ranker import CandidateChunk, CandidateDocument, Ranker
from app.schemas.retrieval import ParsedQuery, SearchRequest, SearchResponse, SearchResult
from app.vectorstore.qdrant_store import qdrant_store
from app.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)

_ranker = Ranker()

# Candidate pool — fetch more than needed before ranking
CANDIDATE_POOL = 50


class HybridRetrievalEngine:
    """
    Production hybrid retrieval engine combining:
      1. Vector similarity (BAAI/bge-base-en-v1.5)
      2. Qdrant metadata filtering (year, month, document_type, user_id)
      3. Keyword/entity/metadata ranking boost
    """

    async def search(
        self,
        request: SearchRequest,
        user_id: str,
        generate_preview_url_fn=None,
    ) -> SearchResponse:
        """
        Execute a full hybrid search for the given user.

        Args:
            request:                SearchRequest from API layer.
            user_id:                Authenticated user's UUID (from JWT).
            generate_preview_url_fn: Callable(document_id, bucket_path) → str
                                     injected from the API layer to produce
                                     signed S3 preview URLs.
        Returns:
            SearchResponse with ranked results.
        """
        t0 = time.monotonic()

        # ── Step 1: Parse Query ────────────────────────────────────────────────
        parsed: ParsedQuery = await query_parser.parse(request.query)

        # Apply explicit override filters from request body
        if request.document_type_filter:
            parsed = parsed.model_copy(update={"document_type": request.document_type_filter})
        if request.year_filter:
            parsed = parsed.model_copy(update={"year": request.year_filter})
        if request.month_filter:
            parsed = parsed.model_copy(update={"month": request.month_filter})

        # ── Step 2: Embed Query ────────────────────────────────────────────────
        query_vector = await embedding_service.embed_query(parsed.semantic_query)

        # ── Step 3: Build Metadata Filters ────────────────────────────────────
        filters: Dict[str, Any] = {}
        if parsed.document_type:
            filters["document_type"] = parsed.document_type
        if parsed.year is not None:
            filters["year"] = parsed.year
        if parsed.month:
            filters["month"] = parsed.month

        # ── Step 4: Vector Search (50 candidate chunks) ───────────────────────
        raw_chunks = await qdrant_store.search_chunks(
            query_vector=query_vector,
            user_id=user_id,
            filters=filters,
            limit=CANDIDATE_POOL,
            score_threshold=request.score_threshold,
        )

        logger.info("[ENGINE] Vector search returned %d chunks for user=%s", len(raw_chunks), user_id)

        # ── Step 5: Collapse Chunks → Documents ───────────────────────────────
        doc_map: Dict[str, CandidateDocument] = {}

        for hit in raw_chunks:
            p = hit.payload or {}
            doc_id = p.get("document_id", "")
            if not doc_id:
                continue

            chunk = CandidateChunk(
                chunk_id=p.get("chunk_id", ""),
                document_id=doc_id,
                text=p.get("text", ""),
                score=hit.score,
                page_number=p.get("page_number", 1),
                chunk_index=p.get("chunk_index", 0),
                payload=p,
            )

            if doc_id not in doc_map:
                doc_map[doc_id] = CandidateDocument(
                    document_id=doc_id,
                    filename=p.get("filename", ""),
                    document_type=p.get("document_type", ""),
                    summary=p.get("summary", ""),
                    date=p.get("date"),
                    year=p.get("year"),
                    month=p.get("month"),
                    metadata=p.get("metadata", {}),
                    tags=p.get("tags", []),
                    entities=p.get("entities", []),
                    chunks=[],
                )
            doc_map[doc_id].chunks.append(chunk)

        candidates = list(doc_map.values())
        logger.info("[ENGINE] Collapsed to %d unique documents", len(candidates))

        # ── Step 5.5: Legacy Collection Fallback ──────────────────────────────
        if not candidates:
            logger.info("[ENGINE] No chunks found in new collection, falling back to legacy user_documents")
            vector_db = VectorDBService()
            legacy_results = await vector_db.search(
                query=parsed.semantic_query,
                limit=15,
                user_id=user_id
            )
            for hit in legacy_results:
                p = hit.payload or {}
                doc_id = hit.id
                
                # We create a dummy CandidateChunk for the Ranker and Preview URL generation to use
                chunk = CandidateChunk(
                    chunk_id=f"{doc_id}_legacy",
                    document_id=str(doc_id),
                    text=p.get("text_content", ""),
                    score=hit.score,
                    page_number=1,
                    chunk_index=0,
                    payload=p,
                )
                
                doc = CandidateDocument(
                    document_id=str(doc_id),
                    filename=p.get("original_name", "Unknown Document"),
                    document_type=p.get("category", "document"),
                    summary=p.get("summary", ""),
                    date=None,
                    year=None,
                    month=None,
                    metadata=p.get("metadata", {}),
                    tags=[],
                    entities=[],
                    chunks=[chunk]
                )
                candidates.append(doc)
                doc_map[str(doc_id)] = doc
            logger.info("[ENGINE] Legacy fallback yielded %d documents", len(candidates))

        # ── Step 6: Rank ───────────────────────────────────────────────────────
        ranked: List[SearchResult] = _ranker.rank(candidates, parsed)

        # ── Step 7: Pagination ────────────────────────────────────────────────
        page = request.page
        limit = request.limit
        total = len(ranked)
        start = (page - 1) * limit
        paginated = ranked[start : start + limit]

        # ── Step 8: Generate Preview URLs ─────────────────────────────────────
        if generate_preview_url_fn:
            for result in paginated:
                try:
                    bucket_path = doc_map[result.document_id].chunks[0].payload.get("bucket_path", "")
                    result.preview_url = generate_preview_url_fn(
                        document_id=result.document_id,
                        bucket_path=bucket_path,
                        user_id=user_id,
                    )
                except Exception as e:
                    logger.warning("[ENGINE] Failed to generate preview URL: %s", e)
                    result.preview_url = ""

        latency_ms = (time.monotonic() - t0) * 1000

        return SearchResponse(
            query=request.query,
            parsed_query=parsed,
            results=paginated,
            total=total,
            page=page,
            limit=limit,
            latency_ms=round(latency_ms, 2),
        )

    async def parse_only(self, query: str) -> ParsedQuery:
        """Parse query without executing search (for /search/query-parse endpoint)."""
        return await query_parser.parse(query)


# ── Module-level singleton ─────────────────────────────────────────────────────
hybrid_engine = HybridRetrievalEngine()
