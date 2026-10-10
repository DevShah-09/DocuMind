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
import re
from typing import Any, Dict, List, Optional

from app.embeddings.embedding_service import embedding_service
from app.retrieval.dynamic_k import dynamic_k_governor
from app.retrieval.query_parser import query_parser
from app.retrieval.ranker import CandidateChunk, CandidateDocument, Ranker
from app.schemas.retrieval import DynamicKDiagnostics, ParsedQuery, SearchRequest, SearchResponse, SearchResult
from app.vectorstore.qdrant_store import qdrant_store
from app.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)

_ranker = Ranker()

# Candidate pool — fetch more than needed before ranking
CANDIDATE_POOL = 50

# Month names mapping for normalization
MONTH_MAP = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
    "january": "January", "february": "February", "march": "March",
    "april": "April", "june": "June", "july": "July", "august": "August",
    "september": "September", "october": "October", "november": "November",
    "december": "December",
}

_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)

def _extract_year_from_payload(p: dict) -> Optional[int]:
    # 1. Direct checks
    for key in ("year", "Year"):
        val = p.get(key) or p.get("metadata", {}).get(key)
        if val is not None:
            try:
                y = int(val)
                if 1950 <= y <= 2099:
                    return y
            except (TypeError, ValueError):
                pass

    # 2. Check metadata dictionary values
    metadata = p.get("metadata", {})
    for k, v in metadata.items():
        if isinstance(v, (str, int)):
            m = re.search(r"\b(19[5-9]\d|20[0-4]\d)\b", str(v))
            if m:
                return int(m.group(1))

    # 3. Check filename, summary, and text
    text_to_search = f"{p.get('original_name', '')} {p.get('summary', '')} {p.get('text_content', '')}"
    m = re.search(r"\b(19[5-9]\d|20[0-4]\d)\b", text_to_search)
    if m:
        return int(m.group(1))
        
    return None

def _extract_month_from_payload(p: dict) -> Optional[str]:
    # 1. Direct checks
    for key in ("month", "Month"):
        val = p.get(key) or p.get("metadata", {}).get(key)
        if isinstance(val, str):
            m_normalized = MONTH_MAP.get(val.lower().strip())
            if m_normalized:
                return m_normalized

    # 2. Check metadata dictionary values
    metadata = p.get("metadata", {})
    for k, v in metadata.items():
        if isinstance(v, str):
            m = _MONTH_RE.search(v)
            if m:
                m_normalized = MONTH_MAP.get(m.group(1).lower())
                if m_normalized:
                    return m_normalized

    # 3. Check filename, summary, and text
    text_to_search = f"{p.get('original_name', '')} {p.get('summary', '')} {p.get('text_content', '')}"
    m = _MONTH_RE.search(text_to_search)
    if m:
        m_normalized = MONTH_MAP.get(m.group(1).lower())
        if m_normalized:
            return m_normalized

    return None

def _extract_date_from_payload(p: dict) -> Optional[str]:
    for key in ("date", "Date", "bill_date", "statement_date"):
        val = p.get(key) or p.get("metadata", {}).get(key)
        if val is not None:
            return str(val)
    return None


def _is_conversational_query(q: str) -> bool:
    clean = q.lower().strip().rstrip("?!.")
    greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "yo", "sup", "hi there", "hello there", "help",
        "how are you", "what can you do", "who are you", "test"
    }
    return clean in greetings


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

        # ── Step 0: Check if Query is conversational greeting ──────────────────
        if _is_conversational_query(request.query):
            parsed = ParsedQuery(
                document_type=None,
                year=None,
                month=None,
                entities=[],
                amount_filters=None,
                semantic_query="",
                raw_query=request.query
            )
            return SearchResponse(
                query=request.query,
                parsed_query=parsed,
                results=[],
                total=0,
                page=request.page,
                limit=request.limit,
                latency_ms=0.0
            )

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
                
                doc_year = _extract_year_from_payload(p)
                doc_month = _extract_month_from_payload(p)
                doc_type = p.get("category", "document")
                
                # Apply hard filtering on extracted fields for legacy fallback
                if parsed.year is not None and doc_year is not None and doc_year != parsed.year:
                    continue
                if parsed.month is not None and doc_month is not None and doc_month.lower() != parsed.month.lower():
                    continue
                if parsed.document_type and doc_type and doc_type.lower() != parsed.document_type.lower():
                    continue
                
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
                    document_type=doc_type,
                    summary=p.get("summary", ""),
                    date=_extract_date_from_payload(p),
                    year=doc_year,
                    month=doc_month,
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
        # ── Step 9: Dynamic-K Diagnostics (if enabled) ────────────────────────
        dynamic_k_diag: Optional[DynamicKDiagnostics] = None
        if request.dynamic_k_enabled and raw_chunks:
            try:
                decision = dynamic_k_governor.decide(
                    query=request.query,
                    initial_chunks=raw_chunks,
                    parsed_query=parsed,
                )
                dynamic_k_diag = DynamicKDiagnostics(**decision.to_dict())
            except Exception as e:
                logger.warning("[ENGINE] Dynamic K computation failed: %s", e)

        latency_ms = (time.monotonic() - t0) * 1000

        return SearchResponse(
            query=request.query,
            parsed_query=parsed,
            results=paginated,
            total=total,
            page=page,
            limit=limit,
            latency_ms=round(latency_ms, 2),
            dynamic_k=dynamic_k_diag,
        )

    async def parse_only(self, query: str) -> ParsedQuery:
        """Parse query without executing search (for /search/query-parse endpoint)."""
        return await query_parser.parse(query)


# ── Module-level singleton ─────────────────────────────────────────────────────
hybrid_engine = HybridRetrievalEngine()
