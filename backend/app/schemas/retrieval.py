"""
retrieval.py — Pydantic Schemas for Search & Retrieval
=======================================================
Typed request/response models for the hybrid retrieval API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Query Parsing ──────────────────────────────────────────────────────────────

class AmountFilter(BaseModel):
    """Optional amount/price range filter extracted from the user query."""
    operator: str = Field(..., description="One of: gt, lt, gte, lte, eq")
    value: float = Field(..., description="The threshold amount")
    currency: Optional[str] = Field(None, description="Detected currency symbol e.g. ₹, $")


class ParsedQuery(BaseModel):
    """
    Structured representation of a natural language query,
    extracted by the Qwen-based query parser.
    """
    document_type: Optional[str] = Field(None, description="Normalised document type slug")
    year: Optional[int] = Field(None, description="4-digit year e.g. 2017")
    month: Optional[str] = Field(None, description="Full month name e.g. January")
    entities: List[str] = Field(default_factory=list, description="Named entities: vendors, banks, etc.")
    amount_filters: Optional[AmountFilter] = Field(None, description="Amount/price filter if detected")
    semantic_query: str = Field(..., description="Clean semantic search string")
    raw_query: str = Field("", description="Original user query")


# ── Search Request ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Payload for POST /search."""
    query: str = Field(..., min_length=2, max_length=500, description="Natural language query")
    limit: int = Field(3, ge=1, le=50, description="Max number of results to return")
    score_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum similarity score")
    page: int = Field(1, ge=1, description="Pagination page number")

    # Optional explicit overrides (bypass query parser)
    document_type_filter: Optional[str] = Field(None, description="Force filter by document type")
    year_filter: Optional[int] = Field(None, description="Force filter by year")
    month_filter: Optional[str] = Field(None, description="Force filter by month")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace")
        return v.strip()


# ── Search Results ─────────────────────────────────────────────────────────────

class MatchedChunk(BaseModel):
    """A single matched chunk within a document result."""
    chunk_id: str
    text: str
    score: float
    page_number: int
    chunk_index: int


class SearchResult(BaseModel):
    """A single document-level search result."""
    document_id: str
    filename: str
    document_type: str
    summary: str
    score: float = Field(..., description="Composite relevance score [0–1]")
    matched_chunks: List[MatchedChunk] = Field(default_factory=list)
    preview_url: str
    date: Optional[str] = None
    year: Optional[int] = None
    month: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Full response for POST /search."""
    query: str
    parsed_query: ParsedQuery
    results: List[SearchResult]
    total: int
    page: int
    limit: int
    latency_ms: float


# ── Query Parse Endpoint ───────────────────────────────────────────────────────

class QueryParseRequest(BaseModel):
    """Payload for POST /search/query-parse."""
    query: str = Field(..., min_length=2, max_length=500)


class QueryParseResponse(BaseModel):
    """Response for POST /search/query-parse."""
    raw_query: str
    parsed: ParsedQuery


# ── Document Detail ────────────────────────────────────────────────────────────

class DocumentChunkDetail(BaseModel):
    """One chunk record returned from GET /documents/{id}."""
    chunk_id: str
    chunk_index: int
    page_number: int
    text: str


class DocumentDetailResponse(BaseModel):
    """Full document metadata + all chunks from GET /documents/{id}."""
    document_id: str
    filename: str
    document_type: str
    summary: str
    date: Optional[str] = None
    year: Optional[int] = None
    month: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    chunks: List[DocumentChunkDetail] = Field(default_factory=list)
    total_chunks: int = 0


class DocumentPreviewResponse(BaseModel):
    """Signed preview URL from GET /documents/{id}/preview."""
    document_id: str
    filename: str
    preview_url: str
    expires_in_seconds: int = 900


# ── Batch Search ───────────────────────────────────────────────────────────────

class BatchSearchRequest(BaseModel):
    """Payload for POST /search/batch."""
    queries: List[str] = Field(..., min_length=1, max_length=10, description="List of queries")
    limit_per_query: int = Field(5, ge=1, le=20)
    score_threshold: float = Field(0.25, ge=0.0, le=1.0)


class BatchSearchResponse(BaseModel):
    """Response for POST /search/batch."""
    results: Dict[str, SearchResponse]
    total_queries: int
    latency_ms: float
