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
    dynamic_k_enabled: bool = Field(True, description="Enable adaptive Dynamic Top-K governor")

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


# ── Dynamic Top-K Diagnostics ──────────────────────────────────────────────────

class DynamicKDiagnostics(BaseModel):
    """Explainable diagnostics emitted by the Dynamic Top-K Retrieval Governor."""
    recommended_k: int = Field(..., description="Calculated total chunk retrieval quota")
    target_documents: int = Field(..., description="Expected number of unique documents needed")
    chunks_per_document: int = Field(..., description="Max chunk quota per document for diversity")
    c_q: float = Field(..., description="Query complexity score [0–1]")
    r_c: float = Field(..., description="Retrieval confidence score [0–1]")
    s_top: float = Field(..., description="Highest chunk similarity score")
    s_avg: float = Field(..., description="Average similarity score across top candidates")
    knee_gap: float = Field(..., description="Maximum inflection drop between adjacent ranks")
    knee_index: int = Field(..., description="1-based rank index immediately before the cliff")
    evidence_alignment: float = Field(..., description="Grounding match between query constraints and candidates")
    features: Dict[str, float] = Field(default_factory=dict, description="Normalized f1..f8 complexity features")
    formula_mode: str = Field("normalized_demand", description="Evaluation formula mode")
    is_abstained: bool = Field(False, description="Whether retrieval expansion was abstained due to sub-floor similarity")
    abstention_reason: str = Field("", description="Reason for abstention if applicable")
    stop_reason: str = Field("", description="Termination reason for K decision")
    explanation: str = Field("", description="Human-readable decision explanation")


class SearchResponse(BaseModel):
    """Full response for POST /search."""
    query: str
    parsed_query: ParsedQuery
    results: List[SearchResult]
    total: int
    page: int
    limit: int
    latency_ms: float
    dynamic_k: Optional[DynamicKDiagnostics] = Field(None, description="Dynamic Top-K governor diagnostics")


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


# ── Chat (RAG) ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Payload for POST /chat."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query or instruction")
    limit: int = Field(5, ge=1, le=20, description="Max source documents to retrieve")
    score_threshold: float = Field(0.10, ge=0.0, le=1.0, description="Minimum similarity score for retrieval")
    dynamic_k_enabled: bool = Field(True, description="Enable adaptive Dynamic Top-K governor")


class ChatSource(BaseModel):
    """A single source document cited in the chat answer."""
    document_id: str
    filename: str
    document_type: str
    summary: str
    score: float
    year: Optional[int] = None
    month: Optional[str] = None
    preview_url: str = ""


class ChatResponse(BaseModel):
    """Full response for POST /chat."""
    answer: str
    sources: List[ChatSource] = Field(default_factory=list)
    parsed_query: Optional[ParsedQuery] = None
    latency_ms: float
    dynamic_k: Optional[DynamicKDiagnostics] = Field(None, description="Dynamic Top-K governor diagnostics")


