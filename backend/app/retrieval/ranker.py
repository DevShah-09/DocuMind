"""
ranker.py — Hybrid Relevance Ranker
=====================================
Combines Qdrant cosine scores with metadata-match boosts
to produce a final confidence score per document.

Boost table:
  +0.15  exact document_type match
  +0.10  exact year match
  +0.08  exact month match
  +0.05  per entity/keyword hit in chunk text (capped at +0.15 total)
  Score clipped to [0.0, 1.0]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.schemas.retrieval import MatchedChunk, ParsedQuery, SearchResult

logger = logging.getLogger(__name__)

# ── Boost weights ──────────────────────────────────────────────────────────────
BOOST_DOC_TYPE  = 0.15
BOOST_YEAR      = 0.10
BOOST_MONTH     = 0.08
BOOST_KEYWORD   = 0.05
BOOST_ENTITY    = 0.05
MAX_KW_BOOST    = 0.15   # cap total keyword boost


@dataclass
class CandidateChunk:
    """Internal representation of a scored chunk from Qdrant."""
    chunk_id:      str
    document_id:   str
    text:          str
    score:         float
    page_number:   int
    chunk_index:   int
    payload:       dict = field(default_factory=dict)


@dataclass
class CandidateDocument:
    """Aggregated document-level candidate from multiple chunks."""
    document_id:   str
    filename:      str
    document_type: str
    summary:       str
    date:          Optional[str]
    year:          Optional[int]
    month:         Optional[str]
    metadata:      dict
    tags:          List[str]
    entities:      List[str]
    chunks:        List[CandidateChunk] = field(default_factory=list)

    @property
    def best_chunk_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)

    @property
    def avg_chunk_score(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.score for c in self.chunks) / len(self.chunks)


class Ranker:
    """
    Ranks a list of CandidateDocument objects using composite scoring.

    Score = base_score + metadata_boosts, clipped to [0, 1].
    """

    def rank(
        self,
        candidates: List[CandidateDocument],
        parsed_query: ParsedQuery,
    ) -> List[SearchResult]:
        """
        Score and sort candidates. Returns ranked SearchResult list.

        Args:
            candidates:   Documents with their matched chunks.
            parsed_query: Structured query used for boost logic.
        """
        scored: List[tuple[float, CandidateDocument]] = []

        for doc in candidates:
            score = self._compute_score(doc, parsed_query)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        logger.debug("[RANKER] Ranked %d candidates", len(scored))

        results: List[SearchResult] = []
        for score, doc in scored:
            matched_chunks = [
                MatchedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text[:400],          # truncate for response size
                    score=round(c.score, 4),
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                )
                for c in sorted(doc.chunks, key=lambda x: x.score, reverse=True)[:3]
            ]
            results.append(
                SearchResult(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    document_type=doc.document_type,
                    summary=doc.summary,
                    score=round(score, 4),
                    matched_chunks=matched_chunks,
                    preview_url="",   # filled by retrieval engine
                    date=doc.date,
                    year=doc.year,
                    month=doc.month,
                    metadata=doc.metadata,
                    tags=doc.tags,
                    entities=doc.entities,
                )
            )

        return results

    # ── Scoring logic ──────────────────────────────────────────────────────────

    def _compute_score(self, doc: CandidateDocument, pq: ParsedQuery) -> float:
        base = doc.best_chunk_score   # highest cosine similarity among chunks
        boost = 0.0

        # Document type boost
        if pq.document_type:
            doc_type_matches = bool(doc.document_type and pq.document_type.lower() == doc.document_type.lower())
            tag_matches = any(str(t).lower() == pq.document_type.lower() for t in doc.tags) if doc.tags else False
            if doc_type_matches or tag_matches:
                boost += BOOST_DOC_TYPE

        # Year boost
        if pq.year is not None and doc.year is not None:
            if pq.year == doc.year:
                boost += BOOST_YEAR

        # Month boost
        if pq.month and doc.month:
            if pq.month.lower() == doc.month.lower():
                boost += BOOST_MONTH

        # Keyword/entity boost across best chunks
        keyword_boost = 0.0
        all_chunk_text = f"{doc.filename.lower()} {doc.summary.lower()} " + " ".join(c.text.lower() for c in doc.chunks)

        for entity in pq.entities:
            if entity.lower() in all_chunk_text:
                keyword_boost += BOOST_ENTITY

        # Semantic query keyword hits
        query_words = [
            w for w in re.split(r"\W+", pq.semantic_query.lower()) if len(w) > 2 and w not in {"give", "show", "find", "get", "the", "my"}
        ]
        for word in query_words:
            if word in all_chunk_text:
                keyword_boost += BOOST_KEYWORD

        boost += min(keyword_boost, MAX_KW_BOOST)

        return min(base + boost, 1.0)
