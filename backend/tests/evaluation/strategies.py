"""
strategies.py — Retrieval Strategy Simulators
==============================================
Implements four retrieval strategies against the SAME mock chunk pool so
that evaluation results are directly comparable.

Strategies
----------
  A. FixedK5Strategy      — always retrieve exactly K=5 chunks
  B. FixedK12Strategy     — always retrieve exactly K=12 chunks
  C. ComplexityOnlyStrategy — uses C_q alone; ignores R_c (beta=0 in formula)
  D. FullDynamicKStrategy   — uses both C_q and R_c (the full system)

Each strategy exposes:
  retrieve(query, chunk_pool, parsed_query) → List[MockChunk]

All strategies share the same chunk pool and the same score_threshold so
that differences in results are attributable solely to K selection, not to
different vector searches.

Design note
-----------
In a full production evaluation you would replace the MockChunk pool with
real Qdrant search results.  The harness in run_evaluation.py shows how to
plug in a real QdrantStore if one is available.
"""

from __future__ import annotations

from typing import Any, List, Optional

from app.retrieval.dynamic_top_k import DynamicTopKController, TopKConfig
from app.retrieval.query_complexity import QueryComplexityEstimator, ComplexityConfig, DEFAULT_WEIGHTS


class MockChunk:
    """Lightweight mock of a Qdrant ScoredPoint for offline evaluation."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        score: float,
        text: str = "",
        doc_type: str = "document",
        year: Optional[int] = None,
        month: Optional[str] = None,
    ) -> None:
        self.score   = score
        self.payload = {
            "chunk_id":      chunk_id,
            "document_id":   doc_id,
            "document_type": doc_type,
            "text":          text,
            "year":          year,
            "month":         month,
            "page_number":   1,
            "chunk_index":   0,
        }


# ── Strategy base ─────────────────────────────────────────────────────────────

class BaseStrategy:
    name: str = "base"

    def retrieve(
        self,
        query: str,
        chunk_pool: List[MockChunk],
        parsed_query: Optional[Any] = None,
    ) -> List[MockChunk]:
        raise NotImplementedError

    def recommended_k(
        self,
        query: str,
        chunk_pool: List[MockChunk],
        parsed_query: Optional[Any] = None,
    ) -> int:
        raise NotImplementedError


# ── A: Fixed K = 5 ───────────────────────────────────────────────────────────

class FixedK5Strategy(BaseStrategy):
    name = "fixed_k5"
    _K   = 5

    def retrieve(self, query, chunk_pool, parsed_query=None):
        sorted_pool = sorted(chunk_pool, key=lambda c: c.score, reverse=True)
        return sorted_pool[: self._K]

    def recommended_k(self, query, chunk_pool, parsed_query=None):
        return self._K


# ── B: Fixed K = 12 ──────────────────────────────────────────────────────────

class FixedK12Strategy(BaseStrategy):
    name = "fixed_k12"
    _K   = 12

    def retrieve(self, query, chunk_pool, parsed_query=None):
        sorted_pool = sorted(chunk_pool, key=lambda c: c.score, reverse=True)
        return sorted_pool[: self._K]

    def recommended_k(self, query, chunk_pool, parsed_query=None):
        return self._K


# ── C: Complexity-Only Dynamic K (beta = 0) ───────────────────────────────────

class ComplexityOnlyStrategy(BaseStrategy):
    """
    Uses C_q only; forces beta = 0 so R_c has no influence on K.

    Formula: K = K_min + C_q * (K_max - K_min)
    This is a degenerate case of the full formula with alpha=1, beta=0.
    """
    name = "complexity_only"

    def __init__(self, k_min: int = 2, k_max: int = 15) -> None:
        # beta=0, alpha=1 (auto-normalised in TopKConfig)
        self._controller = DynamicTopKController(
            config=TopKConfig(k_min=k_min, k_max=k_max, alpha=1.0, beta=0.0),
        )

    def retrieve(self, query, chunk_pool, parsed_query=None):
        k = self.recommended_k(query, chunk_pool, parsed_query)
        return sorted(chunk_pool, key=lambda c: c.score, reverse=True)[:k]

    def recommended_k(self, query, chunk_pool, parsed_query=None):
        decision = self._controller.decide(
            query=query,
            initial_chunks=chunk_pool,
            parsed_query=parsed_query,
        )
        return decision.recommended_k


# ── D: Full Confidence-Aware Dynamic K (alpha=beta=0.5) ───────────────────────

class FullDynamicKStrategy(BaseStrategy):
    """Full system: combines C_q and R_c with equal weight (alpha = beta = 0.5)."""
    name = "full_dynamic_k"

    def __init__(self, k_min: int = 2, k_max: int = 15) -> None:
        self._controller = DynamicTopKController(
            config=TopKConfig(k_min=k_min, k_max=k_max, alpha=0.5, beta=0.5),
        )

    def retrieve(self, query, chunk_pool, parsed_query=None):
        k = self.recommended_k(query, chunk_pool, parsed_query)
        return sorted(chunk_pool, key=lambda c: c.score, reverse=True)[:k]

    def recommended_k(self, query, chunk_pool, parsed_query=None):
        decision = self._controller.decide(
            query=query,
            initial_chunks=chunk_pool,
            parsed_query=parsed_query,
        )
        return decision.recommended_k


# ── Strategy registry ─────────────────────────────────────────────────────────

ALL_STRATEGIES = [
    FixedK5Strategy(),
    FixedK12Strategy(),
    ComplexityOnlyStrategy(),
    FullDynamicKStrategy(),
]
