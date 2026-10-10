"""
dynamic_k.py — Dynamic Top-K Retrieval Governor (compatibility shim)
=====================================================================
This module is the original monolithic implementation.  It has been
refactored into three dedicated modules:

  - app.retrieval.query_complexity       → QueryComplexityEstimator (C_q)
  - app.retrieval.retrieval_confidence   → RetrievalConfidenceEstimator (R_c)
  - app.retrieval.dynamic_top_k          → DynamicTopKController (K decision)

All existing callers (chat.py, hybrid_engine.py, tests) continue to work
unchanged because this module re-exports the same public symbols:

  DynamicKGovernor, dynamic_k_governor,
  W_STOP, W_SAVG, W_KNEE, W_EVIDENCE, K_MIN
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Re-export the new modular components ──────────────────────────────────────
from app.retrieval.query_complexity import (
    QueryComplexityEstimator,
    ComplexityConfig,
    ComplexityResult,
    DEFAULT_WEIGHTS,
    complexity_estimator,
)
from app.retrieval.retrieval_confidence import (
    RetrievalConfidenceEstimator,
    ConfidenceConfig,
    ConfidenceResult,
    DEFAULT_W_STOP,
    DEFAULT_W_SAVG,
    DEFAULT_W_KNEE,
    DEFAULT_W_EVIDENCE,
    confidence_estimator,
)
from app.retrieval.dynamic_top_k import (
    DynamicTopKController,
    TopKConfig,
    DynamicKDecision,
    dynamic_top_k_controller,
)

logger = logging.getLogger(__name__)

# ── Public constants (kept for backward-compatibility with existing tests) ────
W_STOP     = DEFAULT_W_STOP
W_SAVG     = DEFAULT_W_SAVG
W_KNEE     = DEFAULT_W_KNEE
W_EVIDENCE = DEFAULT_W_EVIDENCE

K_MIN   = 2
K_MAX   = 15
K_RANGE = 8     # legacy parameter — preserved for test compatibility
GAMMA   = 5.0   # legacy parameter — preserved for test compatibility


class DynamicKGovernor:
    """
    Public facade that delegates to the three new modular components.

    Provides the same interface as the original DynamicKGovernor so that
    all existing callers require no changes:

      governor.decide(query, initial_chunks, parsed_query) → DynamicKDecision
      governor.extract_features(query, parsed_query)       → Dict[str, float]
      governor.compute_complexity(features)                → float
      governor.compute_confidence(chunks, parsed_query)    → Tuple[...]
    """

    def __init__(
        self,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        k_range: int = K_RANGE,
        gamma: float = GAMMA,
    ) -> None:
        self.k_min   = k_min
        self.k_max   = k_max
        self.k_range = k_range
        self.gamma   = gamma

        # Instantiate the new modular components with matching K bounds
        self._controller = DynamicTopKController(
            config=TopKConfig(k_min=k_min, k_max=k_max),
        )
        self._complexity = self._controller._complexity
        self._confidence = self._controller._confidence

    # ── Phase 1: Query Complexity ──────────────────────────────────────────────

    def extract_features(
        self,
        query: str,
        parsed_query: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Extract 8 normalized features f1..f8 for C_q computation."""
        return self._complexity.extract_features(query, parsed_query)

    def compute_complexity(self, features: Dict[str, float]) -> float:
        """Compute C_q from a pre-extracted feature dict."""
        return self._complexity.compute_score(features)

    # ── Phase 2: Retrieval Confidence ─────────────────────────────────────────

    def compute_confidence(
        self,
        chunks: List[Any],
        parsed_query: Optional[Any] = None,
    ) -> Tuple[float, float, float, float, int, float]:
        """
        Compute R_c and its component signals.

        Returns
        -------
        (r_c, s_top, s_avg, knee_gap, knee_index, evidence_alignment)
        """
        result = self._confidence.estimate(chunks, parsed_query)
        return (
            result.score,
            result.s_top,
            result.s_avg,
            result.knee_gap,
            result.knee_index,
            result.evidence_alignment,
        )

    # ── Phase 3: Full Decision ─────────────────────────────────────────────────

    def decide(
        self,
        query: str,
        initial_chunks: List[Any],
        parsed_query: Optional[Any] = None,
    ) -> DynamicKDecision:
        """
        Execute the full Dynamic-K decision process.

        Delegates entirely to DynamicTopKController.decide() while preserving
        the original return type (DynamicKDecision) and its .to_dict() method.
        """
        return self._controller.decide(
            query=query,
            initial_chunks=initial_chunks,
            parsed_query=parsed_query,
        )


# ── Module-level singleton (backward-compatible name) ─────────────────────────
dynamic_k_governor = DynamicKGovernor()
