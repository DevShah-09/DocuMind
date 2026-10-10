"""
dynamic_top_k.py — Confidence-Aware Dynamic Top-K Controller
=============================================================
Combines Query Complexity (C_q) and Retrieval Confidence (R_c) to determine
the optimal number of chunks to retrieve for a given query.

Mathematical Formula
--------------------
Step 1 — Compute demand signals:

    complexity_demand  = C_q              (how much evidence is required)
    confidence_demand  = 1 - R_c          (how insufficient current evidence is)

Step 2 — Combine with normalised weights (alpha + beta = 1):

    combined_demand = alpha * complexity_demand + beta * confidence_demand

Step 3 — Map to the configured K range:

    raw_K = K_min + combined_demand * (K_max - K_min)

Step 4 — Apply coverage safeguard and round:

    If the query has multi-part / comparative signals AND evidence_alignment < 0.5,
    raw_K is boosted by a coverage_penalty so high confidence cannot prematurely
    stop retrieval for incompletely covered multi-document questions.

Step 5 — Clamp:

    recommended_K = clamp(round(raw_K), K_min, K_max)

Defaults
--------
    alpha   = 0.50   (complexity weight)
    beta    = 0.50   (confidence weight, auto-normalised with alpha)
    K_min   = 2
    K_max   = 15
    coverage_penalty = 3  (extra chunks when multi-part query is under-covered)

High-confidence factual shortcut
---------------------------------
If C_q <= 0.25 AND R_c >= 0.80 AND knee_index <= 2 AND knee_gap >= 0.20,
recommended_K is forced to K_min.  This prevents over-retrieval when one
document clearly dominates and the query is simple.

Document diversity allocation
------------------------------
After choosing recommended_K, the controller plans how many distinct
documents (target_documents) and how many chunks per document
(chunks_per_document) to allocate to ensure diverse grounding for
comparative / aggregative queries.

All results are returned as a DynamicKDecision dataclass that matches the
existing DynamicKDiagnostics Pydantic schema so it can be serialised directly
into the API response.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.retrieval.query_complexity import (
    ComplexityConfig,
    QueryComplexityEstimator,
    complexity_estimator as _default_complexity,
)
from app.retrieval.retrieval_confidence import (
    ConfidenceConfig,
    RetrievalConfidenceEstimator,
    confidence_estimator as _default_confidence,
)

logger = logging.getLogger(__name__)


# ── Default controller parameters ─────────────────────────────────────────────
DEFAULT_K_MIN             = 2
DEFAULT_K_MAX             = 15
DEFAULT_ALPHA             = 0.50   # complexity weight
DEFAULT_BETA              = 0.50   # confidence weight (auto-normalised with alpha)
DEFAULT_COVERAGE_PENALTY  = 3      # extra chunks injected for under-covered multi-part queries
DEFAULT_COVERAGE_THRESHOLD = 0.50  # evidence_alignment below this → apply penalty


@dataclass
class TopKConfig:
    """
    All configurable parameters for the Dynamic Top-K controller.

    Attributes
    ----------
    k_min : int
        Minimum retrieval quota. Guarantees at least this many chunks for
        cross-verification even on simple queries.
    k_max : int
        Hard upper bound on chunk retrieval. Never exceeded regardless of
        demand signals.
    alpha : float
        Weight for complexity_demand in the combined_demand formula.
        Auto-normalised with beta so their sum = 1.0.
    beta : float
        Weight for confidence_demand (1 - R_c) in the combined_demand formula.
    coverage_penalty : int
        Extra chunks added to raw_K when a multi-part query is under-covered
        by the initial retrieval (evidence_alignment < coverage_threshold).
        This prevents high confidence from masking incomplete coverage.
    coverage_threshold : float
        evidence_alignment below this value triggers the coverage penalty for
        multi-part / comparative queries.
    """
    k_min:              int   = DEFAULT_K_MIN
    k_max:              int   = DEFAULT_K_MAX
    alpha:              float = DEFAULT_ALPHA
    beta:               float = DEFAULT_BETA
    coverage_penalty:   int   = DEFAULT_COVERAGE_PENALTY
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD

    def __post_init__(self) -> None:
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {self.k_min}")
        if self.k_max < self.k_min:
            raise ValueError(f"k_max ({self.k_max}) must be >= k_min ({self.k_min})")
        # Auto-normalise alpha + beta
        ab_total = self.alpha + self.beta
        if ab_total <= 0:
            self.alpha = 0.5
            self.beta  = 0.5
        else:
            self.alpha /= ab_total
            self.beta  /= ab_total


@dataclass
class DynamicKDecision:
    """
    Complete diagnostic and operational decision from the Dynamic Top-K controller.

    Field names intentionally match those of DynamicKDiagnostics in
    app/schemas/retrieval.py so that ``DynamicKDiagnostics(**decision.to_dict())``
    works without transformation.
    """
    recommended_k:      int
    target_documents:   int
    chunks_per_document: int
    c_q:                float
    r_c:                float
    s_top:              float
    s_avg:              float
    knee_gap:           float
    knee_index:         int
    evidence_alignment: float
    features:           Dict[str, float] = field(default_factory=dict)
    explanation:        str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_k":       self.recommended_k,
            "target_documents":    self.target_documents,
            "chunks_per_document": self.chunks_per_document,
            "c_q":                 round(self.c_q, 4),
            "r_c":                 round(self.r_c, 4),
            "s_top":               round(self.s_top, 4),
            "s_avg":               round(self.s_avg, 4),
            "knee_gap":            round(self.knee_gap, 4),
            "knee_index":          self.knee_index,
            "evidence_alignment":  round(self.evidence_alignment, 4),
            "features":            {k: round(v, 4) for k, v in self.features.items()},
            "explanation":         self.explanation,
        }


# ── Regex helpers (re-used from complexity estimator) ─────────────────────────
_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|contrasting?|difference|differ|vs\.?|versus|"
    r"higher than|lower than|more than|less than|"
    r"increased?|decreased?|growth|change|variance)\b",
    re.IGNORECASE,
)
_AGGREGATION_RE = re.compile(
    r"\b(total|sum|average|avg|overall|cumulative|aggregate|all|"
    r"breakdown|history|trend|expenses?|spending|annual|quarterly|monthly)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_TEMPORAL_RANGE_RE = re.compile(
    r"\b(between|from\s+\w+\s+to\s+\w+|across\s+(?:the\s+)?"
    r"(?:months?|years?)|q[1-4]|quarter|h[1-2]|ytd|"
    r"over\s+time|last\s+\d+\s+months?)\b",
    re.IGNORECASE,
)

_DOC_BROWSE_RE = re.compile(
    r"\b(documents?|files?|bills?|invoices?|statements?|resumes?|"
    r"cards?|reports?|receipts?|marksheets?|certificates?)\b",
    re.IGNORECASE,
)


class DynamicTopKController:
    """
    Confidence-Aware Dynamic Top-K Controller.

    Combines QueryComplexityEstimator and RetrievalConfidenceEstimator to
    produce a retrieval budget decision.  All computation is deterministic
    and requires no LLM calls.

    Usage
    -----
    ::

        controller = DynamicTopKController()
        decision = controller.decide(
            query="Compare my electricity bills in 2023 and 2024",
            initial_chunks=raw_chunks,
            parsed_query=parsed,
        )
        # decision.recommended_k → e.g. 10
        # decision.explanation   → human-readable reasoning

    Thread-safety
    -------------
    All state is read-only after construction; safe to share across async tasks.
    """

    def __init__(
        self,
        config: Optional[TopKConfig] = None,
        complexity_estimator: Optional[QueryComplexityEstimator] = None,
        confidence_estimator: Optional[RetrievalConfidenceEstimator] = None,
    ) -> None:
        self._cfg        = config or TopKConfig()
        self._complexity = complexity_estimator or _default_complexity
        self._confidence = confidence_estimator or _default_confidence

    # ── Public API ─────────────────────────────────────────────────────────────

    def decide(
        self,
        query: str,
        initial_chunks: List[Any],
        parsed_query: Optional[Any] = None,
    ) -> DynamicKDecision:
        """
        Execute the full Dynamic Top-K decision pipeline.

        Parameters
        ----------
        query : str
            Raw user query.
        initial_chunks : list
            Initial retrieval probe results (Qdrant ScoredPoint objects or
            any objects with a ``.score`` float attribute).
        parsed_query : ParsedQuery, optional
            Structured query output from the existing query parser.  Improves
            both complexity and confidence estimates when available.

        Returns
        -------
        DynamicKDecision with recommended_k and full diagnostics.
        """
        cfg = self._cfg

        # ── Phase 1: Query Complexity ─────────────────────────────────────────
        complexity_result = self._complexity.estimate(query, parsed_query)
        c_q      = complexity_result.score
        features = complexity_result.features

        # ── Phase 2: Retrieval Confidence ─────────────────────────────────────
        confidence_result = self._confidence.estimate(initial_chunks, parsed_query)
        r_c               = confidence_result.score
        s_top             = confidence_result.s_top
        s_avg             = confidence_result.s_avg
        knee_gap          = confidence_result.knee_gap
        knee_index        = confidence_result.knee_index
        evidence          = confidence_result.evidence_alignment

        # ── Phase 3: Combined Demand → K ─────────────────────────────────────
        #
        # Formula (see module docstring):
        #   combined_demand = alpha * C_q + beta * (1 - R_c)
        #   raw_K           = K_min + combined_demand * (K_max - K_min)
        #
        complexity_demand = c_q
        confidence_demand = 1.0 - r_c
        combined_demand   = cfg.alpha * complexity_demand + cfg.beta * confidence_demand

        k_range = cfg.k_max - cfg.k_min
        raw_k   = cfg.k_min + combined_demand * k_range

        # ── Phase 4: Coverage safeguard ───────────────────────────────────────
        # Multi-part / comparative queries with low evidence_alignment may still
        # score a high R_c if individual chunk scores are strong.  In that case,
        # we add a coverage_penalty to prevent premature termination.
        is_multi_part = self._is_multi_part(query, features)
        coverage_penalty_applied = 0
        if is_multi_part and evidence < cfg.coverage_threshold:
            raw_k += cfg.coverage_penalty
            coverage_penalty_applied = cfg.coverage_penalty
            logger.debug(
                "[TOPK] Coverage penalty +%d applied: multi_part=True, evidence=%.2f",
                cfg.coverage_penalty, evidence,
            )

        # ── Phase 5: High-confidence factual shortcut ─────────────────────────
        # Force K = K_min when: simple query + very confident + clear knee cliff
        is_shortcut = (
            c_q <= 0.25
            and r_c >= 0.80
            and knee_index <= 2
            and knee_gap >= 0.20
        )
        if is_shortcut:
            recommended_k = cfg.k_min
            stop_reason   = "high-confidence factual shortcut"
        else:
            recommended_k = max(cfg.k_min, min(round(raw_k), cfg.k_max))
            stop_reason   = "combined demand formula"

        # ── Phase 6: Document diversity allocation ────────────────────────────
        target_docs, chunks_per_doc = self._allocate_diversity(
            query, features, recommended_k, parsed_query
        )

        # ── Explanation ───────────────────────────────────────────────────────
        explanation = self._build_explanation(
            c_q, r_c, s_top, knee_gap, knee_index,
            combined_demand, coverage_penalty_applied,
            recommended_k, target_docs, chunks_per_doc,
            stop_reason, is_shortcut,
        )

        logger.info("[TOPK] %s", explanation)

        return DynamicKDecision(
            recommended_k=recommended_k,
            target_documents=target_docs,
            chunks_per_document=chunks_per_doc,
            c_q=round(c_q, 4),
            r_c=round(r_c, 4),
            s_top=round(s_top, 4),
            s_avg=round(s_avg, 4),
            knee_gap=round(knee_gap, 4),
            knee_index=knee_index,
            evidence_alignment=round(evidence, 4),
            features=features,
            explanation=explanation,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_multi_part(query: str, features: Dict[str, float]) -> bool:
        """Return True if the query appears to span multiple documents or time periods."""
        is_comparative   = features.get("f5_comparison",  0.0) >= 0.5
        is_multi_temporal = features.get("f4_temporal",   0.0) >= 0.5
        is_aggregative   = features.get("f6_aggregation", 0.0) >= 0.5
        is_multi_q       = features.get("f7_subquestions",0.0) >= 0.5
        return is_comparative or is_multi_temporal or is_aggregative or is_multi_q

    def _allocate_diversity(
        self,
        query: str,
        features: Dict[str, float],
        recommended_k: int,
        parsed_query: Optional[Any],
    ) -> tuple[int, int]:
        """
        Decide how many distinct documents (target_docs) and how many
        chunks per document (chunks_per_doc) to allocate.

        Rules:
          - Comparative / multi-temporal queries → at least 2 documents.
          - Aggregative / general-browse queries → at least 3 documents.
          - Specific entity + factual queries    → 1 document.
        """
        is_comparative    = features.get("f5_comparison",  0.0) >= 0.5
        is_multi_temporal = features.get("f4_temporal",    0.0) >= 0.5
        is_aggregative    = features.get("f6_aggregation", 0.0) >= 0.5

        # Detect general browse (category without specific entity / date)
        raw_entities = (getattr(parsed_query, "entities", []) or []) if parsed_query else []
        doc_type_slug = (getattr(parsed_query, "document_type", None) or "") if parsed_query else ""
        _stop_words = {
            doc_type_slug, "resume", "aadhaar", "aadhar", "pan", "bill",
            "invoice", "document", "documents", "file", "id", "card",
            "student_id_card", "",
        }
        filtered_entities = [e for e in raw_entities if e.lower().strip() not in _stop_words]
        has_specific_entity = bool(filtered_entities)
        has_specific_date   = bool(
            parsed_query and (
                getattr(parsed_query, "year", None) is not None
                or getattr(parsed_query, "month", None)
            )
        )

        q_lower = query.lower()
        is_factual = bool(re.search(
            r"\b(what is|what are|how much|when was|due date|total amount|gpa|number)\b",
            q_lower,
        ))

        is_category = bool(
            doc_type_slug or _DOC_BROWSE_RE.search(q_lower)
        )
        is_general_browse = (
            is_category
            and not has_specific_entity
            and not has_specific_date
            and not is_factual
        )

        if is_comparative or is_multi_temporal:
            target_docs = max(2, min(4, math.ceil(recommended_k / 2.5)))
        elif is_aggregative or is_general_browse:
            target_docs = max(3, min(5, math.ceil(recommended_k / 2.0)))
        else:
            target_docs = 1

        chunks_per_doc = max(1, math.ceil(recommended_k / target_docs))
        return target_docs, chunks_per_doc

    @staticmethod
    def _build_explanation(
        c_q: float,
        r_c: float,
        s_top: float,
        knee_gap: float,
        knee_index: int,
        combined_demand: float,
        coverage_penalty: int,
        recommended_k: int,
        target_docs: int,
        chunks_per_doc: int,
        stop_reason: str,
        is_shortcut: bool,
    ) -> str:
        cq_label = "High" if c_q >= 0.6 else ("Med" if c_q >= 0.3 else "Low")
        rc_label = "High" if r_c >= 0.7 else ("Med" if r_c >= 0.4 else "Low")
        penalty_str = f", coverage_penalty=+{coverage_penalty}" if coverage_penalty else ""
        shortcut_str = " [SHORTCUT]" if is_shortcut else ""
        return (
            f"C_q={c_q:.2f}({cq_label}), R_c={r_c:.2f}({rc_label}) "
            f"[S_top={s_top:.2f}, knee={knee_gap:.2f}@{knee_index}] "
            f"→ demand={combined_demand:.2f}{penalty_str} "
            f"→ K={recommended_k}{shortcut_str} "
            f"across {target_docs} doc(s) (max {chunks_per_doc}/doc) "
            f"[{stop_reason}]"
        )


# ── Module-level singleton (matches existing dynamic_k_governor interface) ────
# Uses the same default K bounds as the original dynamic_k.py so that callers
# importing dynamic_top_k_controller instead of dynamic_k_governor get
# identical behaviour.
dynamic_top_k_controller = DynamicTopKController()
