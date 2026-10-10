"""
retrieval_confidence.py — Retrieval Confidence Estimator (R_c)
===============================================================
Estimates how sufficient the initial retrieval results are, returning R_c ∈ [0, 1].

Algorithm
---------
Four signal components are computed from the initial chunk list and combined
with a convex weighting (weights sum to 1.0):

    R_c = W_stop · S_top
        + W_savg · S_avg
        + W_knee · G_knee
        + W_evidence · E

Signal definitions
~~~~~~~~~~~~~~~~~~
  S_top   — Calibrated top similarity score.
             Qdrant returns raw cosine similarity ∈ [-1, 1] for cosine-distance
             collections, but in practice the store is configured with
             COSINE distance and the returned scores are in [0, 1] (Qdrant
             normalises them via (1 + cosine) / 2 internally).  We clip to
             [0, 1] and do NOT treat the raw value as a calibrated probability.

  S_avg   — Average similarity over the top-min(N, 4) chunks.
             Using only the top slice prevents a long tail of weak candidates
             from diluting a genuinely strong match.

  G_knee  — Knee-point gap: maximum consecutive score drop between adjacent
             ranked candidates.
               G_knee = max_i ( score_i - score_{i+1} )   for i = 1 … N-1
             A large gap signals a clear boundary between relevant and
             irrelevant chunks, which correlates with high precision in the
             top slice.
             LIMITATION: a high G_knee can be misleading if S_top is also
             weak (e.g. [0.45, 0.10]).  The S_top and S_avg terms prevent
             this from inflating R_c unduly; see _guard_low_scores().

  E       — Evidence alignment score.
             Deterministic metadata check: how many of the query's explicit
             constraints (document_type, year, month, entities) are satisfied
             by the top-3 chunks.  Falls back to S_top when no metadata
             constraints exist.
             LIMITATION: not all metadata is populated for every document.
             When metadata is sparse the score degrades gracefully to S_top.

Normalisation
~~~~~~~~~~~~~
  S_top, S_avg ∈ [0, 1] by construction (Qdrant cosine similarity clipped).
  G_knee is clipped to [0, 1].
  E ∈ [0, 1] by construction.
  Therefore R_c ∈ [0, 1] and no further normalisation is required.

Configurable parameters
~~~~~~~~~~~~~~~~~~~~~~~
  See ConfidenceConfig.

Limitations explicitly documented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Qdrant cosine scores are NOT calibrated probabilities.  We treat them as
     relative relevance proxies, not absolute confidence values.
  2. Evidence alignment relies on chunk payload metadata.  Sparsely indexed
     documents (legacy uploads without structured metadata) will fall back to
     the score-based baseline.
  3. Duplicate chunks from the same document inflate S_avg.  The estimator
     deduplicates by chunk_id before computing score statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Default Confidence Weights ─────────────────────────────────────────────────
# Strictly convex: must sum to 1.0.
DEFAULT_W_STOP     = 0.35  # weight for S_top
DEFAULT_W_SAVG     = 0.25  # weight for S_avg
DEFAULT_W_KNEE     = 0.20  # weight for G_knee
DEFAULT_W_EVIDENCE = 0.20  # weight for E


@dataclass
class ConfidenceConfig:
    """
    Configurable parameters for the retrieval confidence estimator.

    Attributes
    ----------
    w_stop : float
        Weight for S_top (top similarity score).
    w_savg : float
        Weight for S_avg (average over top-k similarities).
    w_knee : float
        Weight for G_knee (maximum consecutive score gap).
    w_evidence : float
        Weight for E (metadata evidence alignment).
    top_k_avg : int
        Number of top chunks to average for S_avg (default 4).
    min_score_guard : float
        If S_top is below this threshold, G_knee is suppressed to prevent
        a high knee gap with weak scores from falsely boosting R_c.
    """
    w_stop:          float = DEFAULT_W_STOP
    w_savg:          float = DEFAULT_W_SAVG
    w_knee:          float = DEFAULT_W_KNEE
    w_evidence:      float = DEFAULT_W_EVIDENCE
    top_k_avg:       int   = 4
    min_score_guard: float = 0.50   # S_top below this → G_knee contribution zeroed

    def __post_init__(self) -> None:
        total = self.w_stop + self.w_savg + self.w_knee + self.w_evidence
        if not (0.9999 <= total <= 1.0001):
            # Auto-normalise rather than raise, but warn loudly
            logger.warning(
                "[CONFIDENCE] Weights do not sum to 1.0 (got %.4f); normalising.", total
            )
            self.w_stop     /= total
            self.w_savg     /= total
            self.w_knee     /= total
            self.w_evidence /= total


@dataclass
class ConfidenceResult:
    """
    Full output of the retrieval confidence estimator.

    Attributes
    ----------
    score : float
        R_c ∈ [0, 1].  Higher means the initial retrieval looks sufficient.
    s_top : float
        Best similarity score (clipped to [0, 1]).
    s_avg : float
        Average similarity over the top slice.
    knee_gap : float
        Maximum consecutive score drop between adjacent ranks.
    knee_index : int
        1-based rank index immediately before the largest drop (knee cliff).
    evidence_alignment : float
        Metadata/entity coverage score E ∈ [0, 1].
    distinct_docs : int
        Number of unique document_id values in the initial result set.
    duplicate_chunks_removed : int
        Number of duplicate chunk_ids filtered before scoring.
    breakdown : Dict[str, float]
        Per-signal contributions before weighting: {'s_top': …, 's_avg': …,
        'knee_gap': …, 'evidence': …}.
    weighted_contributions : Dict[str, float]
        Per-signal weighted contributions (w_i * signal_i).
    explanation : str
        Human-readable summary.
    """
    score:                    float
    s_top:                    float
    s_avg:                    float
    knee_gap:                 float
    knee_index:               int
    evidence_alignment:       float
    distinct_docs:            int
    duplicate_chunks_removed: int
    breakdown:                Dict[str, float] = field(default_factory=dict)
    weighted_contributions:   Dict[str, float] = field(default_factory=dict)
    explanation:              str = ""


class RetrievalConfidenceEstimator:
    """
    Estimates retrieval sufficiency from an initial chunk list.

    Parameters
    ----------
    config : ConfidenceConfig, optional
        Supply custom weights or thresholds.  Defaults apply when omitted.

    Thread-safety
    -------------
    Stateless after construction; safe to share across async tasks.
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None) -> None:
        self._cfg = config or ConfidenceConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def estimate(
        self,
        chunks: List[Any],
        parsed_query: Optional[Any] = None,
    ) -> ConfidenceResult:
        """
        Estimate retrieval confidence from an initial chunk list.

        Parameters
        ----------
        chunks :
            List of Qdrant ScoredPoint objects (or any objects with a
            ``.score`` float attribute and an optional ``.payload`` dict).
            An empty list → R_c = 0.0.
        parsed_query :
            Optional ParsedQuery from the existing query parser.  Used for
            the evidence-alignment signal (E).

        Returns
        -------
        ConfidenceResult with all diagnostic fields populated.
        """
        if not chunks:
            return self._zero_result("No chunks retrieved: R_c = 0.0")

        # ── Deduplication ─────────────────────────────────────────────────────
        deduped, n_dupes = self._deduplicate(chunks)
        if not deduped:
            return self._zero_result(
                f"All {n_dupes} chunks were duplicates after dedup: R_c = 0.0"
            )

        # ── Score extraction ──────────────────────────────────────────────────
        scores = self._extract_scores(deduped)
        distinct_docs = self._count_distinct_docs(deduped)

        # ── Signal computation ────────────────────────────────────────────────
        s_top  = self._compute_s_top(scores)
        s_avg  = self._compute_s_avg(scores)
        knee_gap, knee_index = self._compute_knee(scores)
        evidence = self._compute_evidence_alignment(deduped, parsed_query, s_top)

        # ── Low-score guard: suppress G_knee when S_top is weak ──────────────
        # Prevents the knee gap from inflating R_c when all scores are low
        # (e.g. scores [0.42, 0.10] have a large gap but weak top score).
        effective_knee = knee_gap if s_top >= self._cfg.min_score_guard else 0.0

        # ── Convex combination ────────────────────────────────────────────────
        cfg = self._cfg
        wc = {
            "s_top":   cfg.w_stop     * s_top,
            "s_avg":   cfg.w_savg     * s_avg,
            "knee":    cfg.w_knee     * effective_knee,
            "evidence": cfg.w_evidence * evidence,
        }
        r_c = max(0.0, min(sum(wc.values()), 1.0))

        breakdown = {
            "s_top":    round(s_top,    4),
            "s_avg":    round(s_avg,    4),
            "knee_gap": round(knee_gap, 4),
            "evidence": round(evidence, 4),
        }
        explanation = self._build_explanation(
            r_c, s_top, s_avg, knee_gap, effective_knee, evidence, distinct_docs
        )

        return ConfidenceResult(
            score=round(r_c, 4),
            s_top=round(s_top, 4),
            s_avg=round(s_avg, 4),
            knee_gap=round(knee_gap, 4),
            knee_index=knee_index,
            evidence_alignment=round(evidence, 4),
            distinct_docs=distinct_docs,
            duplicate_chunks_removed=n_dupes,
            breakdown=breakdown,
            weighted_contributions={k: round(v, 4) for k, v in wc.items()},
            explanation=explanation,
        )

    # ── Signal helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_scores(chunks: List[Any]) -> List[float]:
        """Extract and sort similarity scores descending."""
        raw = []
        for c in chunks:
            s = getattr(c, "score", None)
            if s is not None:
                # Clip to [0, 1]; Qdrant cosine scores should already be in range
                raw.append(float(max(0.0, min(s, 1.0))))
        raw.sort(reverse=True)
        return raw

    def _compute_s_top(self, scores: List[float]) -> float:
        return scores[0] if scores else 0.0

    def _compute_s_avg(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        k = min(len(scores), self._cfg.top_k_avg)
        return sum(scores[:k]) / k

    @staticmethod
    def _compute_knee(scores: List[float]) -> Tuple[float, int]:
        """
        Detect the sharpest consecutive score drop (knee cliff).

        Returns (knee_gap, knee_index) where knee_index is the 1-based
        rank of the last strong candidate (the one before the cliff).
        """
        if len(scores) < 2:
            return 0.0, 1

        max_drop  = -1.0
        best_idx  = 1  # 1-indexed position before the cliff
        for i in range(len(scores) - 1):
            drop = scores[i] - scores[i + 1]
            if drop > max_drop:
                max_drop = drop
                best_idx = i + 1  # 1-indexed

        knee_gap = max(0.0, min(max_drop, 1.0))
        return round(knee_gap, 4), best_idx

    def _compute_evidence_alignment(
        self,
        chunks: List[Any],
        parsed_query: Optional[Any],
        s_top: float,
    ) -> float:
        """
        Evaluate structural alignment between query constraints and top chunks.

        If no metadata constraints exist in the query, falls back to S_top
        (score-based baseline) since there is nothing to verify.

        Coverage logic: for each explicit constraint in the query, check
        whether it is satisfied by at least one of the top-3 chunks.
        Returns the fraction of satisfied constraints.

        LIMITATION: relies on payload metadata; sparse/legacy documents may
        lack year/month/document_type fields, causing under-counting.
        """
        if not parsed_query:
            return s_top

        target_type   = getattr(parsed_query, "document_type", None)
        target_year   = getattr(parsed_query, "year", None)
        target_month  = getattr(parsed_query, "month", None)
        entities      = getattr(parsed_query, "entities", None) or []

        # If the query has no metadata constraints, the vector score is our
        # only evidence signal.
        has_constraints = bool(target_type or target_year or target_month or entities)
        if not has_constraints:
            return s_top

        # Inspect top-3 chunks only (avoids dilution by weak candidates)
        top_chunks = chunks[:3]
        best_alignment = 0.0

        for chunk in top_chunks:
            p = getattr(chunk, "payload", {}) or {}
            chunk_text  = (p.get("text", "") or p.get("text_content", "") or "").lower()
            chunk_type  = (p.get("document_type") or p.get("category") or "").lower()
            chunk_year  = p.get("year")
            chunk_month = (p.get("month") or "").lower()

            satisfied  = 0.0
            total_chks = 0.0

            if target_type:
                total_chks += 1.0
                if target_type.lower() == chunk_type:
                    satisfied += 1.0

            if target_year is not None:
                total_chks += 1.0
                if chunk_year == target_year or str(target_year) in chunk_text:
                    satisfied += 1.0

            if target_month:
                total_chks += 1.0
                if (chunk_month and target_month.lower() == chunk_month) or \
                        target_month.lower() in chunk_text:
                    satisfied += 1.0

            if entities:
                total_chks += 1.0
                if any(e.lower() in chunk_text for e in entities):
                    satisfied += 1.0

            alignment = (satisfied / total_chks) if total_chks > 0 else s_top
            if alignment > best_alignment:
                best_alignment = alignment

        return round(max(0.0, min(best_alignment, 1.0)), 4)

    # ── Utility helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(chunks: List[Any]) -> Tuple[List[Any], int]:
        """
        Remove duplicate chunks by chunk_id.
        Returns (deduped_list, num_removed).
        """
        seen_ids: set = set()
        deduped: List[Any] = []
        for chunk in chunks:
            p  = getattr(chunk, "payload", {}) or {}
            cid = p.get("chunk_id") or p.get("id") or id(chunk)
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append(chunk)
        return deduped, len(chunks) - len(deduped)

    @staticmethod
    def _count_distinct_docs(chunks: List[Any]) -> int:
        """Count distinct document_id values in the chunk list."""
        doc_ids: set = set()
        for chunk in chunks:
            p = getattr(chunk, "payload", {}) or {}
            did = p.get("document_id") or p.get("id")
            if did:
                doc_ids.add(did)
        return len(doc_ids)

    @staticmethod
    def _zero_result(explanation: str) -> ConfidenceResult:
        return ConfidenceResult(
            score=0.0,
            s_top=0.0,
            s_avg=0.0,
            knee_gap=0.0,
            knee_index=0,
            evidence_alignment=0.0,
            distinct_docs=0,
            duplicate_chunks_removed=0,
            breakdown={"s_top": 0.0, "s_avg": 0.0, "knee_gap": 0.0, "evidence": 0.0},
            weighted_contributions={"s_top": 0.0, "s_avg": 0.0, "knee": 0.0, "evidence": 0.0},
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(
        r_c: float,
        s_top: float,
        s_avg: float,
        knee_gap: float,
        eff_knee: float,
        evidence: float,
        distinct_docs: int,
    ) -> str:
        label = "High" if r_c >= 0.70 else ("Medium" if r_c >= 0.40 else "Low")
        guarded = " [knee suppressed: weak S_top]" if eff_knee == 0.0 and knee_gap > 0 else ""
        return (
            f"{label} confidence (R_c={r_c:.3f}): "
            f"S_top={s_top:.2f}, S_avg={s_avg:.2f}, "
            f"knee={knee_gap:.2f}{guarded}, evidence={evidence:.2f}, "
            f"docs={distinct_docs}"
        )


# ── Module-level default instance ─────────────────────────────────────────────
confidence_estimator = RetrievalConfidenceEstimator()
