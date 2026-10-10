"""
query_complexity.py — Query Complexity Estimator (C_q)
=======================================================
Estimates how much evidence a query requires, returning a score C_q ∈ [0, 1].

Algorithm
---------
Eight deterministic features f1..f8 are extracted from the raw query text and
the structured ParsedQuery (when available).  Each feature is independently
normalized to [0, 1] so that no single signal can dominate, and a weighted sum
is computed:

    C_q = Σ  w_i · f_i      where  Σ w_i = 1.0

Feature table
~~~~~~~~~~~~~
  f1  query_length        — normalized character count; damped to avoid length bias
  f2  entity_count        — number of distinct named entities detected
  f3  constraint_count    — structured filters: doc_type, year, month, amount
  f4  temporal_complexity — multi-period / date-range signals
  f5  comparison_intent   — comparative language (compare, vs, difference …)
  f6  aggregation_intent  — aggregation language (total, sum, average, history …)
  f7  multi_subquestion   — multiple question marks or coordinating conjunctions
  f8  reasoning_required  — explanation / analysis intent (why, explain, factors …)

Default weights (sum = 1.0):
  f1=0.05  f2=0.15  f3=0.10  f4=0.20  f5=0.20  f6=0.15  f7=0.05  f8=0.10

Normalization
~~~~~~~~~~~~~
Every feature is already in [0, 1] before weighting, so C_q ∈ [0, 1] is
guaranteed by construction (no post-hoc sigmoid or clipping needed beyond
floating-point guard).

Design notes
~~~~~~~~~~~~
- Zero LLM overhead: entirely regex + arithmetic.
- Reuses ParsedQuery output from the existing query parser instead of
  re-extracting entities/constraints from scratch (avoids double-counting).
- Weights are configurable through ComplexityConfig; tune on dev data only.
- Returns a ComplexityResult with the full feature breakdown for debugging,
  evaluation, and audit logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ── Default Feature Weights ────────────────────────────────────────────────────
# Must sum to 1.0.  Documented as explicit design parameters, not empirically
# proven optima.  Tune on held-out dev data before changing defaults.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "f1_length":      0.05,   # damped length signal
    "f2_entities":    0.15,   # named-entity count
    "f3_constraints": 0.10,   # structured metadata filters
    "f4_temporal":    0.20,   # temporal / multi-period complexity
    "f5_comparison":  0.20,   # comparative intent
    "f6_aggregation": 0.15,   # aggregation intent
    "f7_subquestions":0.05,   # multiple sub-questions
    "f8_reasoning":   0.10,   # explanation / analysis requirement
}

# ── Regex Patterns ─────────────────────────────────────────────────────────────

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

_REASONING_RE = re.compile(
    r"\b(why|explain|reason|cause|how did|how come|elaborate|"
    r"analyze|analysis|factors?)\b",
    re.IGNORECASE,
)

_TEMPORAL_RANGE_RE = re.compile(
    r"\b(between|from\s+\w+\s+to\s+\w+|across\s+(?:the\s+)?"
    r"(?:months?|years?)|q[1-4]|quarter|h[1-2]|ytd|"
    r"over\s+time|last\s+\d+\s+months?)\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)

# ── Normalization constants ─────────────────────────────────────────────────────
_LENGTH_MIN = 10    # chars below this → f1 = 0.0
_LENGTH_MAX = 150   # chars at or above this → f1 = 1.0
_ENTITY_MAX = 3     # >3 entities → f2 = 1.0
_CONSTRAINT_MAX = 3 # >3 constraints → f3 = 1.0


@dataclass
class ComplexityConfig:
    """
    All configurable parameters for the complexity estimator.

    Attributes
    ----------
    weights:
        Per-feature weights dict.  Must contain all eight keys f1_length …
        f8_reasoning.  The estimator normalises the weights to sum to 1.0
        automatically, so absolute scale does not matter.
    length_min:
        Query character count mapped to f1 = 0.0.
    length_max:
        Query character count mapped to f1 = 1.0.
    entity_max:
        Entity count mapped to f2 = 1.0.
    constraint_max:
        Constraint count mapped to f3 = 1.0.
    """
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    length_min: int = _LENGTH_MIN
    length_max: int = _LENGTH_MAX
    entity_max: int = _ENTITY_MAX
    constraint_max: int = _CONSTRAINT_MAX


@dataclass
class ComplexityResult:
    """
    Full output of the complexity estimator.

    Attributes
    ----------
    score:
        C_q ∈ [0, 1].  Higher means the query is more complex and likely
        requires more document evidence.
    features:
        Dict mapping feature name → normalized value in [0, 1].
    weighted_contributions:
        Dict mapping feature name → w_i * f_i (useful for debugging which
        signal drove the score).
    explanation:
        Human-readable summary of the dominant contributing features.
    """
    score: float
    features: Dict[str, float]
    weighted_contributions: Dict[str, float]
    explanation: str


class QueryComplexityEstimator:
    """
    Deterministic, LLM-free query complexity estimator.

    Usage
    -----
    ::

        estimator = QueryComplexityEstimator()
        result = estimator.estimate("Compare my bank statements from 2022 and 2023")
        print(result.score)          # e.g. 0.72
        print(result.features)       # {'f1_length': 0.25, 'f5_comparison': 1.0, …}
        print(result.explanation)    # 'High: comparison_intent (0.20), temporal (0.20)'

    Thread-safety
    -------------
    All state is read-only after construction; safe to share across threads.
    """

    def __init__(self, config: Optional[ComplexityConfig] = None) -> None:
        self._cfg = config or ComplexityConfig()
        self._weights = self._normalize_weights(self._cfg.weights)

    # ── Public API ─────────────────────────────────────────────────────────────

    def estimate(
        self,
        query: str,
        parsed_query: Optional[Any] = None,
    ) -> ComplexityResult:
        """
        Estimate query complexity.

        Parameters
        ----------
        query:
            Raw user query string.  Empty / whitespace-only → score = 0.0.
        parsed_query:
            Optional ParsedQuery from the existing query parser.  When
            provided, entity and constraint counts are taken directly from
            its structured fields to avoid double-extraction.

        Returns
        -------
        ComplexityResult with score, features, contributions, explanation.
        """
        if not query or not query.strip():
            return ComplexityResult(
                score=0.0,
                features={k: 0.0 for k in self._weights},
                weighted_contributions={k: 0.0 for k in self._weights},
                explanation="Empty query: complexity = 0.0",
            )

        features = self.extract_features(query, parsed_query)
        score, contributions = self._compute_score(features)
        explanation = self._build_explanation(score, contributions)

        return ComplexityResult(
            score=score,
            features=features,
            weighted_contributions=contributions,
            explanation=explanation,
        )

    def extract_features(
        self,
        query: str,
        parsed_query: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Extract all eight normalized features from the query.

        Each value is in [0, 1].  See module docstring for feature definitions.
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()
        cfg = self._cfg

        # ── f1: Query length ──────────────────────────────────────────────────
        # Linear interpolation from [length_min, length_max] → [0, 1]
        span = max(cfg.length_max - cfg.length_min, 1)
        f1 = min(max(len(q_clean) - cfg.length_min, 0) / span, 1.0)

        # ── f2: Entity count ─────────────────────────────────────────────────
        # Prefer parsed_query.entities if available; avoids NER overhead.
        if parsed_query is not None:
            entities = getattr(parsed_query, "entities", None) or []
        else:
            # Heuristic fallback: capitalize-pattern words (e.g. proper nouns)
            entities = re.findall(r"\b[A-Z][a-z]{2,}\b", query)
        f2 = min(len(entities) / max(cfg.entity_max, 1), 1.0)

        # ── f3: Structured constraint count ──────────────────────────────────
        constraints = 0
        if parsed_query is not None:
            if getattr(parsed_query, "document_type", None):
                constraints += 1
            if getattr(parsed_query, "year", None) is not None:
                constraints += 1
            if getattr(parsed_query, "month", None):
                constraints += 1
            if getattr(parsed_query, "amount_filters", None):
                constraints += 1
        else:
            # Regex-based fallback
            if _YEAR_RE.search(q_lower):
                constraints += 1
            if _MONTH_RE.search(q_lower):
                constraints += 1
        f3 = min(constraints / max(cfg.constraint_max, 1), 1.0)

        # ── f4: Temporal complexity ───────────────────────────────────────────
        # Multi-period = multiple distinct years OR months, or an explicit range.
        # Single point-in-time (one year + one month) = 0.5
        # Single temporal token (just year or just month) = 0.3
        years  = list(set(_YEAR_RE.findall(q_lower)))
        months = list(set(_MONTH_RE.findall(q_lower)))
        has_range = bool(_TEMPORAL_RANGE_RE.search(q_lower))

        if len(years) > 1 or len(months) > 1 or has_range:
            f4 = 1.0
        elif len(years) == 1 and len(months) == 1:
            f4 = 0.5
        elif len(years) == 1 or len(months) == 1:
            f4 = 0.3
        else:
            f4 = 0.0

        # ── f5: Comparison intent ─────────────────────────────────────────────
        f5 = 1.0 if _COMPARISON_RE.search(q_lower) else 0.0

        # ── f6: Aggregation intent ────────────────────────────────────────────
        f6 = 1.0 if _AGGREGATION_RE.search(q_lower) else 0.0

        # ── f7: Multi-sub-question / multi-part ───────────────────────────────
        qmark_count = q_clean.count("?")
        has_coord   = bool(re.search(
            r"\b(and also|as well as|in addition|along with)\b", q_lower
        ))
        if qmark_count > 1 or (qmark_count >= 1 and has_coord):
            f7 = 1.0
        elif has_coord:
            f7 = 0.5
        else:
            f7 = 0.0

        # ── f8: Reasoning / analysis requirement ─────────────────────────────
        f8 = 1.0 if _REASONING_RE.search(q_lower) else 0.0

        return {
            "f1_length":       round(f1, 4),
            "f2_entities":     round(f2, 4),
            "f3_constraints":  round(f3, 4),
            "f4_temporal":     round(f4, 4),
            "f5_comparison":   round(f5, 4),
            "f6_aggregation":  round(f6, 4),
            "f7_subquestions": round(f7, 4),
            "f8_reasoning":    round(f8, 4),
        }

    def compute_score(self, features: Dict[str, float]) -> float:
        """
        Compute C_q from a pre-extracted feature dict.

        Returns C_q ∈ [0, 1].
        """
        score, _ = self._compute_score(features)
        return score

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_score(
        self, features: Dict[str, float]
    ) -> tuple[float, Dict[str, float]]:
        """Compute weighted sum + per-feature contributions."""
        contributions: Dict[str, float] = {}
        total = 0.0
        for key, weight in self._weights.items():
            fval = float(features.get(key, 0.0))
            contrib = weight * fval
            contributions[key] = round(contrib, 4)
            total += contrib
        # Guard floating-point drift
        score = max(0.0, min(total, 1.0))
        return round(score, 4), contributions

    @staticmethod
    def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
        """
        Normalise weight dict so that values sum to 1.0.
        Falls back to uniform weights if all values are zero.
        """
        total = sum(raw.values())
        if total <= 0.0:
            n = max(len(raw), 1)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    @staticmethod
    def _build_explanation(score: float, contributions: Dict[str, float]) -> str:
        """Build a concise human-readable explanation of the dominant signals."""
        label = "High" if score >= 0.6 else ("Medium" if score >= 0.3 else "Low")
        top = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k.split('_', 1)[1]}={v:.3f}" for k, v in top if v > 0)
        if top_str:
            return f"{label} complexity (C_q={score:.3f}): {top_str}"
        return f"{label} complexity (C_q={score:.3f}): no dominant signals"


# ── Module-level default instance ─────────────────────────────────────────────
# Can be overridden by injecting a custom ComplexityConfig in tests or config.
complexity_estimator = QueryComplexityEstimator()
