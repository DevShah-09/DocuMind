"""
test_query_complexity.py — Unit Tests for QueryComplexityEstimator
==================================================================
Covers:
  - Empty / whitespace queries
  - Simple factual queries (expected C_q ≤ 0.30)
  - Complex comparison queries (expected C_q ≥ 0.55)
  - Aggregation queries
  - Temporal single-point vs. multi-period
  - Multi-sub-question detection
  - Reasoning requirement detection
  - ParsedQuery integration (entities, constraints)
  - Malformed / None parsed_query fallback
  - Weight normalisation invariant
  - Determinism (identical inputs → identical outputs)
  - Feature range invariant (all features in [0, 1])
  - Score range invariant (C_q in [0, 1])
  - Custom ComplexityConfig weights
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.retrieval.query_complexity import (
    ComplexityConfig,
    QueryComplexityEstimator,
    DEFAULT_WEIGHTS,
    complexity_estimator,
)
from app.schemas.retrieval import ParsedQuery


def _parsed(
    doc_type=None, year=None, month=None, entities=None, amount=None, sem_q=""
) -> ParsedQuery:
    return ParsedQuery(
        document_type=doc_type,
        year=year,
        month=month,
        entities=entities or [],
        amount_filters=amount,
        semantic_query=sem_q or "query",
        raw_query=sem_q or "query",
    )


class TestQueryComplexityEdgeCases(unittest.TestCase):

    def setUp(self):
        self.est = QueryComplexityEstimator()

    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_empty_query_returns_zero(self):
        result = self.est.estimate("")
        self.assertEqual(result.score, 0.0)
        self.assertIn("Empty", result.explanation)

    def test_whitespace_only_returns_zero(self):
        result = self.est.estimate("   ")
        self.assertEqual(result.score, 0.0)

    def test_none_parsed_query_does_not_crash(self):
        result = self.est.estimate("What is my PAN number?", parsed_query=None)
        self.assertIsInstance(result.score, float)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_malformed_parsed_query_namespace_fallback(self):
        # Simulate a partial parsed_query object missing some attributes
        partial = SimpleNamespace(entities=["John"], document_type=None)
        result  = self.est.estimate("Find John's resume", parsed_query=partial)
        self.assertIsInstance(result.score, float)

    # ── Score range invariant ──────────────────────────────────────────────────

    def test_score_always_in_0_1(self):
        queries = [
            "hi",
            "What is my PAN number?",
            "Compare and analyze all my bank statements from 2019 to 2024 "
            "and explain the trend in monthly expenses and why total spending increased.",
            "a" * 500,  # very long
        ]
        for q in queries:
            result = self.est.estimate(q)
            self.assertGreaterEqual(result.score, 0.0, f"Failed for: {q[:50]}")
            self.assertLessEqual(result.score, 1.0,   f"Failed for: {q[:50]}")

    def test_all_features_in_0_1(self):
        q      = "Compare bank statements from 2022 and 2023 and explain the trend."
        result = self.est.estimate(q)
        for feat_name, val in result.features.items():
            self.assertGreaterEqual(val, 0.0, f"{feat_name} < 0")
            self.assertLessEqual(val,   1.0, f"{feat_name} > 1")

    # ── Determinism ────────────────────────────────────────────────────────────

    def test_deterministic_identical_inputs(self):
        q = "Summarize my electricity bill for January 2024."
        r1 = self.est.estimate(q)
        r2 = self.est.estimate(q)
        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.features, r2.features)

    # ── Weight normalisation ───────────────────────────────────────────────────

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_custom_weights_auto_normalised(self):
        cfg  = ComplexityConfig(weights={k: 2.0 for k in DEFAULT_WEIGHTS})
        est  = QueryComplexityEstimator(cfg)
        total = sum(est._weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_zero_weights_normalised_to_uniform(self):
        cfg = ComplexityConfig(weights={k: 0.0 for k in DEFAULT_WEIGHTS})
        est = QueryComplexityEstimator(cfg)
        # uniform: 1/8 per feature
        for v in est._weights.values():
            self.assertAlmostEqual(v, 1.0 / 8, places=5)


class TestQueryComplexityLowComplexity(unittest.TestCase):

    def setUp(self):
        self.est = QueryComplexityEstimator()

    def test_simple_factual_pan(self):
        parsed = _parsed(doc_type="pan_card", entities=["PAN"])
        result = self.est.estimate("What is my PAN number?", parsed)
        self.assertLessEqual(result.score, 0.30,
            f"Expected C_q ≤ 0.30 for simple factual, got {result.score}")

    def test_simple_show_document(self):
        parsed = _parsed(doc_type="aadhaar")
        result = self.est.estimate("Show me my Aadhaar card.", parsed)
        self.assertLessEqual(result.score, 0.30)

    def test_single_date_constraint(self):
        parsed = _parsed(doc_type="electricity_bill", year=2024, month="January")
        result = self.est.estimate("Electricity bill January 2024", parsed)
        # f4 partial credit + f3 constraints, still should be low
        self.assertLessEqual(result.score, 0.45)


class TestQueryComplexityHighComplexity(unittest.TestCase):

    def setUp(self):
        self.est = QueryComplexityEstimator()

    def test_comparison_query(self):
        q      = "Compare my electricity bills from January 2023 and January 2024."
        parsed = _parsed(doc_type="electricity_bill")
        result = self.est.estimate(q, parsed)
        self.assertGreaterEqual(result.score, 0.40,
            f"Expected C_q ≥ 0.40 for comparison, got {result.score}")
        self.assertEqual(result.features["f5_comparison"], 1.0)

    def test_aggregation_query(self):
        q      = "What is the total of all my electricity bills in 2023?"
        result = self.est.estimate(q)
        self.assertEqual(result.features["f6_aggregation"], 1.0)

    def test_reasoning_query(self):
        q      = "Why did my electricity bill increase in December 2023?"
        result = self.est.estimate(q)
        self.assertEqual(result.features["f8_reasoning"], 1.0)

    def test_multi_year_temporal(self):
        q = "Show all bank statements from 2021, 2022, and 2023."
        result = self.est.estimate(q)
        self.assertEqual(result.features["f4_temporal"], 1.0)

    def test_temporal_range_keyword(self):
        q = "Show my expenses from January to June 2024."
        result = self.est.estimate(q)
        self.assertEqual(result.features["f4_temporal"], 1.0)

    def test_full_complex_query(self):
        q = (
            "Compare my bank statements from 2022 and 2023 and explain "
            "why the total spending increased significantly each quarter."
        )
        parsed = _parsed(entities=["bank", "SBI"])
        result = self.est.estimate(q, parsed)
        self.assertGreaterEqual(result.score, 0.55,
            f"Expected C_q ≥ 0.55 for full complex query, got {result.score}")

    def test_multi_subquestion_two_qmarks(self):
        q = "What is my PAN number? What is my Aadhaar number?"
        result = self.est.estimate(q)
        self.assertGreaterEqual(result.features["f7_subquestions"], 0.9)

    def test_multi_subquestion_coordinating_conjunction(self):
        q = "Show my resume as well as my bank statement."
        result = self.est.estimate(q)
        self.assertGreater(result.features["f7_subquestions"], 0.0)


class TestQueryComplexityMonotonicity(unittest.TestCase):
    """Higher complexity signals must produce higher C_q."""

    def setUp(self):
        self.est = QueryComplexityEstimator()

    def test_comparison_raises_score_vs_simple(self):
        simple  = self.est.estimate("Show my bank statement.")
        complex_ = self.est.estimate("Compare my bank statements from 2022 and 2023.")
        self.assertGreater(complex_.score, simple.score)

    def test_entity_count_raises_score(self):
        few = _parsed(entities=["Alice"])
        many = _parsed(entities=["Alice", "Bob", "Charlie"])
        r_few  = self.est.estimate("Find documents for Alice.", few)
        r_many = self.est.estimate("Find documents for Alice, Bob, and Charlie.", many)
        self.assertGreaterEqual(r_many.score, r_few.score)

    def test_constraint_count_raises_score(self):
        no_constraints   = _parsed()
        with_constraints = _parsed(doc_type="bank_statement", year=2023, month="March")
        r_none = self.est.estimate("Show me a document.", no_constraints)
        r_full = self.est.estimate("Show bank statement March 2023.", with_constraints)
        self.assertGreater(r_full.score, r_none.score)


class TestComplexityContributions(unittest.TestCase):
    """Verify that weighted_contributions sum to approximately the score."""

    def setUp(self):
        self.est = QueryComplexityEstimator()

    def test_contributions_sum_to_score(self):
        # weighted_contributions are individually rounded to 4 dp before storage,
        # so their sum can differ from the final rounded score by up to ~0.0001
        # per feature (8 features × 0.00005 max rounding error ≈ 0.0004 max drift).
        # We therefore compare at 3 decimal places.
        q      = "Compare my bills from 2022 and 2023 and explain the trend."
        result = self.est.estimate(q)
        contrib_sum = sum(result.weighted_contributions.values())
        self.assertAlmostEqual(contrib_sum, result.score, places=3)

    def test_explanation_non_empty(self):
        result = self.est.estimate("What is my PAN number?")
        self.assertIsInstance(result.explanation, str)
        self.assertGreater(len(result.explanation), 0)


if __name__ == "__main__":
    unittest.main()
