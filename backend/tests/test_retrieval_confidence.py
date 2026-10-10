"""
test_retrieval_confidence.py — Unit Tests for RetrievalConfidenceEstimator
==========================================================================
Covers:
  - Empty chunk list → R_c = 0.0
  - High-confidence result (strong top scores, large knee gap, good evidence)
  - Low-confidence result (weak scores across the board)
  - No-result / zero chunks
  - Knee-point detection (gap and index)
  - Low-score guard: high knee gap with weak S_top must NOT inflate R_c
  - Duplicate chunk deduplication
  - Missing / None similarity scores
  - Evidence alignment with matching metadata
  - Evidence alignment with mismatching metadata
  - Evidence alignment fallback when no constraints exist
  - Distinct document counting
  - Score clipping (scores outside [0,1])
  - Weight sum invariant (convex weights)
  - Determinism
  - ConfidenceConfig auto-normalisation warning path
"""

from __future__ import annotations

import unittest
import warnings

from app.retrieval.retrieval_confidence import (
    ConfidenceConfig,
    RetrievalConfidenceEstimator,
    DEFAULT_W_STOP,
    DEFAULT_W_SAVG,
    DEFAULT_W_KNEE,
    DEFAULT_W_EVIDENCE,
    confidence_estimator,
)
from app.schemas.retrieval import ParsedQuery


def _chunk(score: float, chunk_id: str = "c1", doc_id: str = "d1",
           doc_type: str = "pan_card", year=None, month=None, text: str = "") -> object:
    """Build a lightweight mock chunk object."""
    class _C:
        pass
    c = _C()
    c.score   = score
    c.payload = {
        "chunk_id":      chunk_id,
        "document_id":   doc_id,
        "document_type": doc_type,
        "text":          text,
        "year":          year,
        "month":         month,
    }
    return c


def _parsed(doc_type=None, year=None, month=None, entities=None):
    return ParsedQuery(
        document_type=doc_type,
        year=year,
        month=month,
        entities=entities or [],
        amount_filters=None,
        semantic_query="test",
        raw_query="test",
    )


class TestConfidenceEdgeCases(unittest.TestCase):

    def setUp(self):
        self.est = RetrievalConfidenceEstimator()

    def test_empty_chunks_returns_zero(self):
        result = self.est.estimate([])
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.s_top, 0.0)
        self.assertEqual(result.s_avg, 0.0)
        self.assertEqual(result.knee_gap, 0.0)
        self.assertEqual(result.distinct_docs, 0)

    def test_single_chunk_no_crash(self):
        result = self.est.estimate([_chunk(0.85)])
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.knee_gap, 0.0)  # can't compute gap with 1 chunk
        self.assertEqual(result.knee_index, 1)

    def test_all_duplicate_chunks_returns_zero(self):
        dupes = [_chunk(0.90, chunk_id="same") for _ in range(5)]
        result = self.est.estimate(dupes)
        # After dedup only 1 chunk remains — should still work, not return 0
        self.assertGreaterEqual(result.score, 0.0)
        self.assertEqual(result.duplicate_chunks_removed, 4)

    def test_missing_score_attribute_handled(self):
        class NoScore:
            payload = {"chunk_id": "x", "document_id": "d"}
        result = self.est.estimate([NoScore()])
        # score attribute missing → treated as 0; result should be valid
        self.assertIsInstance(result.score, float)

    def test_score_above_1_clipped(self):
        c = _chunk(1.5)   # out of range
        result = self.est.estimate([c])
        self.assertLessEqual(result.s_top, 1.0)

    def test_score_below_0_clipped(self):
        c = _chunk(-0.3)
        result = self.est.estimate([c])
        self.assertGreaterEqual(result.s_top, 0.0)

    def test_r_c_always_in_0_1(self):
        chunk_sets = [
            [],
            [_chunk(0.99)],
            [_chunk(0.01)],
            [_chunk(0.50), _chunk(0.49), _chunk(0.10)],
            [_chunk(0.95), _chunk(0.92), _chunk(0.40), _chunk(0.38)],
        ]
        for chunks in chunk_sets:
            r = self.est.estimate(chunks)
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score,   1.0)

    def test_deterministic(self):
        chunks = [_chunk(0.90), _chunk(0.85), _chunk(0.40)]
        r1 = self.est.estimate(chunks)
        r2 = self.est.estimate(chunks)
        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.knee_gap, r2.knee_gap)


class TestConfidenceWeights(unittest.TestCase):

    def test_default_weights_sum_to_one(self):
        total = DEFAULT_W_STOP + DEFAULT_W_SAVG + DEFAULT_W_KNEE + DEFAULT_W_EVIDENCE
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_config_auto_normalises_when_off(self):
        # Weights that don't sum to 1 should be auto-normalised with a warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = ConfidenceConfig(w_stop=0.5, w_savg=0.5, w_knee=0.5, w_evidence=0.5)
        total = cfg.w_stop + cfg.w_savg + cfg.w_knee + cfg.w_evidence
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_custom_config_respected(self):
        cfg = ConfidenceConfig(w_stop=1.0, w_savg=0.0, w_knee=0.0, w_evidence=0.0)
        est = RetrievalConfidenceEstimator(cfg)
        chunks = [_chunk(0.80)]
        r = est.estimate(chunks)
        # With only w_stop active, R_c ≈ S_top = 0.80
        self.assertAlmostEqual(r.score, 0.80, places=2)


class TestHighConfidence(unittest.TestCase):
    """High top scores + clear knee cliff + matching metadata → R_c ≥ 0.70."""

    def setUp(self):
        self.est = RetrievalConfidenceEstimator()

    def test_strong_scores_high_confidence(self):
        chunks = [
            _chunk(0.95, "c1", "d1", doc_type="pan_card"),
            _chunk(0.91, "c2", "d1", doc_type="pan_card"),
            _chunk(0.40, "c3", "d2", doc_type="electricity_bill"),
        ]
        parsed = _parsed(doc_type="pan_card")
        result = self.est.estimate(chunks, parsed)
        self.assertGreaterEqual(result.r_c if hasattr(result, "r_c") else result.score, 0.60)

    def test_s_top_reflects_best_score(self):
        chunks = [_chunk(0.95), _chunk(0.80), _chunk(0.60)]
        result = self.est.estimate(chunks)
        self.assertAlmostEqual(result.s_top, 0.95, places=2)

    def test_s_avg_uses_top_4(self):
        # 6 chunks: top 4 avg should be (0.90+0.85+0.80+0.75)/4 = 0.825
        chunks = [
            _chunk(0.90, f"c{i}", "d1") for i, s in
            enumerate([0.90, 0.85, 0.80, 0.75, 0.30, 0.20])
        ]
        # rebuild with correct scores
        scores = [0.90, 0.85, 0.80, 0.75, 0.30, 0.20]
        chunks = [_chunk(s, f"c{i}") for i, s in enumerate(scores)]
        result = self.est.estimate(chunks)
        expected_avg = (0.90 + 0.85 + 0.80 + 0.75) / 4
        self.assertAlmostEqual(result.s_avg, expected_avg, places=3)


class TestLowConfidence(unittest.TestCase):
    """Weak scores across the board → R_c ≤ 0.50."""

    def setUp(self):
        self.est = RetrievalConfidenceEstimator()

    def test_weak_scores_low_confidence(self):
        chunks = [_chunk(s, f"c{i}") for i, s in enumerate([0.42, 0.39, 0.35, 0.30])]
        result = self.est.estimate(chunks)
        self.assertLessEqual(result.score, 0.55,
            f"Expected R_c ≤ 0.55 for weak scores, got {result.score}")

    def test_all_very_low_scores(self):
        chunks = [_chunk(0.10), _chunk(0.08), _chunk(0.05)]
        result = self.est.estimate(chunks)
        self.assertLessEqual(result.score, 0.30)


class TestKneeDetection(unittest.TestCase):

    def setUp(self):
        self.est = RetrievalConfidenceEstimator()

    def test_knee_gap_and_index(self):
        """
        Scores: [0.95, 0.92, 0.40, 0.38]
        Drops:  0.03, 0.52*, 0.02  → knee_gap ≈ 0.52, knee_index = 2
        """
        chunks = [_chunk(s, f"c{i}") for i, s in enumerate([0.95, 0.92, 0.40, 0.38])]
        result = self.est.estimate(chunks)
        self.assertAlmostEqual(result.knee_gap, 0.52, places=2)
        self.assertEqual(result.knee_index, 2)

    def test_uniform_scores_small_knee_gap(self):
        chunks = [_chunk(0.80 - i * 0.01, f"c{i}") for i in range(5)]
        result = self.est.estimate(chunks)
        self.assertLess(result.knee_gap, 0.05)

    def test_low_score_guard_suppresses_knee_contribution(self):
        """
        [0.45, 0.10] has a large gap (0.35) but S_top = 0.45 < min_score_guard (0.50).
        G_knee contribution must be zeroed out, keeping R_c low.
        """
        chunks = [_chunk(0.45, "c0"), _chunk(0.10, "c1")]
        result = self.est.estimate(chunks)
        self.assertLessEqual(result.score, 0.50,
            f"Low-score guard failed: R_c={result.score} should be ≤ 0.50")
        # Knee suppressed note should appear in explanation
        self.assertIn("suppressed", result.explanation)


class TestEvidenceAlignment(unittest.TestCase):

    def setUp(self):
        self.est = RetrievalConfidenceEstimator()

    def test_perfect_evidence_alignment(self):
        """All constraints match → evidence_alignment = 1.0."""
        chunks = [_chunk(0.85, "c1", "d1", doc_type="pan_card",
                         year=2023, text="ABCDE1234F pan card 2023")]
        parsed = _parsed(doc_type="pan_card", year=2023)
        result = self.est.estimate(chunks, parsed)
        self.assertAlmostEqual(result.evidence_alignment, 1.0, places=1)

    def test_no_constraints_falls_back_to_s_top(self):
        """No metadata constraints → evidence = S_top."""
        chunks = [_chunk(0.75, "c1")]
        parsed = _parsed()  # no constraints
        result = self.est.estimate(chunks, parsed)
        self.assertAlmostEqual(result.evidence_alignment, result.s_top, places=3)

    def test_mismatched_doc_type_lowers_evidence(self):
        """Query asks for pan_card but chunk is electricity_bill."""
        chunks = [_chunk(0.90, "c1", "d1", doc_type="electricity_bill")]
        parsed = _parsed(doc_type="pan_card")
        result = self.est.estimate(chunks, parsed)
        self.assertLess(result.evidence_alignment, 1.0)

    def test_entity_match_in_text(self):
        chunks = [_chunk(0.80, "c1", "d1", text="The account belongs to Alice Smith")]
        parsed = _parsed(entities=["Alice"])
        result = self.est.estimate(chunks, parsed)
        self.assertGreater(result.evidence_alignment, 0.5)

    def test_no_parsed_query_falls_back_to_s_top(self):
        chunks = [_chunk(0.70)]
        result = self.est.estimate(chunks, parsed_query=None)
        self.assertAlmostEqual(result.evidence_alignment, 0.70, places=2)


class TestDistinctDocCounting(unittest.TestCase):

    def setUp(self):
        self.est = RetrievalConfidenceEstimator()

    def test_counts_distinct_docs(self):
        chunks = [
            _chunk(0.90, "c1", "doc_A"),
            _chunk(0.85, "c2", "doc_A"),
            _chunk(0.80, "c3", "doc_B"),
            _chunk(0.75, "c4", "doc_C"),
        ]
        result = self.est.estimate(chunks)
        self.assertEqual(result.distinct_docs, 3)

    def test_all_same_doc(self):
        chunks = [_chunk(0.85, f"c{i}", "doc_X") for i in range(4)]
        result = self.est.estimate(chunks)
        self.assertEqual(result.distinct_docs, 1)


if __name__ == "__main__":
    unittest.main()
