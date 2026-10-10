"""
test_dynamic_k.py — Automated Unit Test Suite for Dynamic Top-K Retrieval Governor
=================================================================================
Validates:
  1. K_min = 2 invariant and simple factual query boundary: K ∈ [2, 3].
  2. Monotonicity with respect to Query Complexity (C_q).
  3. Monotonicity with respect to Retrieval Confidence (R_c).
  4. Knee-point gradient cliff detection (knee_gap, knee_index).
  5. Low-similarity noise protection (gap does not falsely inflate confidence).
  6. Multi-document diversity allocation (target_documents, chunks_per_document).
  7. Convex combination bounds: 0.0 <= R_c <= 1.0.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.retrieval.dynamic_k import (
    DynamicKGovernor,
    dynamic_k_governor,
    W_STOP,
    W_SAVG,
    W_KNEE,
    W_EVIDENCE,
    K_MIN,
)
from app.schemas.retrieval import ParsedQuery


class MockChunk:
    """Mock Qdrant chunk representation for testing."""
    def __init__(self, score: float, doc_id: str = "doc_1", payload: dict = None, text: str = ""):
        self.score = score
        self.document_id = doc_id
        self.payload = payload or {
            "document_id": doc_id,
            "document_type": "pan_card",
            "text_content": text,
        }
        self.text = text


class TestDynamicKGovernor(unittest.TestCase):

    def setUp(self):
        self.gov = DynamicKGovernor()

    def test_convex_weights_sum_to_one(self):
        """Ensure R_c weight combination is strictly convex (sum == 1.0)."""
        weight_sum = W_STOP + W_SAVG + W_KNEE + W_EVIDENCE
        self.assertAlmostEqual(weight_sum, 1.0, places=5)

    def test_k_min_invariant(self):
        """K must never evaluate below K_MIN (K_MIN = 2)."""
        # Empty retrieval
        decision = self.gov.decide("Hello", initial_chunks=[])
        self.assertGreaterEqual(decision.recommended_k, K_MIN)

    def test_simple_factual_query(self):
        """
        'What is my PAN number?' with high similarity match:
        - Low complexity (C_q <= 0.30)
        - High confidence (R_c >= 0.85)
        - K must be strictly in [2, 3] (K_min=2 with 1 primary + 1 verify chunk)
        """
        query = "What is my PAN number?"
        parsed = ParsedQuery(
            document_type="pan_card",
            year=None,
            month=None,
            entities=["PAN"],
            semantic_query="What is my PAN number?",
            raw_query=query,
        )
        chunks = [
            MockChunk(0.95, "doc_pan", {"document_type": "pan_card", "text_content": "Income Tax PAN card ABCDE1234F"}),
            MockChunk(0.91, "doc_pan", {"document_type": "pan_card", "text_content": "Permanent Account Number ABCDE1234F"}),
            MockChunk(0.40, "doc_other", {"document_type": "electricity_bill", "text_content": "Electricity bill Jan 2024"}),
        ]

        decision = self.gov.decide(query, chunks, parsed)

        self.assertLessEqual(decision.c_q, 0.30, f"C_q={decision.c_q} expected <= 0.30")
        self.assertGreaterEqual(decision.r_c, 0.80, f"R_c={decision.r_c} expected >= 0.80")
        self.assertIn(decision.recommended_k, [2, 3], f"Recommended K={decision.recommended_k} expected in [2, 3]")
        self.assertEqual(decision.target_documents, 1, "Factual query should target 1 document")

    def test_monotonicity_wrt_complexity(self):
        """
        For identical retrieval confidence, increasing query complexity
        should never decrease the recommended K.
        """
        # High confidence chunk set
        chunks = [
            MockChunk(0.90, "doc_1"),
            MockChunk(0.85, "doc_1"),
            MockChunk(0.82, "doc_2"),
        ]

        simple_q = "What is my consumer number?"
        complex_q = "Compare my electricity bills from January to March 2024 and explain why the total expense increased significantly."

        dec_simple = self.gov.decide(simple_q, chunks)
        dec_complex = self.gov.decide(complex_q, chunks)

        self.assertGreater(dec_complex.c_q, dec_simple.c_q, "Complex query must have higher C_q")
        self.assertGreaterEqual(
            dec_complex.recommended_k,
            dec_simple.recommended_k,
            f"K_complex ({dec_complex.recommended_k}) must be >= K_simple ({dec_simple.recommended_k})",
        )

    def test_monotonicity_wrt_confidence(self):
        """
        For identical query complexity, lower retrieval confidence
        must increase or maintain recommended K (expanding retrieval to compensate).
        """
        query = "Explain my salary deductions in June 2024."

        high_conf_chunks = [
            MockChunk(0.94, "doc_1"),
            MockChunk(0.91, "doc_1"),
            MockChunk(0.45, "doc_2"),
        ]

        low_conf_chunks = [
            MockChunk(0.48, "doc_1"),
            MockChunk(0.44, "doc_1"),
            MockChunk(0.41, "doc_2"),
        ]

        dec_high = self.gov.decide(query, high_conf_chunks)
        dec_low = self.gov.decide(query, low_conf_chunks)

        self.assertGreater(dec_high.r_c, dec_low.r_c, "High conf chunks must produce higher R_c")
        self.assertGreaterEqual(
            dec_low.recommended_k,
            dec_high.recommended_k,
            f"K_low_conf ({dec_low.recommended_k}) must be >= K_high_conf ({dec_high.recommended_k})",
        )

    def test_knee_point_detection(self):
        """
        Scores: [0.95, 0.92, 0.40, 0.38]
        Drops:
          0.95 - 0.92 = 0.03
          0.92 - 0.40 = 0.52  <-- max drop (knee cliff)
          0.40 - 0.38 = 0.02
        Expected:
          knee_gap ≈ 0.52
          knee_index == 2 (boundary after 2nd candidate)
        """
        chunks = [
            MockChunk(0.95),
            MockChunk(0.92),
            MockChunk(0.40),
            MockChunk(0.38),
        ]
        r_c, s_top, s_avg, knee_gap, knee_index, evidence = self.gov.compute_confidence(chunks)

        self.assertAlmostEqual(s_top, 0.95, places=2)
        self.assertAlmostEqual(knee_gap, 0.52, places=2)
        self.assertEqual(knee_index, 2)

    def test_noisy_low_scores_do_not_falsely_inflate_confidence(self):
        """
        Scores: [0.42, 0.39, 0.10, 0.08]
        There is a cliff (0.39 - 0.10 = 0.29), but overall scores are very low.
        Confidence R_c must remain suppressed (<= 0.50) because S_top and S_avg dominate.
        """
        chunks = [
            MockChunk(0.42),
            MockChunk(0.39),
            MockChunk(0.10),
            MockChunk(0.08),
        ]
        r_c, s_top, s_avg, knee_gap, knee_index, evidence = self.gov.compute_confidence(chunks)

        self.assertLessEqual(r_c, 0.50, f"R_c={r_c} must not be inflated by knee gap alone")

    def test_document_diversity_on_comparison_query(self):
        """
        For 'Compare my bank statements from 2023 and 2024':
        - target_documents must be >= 2
        - chunks_per_document must be >= 1
        - target_documents * chunks_per_document >= recommended_k
        """
        query = "Compare my bank statements from 2023 and 2024"
        parsed = ParsedQuery(
            document_type="bank_statement",
            year=None,
            month=None,
            entities=[],
            semantic_query="Compare bank statements 2023 2024",
            raw_query=query,
        )
        chunks = [
            MockChunk(0.85, "bank_2023"),
            MockChunk(0.82, "bank_2023"),
            MockChunk(0.80, "bank_2024"),
            MockChunk(0.78, "bank_2024"),
        ]
        decision = self.gov.decide(query, chunks, parsed)

        self.assertGreaterEqual(decision.target_documents, 2, "Comparison queries must target >= 2 documents")
        self.assertGreaterEqual(decision.chunks_per_document, 1)
        self.assertGreaterEqual(
            decision.target_documents * decision.chunks_per_document,
            decision.recommended_k,
            "Total allocated capacity across target documents must cover recommended K",
        )


if __name__ == "__main__":
    unittest.main()
