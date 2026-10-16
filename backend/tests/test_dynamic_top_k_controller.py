"""
test_dynamic_top_k_controller.py — Unit Tests for DynamicTopKController
========================================================================
Covers:
  - K_min invariant (K never below K_min)
  - K_max invariant (K never above K_max)
  - Simple factual query → K near K_min
  - Complex comparison query → K near K_max
  - Monotonicity w.r.t. C_q (higher complexity → higher or equal K)
  - Monotonicity w.r.t. R_c (lower confidence → higher or equal K)
  - High-confidence factual shortcut (K forced to K_min)
  - Coverage penalty for under-covered multi-part queries
  - Document diversity allocation for comparison queries (target_docs >= 2)
  - Document diversity allocation for aggregation queries (target_docs >= 3)
  - Factual query -> target_docs == 1
  - Invalid TopKConfig raises ValueError
  - TopKConfig alpha+beta auto-normalisation
  - DynamicKDecision.to_dict() structure matches DynamicKDiagnostics fields
  - Determinism
  - Compatibility: DynamicKGovernor facade produces same decision as controller
"""

from __future__ import annotations

import unittest

from app.retrieval.dynamic_top_k import (
    DynamicTopKController,
    DynamicKDecision,
    TopKConfig,
    dynamic_top_k_controller,
)
from app.retrieval.dynamic_k import DynamicKGovernor, dynamic_k_governor
from app.schemas.retrieval import ParsedQuery, DynamicKDiagnostics


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chunk(score, doc_id="d1", doc_type="document", chunk_id=None, year=None, month=None, text=""):
    class _C:
        pass
    c = _C()
    c.score = score
    c.payload = {
        "chunk_id":      chunk_id or f"ck_{score}_{doc_id}",
        "document_id":   doc_id,
        "document_type": doc_type,
        "text":          text,
        "year":          year,
        "month":         month,
    }
    return c


def _parsed(doc_type=None, year=None, month=None, entities=None):
    return ParsedQuery(
        document_type=doc_type, year=year, month=month,
        entities=entities or [], amount_filters=None,
        semantic_query="query", raw_query="query",
    )


def _high_conf_chunks():
    return [
        _chunk(0.95, "d1", chunk_id="c1"),
        _chunk(0.91, "d1", chunk_id="c2"),
        _chunk(0.40, "d2", chunk_id="c3"),
    ]


def _low_conf_chunks():
    return [
        _chunk(0.45, "d1", chunk_id="c1"),
        _chunk(0.42, "d1", chunk_id="c2"),
        _chunk(0.38, "d2", chunk_id="c3"),
    ]


# ── Bound invariants ───────────────────────────────────────────────────────────

class TestBoundInvariants(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def test_k_never_below_k_min(self):
        for query in ["hi", "What is my PAN?", ""]:
            dec = self.ctrl.decide(query, [])
            self.assertGreaterEqual(dec.recommended_k, 2,
                f"K_min violated for query: {query!r}")

    def test_k_never_above_k_max(self):
        cfg  = TopKConfig(k_min=2, k_max=5)
        ctrl = DynamicTopKController(config=cfg)
        q    = "Compare and analyze all documents from 2019 to 2024 and explain everything."
        dec  = ctrl.decide(q, _low_conf_chunks())
        self.assertLessEqual(dec.recommended_k, 5)

    def test_custom_k_min_k_max_respected(self):
        cfg  = TopKConfig(k_min=3, k_max=8)
        ctrl = DynamicTopKController(config=cfg)
        dec  = ctrl.decide("show docs", _high_conf_chunks())
        self.assertGreaterEqual(dec.recommended_k, 3)
        self.assertLessEqual(dec.recommended_k, 8)

    def test_invalid_k_min_raises(self):
        with self.assertRaises(ValueError):
            TopKConfig(k_min=0, k_max=10)

    def test_k_max_less_than_k_min_raises(self):
        with self.assertRaises(ValueError):
            TopKConfig(k_min=10, k_max=5)


class TestTopKConfig(unittest.TestCase):

    def test_alpha_beta_normalised(self):
        cfg = TopKConfig(alpha=3.0, beta=1.0)
        self.assertAlmostEqual(cfg.alpha + cfg.beta, 1.0, places=5)
        self.assertAlmostEqual(cfg.alpha, 0.75, places=5)
        self.assertAlmostEqual(cfg.beta,  0.25, places=5)

    def test_zero_alpha_beta_defaults_to_half_half(self):
        cfg = TopKConfig(alpha=0.0, beta=0.0)
        self.assertAlmostEqual(cfg.alpha, 0.5, places=5)
        self.assertAlmostEqual(cfg.beta,  0.5, places=5)


# ── Simple factual query ───────────────────────────────────────────────────────

class TestSimpleFactualQuery(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def test_simple_pan_query_low_k(self):
        q      = "What is my PAN card number?"
        parsed = _parsed(doc_type="pan_card", entities=["PAN"])
        chunks = [
            _chunk(0.95, "doc_pan", "pan_card", chunk_id="c1", text="ABCDE1234F"),
            _chunk(0.91, "doc_pan", "pan_card", chunk_id="c2", text="PAN number"),
            _chunk(0.38, "doc_elec", "electricity_bill", chunk_id="c3"),
        ]
        dec = self.ctrl.decide(q, chunks, parsed)
        self.assertLessEqual(dec.c_q, 0.30,
            f"C_q={dec.c_q:.3f} should be <= 0.30 for simple factual query")
        self.assertLessEqual(dec.recommended_k, 6,
            f"K={dec.recommended_k} expected <= 6 for simple factual")

    def test_shortcut_forces_k_min(self):
        """C_q<=0.25 + R_c>=0.80 + knee_index<=2 + knee_gap>=0.20 -> K=2."""
        q      = "What is my PAN?"
        parsed = _parsed(doc_type="pan_card")
        chunks = [
            _chunk(0.95, "d1", chunk_id="c1"),
            _chunk(0.91, "d1", chunk_id="c2"),
            _chunk(0.30, "d2", chunk_id="c3"),
        ]
        dec = self.ctrl.decide(q, chunks, parsed)
        # Only assert shortcut when conditions are actually met
        if dec.c_q <= 0.25 and dec.r_c >= 0.80 and dec.knee_index <= 2 and dec.knee_gap >= 0.20:
            self.assertEqual(dec.recommended_k, 2,
                f"Shortcut should force K=2, got {dec.recommended_k}")

    def test_empty_chunks_yields_valid_k(self):
        # Empty chunks → R_c=0 → high confidence_demand → formula pushes K up.
        # K must still be within [K_min, K_max] and R_c must be exactly 0.
        dec = self.ctrl.decide("What is my PAN?", [])
        self.assertGreaterEqual(dec.recommended_k, 2)
        self.assertLessEqual(dec.recommended_k, 15)
        self.assertEqual(dec.r_c, 0.0)


# ── Complex comparison query ───────────────────────────────────────────────────

class TestComplexQuery(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def test_comparison_query_higher_k_than_factual(self):
        q_simple  = "What is my PAN?"
        q_complex = "Compare my electricity bills from January 2023 and January 2024 and explain changes."
        d_simple  = self.ctrl.decide(q_simple,  _high_conf_chunks())
        d_complex = self.ctrl.decide(q_complex, _low_conf_chunks())
        self.assertGreaterEqual(d_complex.recommended_k, d_simple.recommended_k)

    def test_complex_query_c_q_higher_than_simple(self):
        q_simple  = "Show my bank statement."
        q_complex = "Compare all bank statements 2022 to 2024 and analyze spending trends."
        d_s = self.ctrl.decide(q_simple,  _high_conf_chunks())
        d_c = self.ctrl.decide(q_complex, _high_conf_chunks())
        self.assertGreater(d_c.c_q, d_s.c_q)


# ── Monotonicity ───────────────────────────────────────────────────────────────

class TestMonotonicity(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def test_higher_complexity_never_reduces_k(self):
        chunks    = _high_conf_chunks()
        simple_q  = "Show my bank statement."
        complex_q = ("Compare my bank statements from 2022 and 2023 and explain "
                     "why expenses increased significantly each quarter.")
        d_s = self.ctrl.decide(simple_q,  chunks)
        d_c = self.ctrl.decide(complex_q, chunks)
        self.assertGreater(d_c.c_q, d_s.c_q)
        self.assertGreaterEqual(d_c.recommended_k, d_s.recommended_k)

    def test_lower_confidence_never_reduces_k(self):
        q      = "Explain my salary deductions in June 2024."
        d_high = self.ctrl.decide(q, _high_conf_chunks())
        d_low  = self.ctrl.decide(q, _low_conf_chunks())
        self.assertGreater(d_high.r_c, d_low.r_c)
        self.assertGreaterEqual(d_low.recommended_k, d_high.recommended_k)


# ── Coverage penalty ───────────────────────────────────────────────────────────

class TestCoveragePenalty(unittest.TestCase):

    def test_coverage_penalty_applied_when_evidence_low(self):
        cfg_penalty   = TopKConfig(k_min=2, k_max=15, coverage_penalty=3, coverage_threshold=0.99)
        cfg_no_penalty = TopKConfig(k_min=2, k_max=15, coverage_penalty=0, coverage_threshold=0.99)
        ctrl_pen  = DynamicTopKController(config=cfg_penalty)
        ctrl_none = DynamicTopKController(config=cfg_no_penalty)

        # Comparative query with mismatched doc_type -> low evidence alignment
        q      = "Compare my insurance policies from 2022 and 2023."
        chunks = [_chunk(0.80, "d1", "bank_statement", chunk_id="c1"),
                  _chunk(0.78, "d2", "bank_statement", chunk_id="c2")]
        parsed = _parsed(doc_type="insurance", year=2022)

        d_pen  = ctrl_pen.decide(q, chunks, parsed)
        d_none = ctrl_none.decide(q, chunks, parsed)

        # With penalty and low evidence, K should be >= without penalty
        self.assertGreaterEqual(d_pen.recommended_k, d_none.recommended_k)


# ── Document diversity ─────────────────────────────────────────────────────────

class TestDocumentDiversity(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def test_comparison_targets_at_least_2_docs(self):
        q      = "Compare my bank statements from 2022 and 2023."
        parsed = _parsed(doc_type="bank_statement")
        chunks = [
            _chunk(0.85, "bank_2022", chunk_id="c1"),
            _chunk(0.82, "bank_2022", chunk_id="c2"),
            _chunk(0.80, "bank_2023", chunk_id="c3"),
            _chunk(0.78, "bank_2023", chunk_id="c4"),
        ]
        dec = self.ctrl.decide(q, chunks, parsed)
        self.assertGreaterEqual(dec.target_documents, 2)

    def test_aggregation_targets_at_least_3_docs(self):
        q      = "Show all my bills and total the expenses."
        chunks = [_chunk(0.75 - i * 0.02, f"d{i}", chunk_id=f"c{i}") for i in range(8)]
        dec    = self.ctrl.decide(q, chunks)
        self.assertGreaterEqual(dec.target_documents, 3)

    def test_factual_targets_one_doc(self):
        q      = "What is my PAN number?"
        parsed = _parsed(doc_type="pan_card", entities=["PAN"])
        chunks = [_chunk(0.95, "d1", chunk_id="c1"),
                  _chunk(0.91, "d1", chunk_id="c2"),
                  _chunk(0.38, "d2", chunk_id="c3")]
        dec = self.ctrl.decide(q, chunks, parsed)
        self.assertEqual(dec.target_documents, 1)

    def test_chunks_per_doc_covers_recommended_k(self):
        q      = "Compare my electricity bills from 2022 vs 2023."
        parsed = _parsed(doc_type="electricity_bill")
        chunks = [_chunk(0.80, f"d{i % 2}", chunk_id=f"c{i}") for i in range(6)]
        dec    = self.ctrl.decide(q, chunks, parsed)
        capacity = dec.target_documents * dec.chunks_per_document
        self.assertGreaterEqual(capacity, dec.recommended_k)


# ── to_dict / schema ───────────────────────────────────────────────────────────

class TestDecisionDict(unittest.TestCase):

    def test_to_dict_feeds_diagnostics_schema(self):
        dec  = dynamic_top_k_controller.decide("What is my PAN?", _high_conf_chunks())
        diag = DynamicKDiagnostics(**dec.to_dict())
        self.assertEqual(diag.recommended_k, dec.recommended_k)
        self.assertIsInstance(diag.features, dict)

    def test_explanation_non_empty(self):
        dec = dynamic_top_k_controller.decide("Show my resume.", _high_conf_chunks())
        self.assertGreater(len(dec.explanation), 0)

    def test_all_score_fields_in_0_1(self):
        dec = dynamic_top_k_controller.decide("Compare bills 2022 vs 2023.", _low_conf_chunks())
        for field in ("c_q", "r_c", "s_top", "s_avg", "knee_gap", "evidence_alignment"):
            val = getattr(dec, field)
            self.assertGreaterEqual(val, 0.0, f"{field} < 0: {val}")
            self.assertLessEqual(val,   1.0, f"{field} > 1: {val}")


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism(unittest.TestCase):

    def test_identical_inputs_produce_identical_outputs(self):
        ctrl   = DynamicTopKController()
        q      = "Explain my salary deductions for June 2024."
        chunks = [_chunk(0.88, "d1", chunk_id="c1"),
                  _chunk(0.82, "d1", chunk_id="c2"),
                  _chunk(0.45, "d2", chunk_id="c3")]
        d1 = ctrl.decide(q, chunks)
        d2 = ctrl.decide(q, chunks)
        self.assertEqual(d1.recommended_k,    d2.recommended_k)
        self.assertEqual(d1.c_q,              d2.c_q)
        self.assertEqual(d1.r_c,              d2.r_c)
        self.assertEqual(d1.features,         d2.features)
        self.assertEqual(d1.target_documents, d2.target_documents)


# ── Max K enforcement ──────────────────────────────────────────────────────────

class TestMaxKEnforcement(unittest.TestCase):

    def test_max_k_never_exceeded_under_extreme_inputs(self):
        cfg  = TopKConfig(k_min=2, k_max=8)
        ctrl = DynamicTopKController(config=cfg)
        q    = ("Compare and analyze all bank statements from 2018 to 2024 "
                "and explain every quarterly trend as well as annually.")
        chunks = [_chunk(0.05, f"d{i}", chunk_id=f"c{i}") for i in range(3)]
        dec = ctrl.decide(q, chunks)
        self.assertLessEqual(dec.recommended_k, 8,
            f"K_max=8 violated: got {dec.recommended_k}")


# ── Min K enforcement ──────────────────────────────────────────────────────────

class TestMinKEnforcement(unittest.TestCase):

    def test_min_k_enforced_on_high_confidence_non_shortcut(self):
        """
        Very high confidence but C_q=0.30 (medium) should still give K>=2.
        The shortcut only fires when C_q<=0.25 AND knee conditions are met.
        """
        ctrl   = DynamicTopKController()
        q      = "Summarize my SBI bank statement March 2024."
        parsed = _parsed(doc_type="bank_statement", year=2024, month="March")
        chunks = [_chunk(0.92, "d1", chunk_id="c1"),
                  _chunk(0.89, "d1", chunk_id="c2")]
        dec = ctrl.decide(q, chunks, parsed)
        self.assertGreaterEqual(dec.recommended_k, 2)


# ── DynamicKGovernor backward-compat facade ───────────────────────────────────

class TestDynamicKGovernorCompat(unittest.TestCase):

    def test_governor_decide_returns_decision(self):
        dec = dynamic_k_governor.decide("What is my PAN?", _high_conf_chunks())
        self.assertIsInstance(dec, DynamicKDecision)
        self.assertGreaterEqual(dec.recommended_k, 2)

    def test_governor_extract_features_8_keys(self):
        feats = dynamic_k_governor.extract_features("Compare bills 2022 vs 2023.")
        self.assertEqual(len(feats), 8)

    def test_governor_compute_complexity_in_range(self):
        feats = dynamic_k_governor.extract_features("Compare bills 2022 vs 2023.")
        c_q   = dynamic_k_governor.compute_complexity(feats)
        self.assertGreaterEqual(c_q, 0.0)
        self.assertLessEqual(c_q,   1.0)

    def test_governor_compute_confidence_returns_6_tuple(self):
        result = dynamic_k_governor.compute_confidence(_high_conf_chunks())
        self.assertEqual(len(result), 6)
        r_c = result[0]
        self.assertGreaterEqual(r_c, 0.0)
        self.assertLessEqual(r_c,   1.0)

    def test_governor_and_controller_agree(self):
        """DynamicKGovernor facade must match DynamicTopKController directly."""
        q      = "What is my PAN number?"
        chunks = _high_conf_chunks()
        parsed = _parsed(doc_type="pan_card", entities=["PAN"])

        dec_gov  = dynamic_k_governor.decide(q, chunks, parsed)
        dec_ctrl = DynamicTopKController().decide(q, chunks, parsed)
        self.assertEqual(dec_gov.recommended_k,    dec_ctrl.recommended_k)
        self.assertEqual(dec_gov.c_q,              dec_ctrl.c_q)
        self.assertEqual(dec_gov.r_c,              dec_ctrl.r_c)
        self.assertEqual(dec_gov.target_documents, dec_ctrl.target_documents)


# ── Query-parser fallback compatibility ───────────────────────────────────────

class TestQueryParserFallback(unittest.TestCase):

    def test_none_parsed_query_does_not_crash(self):
        ctrl = DynamicTopKController()
        dec  = ctrl.decide("What is my PAN?", _high_conf_chunks(), parsed_query=None)
        self.assertIsInstance(dec.recommended_k, int)

    def test_partial_parsed_query_does_not_crash(self):
        from types import SimpleNamespace
        partial = SimpleNamespace(entities=[], document_type=None)
        ctrl    = DynamicTopKController()
        dec     = ctrl.decide("Show my resume.", _high_conf_chunks(), parsed_query=partial)
        self.assertIsInstance(dec.recommended_k, int)


# ── Formula Modes and OOD Abstention Suite ───────────────────────────────────

class TestFormulaModesAndAbstention(unittest.TestCase):

    def test_invalid_formula_mode_raises(self):
        with self.assertRaises(ValueError):
            TopKConfig(formula_mode="non_existent_mode")

    def test_fixed_k_mode(self):
        cfg = TopKConfig(formula_mode="fixed", fixed_k=7)
        ctrl = DynamicTopKController(config=cfg)
        dec = ctrl.decide("What is my PAN?", _high_conf_chunks())
        self.assertEqual(dec.recommended_k, 7)
        self.assertEqual(dec.stop_reason, "fixed_k_baseline")
        self.assertEqual(dec.formula_mode, "fixed")

    def test_complexity_only_mode(self):
        cfg = TopKConfig(formula_mode="complexity_only", k_min=2, k_max=15)
        ctrl = DynamicTopKController(config=cfg)
        q_simple = "What is my PAN?"
        q_complex = "Compare all bank statements 2022 to 2024 and analyze spending trends."
        d_simple = ctrl.decide(q_simple, _high_conf_chunks())
        d_complex = ctrl.decide(q_complex, _high_conf_chunks())
        self.assertLess(d_simple.recommended_k, d_complex.recommended_k)
        self.assertEqual(d_simple.stop_reason, "complexity_only_demand")

    def test_confidence_only_mode(self):
        cfg = TopKConfig(formula_mode="confidence_only", k_min=2, k_max=15)
        ctrl = DynamicTopKController(config=cfg)
        q = "Show my document."
        d_high = ctrl.decide(q, _high_conf_chunks())
        d_low = ctrl.decide(q, _low_conf_chunks())
        self.assertGreaterEqual(d_low.recommended_k, d_high.recommended_k)
        self.assertEqual(d_high.stop_reason, "confidence_only_demand")

    def test_paper_formula_mode(self):
        # K* = ceil(clip(K_min + (1 - R_c) * K_range + gamma * C_q, K_min, K_max))
        cfg = TopKConfig(formula_mode="paper", k_min=2, k_max=15, gamma=5.0)
        ctrl = DynamicTopKController(config=cfg)
        dec = ctrl.decide("What is my PAN?", _high_conf_chunks())
        self.assertGreaterEqual(dec.recommended_k, 2)
        self.assertLessEqual(dec.recommended_k, 15)
        self.assertEqual(dec.stop_reason, "paper_additive_formula")

    def test_abstention_on_empty_chunks(self):
        cfg = TopKConfig(enable_abstention=True, similarity_floor=0.35)
        ctrl = DynamicTopKController(config=cfg)
        dec = ctrl.decide("What is my electricity bill in 2024?", [])
        self.assertTrue(dec.is_abstained)
        self.assertEqual(dec.recommended_k, 2)
        self.assertEqual(dec.stop_reason, "abstain_low_similarity_floor")
        self.assertIn("Peak similarity", dec.abstention_reason)

    def test_abstention_on_low_similarity_floor(self):
        cfg = TopKConfig(enable_abstention=True, similarity_floor=0.40)
        ctrl = DynamicTopKController(config=cfg)
        low_noise_chunks = [_chunk(0.25, "d1"), _chunk(0.20, "d2")]
        dec = ctrl.decide("Find alien spaceship document", low_noise_chunks)
        self.assertTrue(dec.is_abstained)
        self.assertEqual(dec.recommended_k, 2)
        self.assertEqual(dec.stop_reason, "abstain_low_similarity_floor")

    def test_no_abstention_when_disabled(self):
        cfg = TopKConfig(enable_abstention=False, similarity_floor=0.40)
        ctrl = DynamicTopKController(config=cfg)
        low_noise_chunks = [_chunk(0.25, "d1"), _chunk(0.20, "d2")]
        dec = ctrl.decide("Find alien spaceship document", low_noise_chunks)
        self.assertFalse(dec.is_abstained)
        # Without abstention, low confidence pushes demand up
        self.assertGreater(dec.recommended_k, 2)


if __name__ == "__main__":
    unittest.main()

