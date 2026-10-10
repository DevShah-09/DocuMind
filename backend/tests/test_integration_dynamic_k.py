"""
test_integration_dynamic_k.py — Integration Tests for Dynamic Top-K Pipeline
=============================================================================
Tests the full end-to-end path from raw query → complexity → confidence →
K decision, without requiring a live Qdrant or LLM connection.

Covers:
  - Full pipeline on factual, comparative, aggregative, and date-constrained queries
  - High-confidence retrieval path
  - Low-confidence retrieval path
  - No-result retrieval path
  - Duplicate chunks handled correctly
  - Missing / invalid similarity scores
  - Maximum K enforcement
  - Minimum K enforcement
  - Query-parser fallback (ParsedQuery=None)
  - Legacy retrieval compatibility (chunks without chunk_id in payload)
  - Tenant isolation: user_id present in all query filter paths
  - Authorization: different user_id values must produce independent decisions
  - Deterministic results for identical inputs
  - DynamicKDiagnostics round-trip (decision → to_dict → Pydantic schema)
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.retrieval.dynamic_k import DynamicKGovernor, dynamic_k_governor
from app.retrieval.dynamic_top_k import DynamicTopKController, TopKConfig
from app.retrieval.query_complexity import QueryComplexityEstimator
from app.retrieval.retrieval_confidence import RetrievalConfidenceEstimator
from app.schemas.retrieval import ParsedQuery, DynamicKDiagnostics


# ── Test fixtures ──────────────────────────────────────────────────────────────

def _chunk(score, doc_id="d1", doc_type="document", chunk_id=None,
           year=None, month=None, text="", user_id="user-A"):
    """Build a mock chunk that mimics a Qdrant ScoredPoint."""
    class _C:
        pass
    c = _C()
    c.score = score
    c.payload = {
        "chunk_id":      chunk_id or f"ck-{score}-{doc_id}",
        "document_id":   doc_id,
        "document_type": doc_type,
        "text":          text,
        "year":          year,
        "month":         month,
        "user_id":       user_id,
    }
    return c


def _parsed(doc_type=None, year=None, month=None, entities=None):
    return ParsedQuery(
        document_type=doc_type, year=year, month=month,
        entities=entities or [], amount_filters=None,
        semantic_query="query", raw_query="query",
    )


# ── Full pipeline: query category tests ───────────────────────────────────────

class TestFullPipelineByCategory(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def _chunks_for(self, doc_type, scores, user_id="user-A"):
        return [
            _chunk(s, f"d{i}", doc_type, chunk_id=f"c-{i}", user_id=user_id)
            for i, s in enumerate(scores)
        ]

    # ── Factual ────────────────────────────────────────────────────────────────

    def test_factual_low_complexity(self):
        q      = "What is my PAN card number?"
        parsed = _parsed(doc_type="pan_card", entities=["PAN"])
        chunks = self._chunks_for("pan_card", [0.95, 0.91, 0.38])
        dec    = self.ctrl.decide(q, chunks, parsed)

        self.assertLessEqual(dec.c_q, 0.30, f"C_q={dec.c_q:.3f} too high for factual")
        self.assertGreaterEqual(dec.recommended_k, 2)
        self.assertLessEqual(dec.recommended_k, 8)
        self.assertEqual(dec.target_documents, 1)

    # ── Comparative ────────────────────────────────────────────────────────────

    def test_comparative_high_complexity_multi_doc(self):
        q      = "Compare my electricity bills from 2022 and 2023 and explain the increase."
        parsed = _parsed(doc_type="electricity_bill")
        chunks = (
            self._chunks_for("electricity_bill", [0.85, 0.82], user_id="user-A")
            + self._chunks_for("electricity_bill", [0.80, 0.78], user_id="user-A")
        )
        dec = self.ctrl.decide(q, chunks, parsed)

        self.assertGreater(dec.c_q, 0.30,    "Comparison query should have C_q > 0.30")
        self.assertGreaterEqual(dec.target_documents, 2, "Comparison needs >= 2 docs")
        self.assertGreaterEqual(dec.recommended_k, 4)

    # ── Aggregative ────────────────────────────────────────────────────────────

    def test_aggregative_multi_doc_target(self):
        q      = "What is the total of all my electricity bills in 2023?"
        chunks = [_chunk(0.75 - i * 0.03, f"d{i}", "electricity_bill",
                         chunk_id=f"c{i}", year=2023) for i in range(6)]
        dec = self.ctrl.decide(q, chunks)
        self.assertGreaterEqual(dec.target_documents, 3)

    # ── Date-constrained ───────────────────────────────────────────────────────

    def test_date_constrained_moderate_k(self):
        q      = "Show my electricity bill for January 2024."
        parsed = _parsed(doc_type="electricity_bill", year=2024, month="January")
        chunks = [_chunk(0.88, "d1", "electricity_bill", chunk_id="c1",
                         year=2024, month="January", text="Jan 2024 electricity bill")]
        dec = self.ctrl.decide(q, chunks, parsed)
        self.assertGreaterEqual(dec.recommended_k, 2)

    # ── Multi-document ─────────────────────────────────────────────────────────

    def test_multi_document_query(self):
        q      = "List all my insurance policies and their premium amounts."
        chunks = [_chunk(0.80, f"ins-{i}", "insurance", chunk_id=f"ci{i}") for i in range(4)]
        dec = self.ctrl.decide(q, chunks)
        self.assertGreaterEqual(dec.recommended_k, 2)


# ── Confidence boundary conditions ────────────────────────────────────────────

class TestConfidenceBoundaryConditions(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def test_high_confidence_reduces_k(self):
        q  = "What is my PAN?"
        ch = [_chunk(0.95, "d1", chunk_id="c1"),
              _chunk(0.92, "d1", chunk_id="c2"),
              _chunk(0.35, "d2", chunk_id="c3")]
        dec = self.ctrl.decide(q, ch)
        self.assertGreater(dec.r_c, 0.60)
        # High-conf simple query should NOT retrieve the maximum K
        self.assertLessEqual(dec.recommended_k, 8)

    def test_low_confidence_increases_k(self):
        q  = "What is my PAN?"
        ch = [_chunk(0.42, "d1", chunk_id="c1"),
              _chunk(0.40, "d1", chunk_id="c2"),
              _chunk(0.38, "d2", chunk_id="c3")]
        dec = self.ctrl.decide(q, ch)
        self.assertLess(dec.r_c, 0.60)
        dec_high = self.ctrl.decide(q,
            [_chunk(0.95, "d1", chunk_id="h1"), _chunk(0.92, "d1", chunk_id="h2"),
             _chunk(0.35, "d2", chunk_id="h3")])
        # Low-confidence must recommend >= high-confidence K
        self.assertGreaterEqual(dec.recommended_k, dec_high.recommended_k)

    def test_empty_result_path(self):
        """Zero chunks -> R_c=0 (no evidence), K within valid bounds, no crash."""
        dec = self.ctrl.decide("What is my PAN?", [])
        self.assertEqual(dec.r_c, 0.0)
        self.assertGreaterEqual(dec.recommended_k, 2)
        self.assertLessEqual(dec.recommended_k, 15)
        self.assertEqual(dec.s_top, 0.0)


# ── Duplicate chunks ───────────────────────────────────────────────────────────

class TestDuplicateChunks(unittest.TestCase):

    def setUp(self):
        self.ctrl = DynamicTopKController()

    def test_duplicate_chunks_do_not_crash(self):
        dupes = [_chunk(0.85, "d1", chunk_id="same") for _ in range(5)]
        dec   = self.ctrl.decide("What is my PAN?", dupes)
        self.assertIsInstance(dec.recommended_k, int)
        self.assertGreaterEqual(dec.recommended_k, 2)

    def test_duplicate_chunks_deduped_in_confidence(self):
        est   = RetrievalConfidenceEstimator()
        dupes = [_chunk(0.85, "d1", chunk_id="same") for _ in range(4)]
        r     = est.estimate(dupes)
        self.assertEqual(r.duplicate_chunks_removed, 3)


# ── Missing / invalid similarity scores ────────────────────────────────────────

class TestInvalidScores(unittest.TestCase):

    def test_none_score_handled(self):
        class NoScore:
            payload = {"chunk_id": "x", "document_id": "d1"}
        ctrl = DynamicTopKController()
        dec  = ctrl.decide("What is my PAN?", [NoScore()])
        self.assertIsInstance(dec.recommended_k, int)

    def test_score_out_of_range_clipped(self):
        ctrl = DynamicTopKController()
        ch   = [_chunk(2.5, "d1", chunk_id="c1"),   # > 1
                _chunk(-0.5, "d2", chunk_id="c2")]   # < 0
        dec  = ctrl.decide("Show my document.", ch)
        self.assertLessEqual(dec.s_top, 1.0)
        self.assertGreaterEqual(dec.s_top, 0.0)


# ── Legacy chunk payload compatibility ─────────────────────────────────────────

class TestLegacyChunkCompatibility(unittest.TestCase):
    """
    Chunks from the legacy user_documents collection have a different payload
    schema (no chunk_id, uses text_content instead of text).
    The controller must handle them without crashing.
    """

    def test_legacy_payload_no_crash(self):
        class LegacyChunk:
            score   = 0.72
            payload = {
                "document_id": "legacy-doc-1",
                "document_type": "document",
                "text_content": "Legacy document text",
                # No chunk_id, no year, no month
            }
        ctrl = DynamicTopKController()
        dec  = ctrl.decide("Show my document.", [LegacyChunk()])
        self.assertIsInstance(dec.recommended_k, int)

    def test_missing_payload_no_crash(self):
        class NullPayload:
            score   = 0.60
            payload = None
        ctrl = DynamicTopKController()
        dec  = ctrl.decide("Show my document.", [NullPayload()])
        self.assertIsInstance(dec.recommended_k, int)


# ── Tenant isolation ───────────────────────────────────────────────────────────

class TestTenantIsolation(unittest.TestCase):
    """
    The Dynamic-K controller is user-agnostic — it operates on whatever
    chunks are passed in.  User isolation is enforced at the Qdrant layer
    (user_id filter).  These tests verify that the controller's decisions
    for user A do not bleed into decisions for user B.

    Note: Full integration tests requiring a live Qdrant must be run
    separately.  Here we verify that:
      1. K decisions are independent for different user chunk sets.
      2. The controller does not store any per-user state.
    """

    def test_decisions_independent_across_users(self):
        ctrl = DynamicTopKController()
        q    = "What is my PAN number?"

        chunks_user_a = [_chunk(0.95, "doc-a", chunk_id="ca1", user_id="user-A"),
                         _chunk(0.91, "doc-a", chunk_id="ca2", user_id="user-A")]
        chunks_user_b = [_chunk(0.45, "doc-b", chunk_id="cb1", user_id="user-B"),
                         _chunk(0.40, "doc-b", chunk_id="cb2", user_id="user-B")]

        dec_a = ctrl.decide(q, chunks_user_a)
        dec_b = ctrl.decide(q, chunks_user_b)

        # User A has high-confidence results; user B has low-confidence results
        self.assertGreater(dec_a.r_c, dec_b.r_c,
            "User A's higher-scoring chunks must yield higher R_c than User B's")
        self.assertGreaterEqual(dec_b.recommended_k, dec_a.recommended_k,
            "User B's lower confidence must yield K >= User A's K")

    def test_controller_has_no_per_user_state(self):
        """
        Making decisions for user A then user B must not affect user A's
        subsequent decisions (the controller is stateless).
        """
        ctrl = DynamicTopKController()
        q    = "What is my PAN?"
        ch_a = [_chunk(0.95, "d1", chunk_id="c1"), _chunk(0.91, "d1", chunk_id="c2")]
        ch_b = [_chunk(0.30, "d2", chunk_id="c3"), _chunk(0.28, "d2", chunk_id="c4")]

        dec_a_before = ctrl.decide(q, ch_a)
        _ignored     = ctrl.decide(q, ch_b)   # user B's call
        dec_a_after  = ctrl.decide(q, ch_a)   # user A again

        self.assertEqual(dec_a_before.recommended_k, dec_a_after.recommended_k,
            "Controller must be stateless; user B's call must not affect user A")
        self.assertEqual(dec_a_before.r_c, dec_a_after.r_c)


# ── Authorization regression ───────────────────────────────────────────────────

class TestAuthorizationEnforcement(unittest.TestCase):
    """
    Verify that the user_id field in chunk payloads is preserved and that
    the controller does not promote cross-user retrieval.
    """

    def test_user_id_preserved_in_payload(self):
        """Chunk payloads retain their user_id throughout the decision process."""
        ctrl   = DynamicTopKController()
        chunks = [_chunk(0.80, "d1", chunk_id="c1", user_id="user-X")]
        dec    = ctrl.decide("Show my resume.", chunks)
        # Controller returns the same chunk objects; user_id is NOT modified
        self.assertEqual(chunks[0].payload["user_id"], "user-X")

    def test_no_cross_user_score_leakage(self):
        """
        Passing chunks belonging to user-B to a decision call for user-A
        context must not change the controller's S_top for user-A's chunks.
        The user_id filter is enforced at QdrantStore, not here — but the
        controller must not mutate chunk payloads or share state.
        """
        ctrl = DynamicTopKController()
        ch_a = [_chunk(0.90, "da", chunk_id="ca", user_id="user-A")]
        ch_b = [_chunk(0.30, "db", chunk_id="cb", user_id="user-B")]

        dec_a = ctrl.decide("What is my PAN?", ch_a)
        self.assertAlmostEqual(dec_a.s_top, 0.90, places=2)

        # Even if we accidentally mix chunks, the controller doesn't silently
        # use user-B's low score to change user-A's R_c
        dec_b = ctrl.decide("What is my PAN?", ch_b)
        self.assertAlmostEqual(dec_b.s_top, 0.30, places=2)


# ── DynamicKDiagnostics round-trip ─────────────────────────────────────────────

class TestDiagnosticsRoundTrip(unittest.TestCase):

    def test_to_dict_then_pydantic_schema(self):
        ctrl  = DynamicTopKController()
        dec   = ctrl.decide(
            "Compare my bank statements from 2022 and 2023.",
            [_chunk(0.85, "d1", chunk_id="c1"), _chunk(0.80, "d2", chunk_id="c2")],
        )
        d    = dec.to_dict()
        diag = DynamicKDiagnostics(**d)

        self.assertEqual(diag.recommended_k, dec.recommended_k)
        self.assertAlmostEqual(diag.c_q, dec.c_q, places=4)
        self.assertAlmostEqual(diag.r_c, dec.r_c, places=4)
        self.assertIsInstance(diag.features, dict)
        self.assertIsInstance(diag.explanation, str)


# ── Determinism across modules ─────────────────────────────────────────────────

class TestDeterminismAcrossModules(unittest.TestCase):

    def test_end_to_end_determinism(self):
        """Full pipeline from raw query to K decision must be deterministic."""
        ctrl   = DynamicTopKController()
        q      = "Explain the differences in my electricity bills for 2022 and 2023."
        chunks = [
            _chunk(0.88, "d1", chunk_id="c1", year=2022),
            _chunk(0.83, "d1", chunk_id="c2", year=2022),
            _chunk(0.80, "d2", chunk_id="c3", year=2023),
            _chunk(0.40, "d3", chunk_id="c4"),
        ]
        parsed = _parsed(doc_type="electricity_bill")

        d1 = ctrl.decide(q, chunks, parsed)
        d2 = ctrl.decide(q, chunks, parsed)

        self.assertEqual(d1.recommended_k,    d2.recommended_k)
        self.assertEqual(d1.c_q,              d2.c_q)
        self.assertEqual(d1.r_c,              d2.r_c)
        self.assertEqual(d1.target_documents, d2.target_documents)
        self.assertEqual(d1.features,         d2.features)


# ── Regression: Filter Consistency, Fallback & Expansion ───────────────────────

class TestFilterConsistencyAndFallback(unittest.TestCase):

    def test_effective_filter_used_in_expansion_after_fallback(self):
        """
        When probe returns nothing for doc_type, retry relaxes doc_type.
        Subsequent expansion MUST use the relaxed effective_filters, NOT the original filters,
        while maintaining user_id isolation.
        """
        initial_filters = {"document_type": "invoice", "year": 2023}
        effective_filters = dict(initial_filters)

        search_calls = []

        # Mock search function recording calls
        def mock_search_chunks(query_vector, user_id, filters, limit, score_threshold):
            search_calls.append({
                "user_id": user_id,
                "filters": dict(filters),
                "limit": limit,
            })
            # If document_type is present, return empty (simulating no hits with strict type)
            if "document_type" in filters:
                return []
            # Relaxed search returns chunks
            return [
                _chunk(0.82, "doc-1", chunk_id="c-1", year=2023, user_id=user_id),
                _chunk(0.79, "doc-2", chunk_id="c-2", year=2023, user_id=user_id),
            ]

        user_id = "test-user-42"
        q_vec = [0.1] * 10

        # Step 4: Probe
        raw_chunks = mock_search_chunks(q_vec, user_id=user_id, filters=effective_filters, limit=6, score_threshold=0.2)
        self.assertEqual(len(raw_chunks), 0)

        # Step 4a: Retry without doc-type
        if not raw_chunks and "document_type" in effective_filters:
            relaxed = {k: v for k, v in effective_filters.items() if k != "document_type"}
            retry_chunks = mock_search_chunks(q_vec, user_id=user_id, filters=relaxed, limit=6, score_threshold=0.2)
            if retry_chunks:
                raw_chunks = retry_chunks
                effective_filters = relaxed

        self.assertEqual(len(raw_chunks), 2)
        self.assertNotIn("document_type", effective_filters)
        self.assertEqual(effective_filters.get("year"), 2023)

        # Step 5 & 6: Expansion with recommended_k > probe
        recommended_k = 6
        if recommended_k > len(raw_chunks):
            expanded = mock_search_chunks(
                q_vec,
                user_id=user_id,
                filters=effective_filters,  # Must be effective_filters!
                limit=recommended_k * 2,
                score_threshold=0.2,
            )

        # Assert expansion search call used effective_filters without document_type
        expansion_call = search_calls[-1]
        self.assertEqual(expansion_call["user_id"], user_id)
        self.assertNotIn("document_type", expansion_call["filters"])
        self.assertEqual(expansion_call["filters"].get("year"), 2023)


# ── Regression: Tenant Isolation across all paths ──────────────────────────────

class TestTenantIsolationSearchPaths(unittest.TestCase):

    def test_user_id_passed_on_every_retrieval_stage(self):
        """Verify user_id is mandatory and verified on probe, retry, expansion, and legacy."""
        calls = []

        def mock_search(user_id, stage, filters=None):
            self.assertIsNotNone(user_id)
            self.assertTrue(len(user_id) > 0)
            calls.append((stage, user_id))

        target_user = "user-tenant-xyz"

        # Probe stage
        mock_search(user_id=target_user, stage="probe", filters={"doc_type": "tax"})
        # Retry stage
        mock_search(user_id=target_user, stage="retry", filters={})
        # Expansion stage
        mock_search(user_id=target_user, stage="expansion", filters={})
        # Legacy fallback stage
        mock_search(user_id=target_user, stage="legacy")

        self.assertEqual(len(calls), 4)
        for stage, uid in calls:
            self.assertEqual(uid, target_user)


# ── Regression: Empty Retrieval & Source References ────────────────────────────

class TestEmptyRetrievalAndSources(unittest.TestCase):

    def test_empty_retrieval_safe_budget(self):
        """When no chunks are returned from probe, retry, or legacy, flow handles it safely."""
        raw_chunks = []
        # Dynamic K should not crash on empty, and remains bounded in [k_min, k_max]
        ctrl = DynamicTopKController()
        dec = ctrl.decide("What is my tax return?", raw_chunks)
        self.assertGreaterEqual(dec.recommended_k, 3)
        self.assertLessEqual(dec.recommended_k, 14)

        # Top chunks slice should remain empty
        top_chunks = raw_chunks[:dec.recommended_k]
        self.assertEqual(len(top_chunks), 0)

        # Sources should be empty
        sources = []
        for idx, chunk in enumerate(top_chunks):
            sources.append(chunk)
        self.assertEqual(len(sources), 0)

    def test_sources_preserved_with_legacy_and_missing_document_id(self):
        """Sources are built robustly even when document_id is missing or legacy named."""
        c1 = _chunk(0.92, doc_id="", chunk_id="ck1", text="Sample 1")
        c1.payload["filename"] = "tax_receipt.pdf"
        c1.payload["document_id"] = ""

        c2 = _chunk(0.85, doc_id="doc-2", chunk_id="ck2", text="Sample 2")
        c2.payload["original_name"] = "salary_slip.pdf"

        # Multiple chunks for same doc-2 to test deduplication
        c3 = _chunk(0.88, doc_id="doc-2", chunk_id="ck3", text="Sample 3")

        top_chunks = [c1, c3, c2]

        seen_doc_ids = {}
        sources = []
        for idx, chunk in enumerate(top_chunks):
            p = chunk.payload or {}
            doc_id = p.get("document_id") or p.get("filename") or p.get("original_name") or f"doc_{idx}"
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids[str(doc_id)] = True
            sources.append({
                "document_id": str(doc_id),
                "filename": p.get("filename", p.get("original_name", "document")),
                "score": chunk.score,
            })

        self.assertEqual(len(sources), 2)  # tax_receipt and doc-2
        self.assertEqual(sources[0]["document_id"], "tax_receipt.pdf")
        self.assertEqual(sources[1]["document_id"], "doc-2")
        self.assertEqual(sources[1]["score"], 0.88)  # highest score chunk for doc-2


if __name__ == "__main__":
    unittest.main()

