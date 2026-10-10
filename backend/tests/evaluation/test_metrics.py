"""
test_metrics.py — Unit Tests for Evaluation Metrics
=====================================================
Covers every function in tests/evaluation/metrics.py.
"""

from __future__ import annotations

import math
import unittest

from tests.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_k,
    chunk_reduction_pct,
    empty_result_rate,
    retrieval_failure_rate,
    score_based_faithfulness,
    compute_aggregate_metrics,
)


class TestRecallAtK(unittest.TestCase):

    def test_perfect_recall(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3), 1.0)

    def test_zero_recall(self):
        self.assertEqual(recall_at_k(["x", "y"], {"a", "b"}, k=2), 0.0)

    def test_partial_recall(self):
        r = recall_at_k(["a", "x", "b", "y"], {"a", "b", "c"}, k=4)
        self.assertAlmostEqual(r, 2 / 3, places=5)

    def test_k_smaller_than_list(self):
        # only top-2 considered
        r = recall_at_k(["a", "x", "b"], {"a", "b"}, k=2)
        self.assertAlmostEqual(r, 0.5, places=5)  # only 'a' in top-2

    def test_empty_relevant_returns_zero(self):
        self.assertEqual(recall_at_k(["a", "b"], set(), k=5), 0.0)

    def test_empty_retrieved_returns_zero(self):
        self.assertEqual(recall_at_k([], {"a", "b"}, k=5), 0.0)


class TestReciprocalRank(unittest.TestCase):

    def test_first_item_relevant(self):
        self.assertEqual(reciprocal_rank(["a", "b", "c"], {"a"}), 1.0)

    def test_second_item_relevant(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "a", "b"], {"a"}), 0.5, places=5)

    def test_third_item_relevant(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "y", "a"], {"a"}), 1/3, places=5)

    def test_not_found_returns_zero(self):
        self.assertEqual(reciprocal_rank(["x", "y", "z"], {"a"}), 0.0)

    def test_empty_retrieved_returns_zero(self):
        self.assertEqual(reciprocal_rank([], {"a"}), 0.0)

    def test_empty_relevant_returns_zero(self):
        self.assertEqual(reciprocal_rank(["a"], set()), 0.0)


class TestMRR(unittest.TestCase):

    def test_basic_mrr(self):
        results  = [["a", "b"], ["x", "a"], ["a"]]
        relevant = [{"a"}, {"a"}, {"a"}]
        mrr = mean_reciprocal_rank(results, relevant)
        # RRs: 1.0, 0.5, 1.0 → avg = 0.8333
        self.assertAlmostEqual(mrr, (1.0 + 0.5 + 1.0) / 3, places=5)

    def test_empty_list_returns_zero(self):
        self.assertEqual(mean_reciprocal_rank([], []), 0.0)


class TestNDCG(unittest.TestCase):

    def test_perfect_ndcg(self):
        # Retrieved exactly matches ideal
        retrieved = ["a", "b", "c"]
        relevant  = ["a", "b", "c"]
        score = ndcg_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_zero_ndcg(self):
        retrieved = ["x", "y", "z"]
        relevant  = ["a", "b", "c"]
        score = ndcg_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_partial_ndcg(self):
        # retrieved = ["a", "x", "b"]; relevant = ["a", "b"]
        retrieved = ["a", "x", "b"]
        relevant  = ["a", "b"]
        score = ndcg_at_k(retrieved, relevant, k=3)
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        expected = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(score, expected, places=4)

    def test_empty_relevant_returns_zero(self):
        self.assertEqual(ndcg_at_k(["a"], [], k=5), 0.0)

    def test_k_cutoff_respected(self):
        # The relevant item is at rank 5 but k=3
        retrieved = ["x", "y", "z", "w", "a"]
        relevant  = ["a"]
        score = ndcg_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(score, 0.0, places=5)


class TestAverageK(unittest.TestCase):

    def test_average_k_basic(self):
        self.assertAlmostEqual(average_k([4, 6, 8]), 6.0, places=5)

    def test_average_k_empty(self):
        self.assertEqual(average_k([]), 0.0)

    def test_average_k_single(self):
        self.assertAlmostEqual(average_k([7]), 7.0, places=5)


class TestChunkReduction(unittest.TestCase):

    def test_50_pct_reduction(self):
        ks = [6] * 10   # avg = 6, baseline = 12 → 50% reduction
        self.assertAlmostEqual(chunk_reduction_pct(ks, 12), 50.0, places=2)

    def test_no_reduction(self):
        ks = [12] * 5
        self.assertAlmostEqual(chunk_reduction_pct(ks, 12), 0.0, places=2)

    def test_negative_reduction_system_uses_more(self):
        ks = [14] * 5   # avg = 14 > baseline 12
        self.assertLess(chunk_reduction_pct(ks, 12), 0.0)

    def test_empty_ks_returns_zero(self):
        self.assertEqual(chunk_reduction_pct([], 12), 0.0)

    def test_zero_baseline_returns_zero(self):
        self.assertEqual(chunk_reduction_pct([5, 5], 0), 0.0)


class TestEmptyResultRate(unittest.TestCase):

    def test_all_empty(self):
        self.assertAlmostEqual(empty_result_rate([0, 0, 0]), 1.0, places=5)

    def test_none_empty(self):
        self.assertAlmostEqual(empty_result_rate([3, 5, 2]), 0.0, places=5)

    def test_partial(self):
        self.assertAlmostEqual(empty_result_rate([0, 3, 0, 5]), 0.5, places=5)

    def test_empty_input(self):
        self.assertEqual(empty_result_rate([]), 0.0)

    def test_retrieval_failure_rate_alias(self):
        counts = [0, 1, 0, 2]
        self.assertEqual(empty_result_rate(counts), retrieval_failure_rate(counts))


class TestScoreBasedFaithfulness(unittest.TestCase):

    def test_high_scores(self):
        self.assertAlmostEqual(score_based_faithfulness([0.90, 0.85, 0.80]), 0.85, places=3)

    def test_empty_returns_zero(self):
        self.assertEqual(score_based_faithfulness([]), 0.0)

    def test_scores_clipped_to_1(self):
        result = score_based_faithfulness([1.5, 0.8])
        self.assertLessEqual(result, 1.0)

    def test_scores_clipped_to_0(self):
        result = score_based_faithfulness([-0.5, 0.3])
        self.assertGreaterEqual(result, 0.0)

    def test_all_zero_scores(self):
        self.assertAlmostEqual(score_based_faithfulness([0.0, 0.0]), 0.0, places=5)


class TestComputeAggregateMetrics(unittest.TestCase):

    def _make_inputs(self, n=3):
        """Build minimal parallel inputs for n queries."""
        retrieved = [["a", "b"] for _ in range(n)]
        relevant  = [{"a"} for _ in range(n)]
        ordered   = [["a", "b"] for _ in range(n)]
        rec_ks    = [6] * n
        counts    = [2] * n
        scores    = [[0.85, 0.80] for _ in range(n)]
        return retrieved, relevant, ordered, rec_ks, counts, scores

    def test_returns_all_required_keys(self):
        r, rel, ro, ks, cnts, sc = self._make_inputs()
        result = compute_aggregate_metrics(
            strategy_name="test",
            retrieved_lists=r,
            relevant_sets=rel,
            relevant_ordered=ro,
            recommended_ks=ks,
            result_counts=cnts,
            top_scores_per_query=sc,
        )
        required = {
            "strategy", "n_queries", "recall@10", "mrr", "ndcg@10",
            "avg_faithfulness_proxy", "avg_k",
            "chunk_reduction_vs_k12_pct", "empty_result_rate",
            "retrieval_failure_rate",
        }
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_n_queries_correct(self):
        r, rel, ro, ks, cnts, sc = self._make_inputs(n=5)
        result = compute_aggregate_metrics(
            strategy_name="test",
            retrieved_lists=r, relevant_sets=rel,
            relevant_ordered=ro, recommended_ks=ks,
            result_counts=cnts, top_scores_per_query=sc,
        )
        self.assertEqual(result["n_queries"], 5)

    def test_avg_k_correct(self):
        r, rel, ro, ks, cnts, sc = self._make_inputs(n=4)
        ks = [4, 6, 8, 10]
        result = compute_aggregate_metrics(
            strategy_name="test",
            retrieved_lists=r, relevant_sets=rel,
            relevant_ordered=ro, recommended_ks=ks,
            result_counts=cnts, top_scores_per_query=sc,
        )
        self.assertAlmostEqual(result["avg_k"], 7.0, places=2)


if __name__ == "__main__":
    unittest.main()
