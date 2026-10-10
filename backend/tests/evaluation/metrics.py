"""
metrics.py — Retrieval and Generation Quality Metrics
======================================================
Provides pure-Python implementations of all required metrics that run
without any external API, vector DB, or LLM call:

Retrieval metrics
-----------------
  recall_at_k(retrieved, relevant, k)
      Fraction of relevant items found in the top-k retrieved items.

  reciprocal_rank(retrieved, relevant)
      1 / rank of the first relevant item (0.0 if not found).

  mean_reciprocal_rank(results_list, relevant_list)
      Average MRR over a list of queries.

  ndcg_at_k(retrieved, relevant_ordered, k)
      Normalised Discounted Cumulative Gain using binary relevance grades.
      relevant_ordered: list ordered by descending ground-truth relevance.

Efficiency metrics
------------------
  average_k(ks)
      Mean recommended K across a query set.

  chunk_reduction_pct(ks_system, k_baseline)
      Percentage reduction in average K vs. a fixed-K baseline.

  empty_result_rate(result_counts)
      Fraction of queries that returned zero chunks.

Faithfulness (score-based baseline)
--------------------------------------
  score_based_faithfulness(top_scores)
      Proxy: average of top similarity scores.
      NOTE: NOT equivalent to LLM-judge faithfulness.  Documented as a
      score-based proxy only.  For production evaluation, replace with an
      LLM judge (see run_evaluation.py for guidance).

All functions are pure, deterministic, and dependency-free.
"""

from __future__ import annotations

import math
from typing import List, Optional, Set


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K: fraction of relevant items found in the top-k retrieved list.

    Parameters
    ----------
    retrieved : list of str
        Ordered retrieved item IDs (chunk_id or doc_id), most relevant first.
    relevant : set of str
        Ground-truth relevant item IDs.
    k : int
        Cutoff rank.

    Returns
    -------
    float in [0, 1].  Returns 0.0 when relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Reciprocal Rank: 1 / rank of the first relevant item.

    Returns 0.0 if no relevant item appears in retrieved.
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results_list: List[List[str]],
    relevant_list: List[Set[str]],
) -> float:
    """
    MRR over a list of queries.

    Parameters
    ----------
    results_list : list of list of str
        One ordered retrieved list per query.
    relevant_list : list of set of str
        Ground-truth relevant IDs per query (parallel with results_list).

    Returns
    -------
    float in [0, 1].
    """
    if not results_list:
        return 0.0
    rrs = [
        reciprocal_rank(r, rel)
        for r, rel in zip(results_list, relevant_list)
    ]
    return sum(rrs) / len(rrs)


def ndcg_at_k(
    retrieved: List[str],
    relevant_ordered: List[str],
    k: int,
) -> float:
    """
    nDCG@K with binary relevance (grade = 1 if relevant, 0 otherwise).

    Parameters
    ----------
    retrieved : list of str
        Ordered retrieved item IDs, most relevant first.
    relevant_ordered : list of str
        Ground-truth relevant IDs ordered by descending relevance (ideal ranking).
    k : int
        Cutoff rank.

    Returns
    -------
    float in [0, 1].
    """
    relevant_set = set(relevant_ordered)
    if not relevant_set:
        return 0.0

    def dcg(items: List[str], cutoff: int) -> float:
        gain = 0.0
        for i, item in enumerate(items[:cutoff], start=1):
            if item in relevant_set:
                gain += 1.0 / math.log2(i + 1)
        return gain

    actual_dcg  = dcg(retrieved, k)
    # Ideal DCG: place all relevant items at the top
    ideal_items = [x for x in relevant_ordered if x in relevant_set]
    ideal_dcg   = dcg(ideal_items, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


# ── Efficiency metrics ────────────────────────────────────────────────────────

def average_k(ks: List[int]) -> float:
    """Mean recommended K across a set of queries."""
    if not ks:
        return 0.0
    return sum(ks) / len(ks)


def chunk_reduction_pct(ks_system: List[int], k_baseline: int) -> float:
    """
    Percentage reduction in average K relative to a fixed-K baseline.

    Positive value = the system uses fewer chunks on average.
    Negative value = the system uses more chunks than the baseline.
    """
    if not ks_system or k_baseline <= 0:
        return 0.0
    avg = average_k(ks_system)
    return round((k_baseline - avg) / k_baseline * 100, 2)


def empty_result_rate(result_counts: List[int]) -> float:
    """Fraction of queries that returned zero retrieved chunks."""
    if not result_counts:
        return 0.0
    return sum(1 for c in result_counts if c == 0) / len(result_counts)


def retrieval_failure_rate(result_counts: List[int]) -> float:
    """Alias for empty_result_rate — fraction of zero-result queries."""
    return empty_result_rate(result_counts)


# ── Faithfulness proxy ────────────────────────────────────────────────────────

def score_based_faithfulness(top_scores: List[float]) -> float:
    """
    Score-based faithfulness proxy: average of the top similarity scores.

    LIMITATION
    ----------
    This is NOT equivalent to LLM-judge faithfulness.  It measures how
    strongly retrieved chunks are vectorially aligned with the query, which
    correlates with but does not guarantee answer faithfulness.

    For production use, replace this function with an LLM judge:
      - Model: e.g. gpt-4o-mini or a local Llama
      - Prompt: "Does the following answer faithfully reflect the context?
                 Context: {context}  Answer: {answer}  Score 0–1."
      - Document: model, prompt version, cost per query, and known limitations.

    Returns
    -------
    float in [0, 1].  Returns 0.0 for empty input.
    """
    if not top_scores:
        return 0.0
    valid = [max(0.0, min(s, 1.0)) for s in top_scores]
    return round(sum(valid) / len(valid), 4)


# ── Aggregate report ──────────────────────────────────────────────────────────

def compute_aggregate_metrics(
    *,
    strategy_name: str,
    retrieved_lists:  List[List[str]],   # per query: ordered retrieved chunk/doc IDs
    relevant_sets:    List[Set[str]],    # per query: ground-truth relevant IDs
    relevant_ordered: List[List[str]],   # per query: ordered ground-truth IDs
    recommended_ks:   List[int],
    result_counts:    List[int],
    top_scores_per_query: List[List[float]],
    k_baseline_for_reduction: int = 12,
    ndcg_k: int = 10,
    recall_k: int = 10,
) -> dict:
    """
    Compute all metrics for one retrieval strategy and return a summary dict.

    Parameters mirror the fields populated by run_evaluation.py.
    """
    n = len(retrieved_lists)
    assert len(relevant_sets)    == n, "relevant_sets length mismatch"
    assert len(recommended_ks)   == n, "recommended_ks length mismatch"
    assert len(result_counts)    == n, "result_counts length mismatch"

    recalls   = [recall_at_k(r, rel, recall_k)
                 for r, rel in zip(retrieved_lists, relevant_sets)]
    rrs       = [reciprocal_rank(r, rel)
                 for r, rel in zip(retrieved_lists, relevant_sets)]
    ndcgs     = [ndcg_at_k(r, ro, ndcg_k)
                 for r, ro in zip(retrieved_lists, relevant_ordered)]
    faith     = [score_based_faithfulness(scores)
                 for scores in top_scores_per_query]

    return {
        "strategy":             strategy_name,
        "n_queries":            n,
        f"recall@{recall_k}":   round(sum(recalls) / n, 4) if n else 0.0,
        "mrr":                  round(mean_reciprocal_rank(retrieved_lists, relevant_sets), 4),
        f"ndcg@{ndcg_k}":       round(sum(ndcgs)   / n, 4) if n else 0.0,
        "avg_faithfulness_proxy": round(sum(faith) / n, 4) if n else 0.0,
        "avg_k":                round(average_k(recommended_ks), 2),
        f"chunk_reduction_vs_k{k_baseline_for_reduction}_pct":
                                chunk_reduction_pct(recommended_ks, k_baseline_for_reduction),
        "empty_result_rate":    round(empty_result_rate(result_counts), 4),
        "retrieval_failure_rate": round(retrieval_failure_rate(result_counts), 4),
    }
