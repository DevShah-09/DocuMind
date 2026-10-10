"""
run_evaluation.py — Automated Evaluation Harness
=================================================
Evaluates four retrieval strategies on the same query set and chunk pool,
computing all required metrics and printing a comparison report.

Usage
-----
From the backend/ directory (with the venv active):

    # Run on the demo dataset (offline, no Qdrant required):
    python -m tests.evaluation.run_evaluation

    # Run on only the held-out test split:
    python -m tests.evaluation.run_evaluation --split test

    # Run on the development split:
    python -m tests.evaluation.run_evaluation --split dev

    # Save the report to a JSON file:
    python -m tests.evaluation.run_evaluation --output results.json

Strategies compared
-------------------
  A. fixed_k5       — always retrieve K = 5
  B. fixed_k12      — always retrieve K = 12
  C. complexity_only — Dynamic K using C_q only (beta = 0)
  D. full_dynamic_k  — Dynamic K using C_q + R_c (full system)

Dataset
-------
By default uses the SYNTHETIC demo dataset in dataset.py.  Results are
clearly marked as synthetic and must NOT be reported as real-world findings.

To evaluate against real data, set --dataset-mode real and supply a JSON
file matching the EvalSample schema (see dataset.py for field definitions).

Faithfulness evaluation
-----------------------
The harness uses a SCORE-BASED PROXY (average top similarity score) rather
than an LLM judge.  This is a limitation: the proxy correlates with retrieval
quality but does not measure whether the generated answer is faithful.

To add an LLM judge:
  1. Set OPENROUTER_API_KEY in backend/.env
  2. Implement judge_faithfulness(answer, context) → float in faithfulness_judge.py
  3. Uncomment the LLM judge block in _evaluate_strategy() below.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# sys.path bootstrap — allows running as `python -m tests.evaluation.run_evaluation`
# from the backend/ directory without installing the package.
# ---------------------------------------------------------------------------
import os
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from tests.evaluation.dataset import (
    DEMO_DATASET,
    DEV_DATASET,
    TEST_DATASET,
    EvalSample,
)
from tests.evaluation.metrics import compute_aggregate_metrics
from tests.evaluation.strategies import (
    ALL_STRATEGIES,
    BaseStrategy,
    MockChunk,
)


# ── Mock chunk pool builder ────────────────────────────────────────────────────

def _build_mock_pool(sample: EvalSample, pool_size: int = 20) -> List[MockChunk]:
    """
    Build a synthetic chunk pool for a single evaluation sample.

    Relevant chunks get high similarity scores; irrelevant fillers get low scores.
    This simulates a real Qdrant search result without requiring a live DB.

    NOTE: In a real evaluation, replace this function with an actual Qdrant
    search call and return the real ScoredPoint objects.
    """
    chunks: List[MockChunk] = []

    # Place relevant chunks at the top with high scores
    for i, cid in enumerate(sample.relevant_chunk_ids):
        # Descending scores for ordered relevant chunks
        score = max(0.95 - i * 0.05, 0.60)
        doc_id = next(iter(sample.relevant_doc_ids)) if sample.relevant_doc_ids else "doc-unknown"
        # Use the first doc_id for the first chunk, cycle if multiple docs
        doc_ids_list = list(sample.relevant_doc_ids)
        doc_id = doc_ids_list[i % len(doc_ids_list)] if doc_ids_list else "doc-unknown"
        chunks.append(MockChunk(
            chunk_id=cid,
            doc_id=doc_id,
            score=score,
            text=f"Relevant text for {sample.qid} chunk {i}",
        ))

    # Fill the rest with irrelevant chunks at low scores
    n_filler = max(0, pool_size - len(sample.relevant_chunk_ids))
    for j in range(n_filler):
        chunks.append(MockChunk(
            chunk_id=f"filler-{sample.qid}-{j}",
            doc_id=f"doc-filler-{j}",
            score=max(0.30 - j * 0.01, 0.05),
            text="Irrelevant filler text",
        ))

    return chunks


# ── Single-strategy evaluator ─────────────────────────────────────────────────

def _evaluate_strategy(
    strategy: BaseStrategy,
    dataset: List[EvalSample],
) -> Dict[str, Any]:
    """Run one strategy over the full dataset and collect raw per-query results."""
    retrieved_lists:       List[List[str]] = []
    relevant_sets:         List[Set[str]]  = []
    relevant_ordered_lists: List[List[str]] = []
    recommended_ks:        List[int]       = []
    result_counts:         List[int]       = []
    top_scores:            List[List[float]] = []
    latencies_ms:          List[float]     = []

    for sample in dataset:
        pool = _build_mock_pool(sample)
        t0   = time.monotonic()
        retrieved = strategy.retrieve(query=sample.question, chunk_pool=pool)
        lat_ms = (time.monotonic() - t0) * 1000

        k = strategy.recommended_k(sample.question, pool)

        chunk_ids = [
            (c.payload or {}).get("chunk_id", "") for c in retrieved
        ]
        scores = [c.score for c in retrieved]

        retrieved_lists.append(chunk_ids)
        relevant_sets.append(set(sample.relevant_chunk_ids))
        relevant_ordered_lists.append(sample.relevant_chunk_ids)
        recommended_ks.append(k)
        result_counts.append(len(retrieved))
        top_scores.append(scores[:5])
        latencies_ms.append(lat_ms)

    avg_lat = round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else 0.0

    metrics = compute_aggregate_metrics(
        strategy_name=strategy.name,
        retrieved_lists=retrieved_lists,
        relevant_sets=relevant_sets,
        relevant_ordered=relevant_ordered_lists,
        recommended_ks=recommended_ks,
        result_counts=result_counts,
        top_scores_per_query=top_scores,
        k_baseline_for_reduction=12,
    )
    metrics["avg_retrieval_latency_ms"] = avg_lat
    return metrics


# ── Report printer ─────────────────────────────────────────────────────────────

_COLUMNS = [
    ("strategy",                       "Strategy",           20),
    ("recall@10",                      "Recall@10",          10),
    ("mrr",                            "MRR",                8),
    ("ndcg@10",                        "nDCG@10",            10),
    ("avg_faithfulness_proxy",         "Faith.(proxy)",      14),
    ("avg_k",                          "Avg K",              7),
    ("chunk_reduction_vs_k12_pct",     "D vs K12 (%)",       13),
    ("empty_result_rate",              "Empty%",             8),
    ("avg_retrieval_latency_ms",       "Lat(ms)",            9),
]


def _print_report(results: List[Dict[str, Any]], split_name: str) -> None:
    print()
    print("=" * 110)
    print(f"  Confidence-Aware Dynamic Top-K Evaluation -- {split_name.upper()} SPLIT")
    print("  [!] SYNTHETIC DATA -- results are illustrative only, not real-world measurements")
    print("=" * 110)
    header = "".join(label.ljust(w) for _, label, w in _COLUMNS)
    print(header)
    print("-" * 110)
    for row in results:
        line_parts = []
        for key, _, width in _COLUMNS:
            val = row.get(key, "—")
            if isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = str(val)
            line_parts.append(cell.ljust(width))
        print("".join(line_parts))
    print("=" * 110)
    print()
    print("Metric definitions:")
    print("  Recall@10     -- fraction of ground-truth relevant chunks found in top-10")
    print("  MRR           -- mean reciprocal rank of first relevant chunk")
    print("  nDCG@10       -- normalised DCG at rank 10 (binary relevance)")
    print("  Faith.(proxy) -- avg top-5 similarity scores (NOT LLM-judge faithfulness)")
    print("  Avg K         -- mean recommended K across all queries")
    print("  D vs K12 (%)  -- % reduction in avg K relative to fixed K=12 baseline")
    print("  Empty%        -- fraction of queries returning zero chunks")
    print("  Lat(ms)       -- mean per-query retrieval latency (mock pool, no Qdrant)")
    print()
    print("Faithfulness limitation: the proxy averages similarity scores, which")
    print("correlates with retrieval quality but does NOT measure answer faithfulness.")
    print("For production eval, implement an LLM judge (see module docstring).")
    print()


# ── Main entrypoint ────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Dynamic Top-K retrieval strategies"
    )
    parser.add_argument(
        "--split",
        choices=["all", "dev", "test"],
        default="all",
        help="Which dataset split to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results (e.g. results.json)",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["demo", "real"],
        default="demo",
        help="'demo' uses synthetic data; 'real' requires a real Qdrant instance",
    )
    args = parser.parse_args(argv)

    if args.dataset_mode == "real":
        print(
            "ERROR: Real-data mode requires a live Qdrant connection and annotated dataset.\n"
            "See tests/evaluation/dataset.py for instructions on building a real dataset.\n"
            "Falling back to demo mode."
        )

    split_map = {
        "all":  (DEMO_DATASET,  "full demo"),
        "dev":  (DEV_DATASET,   "development"),
        "test": (TEST_DATASET,  "held-out test"),
    }
    dataset, split_name = split_map[args.split]

    print(f"\nRunning evaluation on {len(dataset)} queries ({split_name} split)…")

    all_results = []
    for strategy in ALL_STRATEGIES:
        print(f"  Strategy: {strategy.name} …", end=" ", flush=True)
        metrics = _evaluate_strategy(strategy, dataset)
        all_results.append(metrics)
        print("done")

    _print_report(all_results, split_name)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
