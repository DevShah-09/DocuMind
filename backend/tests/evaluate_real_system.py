"""
evaluate_real_system.py — Empirical Evaluation of Dynamic Top-K vs Baselines
=============================================================================
Evaluates retrieval on the REAL live Qdrant corpus and live embedding model:
  1. Fixed-K Baseline (K=5)
  2. Complexity-Only Dynamic-K (alpha=1.0, beta=0.0)
  3. Complexity + Confidence Dynamic-K (alpha=0.5, beta=0.5, full controller)
  4. Paper Formula Variant: K* = ceil(clip(K_min + (1-R_c)*K_range + gamma*C_q, K_min, K_max))

Ground-truth relevance is established against the 20 real chunks in Qdrant Cloud.
"""

from __future__ import annotations

import asyncio
import math
import sys
sys.path.insert(0, ".")
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.embeddings.embedding_service import embedding_service
from app.retrieval.dynamic_top_k import DynamicTopKController, TopKConfig
from app.retrieval.query_complexity import QueryComplexityEstimator
from app.retrieval.query_parser import query_parser
from app.retrieval.retrieval_confidence import RetrievalConfidenceEstimator
from app.vectorstore.qdrant_store import qdrant_store
from supabase import create_client


@dataclass
class RealEvalSample:
    qid: str
    query: str
    category: str
    expected_doc_ids: Set[str]
    expected_chunk_ids: List[str]  # ordered by descending relevance
    notes: str


# ── Curated Real Evaluation Dataset (10 queries) ──────────────────────────────
# All document IDs and chunk IDs correspond directly to the live Qdrant points.
REAL_DATASET: List[RealEvalSample] = [
    # 1. Simple factual
    RealEvalSample(
        qid="Q1_FACTUAL_PAN",
        query="What is my PAN card number?",
        category="factual",
        expected_doc_ids={"d2d831a5-81dd-49f8-8bc4-c9ba74828c0a"},
        expected_chunk_ids=["e0ca014af00c371c1eac331fa20a0be9b42eec88feb14f52bae7ececa6002119"],
        notes="Single document lookup for PAN card",
    ),
    # 2. Simple factual
    RealEvalSample(
        qid="Q2_FACTUAL_AADHAAR",
        query="What is my Aadhaar card number?",
        category="factual",
        expected_doc_ids={"0258bd53-c003-4710-84b3-d2437a0277a6"},
        expected_chunk_ids=["6b30331c8c6d1c48325784afc2ca9e4b5e444236ccd3e9da4a4651e375842ddb"],
        notes="Single document lookup for Aadhaar card",
    ),
    # 3. Temporal / Date-constrained
    RealEvalSample(
        qid="Q3_TEMPORAL_MARKSHEET",
        query="Show my secondary school marksheet from March 2022",
        category="temporal",
        expected_doc_ids={"d31be8a8-2642-43d2-b421-c0b4d8478623"},
        expected_chunk_ids=["055c20238759c793e6327814e686c38c98a801dbe30d353687083a57be9d50e1"],
        notes="Date-constrained lookup for 2022 marksheet",
    ),
    # 4. Constrained lookup
    RealEvalSample(
        qid="Q4_CONSTRAINED_RESUME_2024",
        query="Find Shubham Shah resume from 2024",
        category="constrained",
        expected_doc_ids={"cd3baa72-93d3-4745-98de-21437f12e089"},
        expected_chunk_ids=[
            "04196f7b0dcd81e7e4787d210a4356ac41d21924b9791946e370aa7de93e2213",
            "d94943a959349a8e393d60fb0bfae4d3858d0d2166e046407d124488a7e52afb",
        ],
        notes="Constrained resume query with 2 chunks in corpus",
    ),
    # 5. Comparative multi-document
    RealEvalSample(
        qid="Q5_COMPARE_RESUMES",
        query="Compare the resume of Shubham Shah with the resume of Dhruv Soni",
        category="comparative",
        expected_doc_ids={
            "cd3baa72-93d3-4745-98de-21437f12e089",
            "3a253ef5-1b86-4b59-ae07-06d5bdbcaa05",
            "568ff9a4-4c7f-4071-942c-2e983c8c4939",
        },
        expected_chunk_ids=[
            "04196f7b0dcd81e7e4787d210a4356ac41d21924b9791946e370aa7de93e2213",
            "3dbb15c0ac352028b41b0712dab0acb67ec1b69addb3971105de31f7ebba5bfc",
            "0fdfd78d770b4769baf92baf6421f63d9262175238206b0670fff2f258420076",
        ],
        notes="Requires cross-document retrieval across 2 candidates",
    ),
    # 6. Comparative ID cards
    RealEvalSample(
        qid="Q6_COMPARE_ID_CARDS",
        query="Compare the college student ID card and the Aadhaar card",
        category="comparative",
        expected_doc_ids={
            "ae4258ec-0994-45df-ade2-954fc4d3c761",
            "0258bd53-c003-4710-84b3-d2437a0277a6",
        },
        expected_chunk_ids=[
            "22721a40e3e00578e03477110a5a5a726d0eeb458c1a3f5699f2fb122600156c",
            "6b30331c8c6d1c48325784afc2ca9e4b5e444236ccd3e9da4a4651e375842ddb",
        ],
        notes="Compares two distinct identity documents",
    ),
    # 7. Aggregative / Multi-document
    RealEvalSample(
        qid="Q7_AGGREGATE_DIAGRAMS",
        query="Summarize all technical architecture diagrams and system architecture diagrams from August 2026",
        category="aggregative",
        expected_doc_ids={
            "5098d294-37e6-4357-8486-3f646ced9824",
            "1ca9b723-64af-4861-94b2-aaedb222dad2",
        },
        expected_chunk_ids=[
            "2be525d62fa442d6ca636d604bb96b675a1d22efe8373b28174b6e4a27f37f8e",
            "dff311c3b171a37577eddc4ce3b9e8a8c8acc6aed27ee0fffe1cef537e18baed",
        ],
        notes="Aggregative query over multiple August 2026 architecture diagrams",
    ),
    # 8. Complex multi-document browse
    RealEvalSample(
        qid="Q8_COMPLEX_BROWSE",
        query="List all student identity cards and resumes across all years and analyze their differences",
        category="complex",
        expected_doc_ids={
            "ae4258ec-0994-45df-ade2-954fc4d3c761",
            "b55ea0b6-e17e-4e3c-b941-a13f0f126de4",
            "cd3baa72-93d3-4745-98de-21437f12e089",
            "3a253ef5-1b86-4b59-ae07-06d5bdbcaa05",
            "568ff9a4-4c7f-4071-942c-2e983c8c4939",
            "cfee50b4-c077-473e-afda-e63f85167a6a",
        },
        expected_chunk_ids=[
            "22721a40e3e00578e03477110a5a5a726d0eeb458c1a3f5699f2fb122600156c",
            "d68b9d3b19be6eac08391604fec3842fcbba636b8545ceb04ebefe89a8e6df08",
            "04196f7b0dcd81e7e4787d210a4356ac41d21924b9791946e370aa7de93e2213",
            "3dbb15c0ac352028b41b0712dab0acb67ec1b69addb3971105de31f7ebba5bfc",
            "ad63ed4fd47b48af794725da33fc317c6ec936b4128cb5a2481df639e2801643",
            "0fdfd78d770b4769baf92baf6421f63d9262175238206b0670fff2f258420076",
        ],
        notes="Broad multi-document synthesis query",
    ),
    # 9. Out-of-distribution (absent entity)
    RealEvalSample(
        qid="Q9_OOD_ELECTRICITY",
        query="What was my total electricity bill payment in January 2024?",
        category="ood",
        expected_doc_ids=set(),
        expected_chunk_ids=[],
        notes="No electricity bill exists in corpus; tests low-confidence handling",
    ),
    # 10. Out-of-distribution (absent entity)
    RealEvalSample(
        qid="Q10_OOD_MEDICAL",
        query="Show my hospital discharge summary and blood test medical reports",
        category="ood",
        expected_doc_ids=set(),
        expected_chunk_ids=[],
        notes="No medical reports exist in corpus; tests low-confidence handling",
    ),
]


# ── Metrics Helper Functions ──────────────────────────────────────────────────

def compute_recall_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str]) -> float:
    if not relevant_chunk_ids:
        # If no relevant chunks exist in ground truth (e.g. OOD queries):
        # perfect score if nothing retrieved or non-relevant, but typically not counted
        return 1.0 if not retrieved_chunk_ids else 0.0
    relevant_set = set(relevant_chunk_ids)
    hits = sum(1 for cid in retrieved_chunk_ids if cid in relevant_set)
    return hits / len(relevant_set)


def compute_mrr(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str]) -> float:
    if not relevant_chunk_ids or not retrieved_chunk_ids:
        return 0.0
    relevant_set = set(relevant_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, 1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    if not relevant_chunk_ids or not retrieved_chunk_ids:
        return 0.0
    relevance_map = {cid: max(len(relevant_chunk_ids) - rank, 1) for rank, cid in enumerate(relevant_chunk_ids)}
    
    # DCG
    dcg = 0.0
    for i, cid in enumerate(retrieved_chunk_ids[:k]):
        rel = relevance_map.get(cid, 0)
        dcg += rel / math.log2(i + 2)
        
    # IDCG
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# ── Execution Harness ─────────────────────────────────────────────────────────

async def evaluate_all_strategies():
    print("=" * 80)
    print("EMPIRICAL EVALUATION: DYNAMIC TOP-K VS BASELINES ON LIVE QDRANT")
    print("=" * 80)

    # 1. Connect to Supabase to get user ID
    s = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    users = s.auth.admin.list_users()
    user_id = str(users[0].id)
    print(f"[AUTH] User ID: {user_id[:8]}... (20 chunks in Qdrant Cloud)")

    # 2. Warm up embedding service (avoid cold start in timing)
    print("[WARMUP] Pre-loading embedding service...")
    t_warm = time.monotonic()
    await embedding_service.embed_query("warmup query")
    print(f"[WARMUP] Done in {round((time.monotonic() - t_warm)*1000, 1)}ms")

    # 3. Instantiate controllers
    complexity_estimator = QueryComplexityEstimator()
    confidence_estimator = RetrievalConfidenceEstimator()
    
    # Controller for full dynamic K (alpha=0.5, beta=0.5)
    controller_full = DynamicTopKController(
        config=TopKConfig(k_min=2, k_max=15, alpha=0.5, beta=0.5),
        complexity_estimator=complexity_estimator,
        confidence_estimator=confidence_estimator,
    )

    # Controller for complexity only (alpha=1.0, beta=0.0)
    controller_comp_only = DynamicTopKController(
        config=TopKConfig(k_min=2, k_max=15, alpha=1.0, beta=0.0),
        complexity_estimator=complexity_estimator,
        confidence_estimator=confidence_estimator,
    )

    # Strategies:
    # 1: fixed_k5
    # 2: complexity_only
    # 3: full_dynamic_k
    # 4: paper_formula: K* = ceil(clip(K_min + (1-R_c)*K_range + gamma*C_q, K_min, K_max))
    strategy_names = ["fixed_k5", "complexity_only", "full_dynamic_k", "paper_formula"]

    # Results collectors
    results: Dict[str, Dict[str, Any]] = {
        strat: {
            "recalls": [],
            "mrrs": [],
            "ndcgs": [],
            "ks": [],
            "chunk_counts": [],
            "context_chars": [],
            "context_tokens": [],
            "retrieval_latencies": [],
            "end_to_end_latencies": [],
            "c_q_list": [],
            "r_c_list": [],
        }
        for strat in strategy_names
    }

    per_query_records = []

    print("\nRunning evaluation on 10 curated queries...")

    for sample in REAL_DATASET:
        print(f"\n--- Query {sample.qid} [{sample.category}] ---")
        print(f"Query: \"{sample.query}\"")

        # Parse query & embed query (timed)
        t_start_e2e = time.monotonic()
        parsed = await query_parser.parse(sample.query)
        q_vec = await embedding_service.embed_query(sample.query)
        embed_latency_ms = (time.monotonic() - t_start_e2e) * 1000

        # Build metadata filters (matching chat.py)
        filters = {}
        if parsed.document_type:
            filters["document_type"] = parsed.document_type
        if parsed.year is not None:
            filters["year"] = parsed.year
        if parsed.month:
            filters["month"] = parsed.month

        # Probe search (limit=6)
        t_ret_start = time.monotonic()
        probe_chunks = await qdrant_store.search_chunks(
            query_vector=q_vec,
            user_id=user_id,
            filters=filters,
            limit=6,
            score_threshold=0.3,
        )
        # Fallback without document_type if probe returned nothing
        if not probe_chunks and parsed.document_type:
            relaxed = {k: v for k, v in filters.items() if k != "document_type"}
            retry_chunks = await qdrant_store.search_chunks(
                query_vector=q_vec,
                user_id=user_id,
                filters=relaxed,
                limit=6,
                score_threshold=0.3,
            )
            if retry_chunks:
                probe_chunks = retry_chunks
                filters = relaxed

        probe_latency_ms = (time.monotonic() - t_ret_start) * 1000

        # Compute C_q and R_c diagnostics
        comp_res = complexity_estimator.estimate(sample.query, parsed)
        conf_res = confidence_estimator.estimate(probe_chunks, parsed)
        c_q = comp_res.score
        r_c = conf_res.score

        # Decide K for each strategy:
        # A: Fixed K = 5
        k_fixed5 = 5

        # B: Complexity Only
        dec_comp = controller_comp_only.decide(sample.query, probe_chunks, parsed)
        k_comp = dec_comp.recommended_k

        # C: Full Dynamic K
        dec_full = controller_full.decide(sample.query, probe_chunks, parsed)
        k_full = dec_full.recommended_k

        # D: Paper Formula:
        # K* = ceil(clip(K_min + (1 - R_c) * K_range + gamma * C_q, K_min, K_max))
        # With K_min=2, K_max=15, K_range=8, gamma=5.0 (the constants from dynamic_k.py)
        raw_paper = 2 + (1.0 - r_c) * 8.0 + 5.0 * c_q
        clipped_paper = max(2, min(raw_paper, 15))
        k_paper = math.ceil(clipped_paper)

        strat_ks = {
            "fixed_k5": k_fixed5,
            "complexity_only": k_comp,
            "full_dynamic_k": k_full,
            "paper_formula": k_paper,
        }

        print(f"Diagnostics: C_q={c_q:.3f}, R_c={r_c:.3f}, S_top={conf_res.s_top:.3f}, S_avg={conf_res.s_avg:.3f}, knee_gap={conf_res.knee_gap:.3f}")
        print(f"Selected K: fixed_k5={k_fixed5}, complexity_only={k_comp}, full_dynamic_k={k_full}, paper_formula={k_paper}")

        # Now for each strategy, retrieve chunks up to its selected K:
        # If K <= len(probe_chunks), reuse probe chunks; otherwise expand search!
        query_record = {
            "qid": sample.qid,
            "query": sample.query,
            "category": sample.category,
            "c_q": round(c_q, 4),
            "r_c": round(r_c, 4),
            "strategies": {},
        }

        for strat_name, chosen_k in strat_ks.items():
            t_strat_start = time.monotonic()
            if chosen_k <= len(probe_chunks):
                retrieved = probe_chunks[:chosen_k]
                expansion_latency = 0.0
            else:
                # Perform expansion
                t_exp_start = time.monotonic()
                fetch_limit = chosen_k * 2
                expanded = await qdrant_store.search_chunks(
                    query_vector=q_vec,
                    user_id=user_id,
                    filters=filters,
                    limit=fetch_limit,
                    score_threshold=0.3,
                )
                expansion_latency = (time.monotonic() - t_exp_start) * 1000
                # Merge and deduplicate
                seen_keys = set()
                combined = []
                for c in expanded + probe_chunks:
                    p = c.payload or {}
                    cid = p.get("chunk_id", str(getattr(c, "id", "")))
                    if cid not in seen_keys:
                        seen_keys.add(cid)
                        combined.append(c)
                retrieved = sorted(combined, key=lambda c: c.score, reverse=True)[:chosen_k]

            total_retrieval_ms = probe_latency_ms + expansion_latency
            total_e2e_ms = (time.monotonic() - t_start_e2e) * 1000

            retrieved_chunk_ids = [
                (c.payload or {}).get("chunk_id", str(getattr(c, "id", "")))
                for c in retrieved
            ]

            # Context size
            total_chars = sum(len((c.payload or {}).get("text", "")) for c in retrieved)
            approx_tokens = math.ceil(total_chars / 4.0)

            # Metrics
            if sample.expected_chunk_ids:  # Only for in-distribution queries
                recall = compute_recall_at_k(retrieved_chunk_ids, sample.expected_chunk_ids)
                mrr = compute_mrr(retrieved_chunk_ids, sample.expected_chunk_ids)
                ndcg = compute_ndcg_at_k(retrieved_chunk_ids, sample.expected_chunk_ids, k=chosen_k)
            else:
                # OOD queries: 0 expected chunks
                recall = 1.0 if not retrieved_chunk_ids else 0.0
                mrr = 0.0
                ndcg = 0.0

            results[strat_name]["ks"].append(chosen_k)
            results[strat_name]["chunk_counts"].append(len(retrieved))
            results[strat_name]["context_chars"].append(total_chars)
            results[strat_name]["context_tokens"].append(approx_tokens)
            results[strat_name]["retrieval_latencies"].append(total_retrieval_ms)
            results[strat_name]["end_to_end_latencies"].append(total_e2e_ms)
            results[strat_name]["c_q_list"].append(c_q)
            results[strat_name]["r_c_list"].append(r_c)

            if sample.expected_chunk_ids:
                results[strat_name]["recalls"].append(recall)
                results[strat_name]["mrrs"].append(mrr)
                results[strat_name]["ndcgs"].append(ndcg)

            query_record["strategies"][strat_name] = {
                "chosen_k": chosen_k,
                "retrieved_count": len(retrieved),
                "recall": round(recall, 4) if sample.expected_chunk_ids else "N/A (OOD)",
                "mrr": round(mrr, 4) if sample.expected_chunk_ids else "N/A (OOD)",
                "ndcg": round(ndcg, 4) if sample.expected_chunk_ids else "N/A (OOD)",
                "context_tokens": approx_tokens,
                "retrieval_ms": round(total_retrieval_ms, 1),
            }

        per_query_records.append(query_record)

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 90)
    print("COMPARATIVE EVALUATION RESULTS (10 REAL QUERIES ON QDRANT CLOUD)")
    print("=" * 90)
    print(f"{'Strategy':<20} | {'Avg K':>5} | {'Avg Chunks':>10} | {'Avg Tokens':>10} | {'Recall@K':>8} | {'MRR':>6} | {'nDCG':>6} | {'Retr(ms)':>8} | {'E2E(ms)':>8}")
    print("-" * 90)

    for strat in strategy_names:
        r = results[strat]
        avg_k = sum(r["ks"]) / len(r["ks"])
        avg_chunks = sum(r["chunk_counts"]) / len(r["chunk_counts"])
        avg_tokens = sum(r["context_tokens"]) / len(r["context_tokens"])
        avg_recall = sum(r["recalls"]) / len(r["recalls"]) if r["recalls"] else 0.0
        avg_mrr = sum(r["mrrs"]) / len(r["mrrs"]) if r["mrrs"] else 0.0
        avg_ndcg = sum(r["ndcgs"]) / len(r["ndcgs"]) if r["ndcgs"] else 0.0
        avg_ret_lat = sum(r["retrieval_latencies"]) / len(r["retrieval_latencies"])
        avg_e2e_lat = sum(r["end_to_end_latencies"]) / len(r["end_to_end_latencies"])

        print(
            f"{strat:<20} | {avg_k:>5.1f} | {avg_chunks:>10.1f} | {avg_tokens:>10.1f} | "
            f"{avg_recall:>8.3f} | {avg_mrr:>6.3f} | {avg_ndcg:>6.3f} | "
            f"{avg_ret_lat:>8.1f} | {avg_e2e_lat:>8.1f}"
        )
    print("=" * 90)

    # Print detailed per-query table
    print("\n" + "=" * 105)
    print("PER-QUERY BREAKDOWN")
    print("=" * 105)
    print(f"{'QID':<24} | {'Cat':<11} | {'C_q':>5} | {'R_c':>5} | {'Fixed-K (K/Rec)':>15} | {'Comp-Only (K/Rec)':>17} | {'Full-Dyn (K/Rec)':>16} | {'Paper (K/Rec)':>14}")
    print("-" * 105)
    for rec in per_query_records:
        s = rec["strategies"]
        f5_str = f"K={s['fixed_k5']['chosen_k']} (R={s['fixed_k5']['recall']})"
        co_str = f"K={s['complexity_only']['chosen_k']} (R={s['complexity_only']['recall']})"
        fd_str = f"K={s['full_dynamic_k']['chosen_k']} (R={s['full_dynamic_k']['recall']})"
        pa_str = f"K={s['paper_formula']['chosen_k']} (R={s['paper_formula']['recall']})"
        print(f"{rec['qid']:<24} | {rec['category']:<11} | {rec['c_q']:>5.2f} | {rec['r_c']:>5.2f} | {f5_str:>15} | {co_str:>17} | {fd_str:>16} | {pa_str:>14}")
    print("=" * 105)


if __name__ == "__main__":
    asyncio.run(evaluate_all_strategies())
