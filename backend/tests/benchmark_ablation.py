"""
benchmark_ablation.py — Empirical Comparison of Retrieval Governors
===================================================================
Ablation study script comparing:
  - Static K = 2  (Minimal retrieval baseline)
  - Static K = 6  (Default medium baseline)
  - Static K = 12 (High recall baseline)
  - Dynamic Top-K Governor (Two-phase adaptive governor)

Simulates 4 benchmark query archetypes:
  1. Factual Single-Lookup: "What is my PAN card number?"
  2. Constraint Document Search: "Find my SBI bank statement from January 2024"
  3. Multi-Period Comparison: "Compare my electricity bills between Jan 2024 and March 2024"
  4. Complex Aggregation: "Total expense breakdown across all medical reports and explain increases"

Run with:
  python tests/benchmark_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.dynamic_k import dynamic_k_governor
from app.schemas.retrieval import ParsedQuery


@dataclass
class BenchmarkQuery:
    name: str
    query: str
    parsed: ParsedQuery
    mock_chunks: list


def run_ablation():
    # ── Test Queries ────────────────────────────────────────────────────────────
    queries = [
        BenchmarkQuery(
            name="1. Factual (PAN Lookup)",
            query="What is my PAN card number?",
            parsed=ParsedQuery(
                document_type="pan_card",
                year=None,
                month=None,
                entities=["PAN"],
                semantic_query="What is my PAN card number?",
                raw_query="What is my PAN card number?",
            ),
            mock_chunks=[
                type("Chunk", (), {"score": 0.95, "document_id": "pan_1", "payload": {"document_type": "pan_card", "text_content": "PAN ABCDE1234F"}, "text": "PAN ABCDE1234F"}),
                type("Chunk", (), {"score": 0.92, "document_id": "pan_1", "payload": {"document_type": "pan_card", "text_content": "Permanent Account Number ABCDE1234F"}, "text": "PAN ABCDE1234F"}),
                type("Chunk", (), {"score": 0.42, "document_id": "other_1", "payload": {"document_type": "bill", "text_content": "bill"}, "text": "bill"}),
                type("Chunk", (), {"score": 0.38, "document_id": "other_2", "payload": {"document_type": "bill", "text_content": "bill"}, "text": "bill"}),
            ]
        ),
        BenchmarkQuery(
            name="2. Specific Filter (Jan Bill)",
            query="Electricity bill January 2024",
            parsed=ParsedQuery(
                document_type="electricity_bill",
                year=2024,
                month="January",
                entities=[],
                semantic_query="electricity bill",
                raw_query="Electricity bill January 2024",
            ),
            mock_chunks=[
                type("Chunk", (), {"score": 0.91, "document_id": "bill_jan", "payload": {"document_type": "electricity_bill", "year": 2024, "month": "January", "text_content": "Total Rs 2400"}, "text": "bill"}),
                type("Chunk", (), {"score": 0.88, "document_id": "bill_jan", "payload": {"document_type": "electricity_bill", "year": 2024, "month": "January", "text_content": "Due date Jan 25"}, "text": "bill"}),
                type("Chunk", (), {"score": 0.70, "document_id": "bill_feb", "payload": {"document_type": "electricity_bill", "year": 2024, "month": "February", "text_content": "Feb"}, "text": "bill"}),
            ]
        ),
        BenchmarkQuery(
            name="3. Comparative (Jan vs Mar)",
            query="Compare my electricity expenses from January to March 2024 and explain the difference",
            parsed=ParsedQuery(
                document_type="electricity_bill",
                year=2024,
                month=None,
                entities=[],
                semantic_query="compare electricity expenses January March 2024",
                raw_query="Compare my electricity expenses from January to March 2024 and explain the difference",
            ),
            mock_chunks=[
                type("Chunk", (), {"score": 0.85, "document_id": "bill_jan", "payload": {"document_type": "electricity_bill", "year": 2024, "month": "January", "text_content": "Jan bill Rs 1800"}, "text": "bill"}),
                type("Chunk", (), {"score": 0.82, "document_id": "bill_mar", "payload": {"document_type": "electricity_bill", "year": 2024, "month": "March", "text_content": "March bill Rs 3400"}, "text": "bill"}),
                type("Chunk", (), {"score": 0.79, "document_id": "bill_feb", "payload": {"document_type": "electricity_bill", "year": 2024, "month": "February", "text_content": "Feb bill Rs 2100"}, "text": "bill"}),
                type("Chunk", (), {"score": 0.75, "document_id": "bill_jan", "payload": {"document_type": "electricity_bill", "year": 2024, "month": "January", "text_content": "Consumption units"}, "text": "bill"}),
            ]
        ),
        BenchmarkQuery(
            name="4. Analytical Aggregation",
            query="Total expenditure across all medical reports and analyze why hospital expenses increased",
            parsed=ParsedQuery(
                document_type="medical_report",
                year=None,
                month=None,
                entities=["hospital"],
                semantic_query="total expenditure medical reports hospital expenses increased",
                raw_query="Total expenditure across all medical reports and analyze why hospital expenses increased",
            ),
            mock_chunks=[
                type("Chunk", (), {"score": 0.82, "document_id": "med_1", "payload": {"document_type": "medical_report", "text_content": "Pharmacy bill Rs 4500"}, "text": "med"}),
                type("Chunk", (), {"score": 0.80, "document_id": "med_2", "payload": {"document_type": "medical_report", "text_content": "Surgery bill Rs 65000"}, "text": "med"}),
                type("Chunk", (), {"score": 0.77, "document_id": "med_3", "payload": {"document_type": "medical_report", "text_content": "Consultation Rs 1500"}, "text": "med"}),
                type("Chunk", (), {"score": 0.75, "document_id": "med_4", "payload": {"document_type": "medical_report", "text_content": "Diagnostic lab tests"}, "text": "med"}),
            ]
        ),
    ]

    print("\n" + "=" * 105)
    print(f"{'BENCHMARK ABLATION: STATIC K vs DYNAMIC TOP-K GOVERNOR':^105}")
    print("=" * 105)
    header = f"{'Query Archetype':<28} | {'Static K=2':<10} | {'Static K=6':<10} | {'Static K=12':<11} | {'Dynamic K':<10} | {'C_q':<6} | {'R_c':<6} | {'Knee Gap':<8} | {'Docs'}"
    print(header)
    print("-" * 105)

    token_savings = []
    for q in queries:
        dec = dynamic_k_governor.decide(q.query, q.mock_chunks, q.parsed)
        
        # Approx chars/tokens at ~600 tokens per chunk
        static_12_tokens = 12 * 600
        dyn_tokens = dec.recommended_k * 600
        savings_pct = round(((static_12_tokens - dyn_tokens) / static_12_tokens) * 100, 1)
        token_savings.append(savings_pct)

        row = (
            f"{q.name:<28} | "
            f"{'2 chunks':<10} | "
            f"{'6 chunks':<10} | "
            f"{'12 chunks':<11} | "
            f"{f'{dec.recommended_k} chunks':<10} | "
            f"{dec.c_q:<6.2f} | "
            f"{dec.r_c:<6.2f} | "
            f"{dec.knee_gap:<8.2f} | "
            f"{dec.target_documents} doc(s)"
        )
        print(row)

    print("-" * 105)
    avg_savings = sum(token_savings) / len(token_savings)
    print(f"Average context token reduction vs Static K=12: ~{avg_savings:.1f}%\n")


if __name__ == "__main__":
    run_ablation()
