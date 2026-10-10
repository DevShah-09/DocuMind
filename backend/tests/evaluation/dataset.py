"""
dataset.py — Evaluation Dataset Definitions
============================================
Provides a small, clearly labelled demonstration dataset for evaluating the
four retrieval strategies:

  A. Fixed K = 5
  B. Fixed K = 12
  C. Complexity-only Dynamic K
  D. Complexity + Confidence Dynamic K   (full system)

IMPORTANT — Synthetic data notice
-----------------------------------
All queries, document IDs, and relevance labels below are SYNTHETIC.
They are provided solely to demonstrate the evaluation harness format and
confirm that the metrics pipeline executes end-to-end.  They are NOT
real-world retrieval measurements and MUST NOT be interpreted as such.

To produce real-world results, replace the entries in DEMO_DATASET with
queries drawn from your actual document corpus, annotate ground-truth
relevant_chunk_ids from your Qdrant collection, and store expected answers
or answer criteria.

Dataset schema
--------------
Each EvalSample has:
  - qid:                 Unique question ID (e.g. "Q001")
  - question:            Raw query text
  - expected_answer:     Reference answer or None (used for faithfulness eval)
  - relevant_doc_ids:    Set of document_ids that contain the answer
  - relevant_chunk_ids:  Ordered list of chunk_ids by descending relevance
                         (first = most relevant).  Used for MRR and nDCG.
  - category:            One of: factual | comparative | aggregative |
                                  date_constrained | multi_document
  - complexity_label:    One of: low | medium | high
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class EvalSample:
    """One evaluation question with ground-truth labels."""
    qid:                str
    question:           str
    expected_answer:    Optional[str]
    relevant_doc_ids:   Set[str]
    relevant_chunk_ids: List[str]          # ordered: most → least relevant
    category:           str                # factual | comparative | aggregative | …
    complexity_label:   str                # low | medium | high


# ---------------------------------------------------------------------------
# DEMONSTRATION DATASET  (synthetic — see module docstring)
# ---------------------------------------------------------------------------
# To evaluate against a real corpus:
#   1. Replace these entries with actual queries from your document set.
#   2. Populate relevant_chunk_ids from your Qdrant `document_chunks` collection.
#   3. Run: python -m tests.evaluation.run_evaluation --dataset path/to/real.json
# ---------------------------------------------------------------------------

DEMO_DATASET: List[EvalSample] = [
    # ── Factual / Low-complexity ────────────────────────────────────────────
    EvalSample(
        qid="Q001",
        question="What is my PAN card number?",
        expected_answer="ABCDE1234F",
        relevant_doc_ids={"doc-pan-001"},
        relevant_chunk_ids=["chunk-pan-001-0", "chunk-pan-001-1"],
        category="factual",
        complexity_label="low",
    ),
    EvalSample(
        qid="Q002",
        question="What is my Aadhaar number?",
        expected_answer="1234 5678 9012",
        relevant_doc_ids={"doc-aadhaar-001"},
        relevant_chunk_ids=["chunk-aadhaar-001-0"],
        category="factual",
        complexity_label="low",
    ),
    EvalSample(
        qid="Q003",
        question="Show me my electricity bill for January 2024.",
        expected_answer=None,
        relevant_doc_ids={"doc-elec-jan-2024"},
        relevant_chunk_ids=["chunk-elec-jan-2024-0", "chunk-elec-jan-2024-1"],
        category="date_constrained",
        complexity_label="low",
    ),

    # ── Medium-complexity ───────────────────────────────────────────────────
    EvalSample(
        qid="Q004",
        question="What was my total electricity expenditure in 2023?",
        expected_answer=None,
        relevant_doc_ids={"doc-elec-jan-2023", "doc-elec-feb-2023", "doc-elec-dec-2023"},
        relevant_chunk_ids=[
            "chunk-elec-jan-2023-0",
            "chunk-elec-feb-2023-0",
            "chunk-elec-dec-2023-0",
        ],
        category="aggregative",
        complexity_label="medium",
    ),
    EvalSample(
        qid="Q005",
        question="Summarize my SBI bank statement from March 2024.",
        expected_answer=None,
        relevant_doc_ids={"doc-bank-sbi-mar-2024"},
        relevant_chunk_ids=["chunk-bank-sbi-mar-2024-0", "chunk-bank-sbi-mar-2024-1"],
        category="factual",
        complexity_label="medium",
    ),

    # ── High-complexity / Comparative ──────────────────────────────────────
    EvalSample(
        qid="Q006",
        question=(
            "Compare my electricity bills from January 2023 and January 2024 "
            "and explain why the amount changed."
        ),
        expected_answer=None,
        relevant_doc_ids={"doc-elec-jan-2023", "doc-elec-jan-2024"},
        relevant_chunk_ids=[
            "chunk-elec-jan-2023-0",
            "chunk-elec-jan-2024-0",
        ],
        category="comparative",
        complexity_label="high",
    ),
    EvalSample(
        qid="Q007",
        question=(
            "What are the differences between my 2022 and 2023 income tax returns?"
        ),
        expected_answer=None,
        relevant_doc_ids={"doc-itr-2022", "doc-itr-2023"},
        relevant_chunk_ids=["chunk-itr-2022-0", "chunk-itr-2023-0"],
        category="comparative",
        complexity_label="high",
    ),

    # ── Multi-document ──────────────────────────────────────────────────────
    EvalSample(
        qid="Q008",
        question="List all my insurance policies and their premium amounts.",
        expected_answer=None,
        relevant_doc_ids={"doc-insurance-001", "doc-insurance-002"},
        relevant_chunk_ids=[
            "chunk-insurance-001-0",
            "chunk-insurance-002-0",
        ],
        category="multi_document",
        complexity_label="medium",
    ),
    EvalSample(
        qid="Q009",
        question="Do I have any medical reports from 2024?",
        expected_answer=None,
        relevant_doc_ids={"doc-medical-001"},
        relevant_chunk_ids=["chunk-medical-001-0"],
        category="factual",
        complexity_label="low",
    ),
    EvalSample(
        qid="Q010",
        question=(
            "Compare my bank statements from 2022 and 2023 and calculate "
            "the average monthly balance for each year."
        ),
        expected_answer=None,
        relevant_doc_ids={"doc-bank-2022", "doc-bank-2023"},
        relevant_chunk_ids=[
            "chunk-bank-2022-0", "chunk-bank-2022-1",
            "chunk-bank-2023-0", "chunk-bank-2023-1",
        ],
        category="comparative",
        complexity_label="high",
    ),
]

# ── Development / test split ──────────────────────────────────────────────────
# Q001–Q007 are the development set (used to tune weights).
# Q008–Q010 are the held-out test set (never used for tuning).
DEV_QIDS  = {"Q001", "Q002", "Q003", "Q004", "Q005", "Q006", "Q007"}
TEST_QIDS = {"Q008", "Q009", "Q010"}

DEV_DATASET  = [s for s in DEMO_DATASET if s.qid in DEV_QIDS]
TEST_DATASET = [s for s in DEMO_DATASET if s.qid in TEST_QIDS]
