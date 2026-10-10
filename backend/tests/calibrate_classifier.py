"""
calibrate_classifier.py — BGE Local Classifier Calibration Tool
================================================================
Runs the local BGE classifier against real PDF files from a folder and
outputs a diagnostic table so you can pick safe SCORE_THRESHOLD and
MARGIN_THRESHOLD values.

Usage:
    python tests/calibrate_classifier.py <folder_of_pdfs>

    # To label documents with their ground-truth category:
    python tests/calibrate_classifier.py <folder> --labels labels.csv

    # Sweep threshold grid to find the best combination:
    python tests/calibrate_classifier.py <folder> --sweep

Where labels.csv is:
    filename,actual_category
    electricity_jan.pdf,electricity_bill
    pan_card.pdf,pan_card
    ...

Output columns:
    filename | actual | predicted | top_score | second_score | margin | accepted | correct
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── Bootstrap sys.path so imports resolve from backend/ ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF

from app.services.local_classifier import (
    local_classifier,
    SCORE_THRESHOLD,
    MARGIN_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path, max_chars: int = 3000) -> str:
    """Extract leading text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []
        for page in doc:
            t = page.get_text("text").strip()
            if t:
                pages_text.append(t)
        doc.close()
        full = "\n\n".join(pages_text).strip()
        return full[:max_chars]
    except Exception as e:
        print(f"  [WARN] Could not read {pdf_path.name}: {e}")
        return ""


def load_labels(csv_path: Path) -> Dict[str, str]:
    """Load filename → actual_category mapping from a CSV."""
    labels: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"].strip()] = row["actual_category"].strip()
    return labels


def run_calibration(
    folder: Path,
    labels: Dict[str, str],
    score_threshold: float,
    margin_threshold: float,
) -> List[dict]:
    """Run classifier on all PDFs in folder. Return list of result dicts."""
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDF files found in {folder}")
        sys.exit(1)

    results = []
    for pdf in pdfs:
        text = extract_text(pdf)
        if not text:
            continue

        r = local_classifier.classify(
            text,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
        )
        actual = labels.get(pdf.name, "?")
        correct = (actual == r.category) if actual != "?" else None

        results.append({
            "filename":     pdf.name,
            "actual":       actual,
            "predicted":    r.category,
            "top_score":    r.top_score,
            "second_score": r.second_score,
            "margin":       r.margin,
            "accepted":     r.accepted,
            "correct":      correct,
        })

    return results


def print_table(results: List[dict], score_threshold: float, margin_threshold: float) -> None:
    """Print diagnostic table to stdout."""
    col_w = [32, 18, 18, 9, 9, 9, 9, 8]
    header = ["filename", "actual", "predicted", "top", "second", "margin", "accepted", "correct"]
    sep = "-" * (sum(col_w) + len(col_w) * 3)

    print(f"\n{'BGE CLASSIFIER CALIBRATION':^{len(sep)}}")
    print(f"SCORE_THRESHOLD={score_threshold}  MARGIN_THRESHOLD={margin_threshold}")
    print(sep)
    print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(sep)

    accepted_count = 0
    correct_count  = 0
    labeled_count  = 0

    for r in results:
        accepted_str = "YES" if r["accepted"] else "no"
        correct_str  = ("✓" if r["correct"] else "✗") if r["correct"] is not None else "?"

        row = [
            r["filename"][:col_w[0]].ljust(col_w[0]),
            r["actual"][:col_w[1]].ljust(col_w[1]),
            r["predicted"][:col_w[2]].ljust(col_w[2]),
            f"{r['top_score']:.4f}".ljust(col_w[3]),
            f"{r['second_score']:.4f}".ljust(col_w[4]),
            f"{r['margin']:.4f}".ljust(col_w[5]),
            accepted_str.ljust(col_w[6]),
            correct_str.ljust(col_w[7]),
        ]
        print("  ".join(row))

        if r["accepted"]:
            accepted_count += 1
        if r["correct"] is not None:
            labeled_count += 1
            if r["correct"]:
                correct_count += 1

    print(sep)
    n = len(results)
    print(f"\nSummary:")
    print(f"  Total PDFs scanned : {n}")
    print(f"  Local accept rate  : {accepted_count}/{n} ({100*accepted_count/n:.1f}%)")
    if labeled_count > 0:
        accepted_correct = sum(
            1 for r in results if r["accepted"] and r["correct"]
        )
        accepted_wrong = sum(
            1 for r in results if r["accepted"] and r["correct"] is False
        )
        print(f"  Labeled documents  : {labeled_count}")
        print(f"  Overall accuracy   : {correct_count}/{labeled_count} ({100*correct_count/labeled_count:.1f}%)")
        print(f"  Local accept: correct={accepted_correct}  wrong={accepted_wrong}")
    print()


def sweep_thresholds(folder: Path, labels: Dict[str, str]) -> None:
    """Grid search over score/margin threshold combinations."""
    score_values  = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
    margin_values = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

    labeled = {k: v for k, v in labels.items() if v != "?"}
    if not labeled:
        print("[SWEEP] No ground-truth labels provided — cannot compute accuracy.")
        print("        Run with --labels labels.csv to enable sweep.\n")
        return

    print(f"\n{'THRESHOLD SWEEP':=^80}")
    print(f"{'score_thr':>10}  {'margin_thr':>10}  {'accept%':>8}  {'accuracy%':>10}  {'wrong_accepts':>14}")
    print("-" * 60)

    best_row = None
    best_f1  = -1.0

    for s_thr in score_values:
        for m_thr in margin_values:
            results = run_calibration(folder, labels, s_thr, m_thr)
            n = len(results)
            if n == 0:
                continue

            accepted      = [r for r in results if r["accepted"]]
            accept_rate   = len(accepted) / n
            wrong_accepts = sum(1 for r in accepted if r["correct"] is False)
            correct_accepts = sum(1 for r in accepted if r["correct"] is True)
            # Precision of the local path
            prec = correct_accepts / len(accepted) if accepted else 0.0
            print(
                f"{s_thr:>10.2f}  {m_thr:>10.2f}  "
                f"{100*accept_rate:>7.1f}%  {100*prec:>9.1f}%  {wrong_accepts:>14}"
            )

            if prec > best_f1 and wrong_accepts == 0:
                best_f1 = prec
                best_row = (s_thr, m_thr, accept_rate, prec)

    print()
    if best_row:
        s, m, ar, pr = best_row
        print(f"[SWEEP] Recommended thresholds (zero wrong-accepts, best precision):")
        print(f"   SCORE_THRESHOLD  = {s}")
        print(f"   MARGIN_THRESHOLD = {m}")
        print(f"   Local accept rate: {100*ar:.1f}%  |  Precision: {100*pr:.1f}%")
    else:
        print("[SWEEP] No threshold combination achieved zero wrong-accepts on this dataset.")
        print("        Consider adding more/better labeled documents, or lower your precision requirement.")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the BGE local document classifier against real PDFs."
    )
    parser.add_argument("folder", type=Path, help="Folder containing PDF files")
    parser.add_argument(
        "--labels", type=Path, default=None,
        help="CSV with columns: filename,actual_category"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=SCORE_THRESHOLD,
        help=f"Score threshold (default: {SCORE_THRESHOLD})"
    )
    parser.add_argument(
        "--margin-threshold", type=float, default=MARGIN_THRESHOLD,
        help=f"Margin threshold (default: {MARGIN_THRESHOLD})"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Grid search over threshold values (requires --labels)"
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"[ERROR] Not a directory: {args.folder}")
        sys.exit(1)

    labels: Dict[str, str] = {}
    if args.labels:
        if not args.labels.exists():
            print(f"[ERROR] Labels file not found: {args.labels}")
            sys.exit(1)
        labels = load_labels(args.labels)
        print(f"[INFO] Loaded {len(labels)} labels from {args.labels.name}")

    if args.sweep:
        sweep_thresholds(args.folder, labels)
    else:
        results = run_calibration(
            args.folder, labels,
            args.score_threshold,
            args.margin_threshold,
        )
        print_table(results, args.score_threshold, args.margin_threshold)


if __name__ == "__main__":
    main()
