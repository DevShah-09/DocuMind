"""
ingestion_scale_audit.py — Ingestion, Reliability & Scale Benchmark
==================================================================
Priority G & H implementation covering:
  1. Document ingestion edge cases:
     - Invalid file extensions (.exe, .sh, .txt)
     - Empty files (0 bytes)
     - Oversized files (> 20MB)
     - Corrupted PDF binary payload
     - Valid format handling
  2. Scale testing:
     - Synthetic corpus scaling (100, 500, 1000, 5000, 10000 points)
     - Query latency percentiles (p50, p90, p95, p99) vs index size
     - Controller overhead and expansion evaluation
     - Throughput (QPS) and memory consumption
"""

import os
import sys
import io
import csv
import time
import uuid
import psutil
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.core.security import get_current_user
from app.retrieval.dynamic_top_k import DynamicTopKController, TopKConfig
from app.schemas.retrieval import ParsedQuery


class DummyChunk:
    def __init__(self, score: float, document_id: str, chunk_id: str, metadata: dict = None):
        self.score = score
        self.id = chunk_id
        self.payload = {
            "document_id": document_id,
            "filename": f"{document_id}.pdf",
            "document_type": (metadata or {}).get("document_type", "document"),
            "year": (metadata or {}).get("year"),
            "text": "sample chunk content",
        }


INGESTION_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports", "comprehensive_evaluation", "ingestion_test_results.csv")
)
SCALE_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports", "comprehensive_evaluation", "scale_test_results.csv")
)

ingestion_records: List[Dict[str, Any]] = []
scale_records: List[Dict[str, Any]] = []


def record_ingestion(test_name: str, passed: bool, severity: str, details: str):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_name": test_name,
        "status": "PASSED" if passed else "FAILED",
        "severity": severity,
        "details": details,
    }
    ingestion_records.append(record)
    print(f"[{'PASS' if passed else 'FAIL'}] [Ingestion] {test_name}: {details}")


def test_ingestion_edge_cases():
    print("\n--- Testing Ingestion Edge Cases ---")
    client = TestClient(app)
    mock_user_id = str(uuid.uuid4())
    app.dependency_overrides[get_current_user] = lambda: {"sub": mock_user_id, "role": "authenticated"}

    # 1. Unallowed extension: .exe
    res_exe = client.post(
        "/documents/upload",
        files={"file": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-msdownload")},
    )
    exe_blocked = res_exe.status_code == 400 and "not allowed" in res_exe.text.lower()
    record_ingestion("unallowed_extension_exe_rejected", exe_blocked, "HIGH", f"HTTP {res_exe.status_code}: {res_exe.text}")

    # 2. Unallowed extension: .txt
    res_txt = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"plain text notes", "text/plain")},
    )
    txt_blocked = res_txt.status_code == 400 and "not allowed" in res_txt.text.lower()
    record_ingestion("unallowed_extension_txt_rejected", txt_blocked, "MEDIUM", f"HTTP {res_txt.status_code}: {res_txt.text}")

    # 3. Empty file (0 bytes)
    res_empty = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    empty_blocked = res_empty.status_code == 400 and "empty" in res_empty.text.lower()
    record_ingestion("empty_file_rejected", empty_blocked, "HIGH", f"HTTP {res_empty.status_code}: {res_empty.text}")

    # 4. Oversized file (> 20MB)
    # Using 21MB virtual buffer
    oversized_bytes = b"0" * (21 * 1024 * 1024)
    res_oversize = client.post(
        "/documents/upload",
        files={"file": ("huge.pdf", oversized_bytes, "application/pdf")},
    )
    oversize_blocked = res_oversize.status_code == 400 and "too large" in res_oversize.text.lower()
    record_ingestion("oversized_file_rejected", oversize_blocked, "HIGH", f"HTTP {res_oversize.status_code}: {res_oversize.text}")

    # 5. Corrupted PDF byte stream
    res_corrupt = client.post(
        "/documents/upload",
        files={"file": ("corrupt.pdf", b"%PDF-1.4 garbage corrupted byte stream not a valid pdf", "application/pdf")},
    )
    # Orchestrator catches extraction errors and raises 500 or 400 safely
    corrupt_safe = res_corrupt.status_code in (400, 500)
    record_ingestion("corrupted_pdf_safe_error_handling", corrupt_safe, "MEDIUM", f"HTTP {res_corrupt.status_code}: Handled without unhandled crash")

    app.dependency_overrides.clear()


def test_scale_benchmarks():
    print("\n--- Running Scale Benchmark ---")
    scale_points = [100, 500, 1000, 5000, 10000]
    dim = 768  # BGE embedding dimension
    rng = np.random.default_rng(42)

    controller = DynamicTopKController(TopKConfig(formula_mode="paper", similarity_floor=0.35, enable_abstention=True))
    process = psutil.Process(os.getpid())

    for n_points in scale_points:
        mem_before_mb = process.memory_info().rss / (1024 * 1024)
        # Generate synthetic unit vectors
        raw_vecs = rng.standard_normal((n_points, dim), dtype=np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        vecs = raw_vecs / norms

        query_vec = rng.standard_normal(dim, dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec)

        # Measure 100 retrieval probe operations
        n_queries = 100
        latencies_ms = []
        for _ in range(n_queries):
            t0 = time.perf_counter()
            # Cosine similarity matrix multiplication
            sims = np.dot(vecs, query_vec)
            top_indices = np.argpartition(sims, -14)[-14:]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
            top_sims = sims[top_indices]

            # Feed to Dynamic-K controller
            dummy_chunks = [
                DummyChunk(
                    score=float(score),
                    document_id=f"doc_{idx % 10}",
                    chunk_id=f"chunk_{idx}",
                    metadata={"document_type": "invoice", "year": 2024},
                )
                for idx, score in zip(top_indices, top_sims)
            ]
            parsed = ParsedQuery(raw_query="synthetic query for benchmark", semantic_query="synthetic query")
            decision = controller.decide("synthetic query", dummy_chunks, parsed)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

        mem_after_mb = process.memory_info().rss / (1024 * 1024)
        p50 = float(np.percentile(latencies_ms, 50))
        p90 = float(np.percentile(latencies_ms, 90))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        mean_lat = float(np.mean(latencies_ms))
        std_lat = float(np.std(latencies_ms))
        throughput_qps = float(n_queries / (sum(latencies_ms) / 1000.0))

        record = {
            "scale_chunks": n_points,
            "vector_dimension": dim,
            "queries_evaluated": n_queries,
            "p50_latency_ms": round(p50, 3),
            "p90_latency_ms": round(p90, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "mean_latency_ms": round(mean_lat, 3),
            "std_latency_ms": round(std_lat, 3),
            "throughput_qps": round(throughput_qps, 1),
            "memory_usage_mb": round(mem_after_mb, 2),
        }
        scale_records.append(record)
        print(f"Scale: {n_points:5d} vectors | p50: {p50:6.3f}ms | p95: {p95:6.3f}ms | Throughput: {throughput_qps:6.1f} QPS | Mem: {mem_after_mb:6.1f}MB")


def save_csvs():
    os.makedirs(os.path.dirname(INGESTION_CSV), exist_ok=True)
    with open(INGESTION_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "test_name", "status", "severity", "details"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ingestion_records:
            writer.writerow(r)
    print(f"\n[INGESTION] Saved results to {INGESTION_CSV}")

    with open(SCALE_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scale_chunks", "vector_dimension", "queries_evaluated",
            "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "mean_latency_ms", "std_latency_ms", "throughput_qps", "memory_usage_mb"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scale_records:
            writer.writerow(r)
    print(f"[SCALE] Saved scale results to {SCALE_CSV}")


if __name__ == "__main__":
    test_ingestion_edge_cases()
    test_scale_benchmarks()
    save_csvs()
