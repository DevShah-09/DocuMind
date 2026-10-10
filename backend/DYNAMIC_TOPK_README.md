# Confidence-Aware Dynamic Top-K Retrieval — DocuMind AI

## Table of Contents

1. [Overview](#1-overview)
2. [Algorithm and Formula](#2-algorithm-and-formula)
3. [Module Reference](#3-module-reference)
4. [Configuration Parameters](#4-configuration-parameters)
5. [Integration Points](#5-integration-points)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Running the Application](#7-running-the-application)
8. [Running Tests](#8-running-tests)
9. [Experimental Results](#9-experimental-results)
10. [Limitations and Open Questions](#10-limitations-and-open-questions)

---

## 1. Overview

DocuMind AI uses a RAG (Retrieval-Augmented Generation) pipeline to answer questions from a user's uploaded documents. A fixed retrieval budget (e.g. always fetch K=5 chunks) wastes context tokens on simple queries and under-retrieves on complex ones.

**Confidence-Aware Dynamic Top-K** solves this by choosing K per-query using two deterministic, LLM-free signals:

| Signal | Symbol | Meaning |
|---|---|---|
| Query Complexity Score | C_q ∈ [0, 1] | How much evidence the query logically requires |
| Retrieval Confidence Score | R_c ∈ [0, 1] | How sufficient the initial probe results are |

Simple factual queries with strong evidence retrieve fewer chunks. Complex comparative or aggregative queries with weak evidence retrieve more, up to a configurable maximum.

---

## 2. Algorithm and Formula

### 2.1 Phase 1 — Query Complexity (C_q)

Eight deterministic features are extracted from the raw query text and the structured `ParsedQuery` output:

```
f1  query_length        — normalised char count (damped to avoid length bias)
f2  entity_count        — distinct named entities from ParsedQuery.entities
f3  constraint_count    — doc_type + year + month + amount filters
f4  temporal_complexity — multi-period / date-range signals
f5  comparison_intent   — "compare", "vs", "difference", "growth", …
f6  aggregation_intent  — "total", "sum", "average", "history", …
f7  multi_subquestion   — multiple "?" or coordinating conjunctions
f8  reasoning_required  — "why", "explain", "analyze", "factors", …
```

Each feature is independently normalised to [0, 1]. The weighted sum gives C_q:

```
C_q = w1·f1 + w2·f2 + … + w8·f8     where Σ wi = 1.0
```

**Default weights:**

| Feature | Weight |
|---|---|
| f1 query_length | 0.05 |
| f2 entity_count | 0.15 |
| f3 constraint_count | 0.10 |
| f4 temporal_complexity | 0.20 |
| f5 comparison_intent | 0.20 |
| f6 aggregation_intent | 0.15 |
| f7 multi_subquestion | 0.05 |
| f8 reasoning_required | 0.10 |

Weights are explicit design parameters, not empirically proven optima. Tune them on held-out dev data only.

### 2.2 Phase 2 — Retrieval Confidence (R_c)

Four signals are computed from the initial probe chunk list:

```
S_top   — best cosine similarity score (clipped to [0, 1])
S_avg   — average similarity over the top min(N, 4) chunks
G_knee  — max consecutive score drop: max_i(score_i − score_{i+1})
E       — evidence alignment: fraction of query constraints satisfied by top-3 chunks
```

**Important**: Qdrant cosine similarity scores are NOT calibrated probabilities. They are treated as relative relevance proxies only.

**Low-score guard**: If `S_top < 0.50`, the G_knee contribution is zeroed. This prevents a sharp gap between two weak scores (e.g. 0.42 → 0.10) from falsely inflating R_c.

Convex combination (weights sum to 1.0):

```
R_c = W_stop · S_top  +  W_savg · S_avg  +  W_knee · G_knee  +  W_evidence · E
```

**Default weights:**

| Signal | Weight |
|---|---|
| W_stop (S_top) | 0.35 |
| W_savg (S_avg) | 0.25 |
| W_knee (G_knee) | 0.20 |
| W_evidence (E) | 0.20 |

### 2.3 Phase 3 — Dynamic K Controller

**Step 1 — Demand signals:**

```
complexity_demand  = C_q
confidence_demand  = 1 − R_c
```

**Step 2 — Combined demand (alpha + beta normalised to sum to 1.0):**

```
combined_demand = alpha · complexity_demand + beta · confidence_demand
```

Default: `alpha = 0.50`, `beta = 0.50`.

**Step 3 — Map to K range:**

```
raw_K = K_min + combined_demand × (K_max − K_min)
```

**Step 4 — Coverage safeguard:**

If the query is multi-part (comparison, aggregation, multi-temporal) AND `evidence_alignment < coverage_threshold (0.50)`, add `coverage_penalty` (default: 3) to `raw_K`. This prevents high vector similarity from masking incomplete metadata coverage.

**Step 5 — High-confidence factual shortcut:**

If all four conditions hold simultaneously:
- `C_q ≤ 0.25`
- `R_c ≥ 0.80`
- `knee_index ≤ 2`
- `knee_gap ≥ 0.20`

Then `recommended_K = K_min` (no expansion needed).

**Step 6 — Clamp:**

```
recommended_K = clamp(round(raw_K), K_min, K_max)
```

### 2.4 Document Diversity Allocation

After choosing `recommended_K`, the controller plans how to spread chunks across documents:

| Query type | target_documents | chunks_per_document |
|---|---|---|
| Comparative / multi-temporal | max(2, ceil(K / 2.5)) | ceil(K / target_docs) |
| Aggregative / general browse | max(3, ceil(K / 2.0)) | ceil(K / target_docs) |
| Factual / single-document | 1 | K |

The chat pipeline enforces this allocation through a diversity guard that selects the best chunks per document before final re-ranking.

### 2.5 Formula Summary

```
complexity_demand  = C_q
confidence_demand  = 1 − R_c
combined_demand    = 0.5 · C_q + 0.5 · (1 − R_c)     [alpha=beta=0.5]
raw_K              = K_min + combined_demand · (K_max − K_min)
raw_K             += coverage_penalty   if multi_part AND evidence_alignment < 0.5
recommended_K      = K_min              if shortcut conditions met
recommended_K      = clamp(round(raw_K), K_min, K_max)   otherwise
```

### 2.6 Research Paper Discrepancy Note

The task specification references a "Dynamic Top-K equation and associated table" from a research paper. That paper was not available during implementation. The formula above is an independently designed normalised adaptive budget formula consistent with the specification's description. It is explicitly documented as an **unresolved dependency** — conformance to any specific paper equation cannot be claimed.

---

## 3. Module Reference

### New modules created

| File | Purpose |
|---|---|
| `app/retrieval/query_complexity.py` | `QueryComplexityEstimator` — computes C_q from 8 features |
| `app/retrieval/retrieval_confidence.py` | `RetrievalConfidenceEstimator` — computes R_c from 4 signals |
| `app/retrieval/dynamic_top_k.py` | `DynamicTopKController` — combines C_q + R_c → K decision |

### Modified modules

| File | Change |
|---|---|
| `app/retrieval/dynamic_k.py` | Refactored as backward-compatible shim; imports from the three new modules |
| `app/api/chat.py` | Added structured telemetry logging; cleaner two-phase retrieval with explicit stop_reason |

### Evaluation framework

| File | Purpose |
|---|---|
| `tests/evaluation/__init__.py` | Package marker |
| `tests/evaluation/dataset.py` | `EvalSample` schema + 10-query synthetic demo dataset |
| `tests/evaluation/metrics.py` | Pure-Python Recall@K, MRR, nDCG, faithfulness proxy, efficiency metrics |
| `tests/evaluation/strategies.py` | Four strategy classes (A–D) |
| `tests/evaluation/run_evaluation.py` | CLI harness; runs all strategies and prints comparison report |

### Test files

| File | Coverage |
|---|---|
| `tests/test_query_complexity.py` | 26 tests — C_q edge cases, monotonicity, weights |
| `tests/test_retrieval_confidence.py` | 26 tests — R_c signals, knee detection, evidence alignment |
| `tests/test_dynamic_top_k_controller.py` | 32 tests — K bounds, diversity, shortcut, backward compat |
| `tests/test_integration_dynamic_k.py` | 20 tests — full pipeline, tenant isolation, auth, determinism |
| `tests/evaluation/test_metrics.py` | 40 tests — all metric functions |
| `tests/test_dynamic_k.py` | 8 tests — original test suite (unchanged, all passing) |

**Total: 152 tests, 0 failures.**

---

## 4. Configuration Parameters

### TopKConfig (dynamic_top_k.py)

| Parameter | Default | Description |
|---|---|---|
| `k_min` | 2 | Minimum retrieval quota; never goes below this |
| `k_max` | 15 | Hard upper bound on chunk retrieval |
| `alpha` | 0.50 | Weight for complexity_demand; auto-normalised with beta |
| `beta` | 0.50 | Weight for confidence_demand (1 − R_c) |
| `coverage_penalty` | 3 | Extra chunks for under-covered multi-part queries |
| `coverage_threshold` | 0.50 | evidence_alignment below this triggers the penalty |

### ComplexityConfig (query_complexity.py)

| Parameter | Default | Description |
|---|---|---|
| `weights` | See §2.1 table | Dict of 8 feature weights; auto-normalised |
| `length_min` | 10 | Characters mapped to f1 = 0.0 |
| `length_max` | 150 | Characters mapped to f1 = 1.0 |
| `entity_max` | 3 | Entity count mapped to f2 = 1.0 |
| `constraint_max` | 3 | Constraint count mapped to f3 = 1.0 |

### ConfidenceConfig (retrieval_confidence.py)

| Parameter | Default | Description |
|---|---|---|
| `w_stop` | 0.35 | Weight for S_top |
| `w_savg` | 0.25 | Weight for S_avg |
| `w_knee` | 0.20 | Weight for G_knee |
| `w_evidence` | 0.20 | Weight for evidence alignment E |
| `top_k_avg` | 4 | Number of top chunks averaged for S_avg |
| `min_score_guard` | 0.50 | S_top below this → G_knee contribution zeroed |

### ChatRequest (schemas/retrieval.py)

| Field | Default | Description |
|---|---|---|
| `dynamic_k_enabled` | `true` | Set to `false` for static K = `limit × 4` fallback |
| `limit` | 5 | Static K when dynamic_k_enabled = false |
| `score_threshold` | 0.10 | Minimum similarity score for retrieval |

---

## 5. Integration Points

The pipeline in `app/api/chat.py` follows these steps on every POST `/chat` request:

```
1.  Auth check — extract user_id from JWT; abort if missing
2.  Greeting shortcut — return canned response for conversational inputs
3.  Parse query + embed in parallel — QueryParser + EmbeddingService
4.  Build metadata filters — document_type, year, month
5.  Initial probe retrieval — fetch DYNAMIC_K_PROBE_SIZE=6 chunks from Qdrant
    └─ user_id filter ALWAYS applied (tenant isolation enforced at Qdrant layer)
6.  Retry without doc-type filter if probe returns nothing
7.  Legacy fallback — query user_documents collection if still empty
8.  Dynamic K decision — DynamicKGovernor.decide(query, probe_chunks, parsed)
    ├─ C_q computed via QueryComplexityEstimator
    └─ R_c computed via RetrievalConfidenceEstimator
9.  Expand retrieval — if recommended_k > probe size, fetch recommended_k×2 chunks
    └─ user_id filter ALWAYS applied on expansion
10. Document Diversity Guard — allocate target_documents × chunks_per_document
11. Intent detection — analysis query vs. document fetch
12. Build source list — one ChatSource per unique document
13. Generate answer — LLM call (full RAG) or summary string (fetch-only)
14. Structured telemetry — _log_retrieval_telemetry() at INFO level
15. Return ChatResponse
```

**Security invariants preserved:**
- `user_id` is applied to every Qdrant `search_chunks()` call, including probe and expansion.
- User isolation is never relaxed to improve recall.
- Internal scores and chunk metadata are not exposed to clients beyond the `DynamicKDiagnostics` summary in the response.

**Backward compatibility preserved:**
- `dynamic_k_enabled=false` falls back to the original static path.
- `DynamicKGovernor` in `dynamic_k.py` is a facade with the same public interface.
- All existing API response schemas (`ChatResponse`, `DynamicKDiagnostics`, `ChatSource`) are unchanged.

---

## 6. Evaluation Framework

### Dataset format

Each `EvalSample` in `tests/evaluation/dataset.py` has:

```python
EvalSample(
    qid="Q001",                          # unique question ID
    question="What is my PAN number?",   # raw query text
    expected_answer="ABCDE1234F",        # reference answer or None
    relevant_doc_ids={"doc-pan-001"},    # ground-truth document IDs
    relevant_chunk_ids=["chunk-pan-001-0"],  # ordered by descending relevance
    category="factual",                  # factual|comparative|aggregative|…
    complexity_label="low",              # low|medium|high
)
```

**Dataset splits:**
- `DEV_DATASET` — Q001–Q007 (7 queries); use for weight tuning only
- `TEST_DATASET` — Q008–Q010 (3 queries); held-out, never used for tuning

> **⚠ Synthetic data notice:** The demo dataset is entirely synthetic. All queries, document IDs, and relevance labels are illustrative only. Results from this dataset must not be reported as real-world findings.

**To build a real evaluation dataset:**
1. Collect representative queries from your actual user base.
2. For each query, identify the relevant `chunk_id` values from your Qdrant `document_chunks` collection.
3. Annotate expected answers or answer criteria.
4. Save as a JSON array matching the `EvalSample` schema.
5. Run: `python -m tests.evaluation.run_evaluation --split all --output results.json`

### Strategies evaluated

| Strategy | Description |
|---|---|
| A. `fixed_k5` | Always retrieve K = 5 chunks |
| B. `fixed_k12` | Always retrieve K = 12 chunks |
| C. `complexity_only` | Dynamic K using C_q only (beta = 0) |
| D. `full_dynamic_k` | Dynamic K using C_q + R_c (alpha = beta = 0.5) |

All four strategies operate on the same chunk pool and the same score threshold, so differences are attributable to K selection only.

---

## 7. Running the Application

```powershell
# From the backend/ directory with venv activated:
cd backend
.\.venv\Scripts\activate

# Start the FastAPI server:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The chat endpoint is available at `POST http://localhost:8000/chat`.

**Example request:**

```json
{
  "query": "Compare my electricity bills from January 2023 and January 2024.",
  "limit": 5,
  "score_threshold": 0.10,
  "dynamic_k_enabled": true
}
```

**Example response fields added by this feature:**

```json
{
  "answer": "...",
  "sources": [...],
  "dynamic_k": {
    "recommended_k": 10,
    "target_documents": 2,
    "chunks_per_document": 5,
    "c_q": 0.55,
    "r_c": 0.42,
    "s_top": 0.85,
    "s_avg": 0.79,
    "knee_gap": 0.31,
    "knee_index": 2,
    "evidence_alignment": 0.75,
    "features": { "f5_comparison": 1.0, "f4_temporal": 1.0, ... },
    "explanation": "C_q=0.55(Med), R_c=0.42(Med) [S_top=0.85, knee=0.31@2] → demand=0.57 → K=10 across 2 doc(s) [combined demand formula]"
  }
}
```

To disable Dynamic K (fall back to static retrieval):

```json
{ "query": "...", "dynamic_k_enabled": false }
```

---

## 8. Running Tests

```powershell
cd backend

# All new + existing tests (152 total):
& ".\.venv\Scripts\python.exe" -m pytest tests/test_query_complexity.py tests/test_retrieval_confidence.py tests/test_dynamic_top_k_controller.py tests/test_integration_dynamic_k.py tests/evaluation/test_metrics.py tests/test_dynamic_k.py -v

# Original test suite only (regression check):
& ".\.venv\Scripts\python.exe" -m pytest tests/test_dynamic_k.py -v

# Evaluation harness — demo dataset, all splits:
& ".\.venv\Scripts\python.exe" -m tests.evaluation.run_evaluation --split all

# Evaluation harness — dev split only:
& ".\.venv\Scripts\python.exe" -m tests.evaluation.run_evaluation --split dev

# Evaluation harness — held-out test split:
& ".\.venv\Scripts\python.exe" -m tests.evaluation.run_evaluation --split test

# Save results to JSON:
& ".\.venv\Scripts\python.exe" -m tests.evaluation.run_evaluation --split all --output tests/evaluation/results.json
```

---

## 9. Experimental Results

The table below was produced by running the evaluation harness on the **synthetic demo dataset** (10 queries).

```
⚠  SYNTHETIC DATA — illustrative only, not real-world measurements
```

| Strategy | Recall@10 | MRR | nDCG@10 | Faith.(proxy) | Avg K | Δ vs K12 (%) | Empty% |
|---|---|---|---|---|---|---|---|
| fixed_k5 | 1.0000 | 1.0000 | 1.0000 | 0.5518 | 5.00 | 58.33% | 0% |
| fixed_k12 | 1.0000 | 1.0000 | 1.0000 | 0.5518 | 12.00 | 0.00% | 0% |
| complexity_only | 1.0000 | 1.0000 | 1.0000 | 0.5881 | 6.10 | 49.17% | 0% |
| full_dynamic_k | 1.0000 | 1.0000 | 1.0000 | 0.5658 | 5.20 | 56.67% | 0% |

**Why all retrieval metrics are 1.0:** The mock chunk pool in the evaluation harness places every relevant chunk at the top of the pool with scores ≥ 0.60. All four strategies retrieve K ≥ 5 and the demo dataset has at most 4 relevant chunks per query, so perfect recall/MRR/nDCG is expected by construction. These metrics will differentiate strategies when evaluated on a real Qdrant corpus with genuine retrieval noise.

**Faithfulness proxy note:** The proxy averages top-5 similarity scores. It is not equivalent to LLM-judge faithfulness. `complexity_only` scores slightly higher (0.588) because it selects a wider K on complex queries, which in the mock pool happens to include more high-scoring chunks.

**Key efficiency observation:** Both `full_dynamic_k` (avg K=5.20) and `fixed_k5` (avg K=5.00) use roughly the same number of chunks — about 56–58% fewer than `fixed_k12`. The dynamic system achieves this without a fixed cap, adapting K upward for the high-complexity queries in the demo set.

**To produce real-world results:**
1. Replace the mock chunk pool in `run_evaluation.py` with live Qdrant search calls.
2. Annotate ground-truth relevant_chunk_ids from your actual document corpus.
3. Tune alpha, beta, and weights on `DEV_DATASET` only.
4. Report `TEST_DATASET` results as the held-out evaluation.

---

## 10. Limitations and Open Questions

### Known limitations

1. **Qdrant scores are not calibrated probabilities.** R_c treats cosine similarity as a relative relevance proxy. The low-score guard (min_score_guard=0.50) mitigates the most obvious failure mode (high knee gap with weak overall scores) but does not solve calibration.

2. **Evidence alignment depends on payload metadata.** Documents that were uploaded before the structured metadata indexing (legacy `user_documents` collection) may have sparse `year`, `month`, and `document_type` fields. For these, `E` falls back to `S_top` which under-penalises missing evidence.

3. **C_q features are binary indicators.** f5 (comparison), f6 (aggregation), f8 (reasoning) are 0 or 1. Real queries exist on a spectrum. A more nuanced scorer could use regex match counts or sub-sentence analysis, but this would increase complexity without a proven benefit on current data.

4. **Evaluation dataset is synthetic.** All results in §9 are illustrative. No real-world benchmark has been run. The harness is ready for a real corpus; the dataset simply needs to be annotated.

5. **Research paper equation unresolved.** The task specification references a Dynamic Top-K paper. That paper was not available during implementation. The formula is an independently designed normalised adaptive budget formula. Conformance to any specific published equation cannot be claimed. This is an explicitly unresolved dependency.

6. **LLM-judge faithfulness not implemented.** The faithfulness metric in the evaluation harness is a score-based proxy. A proper LLM judge would require an additional API call per query. The module docstring in `run_evaluation.py` describes how to add one.

7. **No adaptive re-ranking after expansion.** The expansion step fetches `recommended_k × 2` chunks and then takes the top `recommended_k` by score. A future improvement would re-embed and re-rank the expanded set using the full ranker.

### Tuning guidance

- Tune `alpha`, `beta`, `coverage_penalty`, and `coverage_threshold` on `DEV_DATASET` only.
- Tune C_q feature weights in `ComplexityConfig.weights` on `DEV_DATASET` only.
- Never use `TEST_DATASET` for weight selection; report it only as the held-out result.
- Consider increasing `K_MAX` if your documents are large and multi-section answers are common.
- Consider decreasing `min_score_guard` if your embedding model produces consistently lower cosine scores (e.g. domain-specific corpora with limited in-domain training).
