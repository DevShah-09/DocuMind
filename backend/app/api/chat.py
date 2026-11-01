"""
chat.py — LLM-Powered RAG Chat Endpoint
========================================
POST /chat

Pipeline:
  1.  Parse user query (structured filters + semantic query) via QueryParser
  2.  Calculate Query Complexity Score (C_q) — deterministic, no LLM
  3.  Embed + vector-search Qdrant for an initial probe batch
  4.  Calculate Retrieval Confidence Score (R_c) from the probe results
  5.  Run the Dynamic Top-K controller to determine the final K
  6.  Expand retrieval if recommended_k > initial probe size
  7.  Apply Document Diversity Guard (target_documents × chunks_per_document)
  8.  Assemble chunks into a compact context string (with citations)
  9.  Call Qwen 2.5 VL 72B Instruct via OpenRouter to generate a grounded answer
  10. Return ChatResponse { answer, sources, parsed_query, latency_ms, dynamic_k }

Security:
  - Requires valid Supabase JWT (get_current_user)
  - user_id scopes EVERY Qdrant query — users never see each other's documents
  - user_id is re-applied on every expansion search; never relaxed

Observability:
  - Structured retrieval telemetry is logged at INFO level (no document content,
    no user query text in production log lines — privacy-conscious by default)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from openai import OpenAI

from app.core.config import settings
from app.core.security import get_current_user
from app.embeddings.embedding_service import embedding_service
from app.retrieval.dynamic_k import dynamic_k_governor
from app.retrieval.query_parser import query_parser
from app.retrieval.hybrid_engine import _is_conversational_query
from app.schemas.retrieval import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    DynamicKDiagnostics,
    ParsedQuery,
)
from app.vectorstore.qdrant_store import qdrant_store
from app.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

# ── Constants ────────────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS   = 14_000
MAX_CHUNKS_IN_CONTEXT = 16
CHAT_MODEL          = "qwen/qwen2.5-vl-72b-instruct"

# Initial probe size when dynamic-K is enabled.
# Small enough to be cheap, large enough for R_c to be meaningful.
DYNAMIC_K_PROBE_SIZE = 6

# ── OpenRouter client ──────────────────────────────────────────────────────────
_llm_client = OpenAI(
    base_url=settings.OPENROUTER_URL,
    api_key=settings.OPENROUTER_API_KEY,
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are DocuMind AI, an intelligent personal document assistant. "
    "You answer questions, explain concepts, and summarize findings using excerpts from the user's uploaded documents.\n\n"
    "Formatting & Presentation Standards:\n"
    "- Executive Overview: When explaining or summarizing, begin directly with a 1-2 sentence overview.\n"
    "- Headings: Use standard markdown headers (### Section Title) for topics. Do NOT use 5 hashes (#####).\n"
    "- Inline Bullets: Place bullet text on the same line as the bullet (e.g. '• **Initial Retrieval**: starts with...'). Never put an isolated bullet symbol on its own line.\n"
    "- Clean Math: Write math and parameters in clean plain text (e.g. 'K0', 'x', 'theta_floor = 0.35'). Do NOT use LaTeX tags like '\\( K_0 \\)'.\n"
    "- Clean Citations: Cite the source document once cleanly at the end (e.g. '*Source: filename.pdf*'). Do NOT repeat '(Excerpt from File: ...)' across every heading.\n"
    "- Direct & Professional: Avoid conversational meta-preamble like 'To explain the research paper based on the provided excerpts, let\'s break down...'. Jump directly into the findings.\n"
    "- Grounded Accuracy: Use ONLY facts from the provided excerpts."
)

_GREETING_REPLY = (
    "Hello! I'm your Document Intelligence assistant. "
    "You can ask me anything about your uploaded documents — for example:\n"
    "- \"Summarize my SBI bank statement\"\n"
    "- \"What was my electricity bill in January 2024?\"\n"
    "- \"Do I have any medical reports?\"\n\n"
    "How can I help you today?"
)


# ── Helpers ────────────────────────────────────────────────────────────────────
import re

def _extract_user_display_name(current_user: dict) -> str:
    metadata = current_user.get("user_metadata") or {}
    name = metadata.get("full_name") or metadata.get("name")
    if not name and current_user.get("email"):
        name = current_user["email"].split("@")[0]
        name = re.sub(r"[0-9_\.]+", " ", name).strip()
    return (name or "User").title()


def _get_doc_upload_timestamp(doc_id: str, payload_ts: Optional[str]) -> Optional[str]:
    if payload_ts:
        return payload_ts
    try:
        from app.core.qdrant_client_singleton import shared_qdrant_client
        pts = shared_qdrant_client.retrieve(collection_name="user_documents", ids=[doc_id])
        if pts and pts[0].payload:
            return pts[0].payload.get("upload_timestamp")
    except Exception:
        pass
    return None


def _build_context(chunks: list) -> str:
    parts: List[str] = []
    total_chars = 0
    for i, (filename, doc_type, text, page) in enumerate(chunks[:MAX_CHUNKS_IN_CONTEXT], 1):
        snippet = text.strip()
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining]
        entry = f"[{i}] File: {filename} (Type: {doc_type}, Page: {page})\n{snippet}"
        parts.append(entry)
        total_chars += len(snippet)
    return "\n\n".join(parts)


def _call_llm(user_query: str, context: str, history: Optional[List[Any]] = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append recent conversation turns for context continuity (like ChatGPT)
    if history:
        for msg in history[-6:]:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "user")
            text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            if role in ("user", "assistant") and text:
                messages.append({"role": role, "content": text[:1500]})

    if context:
        user_message = (
            f"The following are excerpts from the user's documents:\n\n"
            f"{context}\n\n"
            f"User question: {user_query}"
        )
    else:
        user_message = (
            f"The user asked: {user_query}\n\n"
            "No relevant document excerpts were found in their uploaded documents. "
            "Please let the user know politely and suggest they upload relevant documents."
        )

    messages.append({"role": "user", "content": user_message})

    response = _llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=1200,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _log_retrieval_telemetry(
    *,
    user_id: str,
    initial_k: int,
    final_k: int,
    c_q: float,
    r_c: float,
    unique_chunks: int,
    distinct_docs: int,
    evidence_coverage: float,
    retrieval_latency_ms: float,
    total_latency_ms: float,
    stop_reason: str,
    dynamic_k_enabled: bool,
) -> None:
    """
    Emit a single structured telemetry log line per request.

    Privacy contract:
      - No document text or chunk content is logged.
      - No full query text is logged (privacy-by-default).
      - user_id is included for operational debugging; redact at log sink if needed.
    """
    logger.info(
        "[CHAT_TELEMETRY] user=%s dynamic_k=%s "
        "c_q=%.3f r_c=%.3f "
        "initial_k=%d final_k=%d "
        "unique_chunks=%d distinct_docs=%d "
        "evidence=%.3f "
        "retrieval_ms=%.1f total_ms=%.1f "
        "stop_reason=%s",
        user_id,
        dynamic_k_enabled,
        c_q,
        r_c,
        initial_k,
        final_k,
        unique_chunks,
        distinct_docs,
        evidence_coverage,
        retrieval_latency_ms,
        total_latency_ms,
        stop_reason,
    )


# ── POST /chat ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="LLM-powered RAG chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Ask anything about your documents.

    The pipeline:
      1. Parses query into structured filters + semantic embedding
      2. Retrieves the most relevant document chunks from Qdrant
         using Confidence-Aware Dynamic Top-K selection
      3. Feeds chunks as context to Qwen 2.5 VL 72B
      4. Returns a natural-language answer with source citations
    """
    import asyncio

    # ── 0. Auth check ──────────────────────────────────────────────────────────
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user claims",
        )

    t0 = time.monotonic()

    # ── 1. Greeting shortcut ───────────────────────────────────────────────────
    if _is_conversational_query(request.query):
        return ChatResponse(
            answer=_GREETING_REPLY,
            sources=[],
            parsed_query=None,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    try:
        # ── 2. Parse query & embed in parallel ────────────────────────────────
        user_name = _extract_user_display_name(current_user)
        parsed_task = query_parser.parse(request.query, user_name=user_name)
        embed_task  = embedding_service.embed_query(request.query)

        parsed, query_vector = await asyncio.gather(parsed_task, embed_task)

        # Re-embed cleaned semantic query if it differs from raw query
        if parsed.semantic_query and parsed.semantic_query != request.query:
            query_vector = await embedding_service.embed_query(parsed.semantic_query)

        # ── 3. Build metadata filters ─────────────────────────────────────────
        filters: Dict[str, Any] = {}
        if parsed.document_type:
            filters["document_type"] = parsed.document_type
        if parsed.year is not None:
            filters["year"] = parsed.year
        if parsed.month:
            filters["month"] = parsed.month

        effective_filters: Dict[str, Any] = dict(filters)

        # ── 4. Initial probe retrieval ─────────────────────────────────────────
        t_retrieval_start = time.monotonic()
        dynamic_k_diag: Optional[DynamicKDiagnostics] = None

        is_all_query = bool(re.search(r"\b(all|every|complete|entire)\b", request.query, re.IGNORECASE))
        if is_all_query:
            probe_limit = 35
        else:
            probe_limit = DYNAMIC_K_PROBE_SIZE if request.dynamic_k_enabled else request.limit * 4

        raw_chunks = await qdrant_store.search_chunks(
            query_vector=query_vector,
            user_id=user_id,           # ← user_id ALWAYS applied
            filters=effective_filters,
            limit=probe_limit,
            score_threshold=request.score_threshold,
        )

        # ── Determine intent early: analysis vs. document fetch ──────────────
        q_lower = request.query.lower().strip()
        _analysis_kw = {
            "summarize", "summary", "explain", "what is", "what are", "what was", "what were",
            "how much", "how many", "total", "details", "compare", "why", "list all", "extract",
            "reading", "calculate", "who is", "who was", "when did", "date of", "score",
            "marks", "grade", "result", "percentage", "cgpa", "sgpa", "status", "due",
            "fee", "fees", "cost", "price", "amount", "number", "account number",
            "tell me", "show me details", "find out",
        }
        _question_starters = (
            "what", "who", "when", "where", "why", "how", "which",
            "is ", "are ", "was ", "were ", "can ", "does ", "did ", "do ", "has ", "have "
        )
        requires_llm = any(kw in q_lower for kw in _analysis_kw) or q_lower.startswith(_question_starters) or q_lower.endswith("?")

        # ── 4a. Retry without doc-type filter ONLY if analytical query ────────
        type_relaxed = False
        if not raw_chunks and parsed.document_type and requires_llm:
            logger.info(
                "[CHAT] No chunks for document_type=%r; retrying for LLM question",
                parsed.document_type,
            )
            relaxed = {k: v for k, v in effective_filters.items() if k != "document_type"}
            retry_chunks = await qdrant_store.search_chunks(
                query_vector=query_vector,
                user_id=user_id,       # ← user_id ALWAYS applied
                filters=relaxed,
                limit=probe_limit,
                score_threshold=request.score_threshold,
            )
            if retry_chunks:
                raw_chunks = retry_chunks
                effective_filters = relaxed
                type_relaxed = True

        # If continuing an ongoing conversation, automatically treat as analytical LLM question
        if request.history and len(request.history) > 0:
            requires_llm = True

        # ── 4b. Active Document Continuity (Connect follow-up questions to current topic) ──
        active_doc_ids = list(request.active_document_ids or [])
        if active_doc_ids:
            for active_id in active_doc_ids[:2]:
                active_filters = {"user_id": user_id, "document_id": active_id}
                if parsed.document_type and not type_relaxed:
                    active_filters["document_type"] = parsed.document_type
                active_chunks = await qdrant_store.search_chunks(
                    query_vector=query_vector,
                    user_id=user_id,
                    filters=active_filters,
                    limit=probe_limit,
                    score_threshold=request.score_threshold,
                )
                if active_chunks:
                    existing_cids = {getattr(c, "id", None) or (c.payload or {}).get("chunk_id") for c in raw_chunks}
                    for ac in active_chunks:
                        acid = getattr(ac, "id", None) or (ac.payload or {}).get("chunk_id")
                        if acid not in existing_cids:
                            raw_chunks.append(ac)
                            existing_cids.add(acid)

        # ── 4c. Legacy collection fallback ────────────────────────────────────
        if not raw_chunks:
            vdb    = VectorDBService()
            legacy = await vdb.search(
                query=parsed.semantic_query or request.query,
                limit=request.limit,
                user_id=user_id,
            )

            class _FakeChunk:
                def __init__(self, hit: Any) -> None:
                    self.score = hit.score
                    p = hit.payload or {}
                    self.payload = {
                        "document_id":   str(hit.id),
                        "filename":      p.get("original_name", "document"),
                        "document_type": p.get("category", "document"),
                        "text":          p.get("text_content", ""),
                        "page_number":   1,
                        "chunk_index":   0,
                        "summary":       p.get("summary", ""),
                        "year":          None,
                        "month":         None,
                        "bucket_path":   p.get("bucket_path", ""),
                    }

            raw_chunks = [_FakeChunk(h) for h in legacy]
            logger.info("[CHAT] Legacy fallback: %d chunks for user=%s", len(raw_chunks), user_id)

        initial_k_fetched = len(raw_chunks)
        logger.info("[CHAT] Probe: %d raw chunks for user=%s", initial_k_fetched, user_id)

        # ── 5. Dynamic K decision ──────────────────────────────────────────────
        if request.dynamic_k_enabled and raw_chunks:
            decision = dynamic_k_governor.decide(
                query=request.query,
                initial_chunks=raw_chunks,
                parsed_query=parsed,
            )
            dynamic_k_diag = DynamicKDiagnostics(**decision.to_dict())
            stop_reason    = decision.stop_reason or "sufficient_evidence"

            # ── 6. Expand retrieval if recommended_k > probe size and not abstained ──
            can_expand = (decision.recommended_k > len(raw_chunks)) and (not getattr(decision, "is_abstained", False))
            if can_expand:
                fetch_limit = decision.recommended_k * 2
                logger.info(
                    "[CHAT] Expanding retrieval: recommended_k=%d, fetching up to %d chunks",
                    decision.recommended_k, fetch_limit,
                )
                expanded = await qdrant_store.search_chunks(
                    query_vector=query_vector,
                    user_id=user_id,   # ← user_id ALWAYS applied on expansion
                    filters=effective_filters,  # ← Preserves effective filter from successful search
                    limit=fetch_limit,
                    score_threshold=request.score_threshold,
                )
                if expanded:
                    # Deduplicate merged chunks while preserving highest scores
                    seen_chunk_keys = set()
                    combined_chunks = []
                    for c in expanded + raw_chunks:
                        p = c.payload or {}
                        cid = getattr(c, "id", None) or (
                            p.get("document_id", ""),
                            p.get("chunk_index", 0),
                            p.get("text", "")[:50],
                        )
                        if cid not in seen_chunk_keys:
                            seen_chunk_keys.add(cid)
                            combined_chunks.append(c)
                    raw_chunks  = combined_chunks
                    stop_reason = "max_k_reached" if len(expanded) < fetch_limit else "sufficient_evidence"
            elif getattr(decision, "is_abstained", False):
                logger.info(
                    "[CHAT] Expansion abstained: %s",
                    getattr(decision, "abstention_reason", "Low similarity floor"),
                )

            # ── 7. Document Diversity Guard ────────────────────────────────────
            doc_chunks: Dict[str, list] = defaultdict(list)
            for idx, chunk in enumerate(raw_chunks):
                p = chunk.payload or {}
                did = p.get("document_id") or p.get("filename") or p.get("original_name") or f"doc_{idx}"
                doc_chunks[str(did)].append(chunk)

            # By default (unless user explicitly said 'all' or 'both'), assume user wants the LATEST document
            is_latest_query = not is_all_query or getattr(parsed, "is_latest", False)
            if is_latest_query:
                # Order documents by upload timestamp descending, then by score
                def _doc_recency_key(item):
                    chunks = item[1]
                    max_score = max((c.score for c in chunks), default=0.0)
                    ts = None
                    for c in chunks:
                        cp = c.payload or {}
                        ts = cp.get("upload_timestamp") or _get_doc_upload_timestamp(str(item[0]), None)
                        if ts:
                            break
                    return (ts or "", max_score)
                sorted_docs = sorted(doc_chunks.items(), key=_doc_recency_key, reverse=True)
            else:
                def _doc_relevance_key(item):
                    chunks = item[1]
                    max_score = max((c.score for c in chunks), default=0.0)
                    return max_score

                sorted_docs = sorted(
                    doc_chunks.items(),
                    key=_doc_relevance_key,
                    reverse=True,
                )

            best_doc_score = max(
                (max(c.score for c in item[1]) for item in sorted_docs),
                default=0.0,
            )
            min_doc_score = max(request.score_threshold, best_doc_score * 0.65)

            selected_chunks: list = []
            selected_doc_count = 0
            is_all_query = bool(re.search(r"\b(all|every|complete|entire|both|compare|list all|across)\b", request.query, re.IGNORECASE))
            if is_all_query:
                # User explicitly requested all matching documents
                target_doc_cap = max(15, len(sorted_docs))
                recommended_chunk_cap = max(30, decision.recommended_k)
            else:
                # When user hasn't specified: assume they want the single LATEST document
                target_doc_cap = 1
                recommended_chunk_cap = max(6, decision.recommended_k)

            for doc_id, chunks_for_doc in sorted_docs:
                top_score = max((c.score for c in chunks_for_doc), default=0.0)
                if top_score < min_doc_score:
                    continue
                # If document_type filter is active and not relaxed, ensure document matches it
                if parsed.document_type and not type_relaxed:
                    req_type = parsed.document_type.lower()
                    doc_matches = any(
                        ((c.payload or {}).get("document_type", "").lower() == req_type or
                         req_type in [str(t).lower() for t in (c.payload or {}).get("tags", [])])
                        for c in chunks_for_doc
                    )
                    if not doc_matches:
                        continue
                if selected_doc_count >= target_doc_cap:
                    break
                if len(selected_chunks) >= recommended_chunk_cap:
                    break
                best_for_doc = sorted(chunks_for_doc, key=lambda c: c.score, reverse=True)
                chunks_to_take = (recommended_chunk_cap - len(selected_chunks)) if target_doc_cap == 1 else max(decision.chunks_per_document, 2 if is_all_query else 1)
                selected_chunks.extend(best_for_doc[:chunks_to_take])
                selected_doc_count += 1

            if not selected_chunks and raw_chunks:
                selected_chunks = sorted(raw_chunks, key=lambda c: c.score, reverse=True)[: recommended_chunk_cap]

            top_chunks = sorted(selected_chunks, key=lambda c: c.score, reverse=True)[: recommended_chunk_cap]

        else:
            # ── Static fallback: top-4 per document up to limit * 4 ───────────
            doc_chunks_static: Dict[str, list] = defaultdict(list)
            for idx, chunk in enumerate(raw_chunks):
                p = chunk.payload or {}
                did = p.get("document_id") or p.get("filename") or p.get("original_name") or f"doc_{idx}"
                doc_chunks_static[str(did)].append(chunk)

            flattened: list = []
            for _, chunks_for_doc in doc_chunks_static.items():
                flattened.extend(sorted(chunks_for_doc, key=lambda c: c.score, reverse=True)[:4])

            top_chunks  = sorted(flattened, key=lambda c: c.score, reverse=True)[: request.limit * 4]
            stop_reason = "static_k"

        retrieval_latency_ms = round((time.monotonic() - t_retrieval_start) * 1000, 2)

        # ── 8. Entity & Document Type Verification ────────────────────────────
        # If entities were specified (e.g., 'dhruv'), prioritize chunks matching the entity
        if parsed.entities:
            entity_matched = []
            for c in top_chunks:
                p = c.payload or {}
                haystack = f"{p.get('filename', '')} {p.get('summary', '')} {p.get('text', '')} {' '.join(p.get('entities', []))}".lower()
                if any(e.lower() in haystack for e in parsed.entities):
                    entity_matched.append(c)
            if entity_matched:
                top_chunks = entity_matched
            else:
                logger.info("[CHAT] Entity filter %r had no exact matches; preserving top semantic chunks", parsed.entities)

        # If document_type was specified and not relaxed, ensure chunks match that document_type OR have it in their tags
        if parsed.document_type and not type_relaxed:
            req_type = parsed.document_type.lower()
            def _matches_type(c):
                p = c.payload or {}
                chunk_type = p.get("document_type", "").lower()
                chunk_tags = [str(t).lower() for t in p.get("tags", [])]
                return chunk_type == req_type or req_type in chunk_tags
            type_matched = [c for c in top_chunks if _matches_type(c)]
            if type_matched:
                top_chunks = type_matched
            else:
                raw_type_matched = [c for c in raw_chunks if _matches_type(c)]
                if raw_type_matched:
                    top_chunks = sorted(raw_type_matched, key=lambda c: c.score, reverse=True)[:recommended_chunk_cap]

        # ── 9. Build source list ───────────────────────────────────────────────
        seen_doc_ids: Dict[str, bool] = {}
        sources: List[ChatSource] = []
        for idx, chunk in enumerate(top_chunks):
            p      = chunk.payload or {}
            doc_id = p.get("document_id") or p.get("filename") or p.get("original_name") or f"doc_{idx}"
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids[str(doc_id)] = True
            upload_ts = _get_doc_upload_timestamp(str(doc_id), p.get("upload_timestamp"))
            doc_fn = p.get("filename", p.get("original_name", "document"))
            raw_dtype = p.get("document_type", "document")
            doc_sum = p.get("summary", "")
            if "report" in doc_fn.lower():
                if raw_dtype in ("research_paper", "academic_paper"):
                    raw_dtype = "report"
                doc_sum = re.sub(r"\ba research paper\b", "a project report", doc_sum, flags=re.IGNORECASE)

            sources.append(ChatSource(
                document_id=str(doc_id),
                filename=doc_fn,
                document_type=raw_dtype,
                summary=doc_sum,
                score=round(chunk.score, 4),
                year=p.get("year"),
                month=p.get("month"),
                upload_timestamp=upload_ts,
                preview_url="",
            ))

        # Filter out weak/spurious matches below confidence floor
        MIN_CONFIDENCE_SCORE = max(request.score_threshold, 0.35)
        valid_sources = [s for s in sources if s.score >= MIN_CONFIDENCE_SCORE]
        if valid_sources:
            sources = valid_sources
        elif sources and top_chunks:
            sources = [max(sources, key=lambda s: s.score)]

        # ── Handle default-to-latest document selection ────────────────────────
        if sources:
            # Sort descending by upload_timestamp so newest document is always first
            def _recency_key(s: ChatSource):
                return (s.upload_timestamp or "", s.score)
            sources.sort(key=_recency_key, reverse=True)

            def _chunk_recency_key(c):
                cp = c.payload or {}
                return (cp.get("upload_timestamp") or "", c.score)
            top_chunks.sort(key=_chunk_recency_key, reverse=True)

            # Unless user explicitly requested multiple docs (all/both/compare), isolate to the single LATEST document
            if not is_all_query and not getattr(parsed, "year", None):
                latest_doc = sources[0]
                sources = [latest_doc]
                top_chunks = [c for c in top_chunks if (c.payload or {}).get("document_id") == latest_doc.document_id or (c.payload or {}).get("filename") == latest_doc.filename]
            else:
                max_sources = 25 if is_all_query else max(5, request.limit)
                sources = sources[:max_sources]

        # ── 10. Generate answer ────────────────────────────────────────────────
        if not requires_llm:
            if sources:
                best = sources[0]
                pct  = int(round(best.score * 100))
                if getattr(parsed, "is_latest", False):
                    answer = (
                        f"Found latest matching document for \"{request.query}\". "
                        f"Best match: **{best.filename}** (latest version)."
                    )
                else:
                    answer = (
                        f"Found {len(sources)} matching document"
                        f"{'s' if len(sources) > 1 else ''} for \"{request.query}\". "
                        f"Best match: **{best.filename}** ({pct}% relevance)."
                    )
            else:
                if parsed.document_type and not type_relaxed:
                    pretty_type = parsed.document_type.replace('_', ' ').title()
                    answer = f"No {pretty_type} documents were found in your uploaded collection."
                elif parsed.entities:
                    ents_str = ", ".join(f"'{e}'" for e in parsed.entities)
                    answer = f"No documents found matching {ents_str} in your uploaded collection."
                else:
                    answer = (
                        f"No documents matched \"{request.query}\". "
                        "Try a different query or upload more documents."
                    )
        else:
            if not top_chunks or not sources:
                if parsed.document_type and not type_relaxed:
                    pretty_type = parsed.document_type.replace('_', ' ').title()
                    answer = f"I couldn't find any {pretty_type} documents in your uploaded collection to answer this query."
                elif parsed.entities:
                    ents_str = ", ".join(f"'{e}'" for e in parsed.entities)
                    answer = f"I couldn't find any documents related to {ents_str} in your uploaded collection."
                else:
                    answer = "I couldn't find relevant information in your documents for this query."
            else:
                context_tuples = [
                    (
                        (chunk.payload or {}).get("filename", "document"),
                        (chunk.payload or {}).get("document_type", "document"),
                        (chunk.payload or {}).get("text", (chunk.payload or {}).get("text_content", "")),
                        (chunk.payload or {}).get("page_number", 1),
                    )
                    for chunk in top_chunks
                ]
                context_str = _build_context(context_tuples)
                loop        = asyncio.get_event_loop()
                answer      = await loop.run_in_executor(None, _call_llm, request.query, context_str, request.history)

        total_latency_ms = round((time.monotonic() - t0) * 1000, 2)

        # ── 11. Structured telemetry ──────────────────────────────────────────
        # Compute distinct doc count from top_chunks for telemetry
        distinct_doc_ids = {
            (c.payload or {}).get("document_id", "")
            for c in top_chunks
            if (c.payload or {}).get("document_id")
        }
        _log_retrieval_telemetry(
            user_id=user_id,
            initial_k=initial_k_fetched,
            final_k=len(top_chunks),
            c_q=dynamic_k_diag.c_q       if dynamic_k_diag else 0.0,
            r_c=dynamic_k_diag.r_c       if dynamic_k_diag else 0.0,
            unique_chunks=len(top_chunks),
            distinct_docs=len(distinct_doc_ids),
            evidence_coverage=dynamic_k_diag.evidence_alignment if dynamic_k_diag else 0.0,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
            stop_reason=stop_reason,
            dynamic_k_enabled=request.dynamic_k_enabled,
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            parsed_query=parsed,
            latency_ms=total_latency_ms,
            dynamic_k=dynamic_k_diag,
        )

    except Exception as exc:
        logger.exception("[CHAT] Error for user=%s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(exc)}")
