"""
chat.py — LLM-Powered RAG Chat Endpoint
========================================
POST /chat

Pipeline:
  1. Parse user query (structured filters + semantic query) via QueryParser
  2. Embed + vector-search Qdrant for the most relevant document chunks
  3. Assemble chunks into a compact context string (with citations)
  4. Call Qwen 2.5 VL 72B Instruct via OpenRouter to generate a grounded answer
  5. Return ChatResponse { answer, sources, parsed_query, latency_ms }

Security:
  - Requires valid Supabase JWT (get_current_user)
  - user_id scopes every Qdrant query — users never see each other's documents
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from openai import OpenAI

from app.core.config import settings
from app.core.security import get_current_user
from app.embeddings.embedding_service import embedding_service
from app.retrieval.query_parser import query_parser
from app.retrieval.hybrid_engine import _is_conversational_query
from app.schemas.retrieval import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    ParsedQuery,
)
from app.vectorstore.qdrant_store import qdrant_store
from app.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

# ── Constants ────────────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = 14000
MAX_CHUNKS_IN_CONTEXT = 16
CHAT_MODEL = "qwen/qwen2.5-vl-72b-instruct"

# ── OpenRouter client ──────────────────────────────────────────────────────────
_llm_client = OpenAI(
    base_url=settings.OPENROUTER_URL,
    api_key=settings.OPENROUTER_API_KEY,
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful personal document assistant with access to excerpts from "
    "the user's uploaded documents (PDFs, scans, invoices, ID cards, statements, etc.). \n\n"
    "Guidelines:\n"
    "- Answer the user's question or fulfill their request (summarize, explain, list, compare, extract) "
    "using ONLY the information found in the provided document excerpts.\n"
    "- For summarization requests: produce a clear, structured summary of the document's key contents.\n"
    "- For extraction requests (amounts, dates, names): extract and present the exact values found.\n"
    "- Always mention which document (filename) your answer is drawn from.\n"
    "- If multiple documents are provided, address each one separately when relevant.\n"
    "- If the context does not contain enough information to answer, say: "
    "'I couldn\'t find relevant information in your documents for this query.' "
    "Do NOT make up facts or use external knowledge beyond what is in the excerpts."
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


def _call_llm(user_query: str, context: str) -> str:
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

    response = _llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1200,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


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
      3. Feeds chunks as context to Qwen 2.5 VL 72B
      4. Returns a natural-language answer with source citations
    """
    import asyncio

    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user claims",
        )

    t0 = time.monotonic()

    # 0. Greetings shortcut
    if _is_conversational_query(request.query):
        return ChatResponse(
            answer=_GREETING_REPLY,
            sources=[],
            parsed_query=None,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    try:
        # 1 & 2. Parse query & Embed query in PARALLEL via asyncio.gather
        parsed_task = query_parser.parse(request.query)
        embed_task = embedding_service.embed_query(request.query)
        
        parsed, query_vector = await asyncio.gather(parsed_task, embed_task)
        
        # If semantic query was extracted clean, re-embed only if different
        if parsed.semantic_query and parsed.semantic_query != request.query:
            query_vector = await embedding_service.embed_query(parsed.semantic_query)

        # 3. Build filters
        filters: Dict = {}
        if parsed.document_type:
            filters["document_type"] = parsed.document_type
        if parsed.year is not None:
            filters["year"] = parsed.year
        if parsed.month:
            filters["month"] = parsed.month

        # 4. Vector search with strict metadata filters
        raw_chunks = await qdrant_store.search_chunks(
            query_vector=query_vector,
            user_id=user_id,
            filters=filters,
            limit=request.limit * 4,
            score_threshold=request.score_threshold,
        )

        # 4a. If specific document_type filter returned nothing, retry without document_type filter
        if not raw_chunks and parsed.document_type:
            logger.info("[CHAT] No chunks for document_type '%s', retrying un-filtered search", parsed.document_type)
            relaxed_filters = {k: v for k, v in filters.items() if k != "document_type"}
            raw_chunks = await qdrant_store.search_chunks(
                query_vector=query_vector,
                user_id=user_id,
                filters=relaxed_filters,
                limit=request.limit * 4,
                score_threshold=request.score_threshold,
            )

        logger.info("[CHAT] %d raw chunks retrieved for user=%s", len(raw_chunks), user_id)

        # 4b. Legacy fallback
        if not raw_chunks:
            vdb = VectorDBService()
            legacy = await vdb.search(
                query=parsed.semantic_query or request.query,
                limit=request.limit,
                user_id=user_id,
            )

            class _FakeChunk:
                def __init__(self, hit):
                    self.score = hit.score
                    p = hit.payload or {}
                    self.payload = {
                        "document_id": str(hit.id),
                        "filename": p.get("original_name", "document"),
                        "document_type": p.get("category", "document"),
                        "text": p.get("text_content", ""),
                        "page_number": 1,
                        "chunk_index": 0,
                        "summary": p.get("summary", ""),
                        "year": None,
                        "month": None,
                        "bucket_path": p.get("bucket_path", ""),
                    }

            raw_chunks = [_FakeChunk(h) for h in legacy]
            logger.info("[CHAT] Legacy fallback: %d chunks", len(raw_chunks))

        # 5. Collect ALL top chunks per document (important for summarization)
        #    — group by doc_id, keep top chunks by score, then sort across all
        from collections import defaultdict
        doc_chunks: Dict[str, list] = defaultdict(list)
        for chunk in raw_chunks:
            p = chunk.payload or {}
            doc_id = p.get("document_id", "")
            if not doc_id:
                continue
            doc_chunks[doc_id].append(chunk)

        # For each document, keep up to 4 best chunks; then flatten and re-sort
        flattened = []
        for doc_id, chunks in doc_chunks.items():
            top4 = sorted(chunks, key=lambda c: c.score, reverse=True)[:4]
            flattened.extend(top4)

        top_chunks = sorted(flattened, key=lambda c: c.score, reverse=True)[
            : request.limit * 4  # allow more chunks so context is richer
        ]

        # 6. Check intent: Does the user explicitly ask a question/summary/explanation, or just want to fetch/find documents?
        q_lower = request.query.lower().strip()
        analysis_keywords = {
            "summarize", "summary", "explain", "what is", "what are", "how much", 
            "total", "details", "compare", "why", "list all", "extract", "reading",
            "calculate", "who is", "when did", "date of"
        }
        requires_llm_answer = any(kw in q_lower for kw in analysis_keywords) or q_lower.endswith("?")

        # 7. Build sources — one entry per unique document (best-scoring chunk metadata)
        seen_source_docs: Dict[str, bool] = {}
        sources: List[ChatSource] = []
        for chunk in top_chunks:
            p = chunk.payload or {}
            doc_id = p.get("document_id", "")
            if not doc_id or doc_id in seen_source_docs:
                continue
            seen_source_docs[doc_id] = True
            sources.append(ChatSource(
                document_id=doc_id,
                filename=p.get("filename", p.get("original_name", "document")),
                document_type=p.get("document_type", "document"),
                summary=p.get("summary", ""),
                score=round(chunk.score, 4),
                year=p.get("year"),
                month=p.get("month"),
                preview_url="",
            ))

        # Cap sources to top 3 matches
        sources = sources[:3]

        # 8. If pure document retrieval (e.g. "give me aadhar card", "show bank statement"), return document cards without full LLM generation
        if not requires_llm_answer:
            if sources:
                top_doc = sources[0]
                top_score = int(round(top_doc.score * 100))
                answer = f"Found {len(sources)} matching document{'s' if len(sources) > 1 else ''} for \"{request.query}\". Best match: **{top_doc.filename}** ({top_score}% relevance)."
            else:
                answer = f"No documents matched \"{request.query}\". Try a different query or upload more documents."
        else:
            # Full LLM RAG call when explicitly asked (summarize, explain, questions)
            context_tuples = []
            for chunk in top_chunks:
                p = chunk.payload or {}
                context_tuples.append((
                    p.get("filename", p.get("original_name", "document")),
                    p.get("document_type", "document"),
                    p.get("text", p.get("text_content", "")),
                    p.get("page_number", 1),
                ))

            context_str = _build_context(context_tuples)
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(None, _call_llm, request.query, context_str)

        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.info(
            "[CHAT] user=%s answered in %.0fms, sources=%d",
            user_id, latency_ms, len(sources),
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            parsed_query=parsed,
            latency_ms=latency_ms,
        )

    except Exception as exc:
        logger.exception("[CHAT] Error for user=%s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(exc)}")
