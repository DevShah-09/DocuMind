"""
retrieval.py — Hybrid Search & Document API Routes
===================================================
Endpoints:
  POST /search                — full hybrid search
  POST /search/query-parse    — parse query only (debug / UI autocomplete)
  POST /search/batch          — batch queries
  GET  /documents/{id}        — document detail + all chunks
  GET  /documents/{id}/preview — AWS S3 presigned preview URL

Security:
  - All routes require valid Supabase JWT (get_current_user dependency)
  - user_id extracted from JWT claims and injected into every Qdrant query
  - Users can never retrieve another user's documents
  - Preview URLs are presigned S3 URLs (15-minute expiry) — never raw S3 paths
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import settings
from app.core.security import get_current_user
from app.retrieval.hybrid_engine import hybrid_engine
from app.schemas.retrieval import (
    BatchSearchRequest,
    BatchSearchResponse,
    DocumentChunkDetail,
    DocumentDetailResponse,
    DocumentPreviewResponse,
    QueryParseRequest,
    QueryParseResponse,
    SearchRequest,
    SearchResponse,
)
from app.vectorstore.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search & Retrieval"])

PREVIEW_EXPIRY_SECONDS = 900   # 15 minutes


# ── Signed URL Helper ──────────────────────────────────────────────────────────

def _make_s3_presigned_url(bucket_path: str, user_id: str) -> str:
    """
    Generate a 15-minute HMAC-SHA256 signed URL to our backend decryption proxy.
    Because files in S3 are encrypted, we cannot return an S3 presigned URL directly.
    """
    import time
    from app.core.signed_urls import _sign
    
    expires = int(time.time()) + PREVIEW_EXPIRY_SECONDS
    message = f"{bucket_path}|{user_id}|{expires}"
    sig = _sign(message)
    
    # Return URL to our backend proxy that will verify the signature, fetch from S3, decrypt, and stream
    from urllib.parse import urlencode
    params = urlencode({
        "user_id": user_id,
        "expires": expires,
        "sig": sig,
    })
    
    # Note: we use /documents/stream/{bucket_path} because bucket_path is what the UI gives us from the metadata
    return f"http://localhost:8000/documents/stream/{bucket_path}?{params}"


def _preview_url_fn(document_id: str, bucket_path: str, user_id: str) -> str:
    """Adapter function injected into the hybrid engine."""
    if not bucket_path:
        return ""
    return _make_s3_presigned_url(bucket_path, user_id)


# ── POST /search ───────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse, summary="Hybrid document search")
async def search_documents(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Perform a hybrid search over the authenticated user's documents.

    Accepts natural language queries such as:
    - "Show my 2017 electricity bill"
    - "Find SBI bank statements from January"
    - "Medical reports mentioning diabetes"
    - "Receipts above ₹5000"

    The pipeline:
      1. Parses the query into structured filters using Qwen 2.5
      2. Generates a query embedding with BAAI/bge-base-en-v1.5
      3. Performs filtered vector search in Qdrant (user-scoped)
      4. Ranks results using cosine + metadata boost scoring
      5. Returns paginated SearchResponse with preview URLs
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user claims")

    try:
        response = await hybrid_engine.search(
            request=request,
            user_id=user_id,
            generate_preview_url_fn=_preview_url_fn,
        )
        return response
    except Exception as e:
        logger.exception("[SEARCH] Unexpected error for user=%s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ── POST /search/query-parse ───────────────────────────────────────────────────

@router.post(
    "/search/query-parse",
    response_model=QueryParseResponse,
    summary="Parse a natural language query into structured filters",
)
async def parse_query(
    request: QueryParseRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Parse a query into its structured components without executing search.
    Useful for previewing what filters would be applied, or for UI auto-complete.

    Example input: "Show my 2017 electricity bill"
    Example output:
    {
      "document_type": "electricity_bill",
      "year": 2017,
      "month": null,
      "entities": [],
      "amount_filters": null,
      "semantic_query": "electricity bill"
    }
    """
    try:
        parsed = await hybrid_engine.parse_only(request.query)
        return QueryParseResponse(raw_query=request.query, parsed=parsed)
    except Exception as e:
        logger.exception("[PARSE] Error parsing query: %s", e)
        raise HTTPException(status_code=500, detail=f"Query parsing failed: {str(e)}")


# ── POST /search/batch ────────────────────────────────────────────────────────

@router.post(
    "/search/batch",
    response_model=BatchSearchResponse,
    summary="Execute multiple searches in one request",
)
async def batch_search(
    request: BatchSearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Execute up to 10 searches in a single request.
    Results are keyed by the original query string.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user claims")

    t0 = time.monotonic()
    results = {}

    for query in request.queries:
        try:
            search_req = SearchRequest(
                query=query,
                limit=request.limit_per_query,
                score_threshold=request.score_threshold,
            )
            results[query] = await hybrid_engine.search(
                request=search_req,
                user_id=user_id,
                generate_preview_url_fn=_preview_url_fn,
            )
        except Exception as e:
            logger.error("[BATCH] Query '%s' failed: %s", query, e)

    latency_ms = (time.monotonic() - t0) * 1000
    return BatchSearchResponse(
        results=results,
        total_queries=len(request.queries),
        latency_ms=round(latency_ms, 2),
    )


# ── GET /documents/categories ──────────────────────────────────────────────────────
@router.get(
    "/documents/categories",
    summary="Get document categories and summary",
)
async def get_doc_categories_retrieval(current_user: dict = Depends(get_current_user)):
    from app.api.upload import get_document_categories
    return await get_document_categories(current_user=current_user)


# ── GET /documents/{id} ────────────────────────────────────────────────────────

@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document metadata and all chunks",
)
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve a document's full metadata and all indexed text chunks.
    Scoped to the authenticated user — returns 404 if the document
    belongs to a different user or does not exist.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user claims")

    try:
        # Retrieve all chunks for this document
        chunks = await qdrant_store.get_chunks_by_document(document_id, user_id)

        if not chunks:
            # ── Fallback to legacy collection ──
            from app.services.vector_db import VectorDBService
            vdb = VectorDBService()
            try:
                legacy_doc = vdb.get_document(document_id)
            except Exception:
                legacy_doc = None
            if not legacy_doc or legacy_doc.payload.get("user_id") != user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{document_id}' not found or access denied",
                )
            first = legacy_doc.payload or {}
            chunk_details = [
                DocumentChunkDetail(
                    chunk_id=f"{document_id}_legacy",
                    chunk_index=0,
                    page_number=1,
                    text=first.get("text_content", ""),
                )
            ]
        else:
            # Extract document-level metadata from first chunk
            first = chunks[0].payload or {}

            chunk_details = [
                DocumentChunkDetail(
                    chunk_id=c.payload.get("chunk_id", ""),
                    chunk_index=c.payload.get("chunk_index", 0),
                    page_number=c.payload.get("page_number", 1),
                    text=c.payload.get("text", ""),
                )
                for c in sorted(chunks, key=lambda x: x.payload.get("chunk_index", 0))
            ]

        return DocumentDetailResponse(
            document_id=document_id,
            filename=first.get("filename", first.get("original_name", "")),
            document_type=first.get("document_type", first.get("category", "")),
            summary=first.get("summary", ""),
            date=first.get("date"),
            year=first.get("year"),
            month=first.get("month"),
            metadata=first.get("metadata", {}),
            tags=first.get("tags", []),
            entities=first.get("entities", []),
            chunks=chunk_details,
            total_chunks=len(chunk_details),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[DOC] Error fetching document %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


# ── GET /documents/{id}/preview ───────────────────────────────────────────────

@router.get(
    "/documents/{document_id}/preview",
    response_model=DocumentPreviewResponse,
    summary="Get a signed S3 preview URL for a document",
)
async def get_document_preview(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate a short-lived (15-minute) AWS S3 presigned URL for the document.
    The raw S3 URL is never exposed — only the presigned URL is returned.

    Returns 404 if the document is not found or belongs to another user.
    Returns 502 if the presigned URL could not be generated (S3 error).
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user claims")

    try:
        metadata = await qdrant_store.get_document_metadata(document_id, user_id)
        if not metadata:
            # ── Fallback to legacy collection ──
            from app.services.vector_db import VectorDBService
            vdb = VectorDBService()
            legacy_doc = vdb.get_document(document_id)
            if legacy_doc and legacy_doc.payload.get("user_id") == user_id:
                metadata = legacy_doc.payload
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{document_id}' not found or access denied",
                )

        bucket_path = metadata.get("bucket_path", "")
        filename = metadata.get("filename", metadata.get("original_name", document_id))

        if not bucket_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document has no associated file",
            )

        preview_url = _make_s3_presigned_url(bucket_path, user_id)
        if not preview_url:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not generate preview URL — S3 error",
            )

        return DocumentPreviewResponse(
            document_id=document_id,
            filename=filename,
            preview_url=preview_url,
            expires_in_seconds=PREVIEW_EXPIRY_SECONDS,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[PREVIEW] Error for document %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


@router.get("/documents/stream/{bucket_path:path}")
async def stream_decrypted_document(
    bucket_path: str,
    user_id: str = Query(...),
    expires: int = Query(...),
    sig: str = Query(...),
):
    """
    Decryption Proxy:
    Takes an HMAC signed URL, verifies the signature, fetches the encrypted
    bytes from S3, decrypts them, and returns the plaintext file.
    """
    import time
    if int(time.time()) > expires:
        raise HTTPException(status_code=403, detail="Expired URL")
    
    from app.core.signed_urls import _sign
    import hmac
    message = f"{bucket_path}|{user_id}|{expires}"
    expected_sig = _sign(message)
    if not hmac.compare_digest(expected_sig, sig):
        raise HTTPException(status_code=403, detail="Invalid signature")

    # Fetch encrypted bytes
    from app.services.cloud_storage import S3StorageService
    storage = S3StorageService()
    try:
        encrypted_bytes = await storage.download_file(bucket_path)
    except Exception as e:
        logger.error(f"S3 download failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to read file from storage")

    # Decrypt
    from app.core.crypto import decrypt_file
    try:
        # doc_id is a 36-character UUID at the start of the bucket_path
        # format: "{doc_id}-{original_name}"
        doc_id = bucket_path[:36]
        plaintext = decrypt_file(encrypted_bytes, user_id, doc_id)
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise HTTPException(status_code=500, detail="Decryption failed")

    content_type = "application/pdf"
    if bucket_path.lower().endswith((".png", ".jpg", ".jpeg")):
        content_type = "image/jpeg"

    from fastapi.responses import Response
    return Response(
        content=plaintext,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{bucket_path}"'
        }
    )

