import asyncio
import logging
import traceback
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.services.upload_job_manager import upload_job_manager
from app.core.security import get_current_user, verify_supabase_jwt
from app.core.crypto import encrypt_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Lazy-loaded Orchestrator
_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from app.services.processor import DocumentOrchestrator
        _orchestrator = DocumentOrchestrator()
    return _orchestrator

# Allowed formats and max size
ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def get_user_from_header_or_query(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> dict:
    """Authenticate via standard Authorization header or ?token= query parameter (for native EventSource)."""
    raw_token = None
    if authorization and authorization.startswith("Bearer "):
        raw_token = authorization.split("Bearer ", 1)[1].strip()
    elif token:
        raw_token = token.strip()

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token. Pass Authorization header or ?token= query parameter.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_supabase_jwt(raw_token)


@router.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    sync: bool = Query(False, description="Set to true for synchronous execution (legacy/benchmark mode)"),
    current_user: dict = Depends(get_current_user),
):
    """
    Authenticated Upload Pipeline:
      Frontend → FastAPI (JWT verified) → Encrypt (ChaCha20-Poly1305)
      → PyMuPDF → Qwen2.5-VL → Cloudflare R2 + Qdrant

    When sync=False (default):
      Enqueues background task, registers job, and returns {job_id, status: 'queued'}.
      Client connects to GET /documents/upload/{job_id}/events for SSE progress stream.

    When sync=True:
      Runs synchronously and returns {message: 'Upload successful', data: result}.
    """
    user_id = current_user.get("sub")

    # ── File Validation ────────────────────────────────────────────────────
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{file_extension}' not allowed. Must be one of {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # ── Derive unique file_id ──────────────────────────────────────────────
    file_id = str(uuid.uuid4())
    job_id = file_id  # Match job_id with file_id for clean tracking

    if sync:
        # Legacy synchronous execution
        try:
            print(f"[API] Processing synchronous upload: {file.filename} (user={user_id})")
            encrypted_content = encrypt_file(content, user_id, file_id)
            result = await get_orchestrator().process_and_store(
                file_content=content,
                encrypted_content=encrypted_content,
                original_name=file.filename,
                content_type=file.content_type,
                user_id=user_id,
                file_id=file_id,
            )
            return {"message": "Upload successful", "data": result}
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

    # ── Asynchronous Execution with SSE Milestones ──────────────────────────
    upload_job_manager.create_job(job_id=job_id, user_id=user_id, filename=file.filename)

    async def _run_async_pipeline():
        try:
            print(f"[API] Processing background upload: {file.filename} (job={job_id}, user={user_id})")
            encrypted_content = encrypt_file(content, user_id, file_id)

            async def _on_progress(stage: str, progress: int, message: str):
                await upload_job_manager.emit_progress(job_id, stage, progress, message)

            result = await get_orchestrator().process_and_store(
                file_content=content,
                encrypted_content=encrypted_content,
                original_name=file.filename,
                content_type=file.content_type,
                user_id=user_id,
                file_id=file_id,
                progress_callback=_on_progress,
            )
            await upload_job_manager.complete_job(job_id, result)
        except Exception as exc:
            traceback.print_exc()
            await upload_job_manager.fail_job(job_id, str(exc))

    asyncio.create_task(_run_async_pipeline())

    return {
        "message": "Upload queued",
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
    }


@router.get("/upload/{job_id}/events")
async def stream_upload_events(
    job_id: str,
    current_user: dict = Depends(get_user_from_header_or_query),
):
    """
    SSE stream for upload progress milestones.
    Supports native browser EventSource via ?token=<jwt> or Authorization header.
    Features automatic historical replay so reconnected clients don't miss milestones.
    """
    user_id = current_user.get("sub")
    job = upload_job_manager.get_job(job_id, user_id=user_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload job '{job_id}' not found or access denied.",
        )

    return StreamingResponse(
        upload_job_manager.stream_events(job_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/upload/{job_id}/status")
async def get_upload_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Non-streaming status check for a given upload job."""
    user_id = current_user.get("sub")
    job = upload_job_manager.get_job(job_id, user_id=user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "current_progress": job.current_progress,
        "result": job.result,
        "error": job.error,
        "updated_at": job.updated_at,
    }


@router.get("/categories")
async def get_document_categories(
    current_user: dict = Depends(get_current_user),
):
    """
    Fetch documents belonging to the authenticated user, grouped by AI category.
    Deduplicates identical files, normalizes categories, and returns clean groups.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    from collections import defaultdict
    from app.services.vector_db import VectorDBService
    from app.vectorstore.qdrant_store import qdrant_store
    from qdrant_client.http import models

    # Normalization map for synonymous category slugs
    CATEGORY_NORM = {
        "student_identity_card": "student_id_card",
        "identity_card": "student_id_card",
        "academic_paper": "research_paper",
        "project_report": "report",
        "technical_report": "report",
    }

    all_docs = {}

    # 1. Fetch from legacy vector_db (user_documents)
    try:
        vdb = VectorDBService()
        points, _ = vdb.client.scroll(
            collection_name=vdb.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
            with_payload=True,
            limit=500,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = str(point.id)
            raw_cat = (payload.get("category") or "miscellaneous").strip()
            cat_name = CATEGORY_NORM.get(raw_cat, raw_cat)
            name = (payload.get("original_name") or "Document").strip()
            if "report" in name.lower() and cat_name in ("research_paper", "academic_paper"):
                cat_name = "report" 
            file_key = payload.get("bucket_path", "")
            file_hash = payload.get("file_hash", "")
            all_docs[doc_id] = {
                "id": doc_id,
                "name": name,
                "category": cat_name,
                "url": f"http://localhost:8000/static/{file_key}",
                "file_key": file_key,
                "file_hash": file_hash,
            }
    except Exception as e:
        logger.warning(f"[CATEGORIES] Legacy scroll notice: {e}")

    # 2. Fetch from document_chunks collection
    try:
        chunk_points, _ = qdrant_store.client.scroll(
            collection_name="document_chunks",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
            with_payload=True,
            limit=1000,
        )
        for point in chunk_points:
            payload = point.payload or {}
            doc_id = payload.get("document_id")
            if not doc_id:
                continue
            raw_cat = (payload.get("document_type") or payload.get("category") or "miscellaneous").strip()
            cat_name = CATEGORY_NORM.get(raw_cat, raw_cat)
            name = (payload.get("filename") or payload.get("original_name") or "Document").strip()
            if "report" in name.lower() and cat_name in ("research_paper", "academic_paper"):
                cat_name = "report" 
            file_key = payload.get("bucket_path", "")
            file_hash = payload.get("file_hash", "")

            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "id": doc_id,
                    "name": name,
                    "category": cat_name,
                    "url": f"http://localhost:8000/documents/view/{doc_id}",
                    "file_key": file_key,
                    "file_hash": file_hash,
                }
            else:
                # Upgrade category if legacy was generic 'miscellaneous'
                if all_docs[doc_id]["category"] == "miscellaneous" and cat_name != "miscellaneous":
                    all_docs[doc_id]["category"] = cat_name
                if not all_docs[doc_id]["name"] or all_docs[doc_id]["name"] == "Document":
                    all_docs[doc_id]["name"] = name
    except Exception as e:
        logger.warning(f"[CATEGORIES] Chunk scroll notice: {e}")

    # 3. Global document deduplication across all categories
    # Each physical document must belong to EXACTLY ONE primary category.
    # Priority: Specific document types > bills > miscellaneous fallback
    def category_priority(cat):
        if not cat or cat == "miscellaneous":
            return 0
        if cat in ("mobile_bill", "electricity_bill", "water_bill", "gas_bill"):
            return 1
        return 2

    deduped_docs = {}
    for doc_id, doc in all_docs.items():
        norm_name = doc["name"].lower().strip()
        # Identity is file_hash if present, or lowercase filename
        file_identity = doc.get("file_hash") or norm_name

        if file_identity not in deduped_docs:
            deduped_docs[file_identity] = doc
        else:
            existing = deduped_docs[file_identity]
            # If current doc has higher category priority, upgrade to it
            if category_priority(doc["category"]) > category_priority(existing["category"]):
                deduped_docs[file_identity] = doc
            elif category_priority(doc["category"]) == category_priority(existing["category"]):
                if not existing.get("file_key") and doc.get("file_key"):
                    deduped_docs[file_identity] = doc

    # 4. Group into clean categories
    category_map = defaultdict(list)
    for doc in deduped_docs.values():
        category_map[doc["category"]].append({
            "id": doc["id"],
            "name": doc["name"],
            "url": doc["url"],
            "file_key": doc["file_key"],
        })

    CATEGORY_TITLES = {
        "report": "Project Report",
        "research_paper": "Research Paper",
        "student_id_card": "Student ID Card",
        "secondary_school_certificate": "10th / 12th Marksheet",
        "aadhaar": "Aadhaar",
        "pan_card": "PAN Card",
        "marksheet": "Marksheet",
        "resume": "Resume",
        "certificate": "Certificate",
        "electricity_bill": "Electricity Bill",
        "water_bill": "Water Bill",
        "gas_bill": "Gas Bill",
        "mobile_bill": "Mobile Bill",
        "bank_statement": "Bank Statement",
        "invoice": "Invoice",
        "receipt": "Receipt",
        "miscellaneous": "Miscellaneous",
    }

    # Format result
    categories = []
    for cat, doc_list in category_map.items():
        if not doc_list:
            continue
        pretty_name = CATEGORY_TITLES.get(cat, cat.replace("_", " ").title() if cat else "Document")
        categories.append({
            "category": cat,
            "display_name": pretty_name,
            "count": len(doc_list),
            "documents": doc_list,
        })

    return {"categories": categories}


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Authenticated document deletion.

    Verifies ownership before deleting from storage and vector database.
    """
    import os
    from app.core.security import assert_owns_resource
    from app.vectorstore.qdrant_store import qdrant_store

    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # 1. Look up document metadata from legacy vector_db or document_chunks
        try:
            doc = get_orchestrator().vector_db.get_document(doc_id)
        except Exception:
            doc = None
        doc_metadata = None
        file_key = None

        if doc and doc.payload:
            doc_metadata = doc.payload
            file_key = doc_metadata.get("bucket_path")
            # Verify ownership
            assert_owns_resource(current_user, doc_metadata.get("user_id", ""))
        else:
            # Fallback to chunk collection metadata
            doc_metadata = await qdrant_store.get_document_metadata(doc_id, user_id)
            if doc_metadata:
                file_key = doc_metadata.get("bucket_path")
                assert_owns_resource(current_user, doc_metadata.get("user_id", ""))
            else:
                # If neither has metadata, we still try deleting chunks for this user_id
                logger.warning(f"[API] No metadata found for doc_id={doc_id}, proceeding with chunk purge for user={user_id}")

        # 2. Delete physical file from S3 and local storage if key is known
        if file_key:
            try:
                get_orchestrator().storage.delete_file(file_key)
            except Exception as e:
                logger.warning(f"[API] Could not delete file from S3: {e}")

            local_path = os.path.join("local_storage", file_key)
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"[API] Could not delete local storage file {local_path}: {e}")

        # 3. Delete from Qdrant legacy collection
        try:
            get_orchestrator().vector_db.delete_document(doc_id)
        except Exception as e:
            logger.warning(f"[API] Qdrant legacy delete notice: {e}")

        # 4. Delete all chunks from Qdrant document_chunks collection
        try:
            await qdrant_store.delete_document_chunks(doc_id, user_id)
            logger.info(f"[API] Deleted chunks for document {doc_id}")
        except Exception as e:
            logger.warning(f"[API] Qdrant chunk delete notice: {e}")

        return {"message": "Document deleted successfully", "doc_id": doc_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[API] DELETE ERROR for doc_id={doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
