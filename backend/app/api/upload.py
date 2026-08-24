import logging
import traceback
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.services.processor import DocumentOrchestrator
from app.core.security import get_current_user
from app.core.crypto import encrypt_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Initialize Orchestrator which ties everything together
orchestrator = DocumentOrchestrator()

# Allowed formats and max size
ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Authenticated Upload Pipeline:
      Frontend → FastAPI (JWT verified) → Encrypt (ChaCha20-Poly1305)
      → PyMuPDF → Qwen2.5-VL → Cloudflare R2 + Qdrant

    The user_id is extracted from the verified JWT — never from the request body.
    Files are encrypted with a per-file HKDF-derived key BEFORE upload to R2.
    """
    user_id = current_user.get("sub")

    try:
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

        print(f"[API] Processing upload: {file.filename} (user={user_id})")

        # ── Derive a unique file_id for key derivation ─────────────────────────
        file_id = str(uuid.uuid4())

        # ── Encrypt before processing / storage ───────────────────────────────
        encrypted_content = encrypt_file(content, user_id, file_id)
        print(f"[API] Encrypted {len(content)} -> {len(encrypted_content)} bytes (file_id={file_id})")

        # ── Run the full pipeline (Orchestrator works on original content for
        #    text/vision extraction, but stores the encrypted bytes to R2) ──────
        result = await orchestrator.process_and_store(
            file_content=content,            # plain bytes for AI processing
            encrypted_content=encrypted_content,  # encrypted bytes for R2 storage
            original_name=file.filename,
            content_type=file.content_type,
            user_id=user_id,
            file_id=file_id,
        )

        print(f"[API] Successfully processed {file.filename}.")

        return {
            "message": "Upload successful",
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


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
        doc = orchestrator.vector_db.get_document(doc_id)
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
                orchestrator.storage.delete_file(file_key)
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
            orchestrator.vector_db.delete_document(doc_id)
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