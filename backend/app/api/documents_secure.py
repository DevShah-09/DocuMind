"""
documents_secure.py — Secure Document Access Endpoints
======================================================
Endpoint:
  GET /generate-signed-url/{doc_id}
    → verifies JWT + ownership → returns temporary HMAC-signed R2 URL

Principle: the backend is the sole authority on who can access what.
The frontend never gets a permanent R2 URL — only short-lived signed URLs.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from app.core.security import get_current_user, assert_owns_resource
from app.core.signed_urls import generate_signed_url
from app.core.crypto import decrypt_file
from app.services.vector_db import VectorDBService
from fastapi.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Access"])

vector_db = VectorDBService()


@router.get("/generate-signed-url/{doc_id}")
async def get_signed_url(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate a temporary HMAC-SHA256 signed URL for a document.

    Security steps:
      1. Verify JWT (done by Depends(get_current_user))
      2. Fetch document metadata from Qdrant
      3. Assert that the document's user_id matches the JWT sub
      4. Generate a 5-minute signed URL bound to this user

    The URL expires in 300 seconds and cannot be used by another user_id.

    Args:
        doc_id: Document UUID (Qdrant point ID)

    Returns:
        { signed_url, expires_in_seconds, doc_id, filename }
    """
    user_id = current_user.get("sub")

    # 1. Fetch document metadata from Qdrant
    try:
        doc = vector_db.get_document(doc_id)
    except Exception as exc:
        logger.error("[secure] Failed to fetch doc metadata for doc_id=%s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document metadata",
        )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # 2. Ownership check — prevent horizontal privilege escalation
    doc_user_id = doc.payload.get("user_id")
    assert_owns_resource(current_user, doc_user_id)

    # 3. Generate a 5-minute HMAC-SHA256 signed URL
    bucket_path = doc.payload.get("bucket_path", "")
    if not bucket_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document has no associated storage file",
        )

    signed_url = generate_signed_url(
        bucket_path=bucket_path,
        user_id=user_id,
        expiry_seconds=300,
    )

    logger.info(
        "[secure] Signed URL issued: doc_id=%s user_id=%s bucket_path=%s",
        doc_id, user_id, bucket_path,
    )

    return {
        "signed_url": signed_url,
        "expires_in_seconds": 300,
        "doc_id": doc_id,
        "filename": doc.payload.get("original_name"),
    }


@router.get("/view/{doc_id}")
async def view_document(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Secure Decryption Proxy:
    Fetches encrypted bytes from storage, decrypts them on-the-fly,
    and returns the original file to the browser.
    """
    user_id = current_user.get("sub")

    # 1. Fetch metadata
    doc = vector_db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 2. Ownership check
    assert_owns_resource(current_user, doc.payload.get("user_id"))

    # 3. Read encrypted file from storage
    from app.services.cloud_storage import LocalStorageService
    storage = LocalStorageService()
    
    bucket_path = doc.payload.get("bucket_path")
    try:
        # We need a 'download_file' method in storage
        encrypted_bytes = await storage.download_file(bucket_path)
    except Exception as e:
        logger.error(f"Storage read failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to read file from storage")

    # 4. Decrypt using derived key
    try:
        # file_id here is the doc_id used during encryption
        plaintext = decrypt_file(encrypted_bytes, user_id, doc_id)
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise HTTPException(status_code=500, detail="Decryption failed")

    # 5. Return as proper file response
    content_type = "application/pdf"
    if doc.payload.get("original_name", "").lower().endswith((".png", ".jpg", ".jpeg")):
        content_type = "image/jpeg"

    return Response(
        content=plaintext,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{doc.payload.get("original_name")}"'
        }
    )
