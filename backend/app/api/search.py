"""
search.py — Authenticated Search & Collections
===============================================
All queries are scoped to the authenticated user's documents only.
The user_id from the verified JWT is used to filter Qdrant results —
users can never see other users' documents.
"""

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user
from app.services.vector_db import VectorDBService
from qdrant_client.http import models

router = APIRouter(prefix="/collections", tags=["Collections"])
vector_db = VectorDBService()


@router.get("/summary")
async def get_dynamic_collections(
    current_user: dict = Depends(get_current_user),
):
    """
    Fetch documents belonging to the authenticated user, grouped by AI category.
    Returns full categorized document summary across both legacy and chunk vector stores.
    """
    try:
        from app.api.upload import get_document_categories
        res = await get_document_categories(current_user=current_user)
        categories = res.get("categories", [])
        return [
            {
                "category_name": c["category"],
                "display_name": c["display_name"],
                "count": c["count"],
                "documents": c["documents"],
            }
            for c in categories
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query")
async def semantic_search(
    q: str,
    current_user: dict = Depends(get_current_user),
):
    """
    User-scoped semantic search.

    Searches only within documents belonging to the authenticated user.
    Results from other users are never returned.
    """
    user_id = current_user.get("sub")

    # Search with user_id filter — user-level isolation at vector DB level
    results = await vector_db.search(query=q, limit=5, user_id=user_id)

    return [
        {
            "name": res.payload["original_name"],
            "category": res.payload["category"],
            "url": f"http://localhost:8000/static/{res.payload.get('bucket_path', '')}",
            "score": res.score,
        }
        for res in results
    ]
