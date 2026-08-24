"""
history.py — User Query History API
====================================
Stores and retrieves user query session history.
Persisted in Supabase (query_history table) via the PostgREST REST API
using the service role key for full access.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import httpx

from app.core.security import get_current_user
from app.core.config import settings

router = APIRouter(prefix="/history", tags=["Query History"])

# ── Supabase PostgREST helpers ────────────────────────────────────────────────

_TABLE = "query_history"


def _sb_headers() -> dict:
    """Service-role headers for Supabase PostgREST."""
    return {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _rest_url(path: str = "") -> str:
    return f"{settings.SUPABASE_URL}/rest/v1/{_TABLE}{path}"


# ── Schema ────────────────────────────────────────────────────────────────────

class HistoryItemSchema(BaseModel):
    id: str
    query: str
    timestamp: str
    resultCount: Optional[int] = 0
    topFile: Optional[str] = None
    snapshot: Optional[Dict[str, Any]] = None
    label: Optional[str] = None


# ── GET /history ──────────────────────────────────────────────────────────────

@router.get("", response_model=List[HistoryItemSchema])
async def get_user_history(current_user: dict = Depends(get_current_user)):
    """
    Fetch all query history items for the authenticated user, newest-first.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing from claims")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            _rest_url(),
            headers={**_sb_headers(), "Prefer": ""},
            params={
                "user_id": f"eq.{user_id}",
                "order": "timestamp.desc",
                "limit": "200",
            },
        )

    if resp.status_code not in (200, 206):
        raise HTTPException(status_code=502, detail="Failed to fetch history from database")

    rows = resp.json()

    # Map snake_case DB columns → camelCase schema expected by the frontend
    def _row_to_item(row: dict) -> dict:
        return {
            "id": row["id"],
            "query": row["query"],
            "timestamp": row["timestamp"],
            "resultCount": row.get("result_count", 0),
            "topFile": row.get("top_file"),
            "snapshot": row.get("snapshot"),
            "label": row.get("label"),
        }

    return [_row_to_item(r) for r in rows]


# ── POST /history (upsert) ────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_200_OK)
async def upsert_history_item(
    item: HistoryItemSchema,
    current_user: dict = Depends(get_current_user),
):
    """
    Save or update a query history item for the authenticated user.
    Uses Supabase upsert (ON CONFLICT DO UPDATE) keyed on (id, user_id).
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing from claims")

    payload = {
        "id": item.id,
        "user_id": user_id,
        "query": item.query,
        "label": item.label,
        "timestamp": item.timestamp,
        "result_count": item.resultCount or 0,
        "top_file": item.topFile,
        "snapshot": item.snapshot,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _rest_url(),
            headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates"},
            json=payload,
        )

    if resp.status_code not in (200, 201, 204):
        raise HTTPException(status_code=502, detail=f"Failed to save history: {resp.text}")

    return {"status": "success", "id": item.id}


# ── DELETE /history  (clear all) ──────────────────────────────────────────────

@router.delete("", status_code=status.HTTP_200_OK)
async def clear_all_history(current_user: dict = Depends(get_current_user)):
    """
    Delete ALL query history items for the authenticated user.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing from claims")

    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            _rest_url(),
            headers={**_sb_headers(), "Prefer": ""},
            params={"user_id": f"eq.{user_id}"},
        )

    if resp.status_code not in (200, 204):
        raise HTTPException(status_code=502, detail="Failed to clear history")

    return {"status": "cleared"}


# ── DELETE /history/{item_id}  (delete one) ───────────────────────────────────

@router.delete("/{item_id}", status_code=status.HTTP_200_OK)
async def delete_history_item(
    item_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a single history item belonging to the authenticated user.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing from claims")

    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            _rest_url(),
            headers={**_sb_headers(), "Prefer": ""},
            params={
                "id": f"eq.{item_id}",
                "user_id": f"eq.{user_id}",
            },
        )

    if resp.status_code not in (200, 204):
        raise HTTPException(status_code=502, detail="Failed to delete history item")

    return {"status": "deleted", "id": item_id}


