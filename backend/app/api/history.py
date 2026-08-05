"""
history.py — User Query History API
====================================
Stores and retrieves user query session history.
Utilizes server storage scoped per authenticated user_id.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.security import get_current_user

router = APIRouter(prefix="/history", tags=["Query History"])

# ── User History Storage ──────────────────────────────────────────────────────
# Key: user_id (str) -> List of history item dicts
_USER_HISTORY_STORE: Dict[str, List[Dict[str, Any]]] = {}


class HistoryItemSchema(BaseModel):
    id: str
    query: str
    timestamp: str
    resultCount: Optional[int] = 0
    topFile: Optional[str] = None
    snapshot: Optional[Dict[str, Any]] = None
    label: Optional[str] = None


@router.get("", response_model=List[HistoryItemSchema])
async def get_user_history(current_user: dict = Depends(get_current_user)):
    """
    Fetch all query history items belonging to the authenticated user.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing from claims")
    
    user_items = _USER_HISTORY_STORE.get(user_id, [])
    return sorted(user_items, key=lambda x: x.get("timestamp", ""), reverse=True)


@router.post("", status_code=status.HTTP_200_OK)
async def upsert_history_item(
    item: HistoryItemSchema,
    current_user: dict = Depends(get_current_user)
):
    """
    Save or update a query history item for the authenticated user.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing from claims")
    
    if user_id not in _USER_HISTORY_STORE:
        _USER_HISTORY_STORE[user_id] = []

    user_items = _USER_HISTORY_STORE[user_id]
    
    # Remove existing entry with matching id if updating
    filtered = [existing for existing in user_items if existing.get("id") != item.id]
    
    item_dict = item.model_dump()
    filtered.insert(0, item_dict)
    
    # Keep up to 200 items per user
    _USER_HISTORY_STORE[user_id] = filtered[:200]
    return {"status": "success", "id": item.id}


@router.delete("/{item_id}", status_code=status.HTTP_200_OK)
async def delete_history_item(
    item_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a history item belonging to the authenticated user.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing from claims")

    if user_id in _USER_HISTORY_STORE:
        _USER_HISTORY_STORE[user_id] = [
            existing for existing in _USER_HISTORY_STORE[user_id]
            if existing.get("id") != item_id
        ]
    return {"status": "deleted", "id": item_id}
