"""
qdrant_client_singleton.py — Shared Qdrant Client
===================================================
A single QdrantClient instance shared across all services.

Why this exists:
  Qdrant's LOCAL (file-based) mode acquires an exclusive lock on the
  storage folder. If two QdrantClient instances point at the same path
  the second one raises:
    "Storage folder is already accessed by another instance"

  All services (VectorDBService, QdrantStore) must import the client
  from here instead of constructing their own.
"""

import logging
import os

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_storage_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "qdrant_storage")
)
os.makedirs(_storage_path, exist_ok=True)

logger.info("[QDRANT] Shared client → local storage: %s", _storage_path)
shared_qdrant_client = QdrantClient(path=_storage_path)
