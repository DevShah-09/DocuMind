"""
test_sse_upload.py — Integration and Unit Tests for SSE Upload Progress Lifecycle
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.services.upload_job_manager import upload_job_manager, UploadJob
from app.core.security import get_current_user

app = FastAPI()
app.include_router(upload_router)
client = TestClient(app)

TEST_USER_ID = "00000000-0000-0000-0000-000000000001"
TEST_USER = {"sub": TEST_USER_ID, "email": "test@documind.ai"}


app.dependency_overrides[get_current_user] = lambda: TEST_USER


def test_validation_errors():
    # 1. Invalid extension
    res = client.post(
        "/documents/upload",
        files={"file": ("test.exe", b"binarycontent", "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "not allowed" in res.json()["detail"].lower()

    # 2. Empty file
    res = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


async def _test_job_manager_lifecycle_and_replay_async():
    job_id = "test-job-lifecycle-123"
    job = upload_job_manager.create_job(job_id, TEST_USER_ID, "sample_invoice.pdf")
    assert job.status == "queued"
    assert len(job.history) == 1
    assert job.history[0].stage == "uploading"

    # Emit milestones
    await upload_job_manager.emit_progress(job_id, "reading", 25, "Reading pages")
    await upload_job_manager.emit_progress(job_id, "analyzing", 50, "Analyzing with AI")
    await upload_job_manager.emit_progress(job_id, "indexing", 85, "Indexing Qdrant")
    await upload_job_manager.complete_job(job_id, {"id": job_id, "category": "invoice"})

    assert job.status == "done"
    assert len(job.history) == 5

    # Simulate a client connecting late / reconnecting:
    # All historical events must be replayed up to 'done'
    replayed_events = []
    async for sse_chunk in upload_job_manager.stream_events(job_id, TEST_USER_ID):
        replayed_events.append(sse_chunk)

    # 5 events replayed
    assert len(replayed_events) == 5
    assert "event: done" in replayed_events[-1]
    assert "invoice" in replayed_events[-1]


def test_job_manager_lifecycle_and_replay():
    """Sync pytest wrapper -- runs the async body via asyncio.run() without needing pytest-asyncio."""
    asyncio.run(_test_job_manager_lifecycle_and_replay_async())


def test_status_endpoint():
    job_id = "test-status-job-456"
    upload_job_manager.create_job(job_id, TEST_USER_ID, "statement.pdf")

    res = client.get(f"/documents/upload/{job_id}/status")
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"


def test_unauthorized_access():
    job_id = "test-unauth-job-789"
    upload_job_manager.create_job(job_id, "other-user-999", "secret.pdf")

    # Current user is TEST_USER_ID, so looking up another user's job returns 404
    res = client.get(f"/documents/upload/{job_id}/status")
    assert res.status_code == 404


def test_sse_http_streaming():
    from app.api.upload import get_user_from_header_or_query
    app.dependency_overrides[get_user_from_header_or_query] = lambda: TEST_USER

    job_id = "test-stream-http-101"
    job = upload_job_manager.create_job(job_id, TEST_USER_ID, "tax_receipt.pdf")
    # Finish the job synchronously for HTTP stream test
    job.add_event("done", "indexing", 100, "Done indexing", data={"id": job_id, "category": "tax_document"})
    job.status = "done"

    res = client.get(f"/documents/upload/{job_id}/events")
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "no-cache" in res.headers["cache-control"]
    assert "event: done" in res.text
    assert "tax_document" in res.text


if __name__ == "__main__":
    print("[1/5] Running test_validation_errors()...")
    test_validation_errors()
    print("  -> PASSED")

    print("[2/5] Running test_job_manager_lifecycle_and_replay()...")
    asyncio.run(test_job_manager_lifecycle_and_replay())
    print("  -> PASSED")

    print("[3/5] Running test_status_endpoint()...")
    test_status_endpoint()
    print("  -> PASSED")

    print("[4/5] Running test_unauthorized_access()...")
    test_unauthorized_access()
    print("  -> PASSED")

    print("[5/5] Running test_sse_http_streaming()...")
    test_sse_http_streaming()
    print("  -> PASSED")

    print("\nALL SSE UPLOAD LIFECYCLE TESTS PASSED!")
