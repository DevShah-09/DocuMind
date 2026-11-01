"""
upload_job_manager.py — In-Memory Lifecycle & SSE Event Stream Manager
========================================================================
Manages asynchronous document upload jobs, milestone histories, and SSE
event streams for DocuMind AI.

Features:
  - Replay on reconnect: Missed events are replayed from job history so native
    EventSource reconnections never miss milestones or terminal states.
  - Heartbeat pings: Emits ': ping\\n\\n' every 5s during long LLM/OCR operations.
  - Automatic pruning: Cleans up jobs older than 30 minutes to prevent memory leaks.
  - Per-user scoping: Ensures users can only subscribe to their own upload jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Stages mapping to UI STAGES indices
STAGE_MAP = {
    "uploading": 0,
    "reading": 1,
    "analyzing": 2,
    "categorizing": 3,
    "indexing": 4,
}


@dataclass
class JobEvent:
    event: str          # "milestone" | "done" | "error"
    stage: str          # "uploading" | "reading" | "analyzing" | "categorizing" | "indexing" | "complete"
    stage_index: int    # 0 to 4
    progress: int       # 0 to 100
    message: str
    data: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        payload = {
            "stage": self.stage,
            "stageIndex": self.stage_index,
            "progress": self.progress,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        return f"event: {self.event}\ndata: {json.dumps(payload)}\n\n"


class UploadJob:
    def __init__(self, job_id: str, user_id: str, filename: str):
        self.job_id = job_id
        self.user_id = user_id
        self.filename = filename
        self.status = "queued"  # "queued" | "processing" | "done" | "failed" | "cancelled"
        self.current_stage = "uploading"
        self.current_progress = 5
        self.history: List[JobEvent] = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.listeners: Set[asyncio.Queue] = set()
        self.task: Optional[asyncio.Task] = None

    def add_event(self, event_type: str, stage: str, progress: int, message: str, data: Optional[dict] = None) -> JobEvent:
        stage_idx = STAGE_MAP.get(stage, 0)
        evt = JobEvent(
            event=event_type,
            stage=stage,
            stage_index=stage_idx,
            progress=progress,
            message=message,
            data=data,
        )
        self.history.append(evt)
        self.current_stage = stage
        self.current_progress = progress
        self.updated_at = time.time()

        # Broadcast to all connected SSE subscriber queues
        dead_listeners = set()
        for q in self.listeners:
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                dead_listeners.add(q)
        self.listeners -= dead_listeners
        return evt


class UploadJobManager:
    def __init__(self):
        self._jobs: Dict[str, UploadJob] = {}
        self._lock = asyncio.Lock()

    def create_job(self, job_id: str, user_id: str, filename: str) -> UploadJob:
        self._prune_stale_jobs()
        job = UploadJob(job_id=job_id, user_id=user_id, filename=filename)
        job.add_event(
            event_type="milestone",
            stage="uploading",
            progress=10,
            message=f"Received {filename}. Preparing pipeline...",
        )
        self._jobs[job_id] = job
        logger.info("[JobManager] Created job=%s for user=%s file=%s", job_id, user_id, filename)
        return job

    def get_job(self, job_id: str, user_id: Optional[str] = None) -> Optional[UploadJob]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if user_id and job.user_id != user_id:
            logger.warning("[JobManager] Unauthorized access attempt for job=%s by user=%s", job_id, user_id)
            return None
        return job

    async def emit_progress(self, job_id: str, stage: str, progress: int, message: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = "processing"
        job.add_event(
            event_type="milestone",
            stage=stage,
            progress=progress,
            message=message,
        )
        logger.debug("[JobManager] job=%s progress: %s (%d%%) - %s", job_id, stage, progress, message)

    async def complete_job(self, job_id: str, result: dict) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = "done"
        job.result = result
        job.add_event(
            event_type="done",
            stage="indexing",
            progress=100,
            message="Processing and indexing complete.",
            data=result,
        )
        logger.info("[JobManager] Completed job=%s for file=%s", job_id, job.filename)

    async def fail_job(self, job_id: str, error_msg: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = "failed"
        job.error = error_msg
        job.add_event(
            event_type="error",
            stage=job.current_stage,
            progress=job.current_progress,
            message=error_msg,
            data={"error": error_msg},
        )
        logger.error("[JobManager] Failed job=%s: %s", job_id, error_msg)

    async def cancel_job(self, job_id: str, user_id: Optional[str] = None) -> bool:
        job = self.get_job(job_id, user_id)
        if not job:
            return False
        if job.status in ("done", "failed", "cancelled"):
            return False

        job.status = "cancelled"
        job.error = "Upload cancelled by user."
        job.add_event(
            event_type="cancelled",
            stage=job.current_stage,
            progress=job.current_progress,
            message="Upload cancelled by user.",
            data={"cancelled": True},
        )
        if job.task and not job.task.done():
            job.task.cancel()

        logger.info("[JobManager] Cancelled job=%s for file=%s", job_id, job.filename)
        return True

    async def stream_events(self, job_id: str, user_id: str) -> AsyncGenerator[str, None]:
        """
        SSE Generator with historical replay and keep-alive heartbeats.
        """
        job = self.get_job(job_id, user_id)
        if not job:
            error_payload = {"stage": "error", "progress": 0, "message": "Job not found or access denied."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        queue: asyncio.Queue[Optional[JobEvent]] = asyncio.Queue(maxsize=100)
        job.listeners.add(queue)

        try:
            # 1. Replay historical events (ensures reconnected clients catch up instantly)
            for historical_event in list(job.history):
                yield historical_event.to_sse()

            # If job is already in terminal state, we are done
            if job.status in ("done", "failed", "cancelled"):
                return

            # 2. Stream new live events with keepalive pings
            while True:
                try:
                    # Wait up to 5 seconds for a new milestone
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    if event is None:
                        break
                    yield event.to_sse()
                    if event.event in ("done", "error", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Emit SSE comment heartbeat to keep connection alive
                    yield ": ping\n\n"
                    # Check if status transitioned while waiting
                    if job.status in ("done", "failed", "cancelled"):
                        break
        finally:
            job.listeners.discard(queue)

    def _prune_stale_jobs(self, max_age_seconds: int = 1800) -> None:
        now = time.time()
        to_remove = [
            jid for jid, j in self._jobs.items()
            if now - j.updated_at > max_age_seconds
        ]
        for jid in to_remove:
            del self._jobs[jid]


# Global singleton instance
upload_job_manager = UploadJobManager()
