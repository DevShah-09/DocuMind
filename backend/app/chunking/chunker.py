"""
chunker.py — Intelligent Recursive Text Chunker
================================================
Splits document text into semantic chunks with overlap.

Strategy:
  - Recursive splitting on paragraph → sentence → word boundaries
  - Target: ~600 tokens (~2 400 chars), 150-char overlap
  - SHA-256 per chunk for dedup prevention
  - Preserves chunk order, page association, and document metadata
"""

from __future__ import annotations

import hashlib
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Chunk size constants ────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 2400     # ~600 tokens at ~4 chars/token
DEFAULT_CHUNK_OVERLAP = 200   # overlap to preserve context at boundaries
MIN_CHUNK_SIZE = 100          # skip trivially short chunks


@dataclass
class ChunkResult:
    """Represents a single text chunk with full provenance metadata."""
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        """Convert to a flat dict suitable for Qdrant payload storage."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            **self.metadata,
        }


def _make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Stable SHA-256-based chunk ID — same content always produces same ID."""
    raw = f"{document_id}::{chunk_index}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RecursiveTextChunker:
    """
    Recursive text splitter that mimics LangChain's RecursiveCharacterTextSplitter
    without the dependency.

    Splits preferring: double-newline → single-newline → period+space → space.
    Falls back to hard character splits only if no natural boundary is found.
    """

    # Ordered list of separators — try longest/most semantic first
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ── Public API ──────────────────────────────────────────────────────────────

    def chunk_document(
        self,
        text: str,
        document_id: str,
        page_number: int = 1,
        extra_metadata: Optional[dict] = None,
    ) -> List[ChunkResult]:
        """
        Chunk a document's full text into overlapping semantic chunks.

        Args:
            text:           The raw extracted text of the document.
            document_id:    The parent document UUID.
            page_number:    Page number the text originated from (default 1 for
                            single-page or merged-page documents).
            extra_metadata: Additional fields to embed in each chunk's metadata
                            (e.g. user_id, document_type, filename, summary).

        Returns:
            Ordered list of ChunkResult objects.
        """
        if not text or not text.strip():
            logger.warning("[CHUNKER] Empty text for document_id=%s — skipping", document_id)
            return []

        meta = extra_metadata or {}
        raw_chunks = self._recursive_split(text.strip())

        results: List[ChunkResult] = []
        offset = 0

        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                logger.debug("[CHUNKER] Skipping trivially short chunk (idx=%d)", idx)
                continue

            # Locate char position within the original text for provenance
            try:
                char_start = text.index(chunk_text, offset)
                char_end = char_start + len(chunk_text)
                offset = max(0, char_end - self.chunk_overlap)
            except ValueError:
                char_start = 0
                char_end = len(chunk_text)

            chunk_id = _make_chunk_id(document_id, idx, chunk_text)

            results.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=idx,
                    page_number=page_number,
                    text=chunk_text.strip(),
                    char_start=char_start,
                    char_end=char_end,
                    metadata=meta,
                )
            )

        logger.info("[CHUNKER] document_id=%s → %d chunks", document_id, len(results))
        return results

    def chunk_pages(
        self,
        pages: List[str],
        document_id: str,
        extra_metadata: Optional[dict] = None,
    ) -> List[ChunkResult]:
        """
        Chunk a list of per-page texts, preserving page numbers.
        Useful when the document has been analysed page by page.
        """
        all_chunks: List[ChunkResult] = []
        global_idx = 0

        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = self.chunk_document(
                text=page_text,
                document_id=document_id,
                page_number=page_num,
                extra_metadata=extra_metadata,
            )
            # Re-index so chunk_index is global, not per-page
            for chunk in page_chunks:
                chunk.chunk_index = global_idx
                chunk.chunk_id = _make_chunk_id(document_id, global_idx, chunk.text)
                global_idx += 1
            all_chunks.extend(page_chunks)

        return all_chunks

    # ── Internal splitting logic ────────────────────────────────────────────────

    def _recursive_split(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """Recursively split text using the separator hierarchy."""
        if separators is None:
            separators = self.SEPARATORS

        final_chunks: List[str] = []

        # Find the first separator that exists in this text
        separator = separators[-1]  # fallback: hard char split
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split and process each part
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split as last resort
            splits = list(text)

        # Merge short splits back together using a sliding window
        good_splits: List[str] = []
        current = ""

        for split in splits:
            candidate = (current + separator + split).strip() if current else split.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                # If the individual split itself is too long, recurse
                if len(split) > self.chunk_size and len(separators) > 1:
                    sub_chunks = self._recursive_split(split, separators[1:])
                    good_splits.extend(sub_chunks)
                    current = ""
                else:
                    current = split.strip()

        if current:
            good_splits.append(current)

        # Add overlap between consecutive chunks
        return self._add_overlap(good_splits)

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Re-join with overlap so consecutive chunks share context.
        Each chunk gets a suffix from the previous chunk prepended.
        """
        if len(chunks) <= 1:
            return chunks

        result: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap:].strip()
            combined = (prev_tail + " " + chunks[i]).strip() if prev_tail else chunks[i]
            result.append(combined)

        return result
