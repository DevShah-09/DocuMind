import uuid
import json
import datetime
import hashlib
import base64
import re
import traceback
import fitz  # PyMuPDF
from app.services.vision_model import VisionProcessor
from app.services.cloud_storage import S3StorageService
from app.services.vector_db import VectorDBService

MONTH_MAP = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
    "january": "January", "february": "February", "march": "March",
    "april": "April", "june": "June", "july": "July", "august": "August",
    "september": "September", "october": "October", "november": "November",
    "december": "December",
}

_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)

class DocumentOrchestrator:
    def __init__(self):
        self.vision = VisionProcessor()
        self.storage = S3StorageService()
        self.vector_db = VectorDBService()

    async def process_and_store(
        self, 
        file_content: bytes, 
        encrypted_content: bytes,
        original_name: str, 
        content_type: str, 
        user_id: str,
        file_id: str
    ):
        # Metadata Generation
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check for duplicates before processing — scoped to THIS user only
        existing_doc = await self.vector_db.check_duplicate(file_hash, user_id=user_id)
        if existing_doc:
            print(f"[PROCESSOR] Duplicate detected: {original_name} already exists for user={user_id}.")
            raise ValueError("This file already exists in your collections.")

        doc_id = file_id  # Use the shared file_id from API
        upload_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Step 1: Smart Hybrid Pre-processing (Digital Text Extraction vs Scanned Vision)
        images_base64 = []
        extracted_text = ""

        if "pdf" in content_type.lower() or original_name.lower().endswith(".pdf"):
            try:
                doc = fitz.open(stream=file_content, filetype="pdf")
                # Try extracting digital text directly from pages
                full_text = []
                for page in doc:
                    t = page.get_text("text").strip()
                    if t:
                        full_text.append(t)
                extracted_text = "\n\n".join(full_text).strip()

                # If text is minimal (<50 words), treat as scanned PDF and render images
                if len(extracted_text.split()) < 50:
                    extracted_text = ""
                    # 2-page, 1.25x first-pass optimization:
                    # - Reduced from 3 pages → max 2 pages (less payload to the VL model)
                    # - Reduced from 1.5x → 1.25x zoom (~116 DPI; balances legibility vs size)
                    # This is a page-count + resolution change only — model is unchanged.
                    # Benchmark actual latency before considering further optimizations (e.g. 7B VL).
                    max_pages = min(2, len(doc))
                    for page_num in range(max_pages):
                        page = doc.load_page(page_num)
                        mat = fitz.Matrix(1.25, 1.25)
                        pix = page.get_pixmap(matrix=mat)
                        img_bytes = pix.tobytes("jpeg")
                        images_base64.append(base64.b64encode(img_bytes).decode('utf-8'))
                doc.close()
            except Exception as e:
                print(f"[PROCESSOR] PDF processing error: {e}")
        else:
            # Direct Image upload
            images_base64.append(base64.b64encode(file_content).decode('utf-8'))

        # Step 2: Analyze Document (Fast Text Path vs Vision LLM Path)
        if extracted_text:
            print(f"[PROCESSOR] Fast path: Digital text PDF detected ({len(extracted_text.split())} words). Skipping Vision LLM!")
            analysis_json_str = await self.vision.analyze_text_document(extracted_text, original_name)
        elif images_base64:
            print(f"[PROCESSOR] Scanned path: Calling Vision LLM (Qwen 72B) for {len(images_base64)} image(s)...")
            analysis_json_str = await self.vision.analyze_document(images_base64)
        else:
            raise ValueError("No extractable content or images found in the document.")
        
        try:
            # Clean up potential markdown code block format from the LLM
            clean_json = analysis_json_str
            if clean_json.startswith("```json"):
                clean_json = clean_json.replace("```json", "", 1).replace("```", "").strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.replace("```", "", 1).replace("```", "").strip()
                
            analysis = json.loads(clean_json)
        except Exception as e:
            print(f"JSON Parse Error: {e} -> Raw string: {analysis_json_str}")
            analysis = {}

        # Quality check: Reject if image is blurry or illegible
        if analysis.get("is_blur"):
            reason = analysis.get("blur_reason", "The photo is too blurry or low quality.")
            print(f"[PROCESSOR] Quality check failed: {reason}")
            raise ValueError(f"Image Quality Issue: {reason} Please re-upload a clearer version.")

        category = analysis.get("category", "miscellaneous")
        confidence = analysis.get("confidence", 0)
        metadata_dict = analysis.get("metadata", {})
        summary = analysis.get("summary", "")
        text_content = analysis.get("text", "")

        # Fallback if text is empty so vector isn't empty
        if not text_content:
            text_content = original_name

        # Step 3: Storage in Local Disk (USE ENCRYPTED CONTENT)
        file_key = f"{doc_id}-{original_name}"
        await self.storage.upload_file(encrypted_content, file_key, content_type)

        # Step 4: Index in Qdrant Vector DB (legacy user_documents)
        payload = {
            "original_name": original_name,
            "category": category,
            "confidence": confidence,
            "metadata": metadata_dict,
            "summary": summary,
            "bucket_path": file_key,
            "upload_timestamp": upload_timestamp,
            "file_hash": file_hash,
            "user_id": user_id,
        }
        
        await self.vector_db.index_document(
            doc_id=doc_id,
            text=text_content,
            metadata=payload
        )

        # Step 5: Index chunks in document_chunks collection (BGE 768-dim)
        try:
            from app.chunking.chunker import RecursiveTextChunker
            from app.vectorstore.qdrant_store import qdrant_store
            from app.embeddings.embedding_service import embedding_service
            
            # Extract year, month, and date for chunk-level metadata
            doc_year = None
            for key in ("year", "Year"):
                val = metadata_dict.get(key)
                if val is not None:
                    try:
                        y = int(val)
                        if 1950 <= y <= 2099:
                            doc_year = y
                            break
                    except (TypeError, ValueError):
                        pass
            if not doc_year:
                text_to_search = f"{original_name} {summary} {text_content}"
                m = re.search(r"\b(19[5-9]\d|20[0-4]\d)\b", text_to_search)
                if m:
                    doc_year = int(m.group(1))

            doc_month = None
            for key in ("month", "Month"):
                val = metadata_dict.get(key)
                if isinstance(val, str):
                    m_normalized = MONTH_MAP.get(val.lower().strip())
                    if m_normalized:
                        doc_month = m_normalized
                        break
            if not doc_month:
                text_to_search = f"{original_name} {summary} {text_content}"
                m = _MONTH_RE.search(text_to_search)
                if m:
                    doc_month = MONTH_MAP.get(m.group(1).lower())

            doc_date = None
            for key in ("date", "Date", "bill_date", "statement_date"):
                val = metadata_dict.get(key)
                if val is not None:
                    doc_date = str(val)
                    break

            # Prepare common metadata for all chunks
            chunk_metadata = {
                "user_id": user_id,
                "document_id": doc_id,
                "document_type": category,
                "filename": original_name,
                "summary": summary,
                "date": doc_date,
                "year": doc_year,
                "month": doc_month,
                "entities": metadata_dict.get("entities", []) if isinstance(metadata_dict.get("entities"), list) else [],
                "tags": metadata_dict.get("tags", []) if isinstance(metadata_dict.get("tags"), list) else [],
                "metadata": metadata_dict,
                "bucket_path": file_key,
            }

            # Use chunker to split document text
            chunker = RecursiveTextChunker()
            chunks = chunker.chunk_document(
                text=text_content,
                document_id=doc_id,
                extra_metadata=chunk_metadata,
            )

            if chunks:
                # Embed chunk texts in batch
                chunk_texts = [c.text for c in chunks]
                vectors = await embedding_service.embed_texts(chunk_texts)

                # Prepare payloads for upsert
                chunk_payloads = []
                for idx, chunk in enumerate(chunks):
                    payload = chunk.to_payload()
                    payload["vector"] = vectors[idx]
                    chunk_payloads.append(payload)

                # Upsert into Qdrant document_chunks collection
                await qdrant_store.upsert_chunks(chunk_payloads)
                print(f"[PROCESSOR] Indexed {len(chunks)} chunks in document_chunks collection.")
        except Exception as e:
            print(f"[PROCESSOR] ERROR indexing chunks: {e}")
            traceback.print_exc()

        return {
            "id": doc_id,
            "file_id": file_key,
            "original_name": original_name,
            "category": category,
            "confidence": confidence,
            "summary": summary,
            "storage_provider": "Local Storage"
        }