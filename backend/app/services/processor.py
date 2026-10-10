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
from app.services.local_classifier import local_classifier, SCORE_THRESHOLD, MARGIN_THRESHOLD
from app.retrieval.query_parser import DOC_TYPE_MAP

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

_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            print("[PROCESSOR] Lazy loading EasyOCR model (CPU mode)...")
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"[PROCESSOR] Failed to load EasyOCR: {e}")
            _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader is not False else None

def extract_text_with_easyocr(images_base64: list[str]) -> str:
    reader = get_easyocr_reader()
    if not reader:
        return ""
    try:
        all_text = []
        for b64 in images_base64:
            img_bytes = base64.b64decode(b64)
            results = reader.readtext(img_bytes, detail=0)
            if results:
                all_text.append(" ".join(results))
        return "\n\n".join(all_text).strip()
    except Exception as e:
        print(f"[PROCESSOR] EasyOCR extraction error: {e}")
        return ""

AMBIGUOUS_KEYWORDS = {"mobile", "gas", "water", "bank", "tax", "bill", "pan", "medical", "electric", "statement", "policy"}

def fallback_classify_from_text(text: str, clf_result=None) -> tuple[str, int]:
    """
    When cloud Vision/Text LLM API fails (e.g. 402 Insufficient Credits),
    classify using explicit content keyword scanning or calibrated local BGE prediction.
    """
    text_lower = text.lower()
    
    # Check for Student / College Identity Card specific terms
    student_id_terms = ["identity card", "student id", "college of engineering", "roll no", "enrollment no", "student identity card"]
    if any(term in text_lower for term in student_id_terms):
        return "student_id_card", 85

    # 1. High-precision content keyword scanning against DOC_TYPE_MAP phrases
    for phrase, slug in sorted(DOC_TYPE_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in AMBIGUOUS_KEYWORDS:
            continue
        if phrase in text_lower:
            return slug, 85

    # 2. Local BGE top prediction check (requires score >= 0.58 AND margin >= 0.05)
    if clf_result and clf_result.top_score >= 0.58 and clf_result.margin >= 0.05:
        return clf_result.category, round(clf_result.top_score * 100)

    return "miscellaneous", 50

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

                # If text is short (<50 words), render page images for Vision LLM backup
                # DO NOT erase extracted_text if text exists!
                if len(extracted_text.split()) < 50:
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

        # Step 2: Analyze Document
        # Path A — Digital text PDF: try local BGE classifier first.
        #   If confident (score >= threshold AND margin >= threshold) → skip Qwen-72B entirely.
        #   If uncertain → fall through to Qwen-72B for full analysis.
        # Path B — Scanned/image PDF: always use Qwen-72B VL (needs vision).
        if extracted_text:
            print(f"[PROCESSOR] Fast path: Digital text PDF ({len(extracted_text.split())} words). Running local BGE classifier...")

            clf_result = local_classifier.classify(extracted_text, max_chars=2500)
            print(
                f"[CLASSIFIER] top={clf_result.category} score={clf_result.top_score:.4f} "
                f"second={clf_result.second_score:.4f} margin={clf_result.margin:.4f} "
                f"→ {'LOCAL ACCEPT' if clf_result.accepted else 'FALLBACK→Qwen-72B'}"
            )

            if clf_result.accepted:
                # Build a minimal analysis JSON matching Qwen's schema.
                print(f"[PROCESSOR] Local classification accepted: {clf_result.category} (no API call)")
                analysis_json_str = json.dumps({
                    "is_blur": False,
                    "blur_reason": None,
                    "category": clf_result.category,
                    "confidence": round(clf_result.top_score * 100),
                    "metadata": {},
                    "summary": f"Document classified locally as {clf_result.category} "
                               f"(BGE score={clf_result.top_score:.2f}, margin={clf_result.margin:.2f}).",
                    "text": extracted_text,
                    "_classifier": "local_bge",   # diagnostic tag
                })
            else:
                # Uncertain — delegate to Qwen-72B for full analysis
                print(f"[PROCESSOR] Classifier uncertain — calling Qwen-72B for full analysis...")
                analysis_json_str = await self.vision.analyze_text_document(extracted_text, original_name)

                # Graceful degradation: if Qwen-72B fails (e.g. 402 payment required, timeout)
                try:
                    parsed_check = json.loads(analysis_json_str)
                    if parsed_check.get("category") == "miscellaneous":
                        fb_category, fb_conf = fallback_classify_from_text(extracted_text, clf_result)
                        if fb_category != "miscellaneous":
                            print(
                                f"[PROCESSOR] Qwen API failed/miscellaneous — using content fallback: "
                                f"{fb_category} ({fb_conf}% conf, BGE score={clf_result.top_score:.4f})"
                            )
                            analysis_json_str = json.dumps({
                                "is_blur": False,
                                "blur_reason": None,
                                "category": fb_category,
                                "confidence": fb_conf,
                                "metadata": {},
                                "summary": f"Classified as {fb_category} based on document text content (BGE score={clf_result.top_score:.2f}).",
                                "text": extracted_text,
                                "_classifier": "content_bge_fallback",
                            })
                        else:
                            print(f"[PROCESSOR] Content fallback uncertain — retaining miscellaneous.")
                except (json.JSONDecodeError, Exception) as err:
                    print(f"[PROCESSOR] Degradation parse error: {err}")

        elif images_base64:
            print(f"[PROCESSOR] Scanned image/PDF path ({len(images_base64)} image(s)). Calling fast Vision LLM (Gemini 2.5 Flash)...")
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
        raw_conf = analysis.get("confidence", 0)
        try:
            conf_val = float(raw_conf)
            if 0.0 < conf_val <= 1.0:
                confidence = round(conf_val * 100)
            else:
                confidence = round(conf_val)
        except Exception:
            confidence = 0
        metadata_dict = analysis.get("metadata", {})
        summary = analysis.get("summary", "")
        text_content = analysis.get("text", "")

        # Self-Learning Dynamic Prototype Feedback Loop:
        # When Qwen-72B or Vision LLM classifies a document, automatically train the local BGE model
        # so future uploads of this document type are recognized locally (0 cost, ultra-fast).
        # NEVER learn from fallback heuristic guesses!
        is_fallback = analysis.get("_classifier") in ("content_bge_fallback", "easyocr_bge_content_scan")
        if category and category != "miscellaneous" and text_content and not is_fallback:
            local_classifier.learn_prototype(category, text_content)

        # Fallback if text is empty or short so chunker doesn't discard it (<50 chars)
        if not text_content or len(text_content.strip()) < 50:
            text_content = (
                f"Document Name: {original_name}\n"
                f"Category: {category}\n"
                f"Summary: {summary or original_name}\n"
                f"Content: {text_content or original_name}"
            )

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