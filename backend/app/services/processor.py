import uuid
import json
import datetime
import hashlib
import base64
import fitz  # PyMuPDF
from app.services.vision_model import VisionProcessor
from app.services.cloud_storage import LocalStorageService
from app.services.vector_db import VectorDBService

class DocumentOrchestrator:
    def __init__(self):
        self.vision = VisionProcessor()
        self.storage = LocalStorageService()
        self.vector_db = VectorDBService()

    async def process_and_store(self, file_content: bytes, original_name: str, content_type: str, user_id: str = "test_user_123"):
        # Metadata Generation
        file_hash = hashlib.sha256(file_content).hexdigest()
        doc_id = str(uuid.uuid4())
        upload_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Step 1: Pre-process file into Base64 images
        images_base64 = []
        if "pdf" in content_type.lower() or original_name.lower().endswith(".pdf"):
            try:
                # Open PDF from memory
                doc = fitz.open(stream=file_content, filetype="pdf")
                # Limit to first 3 pages to reduce token usage and speed up
                for page_num in range(min(3, len(doc))):
                    page = doc.load_page(page_num)
                    zoom = 2  # upscale for better OCR/Vision resolution
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("jpeg")
                    images_base64.append(base64.b64encode(img_bytes).decode('utf-8'))
                doc.close()
            except Exception as e:
                print(f"PDF processing error: {e}")
        else:
            # Direct Image
            images_base64.append(base64.b64encode(file_content).decode('utf-8'))

        # Step 2: Analyze Document with Vision Model
        if not images_base64:
            raise ValueError("No extractable images found in the document.")
            
        analysis_json_str = await self.vision.analyze_document(images_base64)
        
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

        category = analysis.get("category", "miscellaneous")
        confidence = analysis.get("confidence", 0)
        metadata_dict = analysis.get("metadata", {})
        summary = analysis.get("summary", "")
        text_content = analysis.get("text", "")

        # Fallback if text is empty so vector isn't empty
        if not text_content:
            text_content = original_name

        # Step 3: Storage in Local Disk
        file_key = f"{doc_id}-{original_name}"
        await self.storage.upload_file(file_content, file_key, content_type)

        # Step 4: Index in Qdrant Vector DB
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

        return {
            "id": doc_id,
            "file_id": file_key,
            "original_name": original_name,
            "category": category,
            "confidence": confidence,
            "summary": summary,
            "storage_provider": "Local Storage"
        }