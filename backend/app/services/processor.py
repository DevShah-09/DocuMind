import uuid
import json
from app.services.vision_model import VisionProcessor
from app.services.cloud_storage import R2Service
from app.services.vector_db import VectorDBService

class DocumentOrchestrator:
    def __init__(self):
        self.vision = VisionProcessor()
        self.r2 = R2Service()
        self.vector_db = VectorDBService()

    async def process_and_store(self, file_content: bytes, original_name: str, content_type: str):
        # Step 1: Analyze Document
        analysis_json = await self.vision.analyze_document(file_content)
        
        try:
            analysis = json.loads(analysis_json)
            category = analysis.get("category", "Unsorted")
            text_content = analysis.get("text", "")
        except Exception:
            category = "Unsorted"
            text_content = ""

        # Step 2: Storage in Cloudflare R2
        doc_id = str(uuid.uuid4())
        r2_key = f"{doc_id}-{original_name}"
        await self.r2.upload_file(file_content, r2_key, content_type)

        # Step 3: Index in Qdrant
        await self.vector_db.index_document(
            doc_id=doc_id,
            text=text_content,
            metadata={"original_name": original_name, "category": category, "r2_key": r2_key}
        )

        return {
            "file_id": r2_key,
            "category": category,
            "qdrant_id": doc_id,
            "storage_provider": "Cloudflare R2"
        }