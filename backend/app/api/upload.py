import uuid
import json
import os
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.vision_model import VisionProcessor
from app.services.vector_db import VectorDBService

router = APIRouter(prefix="/documents", tags=["Documents"])

# Initialize services
vision_processor = VisionProcessor()
vector_db = VectorDBService()

# Local storage configuration
UPLOAD_DIR = "./local_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    """
    Local Pipeline: Frontend -> Local Disk -> Qwen2.5-VL -> Qdrant
    """
    try:
        # Generate unique ID and local path
        doc_id = uuid.uuid4()  # Keep as UUID object, not string — Qdrant requires UUID or int
        doc_id_str = str(doc_id)
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{doc_id_str}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        print(f"[UPLOAD] Starting upload for: {file.filename} -> {unique_filename}")

        # 1. Save file to local storage
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"[UPLOAD] File saved to disk: {file_path}")

        # 2. Dynamic Categorization using Qwen2.5-VL
        raw_analysis = await vision_processor.analyze_document(content)
        print(f"[UPLOAD] Raw vision model response: {raw_analysis}")
        
        # Strip potential markdown formatting before parsing JSON
        if raw_analysis.startswith("```json"):
            raw_analysis = raw_analysis.replace("```json", "", 1).replace("```", "").strip()
        elif raw_analysis.startswith("```"):
            raw_analysis = raw_analysis.replace("```", "", 1).replace("```", "").strip()
            
        try:
            analysis_data = json.loads(raw_analysis)
        except json.JSONDecodeError:
            print(f"[UPLOAD] Failed to parse model JSON output: {raw_analysis}")
            analysis_data = {}
        
        category = analysis_data.get("category", "Unsorted")
        extracted_text = analysis_data.get("text") or file.filename  # Fallback to filename so vector is never empty
        print(f"[UPLOAD] Category: '{category}' | Text length: {len(extracted_text)}")

        # 3. Index in Qdrant with local metadata
        payload = {
            "original_name": file.filename,
            "local_filename": unique_filename,
            "category": category,
            "text_content": extracted_text,
            "content_type": file.content_type
        }
        
        print(f"[UPLOAD] Sending to Qdrant with id={doc_id_str}...")
        await vector_db.index_document(doc_id, extracted_text, payload)
        print(f"[UPLOAD] Successfully indexed in Qdrant!")

        return {
            "id": doc_id_str,
            "filename": file.filename,
            "category": category,
            "status": "Successfully indexed locally"
        }

    except Exception as e:
        print(f"[UPLOAD] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")





#with cloudfare R2
# from fastapi import APIRouter, UploadFile, File
# from app.services.processor import DocumentOrchestrator

# router = APIRouter()
# orchestrator = DocumentOrchestrator()

# @router.post("/process-document")
# async def handle_upload(file: UploadFile = File(...)):
#     content = await file.read()
    
#     # Run the categorization and storage pipeline
#     result = await orchestrator.process_and_store(
#         file_content=content,
#         original_name=file.filename,
#         content_type=file.content_type
#     )
    
#     return {
#         "message": "Upload successful",
#         "data": result
#     }