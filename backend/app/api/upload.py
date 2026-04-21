import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.processor import DocumentOrchestrator

router = APIRouter(prefix="/documents", tags=["Documents"])

# Initialize Orchestrator which ties everything together
orchestrator = DocumentOrchestrator()

# Allowed formats and max size
ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

@router.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    """
    Cloud Pipeline: Frontend -> FastAPI -> PyMuPDF -> Qwen2.5-VL -> Cloudflare R2 + Qdrant
    """
    try:
        # File Validation
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File extension '{file_extension}' not allowed. Must be one of {ALLOWED_EXTENSIONS}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")
        elif len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        print(f"[API] Processing upload for: {file.filename}")

        # Run the full categorization and storage pipeline via Orchestrator
        result = await orchestrator.process_and_store(
            file_content=content,
            original_name=file.filename,
            content_type=file.content_type,
            user_id="test_user_123" # Hardcoded for MVP, should come from Auth
        )

        print(f"[API] Successfully processed {file.filename}.")
        
        return {
            "message": "Upload successful",
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """
    Deletes document from storage and vector database.
    """
    try:
        # 1. Get document metadata from Qdrant
        doc = orchestrator.vector_db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 2. Delete physical file from storage
        file_key = doc.payload.get("bucket_path")
        if file_key:
            orchestrator.storage.delete_file(file_key)
            
        # 3. Delete point from Qdrant
        orchestrator.vector_db.delete_document(doc_id)
        
        return {"message": "Document deleted successfully", "doc_id": doc_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] DELETE ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
