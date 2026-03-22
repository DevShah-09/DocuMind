from fastapi import APIRouter, HTTPException
from app.services.vector_db import VectorDBService
from collections import defaultdict

router = APIRouter(prefix="/collections", tags=["Collections"])
vector_db = VectorDBService()

@router.get("/summary")
async def get_dynamic_collections():
    """
    Fetches all documents and groups them by the categories 
    assigned by the AI for the 'My Collections' view.
    """
    try:
        # 1. Retrieve all stored points from Qdrant [cite: 41, 557]
        points, _ = vector_db.client.scroll(
            collection_name=vector_db.collection_name,
            with_payload=True,
            limit=100
        )

        category_map = defaultdict(list)
        
        for point in points:
            payload = point.payload
            cat_name = payload.get("category", "Unsorted")
            
            # Construct a local URL pointing to your FastAPI static mount
            file_url = f"http://localhost:8000/static/{payload['local_filename']}"
            
            category_map[cat_name].append({
                "id": point.id,
                "name": payload.get("original_name"),
                "url": file_url
            })

        # 2. Format as a list for the frontend .map() function
        summary = []
        for cat, docs in category_map.items():
            summary.append({
                "category_name": cat,
                "count": len(docs),
                "documents": docs
            })

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query")
async def semantic_search(q: str):
    """
    Semantic search across all local documents[cite: 36, 48].
    """
    results = await vector_db.search(query=q, limit=5)
    
    return [{
        "name": res.payload["original_name"],
        "category": res.payload["category"],
        "url": f"http://localhost:8000/static/{res.payload['local_filename']}",
        "score": res.score
    } for res in results]





#with Cloudfare R2
# from fastapi import APIRouter
# from app.services.vector_db import VectorDBService
# from app.services.cloud_storage import R2Service
# from collections import defaultdict

# router = APIRouter(prefix="/collections", tags=["Collections"])
# vector_db = VectorDBService()
# r2_service = R2Service()

# @router.get("/summary")
# async def get_dynamic_summary():
#     # 1. Get all documents from Qdrant
#     points, _ = vector_db.client.scroll(
#         collection_name=vector_db.collection_name,
#         with_payload=True,
#         limit=100
#     )

#     # 2. Dynamic Grouping: We don't know the categories until we see the data
#     category_map = defaultdict(list)
    
#     for point in points:
#         payload = point.payload
#         # If Qwen labeled it "Electricity Bill", that becomes a new category
#         cat_name = payload.get("category", "Unsorted")
        
#         view_url = r2_service.get_signed_url(payload["r2_key"])
        
#         category_map[cat_name].append({
#             "id": point.id,
#             "name": payload.get("original_name"),
#             "url": view_url
#         })

#     # 3. Format for Frontend
#     # This sends a list of categories and their counts
#     summary = []
#     for cat, docs in category_map.items():
#         summary.append({
#             "category_name": cat,
#             "count": len(docs),
#             "documents": docs
#         })

#     return summary