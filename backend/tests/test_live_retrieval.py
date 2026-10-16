import asyncio
import sys
sys.path.insert(0, ".")

from app.core.config import settings
from app.embeddings.embedding_service import embedding_service
from app.vectorstore.qdrant_store import qdrant_store
from supabase import create_client


async def run_live_test():
    print("Connecting to Supabase...")
    s = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    users = s.auth.admin.list_users()
    if not users:
        print("FAIL: No users in Supabase")
        return
    real_user_id = str(users[0].id)
    print(f"Supabase connection: OK, active user: {real_user_id[:8]}...")

    query = "Shubham Shah resume skills and education"
    print(f"Embedding query via BGE: '{query}'...")
    vec = await embedding_service.embed_query(query)
    print(f"Vector produced successfully, dimension: {len(vec)}")

    print("Querying Qdrant Cloud document_chunks with user isolation filter...")
    chunks = await qdrant_store.search_chunks(
        query_vector=vec,
        user_id=real_user_id,
        limit=5,
        score_threshold=0.3,
    )
    print(f"Real retrieval status: SUCCESS! Retrieved {len(chunks)} chunks.")
    for i, c in enumerate(chunks):
        p = c.payload or {}
        print(f"  [{i}] Score: {c.score:.4f} | File: {p.get('filename')} | Type: {p.get('document_type')} | Year: {p.get('year')}")


if __name__ == "__main__":
    asyncio.run(run_live_test())
