from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

class VectorDBService:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2') # Generates 384-dim vectors
        self.collection_name = "user_documents"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            print(f"Created Qdrant collection: '{self.collection_name}'")

    async def index_document(self, doc_id, text: str, metadata: dict):
        """Generates embeddings and stores them in Qdrant."""
        if not text:
            text = metadata.get("original_name", "")
            
        print(f"[QDRANT] Encoding text of length {len(text)}...")
        vector = self.encoder.encode(text).tolist()
        
        # Merge text into metadata payload
        payload = {**metadata, "text_content": text}
        
        print(f"[QDRANT] Upserting point id={doc_id} into collection '{self.collection_name}'...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        print(f"[QDRANT] Upsert complete.")

    async def search(self, query: str, limit: int = 5):
        """Performs semantic search across stored documents."""
        query_vector = self.encoder.encode(query).tolist()
        
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )