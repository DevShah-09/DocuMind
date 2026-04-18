from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os


class VectorDBService:
    """
    Singleton wrapper around Qdrant local storage.
    Using a singleton ensures only ONE QdrantClient instance holds the lock
    on the local storage folder, preventing 'already accessed' errors.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Local file-based Qdrant — no Docker/server required
            storage_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "qdrant_storage")
            )
            os.makedirs(storage_path, exist_ok=True)
            cls._instance.client = QdrantClient(path=storage_path)
            cls._instance.encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
            cls._instance.collection_name = "user_documents"
            cls._instance._ensure_collection_exists()
        return cls._instance

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
            print(f"[QDRANT] Created collection: '{self.collection_name}'")

    async def index_document(self, doc_id: str, text: str, metadata: dict):
        """Encodes text and upserts the vector + payload into Qdrant."""
        if not text:
            text = metadata.get("original_name", "document")

        print(f"[QDRANT] Encoding {len(text)} chars...")
        vector = self.encoder.encode(text).tolist()

        payload = {**metadata, "text_content": text}

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(id=doc_id, vector=vector, payload=payload)
            ],
        )
        print(f"[QDRANT] Upserted doc_id={doc_id}")

    async def search(self, query: str, limit: int = 5):
        """Semantic search across stored documents."""
        query_vector = self.encoder.encode(query).tolist()
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
        )