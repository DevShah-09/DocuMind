from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from app.core.qdrant_client_singleton import shared_qdrant_client
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
            # Use the shared client — prevents storage lock conflict
            cls._instance.client = shared_qdrant_client
            print(f"[QDRANT] VectorDBService using shared client")
            
            print("[AI] Loading SentenceTransformer model (this may take a minute on first run)...")
            cls._instance.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            print("[AI] Model loaded successfully.")
            
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

    async def search(self, query: str, limit: int = 5, user_id: str = None):
        """Semantic search across stored documents."""
        query_vector = self.encoder.encode(query).tolist()
        
        query_filter = None
        if user_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points

    async def check_duplicate(self, file_hash: str):
        """Checks if a document with the same file_hash already exists."""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_hash",
                        match=models.MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        return points[0] if points else None

    def get_document(self, doc_id: str):
        """Retrieves a single document by its point ID."""
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True
            )
            return results[0] if results else None
        except Exception:
            return None

    def delete_document(self, doc_id: str):
        """Deletes a document from Qdrant by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[doc_id],
            ),
        )
        print(f"[QDRANT] Deleted doc_id={doc_id}")