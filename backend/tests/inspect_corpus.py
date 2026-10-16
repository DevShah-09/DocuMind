import sys
sys.path.insert(0, ".")
from app.core.config import settings
from qdrant_client import QdrantClient

qc = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, check_compatibility=False)
records, _ = qc.scroll(collection_name="document_chunks", limit=50, with_payload=True, with_vectors=False)
print(f"Total points in document_chunks: {len(records)}")
for i, r in enumerate(records):
    p = r.payload or {}
    uid = str(p.get("user_id", ""))[:8]
    cid = p.get("chunk_id", "")
    did = p.get("document_id", "")
    fn = p.get("filename", "")
    dt = p.get("document_type", "")
    yr = p.get("year", "")
    mo = p.get("month", "")
    tl = len(p.get("text", ""))
    summary = (p.get("summary") or "")[:60].replace("\n", " ")
    print(f"[{i:02d}] doc_id={did} | chunk_id={cid} | user={uid}... | file={fn} | type={dt} | yr={yr} | mo={mo} | chars={tl} | sum={summary}")
