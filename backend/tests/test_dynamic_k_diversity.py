import asyncio
from app.retrieval.query_parser import query_parser
from app.retrieval.dynamic_k import dynamic_k_governor
from collections import namedtuple

FakeChunk = namedtuple("FakeChunk", ["score", "payload"])


async def _run():
    # 1. Test query parser
    for q in [
        "give me aadhar card",
        "give me adhar card",
        "show my resume",
        "pan card details",
        "give me college id",
    ]:
        parsed = await query_parser.parse(q)
        print(f"Query: '{q}' -> doc_type: {parsed.document_type}, entities: {parsed.entities}")

    res_chunks = [
        FakeChunk(score=0.46, payload={"document_id": "shubham_resume_dd", "document_type": "resume"}),
        FakeChunk(score=0.45, payload={"document_id": "shubham_resume",    "document_type": "resume"}),
        FakeChunk(score=0.43, payload={"document_id": "dhruv_resume",      "document_type": "resume"}),
        FakeChunk(score=0.41, payload={"document_id": "aditya_resume",     "document_type": "resume"}),
    ]

    # 2. General category query (no entity) -> should target >= 3 documents
    q_cat = "give me resume"
    parsed_cat = await query_parser.parse(q_cat)
    dec_cat = dynamic_k_governor.decide(query=q_cat, initial_chunks=res_chunks, parsed_query=parsed_cat)
    print(f"\nCategory Query '{q_cat}' -> target_documents={dec_cat.target_documents}")
    assert dec_cat.target_documents >= 3, (
        f"Expected category query to target >= 3 docs, got {dec_cat.target_documents}"
    )

    # 3. Entity-specific query -> should target 1 document
    q_ent = "give me Shubham''s resume"
    parsed_ent = await query_parser.parse(q_ent)
    dec_ent = dynamic_k_governor.decide(query=q_ent, initial_chunks=res_chunks, parsed_query=parsed_ent)
    print(f"Entity Query '{q_ent}' -> target_documents={dec_ent.target_documents}")
    assert dec_ent.target_documents == 1, (
        f"Expected entity-specific query to target 1 doc, got {dec_ent.target_documents}"
    )

    # 4. Factual question -> should target 1 document
    q_fact = "What is my PAN number?"
    parsed_fact = await query_parser.parse(q_fact)
    dec_fact = dynamic_k_governor.decide(query=q_fact, initial_chunks=res_chunks, parsed_query=parsed_fact)
    print(f"Factual Query '{q_fact}' -> target_documents={dec_fact.target_documents}")
    assert dec_fact.target_documents == 1, (
        f"Expected factual query to target 1 doc, got {dec_fact.target_documents}"
    )

    print("\nALL DIVERSITY & TARGETING TESTS PASSED SUCCESSFULLY!")


def test():
    """Sync pytest wrapper -- runs the async body via asyncio.run() without needing pytest-asyncio."""
    asyncio.run(_run())


if __name__ == "__main__":
    asyncio.run(_run())
