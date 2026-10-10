"""
test_chat_endpoint_integration.py — Endpoint-level integration tests for /chat
================================================================================
Tests the real FastAPI chat handler via TestClient(app):
  1. Unauthenticated requests are rejected (401).
  2. Effective filters are preserved in expansion after document-type fallback.
  3. Legacy fallback is executed with mandatory authenticated user_id.
  4. Empty retrieval yields safe 200 responses with empty sources and polite answer.
  5. Multi-tenant isolation is enforced on every search argument at the HTTP boundary.
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.auth import get_current_user


class MockPoint:
    """Mock Qdrant ScoredPoint."""
    def __init__(self, point_id: str, score: float, payload: dict):
        self.id = point_id
        self.score = score
        self.payload = payload


class TestChatEndpointIntegration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        app.dependency_overrides.clear()

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_unauthenticated_request_rejected(self):
        """Unauthenticated POST /chat must return 401."""
        # Ensure no override
        response = self.client.post("/chat", json={"query": "What is my PAN?"})
        self.assertEqual(response.status_code, 401)

    @patch("app.api.chat._call_llm", return_value="Here is your bill summary.")
    @patch("app.api.chat.qdrant_store.search_chunks", new_callable=AsyncMock)
    def test_effective_filter_used_in_expansion_at_endpoint_level(
        self,
        mock_search_chunks,
        mock_llm,
    ):
        """
        Real endpoint test:
        1. Probe with document_type returns 0 chunks.
        2. Retry with relaxed filter returns 2 chunks.
        3. Expansion must be invoked with effective_filters (no document_type) and user_id.
        4. Final response must contain dynamic_k diagnostics, sources, and answer.
        """
        user_id = "user-auth-101"
        app.dependency_overrides[get_current_user] = lambda: {"sub": user_id}

        recorded_calls = []

        async def fake_search(query_vector, user_id, filters, limit, score_threshold):
            recorded_calls.append({
                "user_id": user_id,
                "filters": dict(filters) if filters else {},
                "limit": limit,
            })
            # 1. Strict doc-type search returns nothing
            if filters and "document_type" in filters:
                return []
            # 2. Relaxed search returns 2 chunks (triggering expansion if recommended_k > 2)
            if limit <= 6:
                return [
                    MockPoint(
                        "p1",
                        0.88,
                        {
                            "document_id": "doc-elec-1",
                            "filename": "bill_2023.pdf",
                            "document_type": "electricity_bill",
                            "text": "Electricity bill for 2023: Total Rs 4500",
                            "page_number": 1,
                            "year": 2023,
                        },
                    ),
                    MockPoint(
                        "p2",
                        0.82,
                        {
                            "document_id": "doc-elec-2",
                            "filename": "bill_2023_pt2.pdf",
                            "document_type": "electricity_bill",
                            "text": "Units consumed: 340 kWh in 2023",
                            "page_number": 2,
                            "year": 2023,
                        },
                    ),
                ]
            # 3. Expansion search returns more chunks
            return [
                MockPoint(
                    "p3",
                    0.79,
                    {
                        "document_id": "doc-elec-3",
                        "filename": "bill_2023_pt3.pdf",
                        "document_type": "electricity_bill",
                        "text": "Payment receipt for bill 2023",
                        "page_number": 1,
                        "year": 2023,
                    },
                ),
            ]

        mock_search_chunks.side_effect = fake_search

        response = self.client.post(
            "/chat",
            json={
                "query": "Compare and explain my electricity bills for 2023 in detail.",
                "dynamic_k_enabled": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check response structure
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Here is your bill summary.")
        self.assertIn("sources", data)
        self.assertGreaterEqual(len(data["sources"]), 1)
        self.assertIsNotNone(data["dynamic_k"])
        self.assertGreaterEqual(data["dynamic_k"]["recommended_k"], 3)

        # Inspect real search call sequence:
        # Call 0: Initial probe with doc_type
        # Call 1: Retry without doc_type
        # Call 2 (if recommended_k > 2): Expansion without doc_type
        self.assertGreaterEqual(len(recorded_calls), 2)
        call_probe = recorded_calls[0]
        call_retry = recorded_calls[1]

        self.assertEqual(call_probe["user_id"], user_id)
        self.assertIn("document_type", call_probe["filters"])

        self.assertEqual(call_retry["user_id"], user_id)
        self.assertNotIn("document_type", call_retry["filters"])

        if len(recorded_calls) >= 3:
            call_expansion = recorded_calls[2]
            self.assertEqual(call_expansion["user_id"], user_id)
            # CRITICAL CHECK: Expansion must NOT use the failed document_type filter!
            self.assertNotIn(
                "document_type",
                call_expansion["filters"],
                "Expansion search incorrectly re-applied the failed document_type filter!",
            )

    @patch("app.api.chat._call_llm", return_value="Found in legacy collection.")
    @patch("app.api.chat.VectorDBService")
    @patch("app.api.chat.qdrant_store.search_chunks", new_callable=AsyncMock)
    def test_legacy_fallback_with_authenticated_user(
        self,
        mock_search_chunks,
        mock_vdb_cls,
        mock_llm,
    ):
        """When Qdrant yields no chunks, legacy search is called with authenticated user_id."""
        user_id = "user-legacy-777"
        app.dependency_overrides[get_current_user] = lambda: {"sub": user_id}

        # Qdrant returns nothing
        mock_search_chunks.return_value = []

        mock_vdb_instance = MagicMock()
        mock_vdb_cls.return_value = mock_vdb_instance

        # Mock legacy point hit
        mock_hit = MagicMock()
        mock_hit.id = "legacy-hit-1"
        mock_hit.score = 0.81
        mock_hit.payload = {
            "original_name": "legacy_doc.pdf",
            "category": "tax",
            "text_content": "Legacy tax content",
            "summary": "Summary of legacy tax",
        }
        mock_vdb_instance.search = AsyncMock(return_value=[mock_hit])

        response = self.client.post(
            "/chat",
            json={"query": "What is my tax summary?", "dynamic_k_enabled": True},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify VectorDBService.search was called with user_id
        mock_vdb_instance.search.assert_called_once()
        _, kwargs = mock_vdb_instance.search.call_args
        self.assertEqual(kwargs.get("user_id"), user_id)

        # Sources should include the legacy doc
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["document_id"], "legacy-hit-1")
        self.assertEqual(data["sources"][0]["filename"], "legacy_doc.pdf")

    @patch("app.api.chat.VectorDBService")
    @patch("app.api.chat.qdrant_store.search_chunks", new_callable=AsyncMock)
    def test_empty_retrieval_at_endpoint_level(
        self,
        mock_search_chunks,
        mock_vdb_cls,
    ):
        """When both Qdrant and legacy return empty, returns safe 200 with polite message."""
        user_id = "user-empty-999"
        app.dependency_overrides[get_current_user] = lambda: {"sub": user_id}

        mock_search_chunks.return_value = []
        mock_vdb_instance = MagicMock()
        mock_vdb_cls.return_value = mock_vdb_instance
        mock_vdb_instance.search = AsyncMock(return_value=[])

        response = self.client.post(
            "/chat",
            json={"query": "Find my nonexistent document", "dynamic_k_enabled": True},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sources"], [])
        self.assertIsNone(data["dynamic_k"])
        self.assertIn("No documents matched", data["answer"])

    @patch("app.api.chat._call_llm", return_value="Tenant verified.")
    @patch("app.api.chat.qdrant_store.search_chunks", new_callable=AsyncMock)
    def test_tenant_isolation_strictly_enforced_between_users(
        self,
        mock_search_chunks,
        mock_llm,
    ):
        """Different users make requests; Qdrant searches receive only the respective user_id."""
        mock_search_chunks.return_value = [
            MockPoint("p1", 0.9, {"document_id": "d1", "filename": "f.pdf", "text": "txt"})
        ]

        # Request 1: User Alice
        app.dependency_overrides[get_current_user] = lambda: {"sub": "user-alice"}
        res_alice = self.client.post("/chat", json={"query": "Show my documents"})
        self.assertEqual(res_alice.status_code, 200)
        alice_user_arg = mock_search_chunks.call_args_list[-1].kwargs["user_id"]
        self.assertEqual(alice_user_arg, "user-alice")

        # Request 2: User Bob
        app.dependency_overrides[get_current_user] = lambda: {"sub": "user-bob"}
        res_bob = self.client.post("/chat", json={"query": "Show my documents"})
        self.assertEqual(res_bob.status_code, 200)
        bob_user_arg = mock_search_chunks.call_args_list[-1].kwargs["user_id"]
        self.assertEqual(bob_user_arg, "user-bob")


if __name__ == "__main__":
    unittest.main()
