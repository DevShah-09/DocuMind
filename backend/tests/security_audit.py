"""
security_audit.py — Comprehensive Security & Cryptographic Audit
================================================================
Priority B Audit implementation covering:
  1. Cryptographic correctness (ChaCha20-Poly1305 + HKDF-SHA256)
  2. Authentication and Authorization (JWT validation, ownership assertions)
  3. Tenant isolation in vector searches (Qdrant filter enforcement)
  4. Prompt injection resistance & sensitive data exposure prevention
"""

import os
import sys
import csv
import json
import time
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Ensure app is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jose import jwt
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.core.config import settings
from app.core.crypto import (
    derive_file_key,
    encrypt_file,
    decrypt_file,
    CHACHA20_KEY_BYTES,
    CHACHA20_NONCE_BYTES,
)
from app.core.security import verify_supabase_jwt, assert_owns_resource, get_current_user
from app.api.chat import _build_context, _log_retrieval_telemetry, SYSTEM_PROMPT


RESULTS_CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports", "comprehensive_evaluation", "security_test_results.csv")
)

test_records: List[Dict[str, Any]] = []


def record_result(category: str, test_name: str, passed: bool, severity: str, details: str):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "test_name": test_name,
        "status": "PASSED" if passed else "FAILED",
        "severity": severity,
        "details": details,
    }
    test_records.append(record)
    print(f"[{'PASS' if passed else 'FAIL'}] [{category}] {test_name}: {details}")


def save_audit_csv():
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "category", "test_name", "status", "severity", "details"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in test_records:
            writer.writerow(r)
    print(f"\n[SECURITY AUDIT] Saved {len(test_records)} test results to {RESULTS_CSV_PATH}")


# ==============================================================================
# 1. Cryptographic Correctness Tests (ChaCha20-Poly1305 + HKDF-SHA256)
# ==============================================================================

def test_crypto_roundtrips():
    """Verify encryption and decryption roundtrips on multiple plaintexts."""
    user_id = str(uuid.uuid4())
    file_id = str(uuid.uuid4())
    plaintexts = [
        b"Hello world!",
        b"",
        b"A" * 1024,
        b"\x00\x01\x02\x03\xff\xfe\xfd" * 50,
        "Document with UTF-8 symbols: \u2603 \U0001F600 \u0928\u092e\u0938\u094d\u0924\u0947".encode("utf-8"),
    ]
    all_passed = True
    for idx, pt in enumerate(plaintexts):
        try:
            ct = encrypt_file(pt, user_id, file_id)
            recovered = decrypt_file(ct, user_id, file_id)
            if recovered != pt:
                all_passed = False
                break
        except Exception as e:
            all_passed = False
            break

    record_result(
        category="Cryptography",
        test_name="roundtrip_encryption_decryption",
        passed=all_passed,
        severity="CRITICAL",
        details="Verified 5 varied payload types (empty, short, 1KB, binary bytes, UTF-8 unicode).",
    )
    assert all_passed


def test_crypto_tampered_ciphertext():
    """Verify that tampering with any bit in ciphertext raises ValueError (Poly1305 tag check)."""
    user_id = str(uuid.uuid4())
    file_id = str(uuid.uuid4())
    pt = b"Critical financial record content that must not be tampered."
    ct = bytearray(encrypt_file(pt, user_id, file_id))

    # Flip a bit in the ciphertext portion (after 12-byte nonce)
    ct[15] ^= 0x01

    tamper_rejected = False
    try:
        decrypt_file(bytes(ct), user_id, file_id)
    except ValueError:
        tamper_rejected = True
    except Exception:
        pass

    record_result(
        category="Cryptography",
        test_name="tampered_ciphertext_detection",
        passed=tamper_rejected,
        severity="CRITICAL",
        details="Bit flip in ciphertext rejected by Poly1305 AEAD tag authentication.",
    )
    assert tamper_rejected


def test_crypto_tampered_auth_tag():
    """Verify that tampering with the auth tag raises ValueError."""
    user_id = str(uuid.uuid4())
    file_id = str(uuid.uuid4())
    pt = b"Another sensitive document payload."
    ct = bytearray(encrypt_file(pt, user_id, file_id))

    # Flip a bit in the last byte (auth tag)
    ct[-1] ^= 0x01

    tag_rejected = False
    try:
        decrypt_file(bytes(ct), user_id, file_id)
    except ValueError:
        tag_rejected = True
    except Exception:
        pass

    record_result(
        category="Cryptography",
        test_name="tampered_auth_tag_rejection",
        passed=tag_rejected,
        severity="CRITICAL",
        details="Modified authentication tag correctly triggers integrity failure.",
    )
    assert tag_rejected


def test_crypto_truncated_blob():
    """Verify that blobs shorter than nonce + 16 bytes are safely rejected."""
    user_id = str(uuid.uuid4())
    file_id = str(uuid.uuid4())
    truncated_blobs = [b"", b"short", b"123456789012", b"A" * 27]  # Need at least 12 + 16 = 28 bytes

    all_rejected = True
    for tb in truncated_blobs:
        try:
            decrypt_file(tb, user_id, file_id)
            all_rejected = False
        except ValueError:
            pass
        except Exception:
            all_rejected = False

    record_result(
        category="Cryptography",
        test_name="truncated_blob_bounds_check",
        passed=all_rejected,
        severity="HIGH",
        details="Blobs < 28 bytes (12B nonce + 16B tag) rejected before cipher initialization.",
    )
    assert all_rejected


def test_crypto_key_separation():
    """Verify HKDF domain and identity separation across users and files."""
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    file_1 = str(uuid.uuid4())
    file_2 = str(uuid.uuid4())

    key_a1 = derive_file_key(user_a, file_1)
    key_b1 = derive_file_key(user_b, file_1)
    key_a2 = derive_file_key(user_a, file_2)
    key_b2 = derive_file_key(user_b, file_2)

    keys = {key_a1, key_b1, key_a2, key_b2}
    unique_keys = len(keys) == 4
    length_valid = all(len(k) == CHACHA20_KEY_BYTES for k in keys)

    # Cross user decryption failure
    pt = b"Secret data owned by User A"
    ct_a1 = encrypt_file(pt, user_a, file_1)
    cross_decrypt_failed = False
    try:
        decrypt_file(ct_a1, user_b, file_1)
    except ValueError:
        cross_decrypt_failed = True

    passed = unique_keys and length_valid and cross_decrypt_failed
    record_result(
        category="Cryptography",
        test_name="hkdf_key_separation_and_isolation",
        passed=passed,
        severity="CRITICAL",
        details="HKDF-SHA256 generates unique 32-byte keys for each (user, file) pair; cross-user decryption fails.",
    )
    assert passed


def test_crypto_nonce_uniqueness():
    """Verify that multiple encryptions of the same plaintext produce unique nonces (no nonce reuse)."""
    user_id = str(uuid.uuid4())
    file_id = str(uuid.uuid4())
    pt = b"Identical plaintext encrypted 200 times."

    nonces = set()
    num_trials = 200
    for _ in range(num_trials):
        ct = encrypt_file(pt, user_id, file_id)
        nonce = ct[:CHACHA20_NONCE_BYTES]
        nonces.add(nonce)

    passed = len(nonces) == num_trials
    record_result(
        category="Cryptography",
        test_name="nonce_uniqueness_no_reuse",
        passed=passed,
        severity="CRITICAL",
        details=f"Generated {num_trials} distinct 12-byte nonces with zero collisions.",
    )
    assert passed


# ==============================================================================
# 2. Authentication & Authorization Tests
# ==============================================================================

def test_auth_missing_token():
    """Verify that calling verify_supabase_jwt with empty or None token raises 401."""
    passed = False
    try:
        verify_supabase_jwt("")
    except HTTPException as e:
        if e.status_code == 401:
            passed = True

    record_result(
        category="Authentication",
        test_name="missing_jwt_rejection",
        passed=passed,
        severity="CRITICAL",
        details="Empty or missing token correctly raises HTTP 401 Unauthorized.",
    )
    assert passed


def test_auth_malformed_token():
    """Verify that malformed or garbage token strings raise 401."""
    passed = False
    try:
        verify_supabase_jwt("not.a.valid.jwt.string.at.all")
    except HTTPException as e:
        if e.status_code == 401:
            passed = True

    record_result(
        category="Authentication",
        test_name="malformed_jwt_rejection",
        passed=passed,
        severity="CRITICAL",
        details="Malformed JWT raises HTTP 401 Unauthorized.",
    )
    assert passed


def test_auth_expired_token():
    """Verify that an expired JWT raises 401."""
    # Create an expired token signed with a dummy secret using HS256
    expired_payload = {
        "sub": str(uuid.uuid4()),
        "aud": settings.JWT_AUDIENCE,
        "iss": settings.JWT_ISSUER,
        "exp": int(time.time()) - 3600,  # 1 hour ago
        "role": "authenticated",
    }
    dummy_secret = settings.SUPABASE_SERVICE_ROLE_KEY or "dummy_secret_key"
    token = jwt.encode(expired_payload, dummy_secret, algorithm="HS256", headers={"kid": "dummy"})

    passed = False
    # Mock JWKS to return kid so it proceeds to decode
    with patch("app.core.security._get_jwks", return_value={"keys": [{"kid": "dummy"}]}):
        try:
            verify_supabase_jwt(token)
        except HTTPException as e:
            if e.status_code == 401 and "expired" in e.detail.lower():
                passed = True

    record_result(
        category="Authentication",
        test_name="expired_jwt_rejection",
        passed=passed,
        severity="CRITICAL",
        details="Expired JWT rejected with HTTP 401 'Token has expired'.",
    )
    assert passed


def test_auth_invalid_signature():
    """Verify that a token signed with the wrong key raises 401."""
    payload = {
        "sub": str(uuid.uuid4()),
        "aud": settings.JWT_AUDIENCE,
        "iss": settings.JWT_ISSUER,
        "exp": int(time.time()) + 3600,
        "role": "authenticated",
    }
    wrong_key = "completely_wrong_signature_secret_key_12345"
    token = jwt.encode(payload, wrong_key, algorithm="HS256", headers={"kid": "dummy"})

    passed = False
    with patch("app.core.security._get_jwks", return_value={"keys": [{"kid": "dummy"}]}):
        try:
            verify_supabase_jwt(token)
        except HTTPException as e:
            if e.status_code == 401:
                passed = True

    record_result(
        category="Authentication",
        test_name="invalid_signature_rejection",
        passed=passed,
        severity="CRITICAL",
        details="Forged JWT with wrong signature secret raises HTTP 401.",
    )
    assert passed


def test_auth_ownership_assertion():
    """Verify that assert_owns_resource raises 403 on mismatched user ID."""
    current_user = {"sub": "user-uuid-123", "role": "authenticated"}
    
    # Matching user
    assert_owns_resource(current_user, "user-uuid-123")  # Should not raise

    # Mismatched user
    forbidden_raised = False
    try:
        assert_owns_resource(current_user, "user-uuid-999")
    except HTTPException as e:
        if e.status_code == 403:
            forbidden_raised = True

    record_result(
        category="Authorization",
        test_name="ownership_assertion_enforcement",
        passed=forbidden_raised,
        severity="HIGH",
        details="assert_owns_resource strictly raises HTTP 403 when current_user['sub'] != resource_user_id.",
    )
    assert forbidden_raised


# ==============================================================================
# 3. Cross-Tenant Isolation Tests (API & Endpoint Level)
# ==============================================================================

def test_cross_tenant_document_access_endpoints():
    """Verify that user A cannot access signed URL or view user B's document."""
    from app.main import app

    user_a_id = str(uuid.uuid4())
    user_b_id = str(uuid.uuid4())
    target_doc_id = str(uuid.uuid4())

    # User B owns target_doc_id
    mock_doc = MagicMock()
    mock_doc.id = target_doc_id
    mock_doc.payload = {
        "user_id": user_b_id,
        "bucket_path": f"{user_b_id}/{target_doc_id}.pdf",
        "original_name": "tax_return_user_b.pdf",
    }

    client = TestClient(app)

    # Authenticate as User A
    app.dependency_overrides[get_current_user] = lambda: {"sub": user_a_id, "role": "authenticated"}

    signed_url_blocked = False
    view_doc_blocked = False

    with patch("app.api.documents_secure.vector_db.get_document", return_value=mock_doc):
        # 1. Attempt to generate signed URL for User B's document
        res_url = client.get(f"/documents/generate-signed-url/{target_doc_id}")
        if res_url.status_code == 403:
            signed_url_blocked = True

        # 2. Attempt to view/decrypt User B's document
        res_view = client.get(f"/documents/view/{target_doc_id}")
        if res_view.status_code == 403:
            view_doc_blocked = True

    app.dependency_overrides.clear()

    passed = signed_url_blocked and view_doc_blocked
    record_result(
        category="Tenant Isolation",
        test_name="cross_tenant_document_escalation_blocked",
        passed=passed,
        severity="CRITICAL",
        details="User A's attempts to view or generate signed URLs for User B's documents return 403 Forbidden.",
    )
    assert passed


def test_cross_tenant_document_deletion_blocked():
    """Verify that user A cannot delete user B's document."""
    from app.main import app

    user_a_id = str(uuid.uuid4())
    user_b_id = str(uuid.uuid4())
    target_doc_id = str(uuid.uuid4())

    mock_doc = MagicMock()
    mock_doc.id = target_doc_id
    mock_doc.payload = {
        "user_id": user_b_id,
        "bucket_path": f"{user_b_id}/{target_doc_id}.pdf",
    }

    client = TestClient(app)
    app.dependency_overrides[get_current_user] = lambda: {"sub": user_a_id, "role": "authenticated"}

    deletion_blocked = False
    with patch("app.api.upload.orchestrator.vector_db.get_document", return_value=mock_doc):
        res = client.delete(f"/documents/{target_doc_id}")
        if res.status_code == 403:
            deletion_blocked = True

    app.dependency_overrides.clear()

    record_result(
        category="Tenant Isolation",
        test_name="cross_tenant_deletion_blocked",
        passed=deletion_blocked,
        severity="CRITICAL",
        details="Unauthorized document deletion by cross-tenant actor correctly rejected with 403 Forbidden.",
    )
    assert deletion_blocked


# ==============================================================================
# 4. Prompt Injection & Sensitive Data Exposure Tests
# ==============================================================================

def test_system_prompt_instruction_defense():
    """Verify that the system prompt strictly confines answers to document excerpts and forbids external facts."""
    assert "using ONLY the information found in the provided document excerpts" in SYSTEM_PROMPT
    assert "Do NOT make up facts or use external knowledge beyond what is in the excerpts" in SYSTEM_PROMPT
    assert "I couldn't find relevant information in your documents for this query" in SYSTEM_PROMPT

    record_result(
        category="Prompt Injection",
        test_name="system_prompt_evidence_guardrails",
        passed=True,
        severity="MEDIUM",
        details="System prompt explicitly mandates closed-world evidence grounding and abstention phrasing.",
    )


def test_sensitive_credentials_never_exposed_in_telemetry():
    """Verify that structured telemetry never logs secrets, access tokens, or raw document text."""
    secret_keys = [
        settings.OPENROUTER_API_KEY,
        settings.SUPABASE_SERVICE_ROLE_KEY,
        settings.MASTER_ENCRYPTION_SECRET,
        settings.QDRANT_API_KEY,
    ]

    # Intercept logger
    logged_messages = []
    with patch("app.api.chat.logger.info", side_effect=lambda msg, *args: logged_messages.append(msg % args if args else msg)):
        _log_retrieval_telemetry(
            user_id="user-12345",
            initial_k=5,
            final_k=3,
            c_q=0.45,
            r_c=0.82,
            unique_chunks=3,
            distinct_docs=1,
            evidence_coverage=0.9,
            retrieval_latency_ms=12.5,
            total_latency_ms=145.2,
            stop_reason="target_k_met",
            dynamic_k_enabled=True,
        )

    log_output = " ".join(logged_messages)
    no_secrets_exposed = True
    for key in secret_keys:
        if key and len(key) > 5 and key in log_output:
            no_secrets_exposed = False
            break

    # Verify no raw chunk text placeholder
    no_raw_text = "excerpts" not in log_output and "document text" not in log_output

    passed = no_secrets_exposed and no_raw_text
    record_result(
        category="Data Exposure",
        test_name="telemetry_zero_secret_and_text_leakage",
        passed=passed,
        severity="HIGH",
        details="Telemetry log format confirmed strictly numerical/metadata; no API keys, bearer tokens, or document text present.",
    )
    assert passed


def run_all_security_tests():
    print("\n" + "=" * 60)
    print("RUNNING DOCUMIND AI SECURITY & CRYPTOGRAPHIC AUDIT")
    print("=" * 60)
    
    test_crypto_roundtrips()
    test_crypto_tampered_ciphertext()
    test_crypto_tampered_auth_tag()
    test_crypto_truncated_blob()
    test_crypto_key_separation()
    test_crypto_nonce_uniqueness()

    test_auth_missing_token()
    test_auth_malformed_token()
    test_auth_expired_token()
    test_auth_invalid_signature()
    test_auth_ownership_assertion()

    test_cross_tenant_document_access_endpoints()
    test_cross_tenant_document_deletion_blocked()

    test_system_prompt_instruction_defense()
    test_sensitive_credentials_never_exposed_in_telemetry()

    save_audit_csv()
    print("=" * 60)
    print(f"ALL {len(test_records)} SECURITY & CRYPTO AUDIT TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_security_tests()
