"""
crypto.py — File Encryption & Key Derivation
=============================================
Implements:
  • HKDF-SHA256 key derivation from a master secret + user_id + file_id
  • ChaCha20-Poly1305 authenticated encryption / decryption

All files are encrypted BEFORE upload to Cloudflare R2.
The nonce (12 bytes) is prepended to the ciphertext:
    stored_bytes = nonce || ciphertext_with_auth_tag

Keys are NEVER stored in plaintext. They are derived on demand.
"""

import os
import logging

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CHACHA20_KEY_BYTES = 32   # 256-bit key required by ChaCha20-Poly1305
CHACHA20_NONCE_BYTES = 12  # 96-bit nonce


# ── HKDF-SHA256 Key Derivation ─────────────────────────────────────────────────

def derive_file_key(user_id: str, file_id: str) -> bytes:
    """
    Derive a 256-bit encryption key unique to this (user, file) pair.

    Uses HKDF-SHA256:
        IKM  = MASTER_ENCRYPTION_SECRET (hex-decoded)
        Salt = user_id (UTF-8 bytes)
        Info = file_id (UTF-8 bytes)
        OKM  = 32 bytes (256-bit ChaCha20 key)

    The same inputs always produce the same key, so we can re-derive
    on read without ever storing the key.

    Args:
        user_id: Supabase user UUID string
        file_id: Unique file identifier (e.g. UUID or bucket path)

    Returns:
        32-byte derived key
    """
    master_secret_bytes = bytes.fromhex(settings.MASTER_ENCRYPTION_SECRET)

    hkdf = HKDF(
        algorithm=SHA256(),
        length=CHACHA20_KEY_BYTES,
        salt=user_id.encode("utf-8"),
        info=file_id.encode("utf-8"),
    )
    derived_key = hkdf.derive(master_secret_bytes)
    return derived_key


# ── ChaCha20-Poly1305 Encryption ───────────────────────────────────────────────

def encrypt_file(plaintext: bytes, user_id: str, file_id: str) -> bytes:
    """
    Encrypt file bytes with ChaCha20-Poly1305.

    Generates a random 12-byte nonce per call.
    Returns: nonce (12 bytes) || ciphertext+auth_tag

    The auth tag (16 bytes) is appended automatically by the AEAD cipher
    and verifies integrity + authenticity on decryption.

    Args:
        plaintext: Raw file bytes (PDF, image, etc.)
        user_id:   Supabase user UUID (used in key derivation)
        file_id:   Unique file ID (used in key derivation)

    Returns:
        Encrypted bytes (nonce prepended)
    """
    key = derive_file_key(user_id, file_id)
    nonce = os.urandom(CHACHA20_NONCE_BYTES)

    cipher = ChaCha20Poly1305(key)
    ciphertext = cipher.encrypt(nonce, plaintext, None)  # No additional data

    logger.debug(
        "[crypto] Encrypted %d bytes → %d bytes (file_id=%s)",
        len(plaintext),
        len(ciphertext),
        file_id,
    )
    return nonce + ciphertext


def decrypt_file(encrypted_bytes: bytes, user_id: str, file_id: str) -> bytes:
    """
    Decrypt file bytes encrypted by encrypt_file().

    Splits the stored blob into nonce + ciphertext, then authenticates
    and decrypts using the re-derived key.

    Raises:
        ValueError: If the ciphertext has been tampered with (auth tag mismatch)
        ValueError: If the blob is too short to contain a valid nonce

    Args:
        encrypted_bytes: Stored bytes (nonce || ciphertext+auth_tag)
        user_id:         Supabase user UUID
        file_id:         Unique file ID

    Returns:
        Original plaintext bytes
    """
    if len(encrypted_bytes) < CHACHA20_NONCE_BYTES + 16:
        raise ValueError("Encrypted blob is too short — data may be corrupt")

    nonce = encrypted_bytes[:CHACHA20_NONCE_BYTES]
    ciphertext = encrypted_bytes[CHACHA20_NONCE_BYTES:]

    key = derive_file_key(user_id, file_id)
    cipher = ChaCha20Poly1305(key)

    try:
        plaintext = cipher.decrypt(nonce, ciphertext, None)
    except Exception as exc:
        logger.error("[crypto] Decryption failed for file_id=%s: %s", file_id, exc)
        raise ValueError("Decryption failed — ciphertext may be corrupt or tampered") from exc

    logger.debug(
        "[crypto] Decrypted %d bytes → %d bytes (file_id=%s)",
        len(encrypted_bytes),
        len(plaintext),
        file_id,
    )
    return plaintext
