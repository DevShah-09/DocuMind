"""
signed_urls.py — HMAC-SHA256 Signed URL Generation
====================================================
Generates temporary, authenticated URLs for Cloudflare R2 objects.

Format:
    https://{r2_endpoint}/{bucket}/{key}?user_id=...&expires=...&sig=HMAC-SHA256(...)

The HMAC signature covers: bucket_path + user_id + expiry timestamp
Any tampering invalidates the signature.
Default expiry: 5 minutes (300 seconds).
"""

import hashlib
import hmac
import time
import logging
from urllib.parse import urlencode

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sign(message: str) -> str:
    """HMAC-SHA256 signature over a message using the SIGNED_URL_SECRET."""
    key = settings.SIGNED_URL_SECRET.encode("utf-8")
    sig = hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()
    return sig


def _build_r2_base_url(bucket_path: str) -> str:
    """Construct the base R2 public/endpoint URL for a bucket object."""
    account_id = settings.R2_ACCOUNT_ID
    bucket = settings.R2_BUCKET_NAME
    # Standard Cloudflare R2 S3-compatible endpoint
    return f"https://{account_id}.r2.cloudflarestorage.com/{bucket}/{bucket_path}"


# ── Signed URL Generation ──────────────────────────────────────────────────────

def generate_signed_url(
    bucket_path: str,
    user_id: str,
    expiry_seconds: int = 300,
) -> str:
    """
    Generate a short-lived HMAC-SHA256 signed URL for a Cloudflare R2 object.

    The signed URL is valid only for `expiry_seconds` seconds and is
    bound to the specific `user_id` — the backend validates both before
    serving the redirect.

    Signature covers: "{bucket_path}|{user_id}|{expires}"

    Args:
        bucket_path:    R2 object key (e.g. "abc123/report.pdf.enc")
        user_id:        Supabase user UUID (binds URL to the user)
        expiry_seconds: Seconds until expiry (default 300 = 5 minutes)

    Returns:
        Full signed URL string
    """
    expires = int(time.time()) + expiry_seconds
    message = f"{bucket_path}|{user_id}|{expires}"
    signature = _sign(message)

    base_url = _build_r2_base_url(bucket_path)
    params = urlencode({
        "user_id": user_id,
        "expires": expires,
        "sig": signature,
    })

    signed_url = f"{base_url}?{params}"
    logger.debug(
        "[signed_url] Generated URL for bucket_path=%s user_id=%s expires_in=%ds",
        bucket_path, user_id, expiry_seconds,
    )
    return signed_url


def verify_signed_url(
    bucket_path: str,
    user_id: str,
    expires: int,
    sig: str,
) -> bool:
    """
    Verify a previously generated signed URL.

    Returns True only if:
      • The signature is valid (HMAC matches)
      • The URL has not expired
      • The user_id matches

    Args:
        bucket_path: R2 object key from the URL path
        user_id:     user_id query param from the URL
        expires:     expires query param (Unix timestamp) from the URL
        sig:         sig query param from the URL

    Returns:
        True if valid, False otherwise
    """
    # Check expiry first (avoids unnecessary HMAC computation)
    if int(time.time()) > expires:
        logger.warning("[signed_url] URL expired: bucket_path=%s", bucket_path)
        return False

    # Re-derive expected signature
    message = f"{bucket_path}|{user_id}|{expires}"
    expected_sig = _sign(message)

    # Constant-time comparison to prevent timing attacks
    valid = hmac.compare_digest(expected_sig, sig)
    if not valid:
        logger.warning("[signed_url] Signature mismatch: bucket_path=%s", bucket_path)
    return valid
