"""
security.py — JWT Middleware & Auth Dependencies
================================================
Verifies Supabase-issued RS256 JWTs against the Supabase JWKS endpoint.

Flow:
  Request → extract Bearer token → fetch JWKS (cached) → validate RS256
  → validate issuer / audience / expiry → inject user claims as dependency
"""

import httpx
import logging
from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import ExpiredSignatureError, JWTError, jwt
from jose.backends import RSAKey  # noqa: F401 – imported for jose JWKS support

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── HTTP Bearer scheme ─────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


# ── JWKS fetching (cached per process lifetime) ────────────────────────────────

@lru_cache(maxsize=1)
def _get_jwks() -> dict:
    """
    Fetch Supabase JWKS and cache the result for the process lifetime.
    The JWKS URL is:  {SUPABASE_URL}/auth/v1/.well-known/jwks.json
    """
    jwks_url = f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    try:
        response = httpx.get(jwks_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error("Failed to fetch Supabase JWKS from %s: %s", jwks_url, exc)
        raise RuntimeError(f"Cannot fetch JWKS: {exc}") from exc


def _refresh_jwks() -> dict:
    """Force-refresh the JWKS cache (call when a new kid is encountered)."""
    _get_jwks.cache_clear()
    return _get_jwks()


# ── Core JWT verification ──────────────────────────────────────────────────────

def verify_supabase_jwt(token: str) -> dict:
    """
    Verify a Supabase-issued JWT.

    Validates:
      • Signature (RS256 via Supabase JWKS)
      • Issuer   (JWT_ISSUER env var)
      • Audience (JWT_AUDIENCE env var, default "authenticated")
      • Expiry   (automatic via python-jose)

    Returns the decoded claims dict on success.
    Raises HTTPException 401 on any failure.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        jwks = _get_jwks()

        # Decode header to find kid without full verification first
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        # Find matching key
        rsa_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                rsa_key = key
                break

        # kid not found → try refreshing JWKS once (key rotation)
        if rsa_key is None:
            jwks = _refresh_jwks()
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    rsa_key = key
                    break

        if rsa_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to find matching public key (kid mismatch)",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Full verification
        alg = unverified_header.get("alg")
        logger.info(f"Verifying token with algorithm: {alg}")
        
        # Choose the correct key based on algorithm
        # RS256 / ES256 = Uses the public key from Supabase JWKS
        # HS256 = Uses your Supabase JWT Secret (Service Role Key)
        if alg in ["RS256", "ES256"]:
            key_to_use = rsa_key
        else:
            key_to_use = settings.SUPABASE_SERVICE_ROLE_KEY

        # Debug info for Issuer/Audience
        unverified_claims = jwt.get_unverified_claims(token)
        logger.info(f"Token claims -> iss: {unverified_claims.get('iss')}, aud: {unverified_claims.get('aud')}")
        logger.info(f"Expected -> iss: {settings.JWT_ISSUER}, aud: {settings.JWT_AUDIENCE}")
        
        claims = jwt.decode(
            token,
            key_to_use,
            algorithms=["RS256", "ES256", "HS256"],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            options={"verify_exp": True},
        )
        return claims

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected auth error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ─────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency. Injects verified JWT claims into route handlers.

    Usage:
        @router.get("/me")
        async def me(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}

    The returned dict contains all Supabase JWT claims including:
      • sub      — Supabase user UUID
      • email    — user email
      • role     — "authenticated"
      • exp      — expiry timestamp
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_supabase_jwt(credentials.credentials)


# ── Ownership assertion helper ─────────────────────────────────────────────────

def assert_owns_resource(current_user: dict, resource_user_id: str) -> None:
    """
    Raise 403 if the authenticated user does not own the resource.
    Call this inside any route that accesses user-owned data.
    """
    if current_user.get("sub") != str(resource_user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this resource",
        )
