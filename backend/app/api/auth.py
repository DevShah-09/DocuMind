"""
auth.py — Authentication API Endpoints
=======================================
Endpoints:
  POST /auth/google          — Returns Supabase OAuth redirect URL for Google
  POST /auth/otp/request     — Triggers Supabase OTP/magic-link email
  POST /auth/otp/verify      — Verifies OTP token, returns session info
  GET  /me                   — Returns current user profile from JWT
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.core.config import settings
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class OTPRequestBody(BaseModel):
    email: EmailStr


class OTPVerifyBody(BaseModel):
    email: EmailStr
    otp: str


# ── Helper: Supabase Admin HTTP client ─────────────────────────────────────────

def _supabase_headers() -> dict:
    """Headers for Supabase Admin API calls (service role key)."""
    return {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


# ── Google OAuth ───────────────────────────────────────────────────────────────

@router.post("/google")
async def google_oauth():
    """
    Returns the Supabase-managed Google OAuth URL.

    The frontend uses this URL to initiate the OAuth flow. Supabase handles
    the full code exchange and issues a JWT session on completion.

    The redirect URL must be configured in Supabase Dashboard →
    Authentication → URL Configuration → Redirect URLs.
    """
    oauth_url = (
        f"{settings.SUPABASE_URL}/auth/v1/authorize"
        f"?provider=google"
        f"&redirect_to={settings.SUPABASE_URL}/auth/v1/callback"
    )
    return {
        "provider": "google",
        "oauth_url": oauth_url,
        "message": "Redirect user to oauth_url to initiate Google login",
    }


# ── Email OTP — Request ────────────────────────────────────────────────────────

@router.post("/otp/request", status_code=status.HTTP_200_OK)
async def request_otp(body: OTPRequestBody):
    """
    Trigger Supabase to send an OTP / magic link to the provided email.

    Supabase will send a 6-digit OTP code (if OTP is enabled) or a magic
    link. Configure in Supabase Dashboard → Auth → Email Templates.
    """
    url = f"{settings.SUPABASE_URL}/auth/v1/otp"
    payload = {
        "email": body.email,
        "create_user": True,   # Register user if they don't exist
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=_supabase_headers(),
                timeout=15,
            )

        if response.status_code not in (200, 204):
            logger.error(
                "[auth] Supabase OTP request failed: %s %s",
                response.status_code,
                response.text,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to send OTP. Please try again.",
            )

        return {"message": f"OTP sent to {body.email}"}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[auth] OTP request error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach authentication service",
        )


# ── Email OTP — Verify ─────────────────────────────────────────────────────────

@router.post("/otp/verify", status_code=status.HTTP_200_OK)
async def verify_otp(body: OTPVerifyBody):
    """
    Verify the OTP code and exchange it for a Supabase session (access_token).

    The frontend should store the returned access_token and use it as the
    Bearer token in the Authorization header for all subsequent API calls.
    """
    url = f"{settings.SUPABASE_URL}/auth/v1/verify"
    payload = {
        "email": body.email,
        "token": body.otp,
        "type": "email",  # 'email' for OTP, 'magiclink' for magic links
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=_supabase_headers(),
                timeout=15,
            )

        if response.status_code != 200:
            logger.warning(
                "[auth] OTP verification failed: %s %s",
                response.status_code,
                response.text,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired OTP code",
            )

        data = response.json()
        return {
            "message": "OTP verified successfully",
            "access_token": data.get("access_token"),
            "token_type": "bearer",
            "expires_in": data.get("expires_in"),
            "refresh_token": data.get("refresh_token"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[auth] OTP verify error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach authentication service",
        )


# ── /me — Current user profile ─────────────────────────────────────────────────

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Return the authenticated user's profile extracted from their JWT.

    No database call needed — all data is embedded in the verified token.

    Requires: Authorization: Bearer <supabase_access_token>
    """
    return {
        "user_id": current_user.get("sub"),
        "email": current_user.get("email"),
        "full_name": current_user.get("user_metadata", {}).get("full_name"),
        "avatar_url": current_user.get("user_metadata", {}).get("avatar_url"),
        "provider": current_user.get("app_metadata", {}).get("provider"),
        "role": current_user.get("role"),
        "created_at": current_user.get("created_at"),
    }
