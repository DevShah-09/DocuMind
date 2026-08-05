import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import upload, search
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents_secure import router as documents_secure_router
from app.api.history import router as history_router
from app.api.retrieval import router as retrieval_router
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind AI",
    description="AI-powered document intelligence platform",
    version="1.0.0",
)

# ── Static file mount (local storage fallback) ─────────────────────────────────
import os as _os
_os.makedirs("local_storage", exist_ok=True)
app.mount("/static", StaticFiles(directory="local_storage"), name="static")

# ── Security Headers Middleware ────────────────────────────────────────────────
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """
    Inject security headers on every response.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Updated CSP to be more permissive for local development
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' http://localhost:* https://*.supabase.co; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "img-src 'self' data: blob: http://localhost:*; "
        "connect-src 'self' http://localhost:* https://*.supabase.co https://*.r2.cloudflarestorage.com"
    )
    return response

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth_router)               # /auth/*  /me
app.include_router(upload.router)             # /documents/upload  /documents/{id}
app.include_router(search.router)             # /collections/*
app.include_router(documents_secure_router)   # /documents/generate-signed-url/*
app.include_router(retrieval_router)          # /search/*  /documents/{id}  /documents/{id}/preview
app.include_router(chat_router)               # /chat
app.include_router(history_router)            # /history

# ── CORS (OUTERMOST MIDDLEWARE) ────────────────────────────────────────────────
# We add this LAST so it's the first thing that catches incoming requests 
# and the last thing that processes outgoing responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def read_root():
    return {
        "message": "DocuMind AI Backend is Running",
        "version": "1.0.0",
        "auth": "Supabase JWT (RS256)",
        "encryption": "ChaCha20-Poly1305 + HKDF-SHA256",
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}