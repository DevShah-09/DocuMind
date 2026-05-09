from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Cloudflare R2 ──────────────────────────────────────────
    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY: str = ""
    R2_SECRET_KEY: str = ""
    R2_BUCKET_NAME: str = ""

    # ── Qdrant ─────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: str = "6333"

    # ── OpenRouter / LLM ───────────────────────────────────────
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    # ── Supabase ───────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # ── JWT Validation ─────────────────────────────────────────
    JWT_ISSUER: str = ""
    JWT_AUDIENCE: str = "authenticated"

    # ── File Encryption ────────────────────────────────────────
    MASTER_ENCRYPTION_SECRET: str = ""

    # ── Signed URL Signing ─────────────────────────────────────
    SIGNED_URL_SECRET: str = ""

    # ── CORS ───────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()