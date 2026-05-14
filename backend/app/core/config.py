from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── AWS S3 ────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "eu-north-1"
    AWS_S3_BUCKET: str = ""

    # ── Qdrant ─────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: str = "6333"
    CHUNKS_COLLECTION: str = "document_chunks"

    # ── Embeddings ────────────────────────────────────────────────────
    BGE_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"

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

    # ── CORS ───────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def aws_region_clean(self) -> str:
        """
        Return just the region code (e.g. 'eu-north-1').
        The .env value may include a display prefix like 'Europe (Stockholm) eu-north-1'.
        We extract the last token which is always the actual region code.
        """
        return self.AWS_REGION.split()[-1]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()