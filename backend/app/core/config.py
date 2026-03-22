from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY: str = ""
    R2_SECRET_KEY: str = ""
    R2_BUCKET_NAME: str = ""
    OPENROUTER_API_KEY: str = ""
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: str = "6333"
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()