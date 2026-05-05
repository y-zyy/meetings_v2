from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://meetings:meetings@postgres:5432/meetings"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"

    # Security
    SECRET_KEY: str = "insecure-default-change-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    ALGORITHM: str = "HS256"

    # File storage
    UPLOAD_DIR: str = "/app/data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 500

    # ASR
    ASR_API_URL: str = "http://asr-server:9000/transcribe"
    ASR_API_KEY: str = ""
    ASR_FILE_FIELD: str = "file"
    ASR_RESPONSE_FIELD: str = "text"
    ASR_TIMEOUT: int = 7200

    # LLM
    LLM_API_BASE_URL: str = "http://llm-server:8000/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    LLM_TIMEOUT: int = 120

    # Admin seed
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "change-me"
    ADMIN_EMAIL: str = "admin@local"


settings = Settings()
