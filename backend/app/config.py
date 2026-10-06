from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://meetings:meetings@postgres:5432/meetings"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_VISIBILITY_TIMEOUT: int = 14400
    CELERY_TASK_SOFT_TIME_LIMIT: int = 7800
    CELERY_TASK_TIME_LIMIT: int = 7920

    # Security
    SECRET_KEY: str = "insecure-default-change-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    COOKIE_SECURE: bool = False  # True in production (HTTPS)

    # File storage
    UPLOAD_DIR: str = "/app/data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 500
    UPLOAD_CHUNK_SIZE_MB: int = 4

    # Rule-based STT 교정 시드 사전 (파일 수정 시 재시작 없이 다음 회의 처리부터 반영)
    CORRECTION_RULES_SEED_PATH: str = "/app/data/correction_rules_seed.json"

    # ASR
    ASR_API_URL: str = "http://asr-server:9000/transcribe"
    ASR_API_KEY: str = ""
    ASR_FILE_FIELD: str = "file"
    ASR_RESPONSE_FIELD: str = "text"
    ASR_TIMEOUT: int = 7200
    # 후처리 이후 정렬/화자 분리 (비어 있으면 ASR_API_URL 의 /transcribe 를 /align_diarize 로 치환)
    ASR_DIARIZE_ENABLED: bool = True
    ASR_DIARIZE_API_URL: str = ""
    ASR_MIN_SPEAKERS: int | None = None
    ASR_MAX_SPEAKERS: int | None = None

    # LLM (OpenAI-compatible local server)
    LLM_API_BASE_URL: str = "http://llm-server:8000/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    LLM_TIMEOUT: int = 120
    LLM_MAX_TOKENS: int = 4096

    # Cloud inference keys (if set, take priority over local servers)
    OPENAI_API_KEY: str = ""           # Whisper ASR via OpenAI
    ANTHROPIC_API_KEY: str = ""        # Meeting-minutes LLM via Claude
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # Admin seed
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "change-me"
    ADMIN_EMAIL: str = "admin@local"


settings = Settings()

