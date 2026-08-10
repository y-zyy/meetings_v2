from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://meetings:meetings@postgres:5432/meetings"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"

    # Security
    SECRET_KEY: str = "insecure-default-change-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    COOKIE_SECURE: bool = False  # True in production (HTTPS)

    # File storage
    UPLOAD_DIR: str = "/app/data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 500

    # Rule-based STT 교정 시드 사전 (파일 수정 시 재시작 없이 다음 회의 처리부터 반영)
    CORRECTION_RULES_SEED_PATH: str = "/app/data/correction_rules_seed.json"

    # ASR
    ASR_API_URL: str = "http://asr-server:9000/transcribe"
    ASR_API_KEY: str = ""
    ASR_FILE_FIELD: str = "file"
    ASR_RESPONSE_FIELD: str = "text"
    ASR_TIMEOUT: int = 7200

    # LLM (OpenAI-compatible local server)
    LLM_API_BASE_URL: str = "http://llm-server:8000/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    LLM_TIMEOUT: int = 120
    LLM_MAX_TOKENS: int = 4096

    # LLM 스트리밍 반복(hallucination) 가드
    # 반복 판정 기준(짧은 단위일수록 더 많은 반복을 요구하도록 내부에서 스케일링됨)
    LLM_REPEAT_MAX: int = 8
    # 몇 글자짜리 단위(phrase)까지 반복을 검사할지 (1=한 글자 ~ N=N글자 문구)
    LLM_REPEAT_NGRAM_MAX_CHARS: int = 12
    # 반복이 감지됐을 때 재생성을 몇 번까지 시도할지
    LLM_STREAM_MAX_RETRIES: int = 2
    # vLLM 등에서 지원하는 repetition_penalty (extra_body로 전달, >1.0일 때만 적용).
    # repetition_penalty를 서버에 고정해둬도 반복이 재발하는 경우가 있어,
    # 재시도할 때마다 이 값을 조금씩 올려서 같은 loop을 더 강하게 억제한다.
    LLM_REPETITION_PENALTY: float = 1.15

    # Cloud inference keys (if set, take priority over local servers)
    OPENAI_API_KEY: str = ""           # Whisper ASR via OpenAI
    ANTHROPIC_API_KEY: str = ""        # Meeting-minutes LLM via Claude
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # Admin seed
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "change-me"
    ADMIN_EMAIL: str = "admin@local"


settings = Settings()
