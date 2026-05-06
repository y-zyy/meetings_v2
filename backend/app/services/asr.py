import httpx

from app.config import settings


def transcribe(file_path: str) -> str:
    """Transcribe audio. Uses OpenAI Whisper when OPENAI_API_KEY is set,
    otherwise falls back to the configured HTTP ASR endpoint."""
    if settings.OPENAI_API_KEY:
        return _transcribe_whisper(file_path)
    return _transcribe_http(file_path)


def _transcribe_whisper(file_path: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.ASR_TIMEOUT)
    with open(file_path, "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f)
    return response.text


def _transcribe_http(file_path: str) -> str:
    headers = {}
    if settings.ASR_API_KEY:
        headers["Authorization"] = f"Bearer {settings.ASR_API_KEY}"

    with open(file_path, "rb") as audio_file:
        with httpx.Client(timeout=settings.ASR_TIMEOUT) as client:
            response = client.post(
                settings.ASR_API_URL,
                files={settings.ASR_FILE_FIELD: audio_file},
                headers=headers,
            )
            response.raise_for_status()

    data = response.json()
    text = data.get(settings.ASR_RESPONSE_FIELD, "")
    if not isinstance(text, str):
        raise ValueError(f"ASR API 응답에서 텍스트 필드({settings.ASR_RESPONSE_FIELD})를 찾을 수 없습니다.")
    return text
