import httpx

from app.config import settings


def transcribe(file_path: str) -> str:
    """POST the audio file to the ASR API and return transcribed text."""
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
