import httpx

from app.config import settings


def transcribe(file_path: str, effective: dict | None = None) -> str:
    """Transcribe audio.

    effective: dict returned by get_effective_settings_sync(); when None the
    module-level settings object is used directly (backward-compat).
    """
    cfg = effective or {k: getattr(settings, k, "") for k in (
        "OPENAI_API_KEY", "ASR_API_URL", "ASR_API_KEY",
        "ASR_FILE_FIELD", "ASR_RESPONSE_FIELD", "ASR_TIMEOUT",
    )}
    if cfg.get("OPENAI_API_KEY"):
        return _transcribe_whisper(file_path, cfg)
    return _transcribe_http(file_path, cfg)


def _transcribe_whisper(file_path: str, cfg: dict) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=cfg["OPENAI_API_KEY"], timeout=int(cfg.get("ASR_TIMEOUT") or settings.ASR_TIMEOUT))
    with open(file_path, "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f)
    return response.text


def _transcribe_http(file_path: str, cfg: dict) -> str:
    headers = {}
    if cfg.get("ASR_API_KEY"):
        headers["Authorization"] = f"Bearer {cfg['ASR_API_KEY']}"

    url = cfg.get("ASR_API_URL") or settings.ASR_API_URL
    file_field = cfg.get("ASR_FILE_FIELD") or settings.ASR_FILE_FIELD
    response_field = cfg.get("ASR_RESPONSE_FIELD") or settings.ASR_RESPONSE_FIELD
    timeout = int(cfg.get("ASR_TIMEOUT") or settings.ASR_TIMEOUT)

    with open(file_path, "rb") as audio_file:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, files={file_field: audio_file}, headers=headers)
            response.raise_for_status()

    data = response.json()
    text = data.get(response_field, "")
    if not isinstance(text, str):
        raise ValueError(f"ASR API 응답에서 텍스트 필드({response_field})를 찾을 수 없습니다.")
    return text
