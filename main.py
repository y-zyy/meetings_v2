"""
FastAPI 회의록 자동 생성 파이프라인

흐름:
  오디오 업로드 → WhisperX (ASR) → LLM (JSON) → python-docx (DOCX) → 파일 반환

실행:
  uvicorn main:app --host 0.0.0.0 --port 8000

환경 변수 (필수):
  LLM_ENDPOINT   - LLM API 엔드포인트  (예: http://10.0.0.1:8080/v1)
  LLM_API_TOKEN  - LLM API 인증 토큰

환경 변수 (선택):
  LLM_MODEL        - 모델 이름 (기본: default)
  ASR_DEVICE       - cuda / cpu (기본: cuda)
  ASR_COMPUTE_TYPE - float16 / int8 (기본: float16)
  ASR_BATCH_SIZE   - 배치 크기 (기본: 16)
"""

import asyncio
import os
import tempfile
from functools import partial
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from asr import transcribe_audio
from generate import build_document
from llm import text_to_meeting_json

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

app = FastAPI(
    title="회의록 자동 생성 API",
    description="음성 파일을 업로드하면 회의록 DOCX 파일을 반환합니다.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/generate-minutes",
    summary="음성 → 회의록 DOCX 생성",
    response_description="회의록 DOCX 파일",
)
async def generate_minutes(audio: UploadFile = File(..., description="회의 녹음 파일")):
    """
    오디오 파일을 받아 다음 순서로 처리합니다:
    1. WhisperX로 음성 → 텍스트 변환
    2. LLM (OpenAI 호환 API)으로 텍스트 → 회의록 JSON 변환
    3. python-docx로 JSON → DOCX 변환
    4. DOCX 파일 반환
    """
    suffix = Path(audio.filename or "audio.mp3").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    audio_path: str | None = None
    docx_path: str | None = None

    try:
        # ── 1. 오디오를 임시 파일로 저장 ──────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
            content = await audio.read()
            tmp_audio.write(content)
            audio_path = tmp_audio.name

        # ── 2. ASR: 음성 → 텍스트 (블로킹 → 스레드풀) ────────────────
        device = os.getenv("ASR_DEVICE", "cuda")
        compute_type = os.getenv("ASR_COMPUTE_TYPE", "float16")
        batch_size = int(os.getenv("ASR_BATCH_SIZE", "16"))

        try:
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None,
                partial(
                    transcribe_audio,
                    audio_path,
                    device=device,
                    batch_size=batch_size,
                    compute_type=compute_type,
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"음성 인식 중 오류: {e}") from e

        if not transcript.strip():
            raise HTTPException(status_code=422, detail="음성에서 텍스트를 인식하지 못했습니다.")

        # ── 3. LLM: 텍스트 → 회의록 JSON ─────────────────────────────
        try:
            meeting_data = await text_to_meeting_json(transcript)
        except EnvironmentError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"회의록 분석 중 오류: {e}") from e

        # ── 4. DOCX 생성 ───────────────────────────────────────────────
        try:
            doc = build_document(meeting_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"문서 생성 중 오류: {e}") from e

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
            docx_path = tmp_docx.name
        doc.save(docx_path)

        # ── 5. 파일 반환 ───────────────────────────────────────────────
        meeting_name = meeting_data.get("회의명", "회의록")
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in meeting_name)
        filename = f"{safe_name}.docx"

        return FileResponse(
            path=docx_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename,
            background=_CleanupFiles(audio_path, docx_path),
        )

    except HTTPException:
        _delete_if_exists(audio_path)
        _delete_if_exists(docx_path)
        raise
    except Exception as e:
        _delete_if_exists(audio_path)
        _delete_if_exists(docx_path)
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}") from e


# ── 임시 파일 정리 ────────────────────────────────────────────────────

def _delete_if_exists(path: str | None):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


class _CleanupFiles:
    """FileResponse 전송 완료 후 임시 파일을 삭제하는 background 태스크."""

    def __init__(self, *paths: str | None):
        self.paths = paths

    async def __call__(self):
        for path in self.paths:
            _delete_if_exists(path)
