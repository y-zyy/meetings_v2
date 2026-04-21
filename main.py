"""
FastAPI 회의록 자동 생성 파이프라인

흐름:
  오디오 업로드 → WhisperX (ASR) → LLM (JSON) → python-docx × 2 → ZIP 반환

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

curl 예시:
  curl -X POST http://localhost:8000/generate-minutes \\
    -F "audio=@meeting.mp3" \\
    -F "title=4분기 로드맵 검토" \\
    -F "date_time=2024-10-21 14:00~16:00" \\
    -F "location=본사 3층 대회의실" \\
    -F "participants=김철수 (PM), 이영희 (개발팀장)" \\
    -F "agenda=신규 기능 우선순위 조정" \\
    -F "agenda=UI/UX 개편 방향 확정" \\
    -o result.zip
"""

import asyncio
import io
import os
import tempfile
import zipfile
from functools import partial
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from asr import transcribe_audio
from generate import build_document, build_transcript_document
from llm import text_to_meeting_json

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

app = FastAPI(
    title="회의록 자동 생성 API",
    description="음성 파일을 업로드하면 회의록과 음성인식 결과 DOCX 2개가 담긴 ZIP을 반환합니다.",
    version="2.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/generate-minutes",
    summary="음성 → 회의록 + 음성인식결과 ZIP 생성",
    response_description="회의록.docx + 음성인식결과.docx 가 담긴 ZIP 파일",
)
async def generate_minutes(
    audio: UploadFile = File(..., description="회의 녹음 파일"),
    title: str | None = Form(default=None, description="회의명"),
    date_time: str | None = Form(default=None, description="일시 (예: 2024-10-21 14:00~16:00)"),
    location: str | None = Form(default=None, description="장소"),
    participants: str | None = Form(default=None, description="참석자 (쉼표 구분)"),
    agenda: list[str] | None = Form(default=None, description="안건 (여러 번 입력 가능)"),
):
    """
    오디오 파일과 회의 메타데이터를 받아 다음 순서로 처리합니다:

    1. WhisperX로 음성 → 텍스트 변환
    2. LLM으로 텍스트 + 메타데이터 → 회의록 JSON 변환
       (안건이 제공된 경우 해당 안건을 기준으로 회의 내용 정리)
    3. python-docx로 DOCX 2개 생성
       - 회의록.docx  : LLM이 정리한 최종 회의록
       - 음성인식결과.docx : ASR 원문 텍스트 (동일 포맷)
    4. ZIP으로 묶어 반환
    """
    suffix = Path(audio.filename or "audio.mp3").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    metadata = {
        "회의명": title or "",
        "일시": date_time or "",
        "장소": location or "",
        "참석자": participants or "",
        "안건": agenda or [],
    }

    audio_path: str | None = None

    try:
        # ── 1. 오디오 임시 저장 ────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            audio_path = tmp.name

        # ── 2. ASR (블로킹 → 스레드풀) ───────────────────────────────
        device = os.getenv("ASR_DEVICE", "cuda")
        compute_type = os.getenv("ASR_COMPUTE_TYPE", "float16")
        batch_size = int(os.getenv("ASR_BATCH_SIZE", "16"))

        try:
            loop = asyncio.get_event_loop()
            transcript: str = await loop.run_in_executor(
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

        # ── 3. LLM: 전사본 + 메타데이터 → 회의록 JSON ────────────────
        try:
            meeting_data = await text_to_meeting_json(transcript, metadata)
        except EnvironmentError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"회의록 분석 중 오류: {e}") from e

        # ── 4. DOCX 2개 생성 (인메모리) ───────────────────────────────
        try:
            minutes_buf = io.BytesIO()
            build_document(meeting_data).save(minutes_buf)
            minutes_buf.seek(0)

            transcript_buf = io.BytesIO()
            build_transcript_document(transcript, metadata).save(transcript_buf)
            transcript_buf.seek(0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"문서 생성 중 오류: {e}") from e

        # ── 5. ZIP으로 묶어 반환 ───────────────────────────────────────
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("회의록.docx", minutes_buf.read())
            zf.writestr("음성인식결과.docx", transcript_buf.read())
        zip_buf.seek(0)

        safe_title = "".join(
            c if c.isalnum() or c in " _-" else "_"
            for c in (title or "회의")
        ).strip("_") or "회의"

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_title}_결과.zip"'
            },
        )

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
