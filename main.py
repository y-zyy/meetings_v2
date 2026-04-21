"""
FastAPI 회의록 자동 생성 파이프라인

흐름:
  오디오 업로드 → WhisperX (ASR)
                → LLM ① 회의록 JSON 생성  ┐ (병렬)
                → LLM ② 전사본 정제       ┘
                → python-docx × 2 → ZIP 반환

실행:
  uvicorn main:app --host 0.0.0.0 --port 8000

환경 변수 (필수):
  LLM_ENDPOINT                  - LLM API 엔드포인트  (예: http://10.0.0.1:8080/v1)
  LLM_API_TOKEN                 - LLM API 인증 토큰

환경 변수 (선택):
  LLM_MODEL                     - 모델 이름 (기본: default)
  ASR_DEVICE                    - cuda / cpu (기본: cuda)
  ASR_COMPUTE_TYPE              - float16 / int8 (기본: float16)
  ASR_BATCH_SIZE                - 배치 크기 (기본: 16)
  TRANSCRIPT_REFINE_PROMPT      - 전사본 정제 시스템 프롬프트 (직접 텍스트)
  TRANSCRIPT_REFINE_PROMPT_FILE - 전사본 정제 프롬프트 파일 경로

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
import logging
import os
import tempfile
import urllib.parse
import zipfile
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

import asr as asr_module
from asr import transcribe_audio
from generate import build_document, build_transcript_document
from llm import refine_transcript, text_to_meeting_json

# ── 로깅 설정 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


# ── 한글 인코딩 복구 ──────────────────────────────────────────────────

def _fix_encoding(value: str | None) -> str | None:
    """python-multipart가 Latin-1로 잘못 디코딩한 한글을 UTF-8로 복구합니다."""
    if not value:
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return value


def _fix_encoding_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [_fix_encoding(v) or v for v in values]


# ── 서버 수명 주기 ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 Whisper 모델을 GPU에 로드하고, 종료 시까지 유지합니다."""
    device = os.getenv("ASR_DEVICE", "cuda")
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "float16")
    batch_size = int(os.getenv("ASR_BATCH_SIZE", "16"))

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        partial(asr_module.init_model, device, compute_type, batch_size),
    )

    yield  # 서버 실행 중


app = FastAPI(
    title="회의록 자동 생성 API",
    description="음성 파일을 업로드하면 회의록과 음성인식 결과 DOCX 2개가 담긴 ZIP을 반환합니다.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── 엔드포인트 ────────────────────────────────────────────────────────

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
    # ── 한글 인코딩 복구 ───────────────────────────────────────────────
    title = _fix_encoding(title)
    date_time = _fix_encoding(date_time)
    location = _fix_encoding(location)
    participants = _fix_encoding(participants)
    agenda = _fix_encoding_list(agenda)

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
        "안건": agenda,
    }

    logger.info("[API] 요청 수신: 파일=%s, 회의명=%s, 안건 수=%d",
                audio.filename, title or "(없음)", len(agenda))

    audio_path: str | None = None

    try:
        # ── STEP 1/4 오디오 저장 ──────────────────────────────────────
        logger.info("[STEP 1/4] 오디오 임시 저장 시작")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            audio_path = tmp.name
        logger.info("[STEP 1/4] 오디오 임시 저장 완료: %s", audio_path)

        # ── STEP 2/4 ASR ──────────────────────────────────────────────
        logger.info("[STEP 2/4] ASR(음성→텍스트) 시작")
        try:
            loop = asyncio.get_event_loop()
            transcript: str = await loop.run_in_executor(
                None, transcribe_audio, audio_path
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"음성 인식 중 오류: {e}") from e

        if not transcript.strip():
            raise HTTPException(status_code=422, detail="음성에서 텍스트를 인식하지 못했습니다.")
        logger.info("[STEP 2/4] ASR 완료: %d자 전사", len(transcript))

        # ── STEP 3/4 LLM (회의록 JSON + 전사본 정제 병렬 실행) ────────
        logger.info("[STEP 3/4] LLM 처리 시작 (회의록 JSON 생성 + 전사본 정제 병렬)")
        try:
            meeting_data, refined_text = await asyncio.gather(
                text_to_meeting_json(transcript, metadata),
                refine_transcript(transcript, metadata),
            )
        except EnvironmentError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM 처리 중 오류: {e}") from e
        logger.info("[STEP 3/4] LLM 처리 완료")

        # ── STEP 4/4 DOCX 생성 및 ZIP 반환 ───────────────────────────
        logger.info("[STEP 4/4] DOCX 생성 시작")
        try:
            minutes_buf = io.BytesIO()
            build_document(meeting_data).save(minutes_buf)
            minutes_buf.seek(0)

            transcript_buf = io.BytesIO()
            build_transcript_document(refined_text, metadata).save(transcript_buf)
            transcript_buf.seek(0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"문서 생성 중 오류: {e}") from e

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("회의록.docx", minutes_buf.read())
            zf.writestr("음성인식결과.docx", transcript_buf.read())
        zip_buf.seek(0)
        logger.info("[STEP 4/4] ZIP 생성 완료, 응답 반환")

        zip_filename = f"{title or '회의'}_결과.zip"
        encoded_filename = urllib.parse.quote(zip_filename)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=\"result.zip\"; "
                    f"filename*=UTF-8''{encoded_filename}"
                )
            },
        )

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
