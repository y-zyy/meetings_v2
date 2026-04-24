"""
FastAPI 회의록 자동 생성 파이프라인

흐름:
  오디오 업로드 → Meeting 레코드 생성 → 백그라운드 처리
    → WhisperX (ASR)
    → LLM ① 회의록 JSON 생성  ┐ (병렬)
    → LLM ② 전사본 정제       ┘
    → DB 저장 (status: completed)

실행:
  uvicorn main:app --host 0.0.0.0 --port 8000

환경 변수 (필수):
  LLM_ENDPOINT                  - LLM API 엔드포인트  (예: http://10.0.0.1:8080/v1)
  LLM_API_TOKEN                 - LLM API 인증 토큰

환경 변수 (선택):
  SECRET_KEY                    - JWT 서명 키 (기본: dev 키, 운영 시 반드시 변경)
  DATABASE_URL                  - DB 연결 문자열 (기본: sqlite+aiosqlite:///./meetings.db)
  LLM_MODEL                     - 모델 이름 (기본: default)
  ASR_DEVICE                    - cuda / cpu (기본: cuda)
  ASR_COMPUTE_TYPE              - float16 / int8 (기본: float16)
  ASR_BATCH_SIZE                - 배치 크기 (기본: 16)
  TRANSCRIPT_REFINE_PROMPT      - 전사본 정제 시스템 프롬프트 (직접 텍스트)
  TRANSCRIPT_REFINE_PROMPT_FILE - 전사본 정제 프롬프트 파일 경로
"""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

import asr as asr_module
from asr import transcribe_audio
from auth import get_current_user
from database import AsyncSessionLocal, get_db, init_db
from llm import refine_transcript, text_to_meeting_json
from models import Meeting, User
from routers.auth import router as auth_router
from routers.meetings import router as meetings_router

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
    """python-multipart가 잘못 디코딩한 한글을 복구합니다."""
    if not value:
        return value
    try:
        raw = value.encode("latin-1")
    except UnicodeEncodeError:
        return value
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return value


def _fix_encoding_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [_fix_encoding(v) or v for v in values]


# ── 백그라운드 처리 ───────────────────────────────────────────────────

async def _process_meeting_bg(meeting_id: int, audio_path: str, metadata: dict) -> None:
    """ASR + LLM 처리 후 DB를 갱신하는 백그라운드 태스크."""
    async with AsyncSessionLocal() as db:
        try:
            meeting = await db.get(Meeting, meeting_id)
            if not meeting:
                return

            meeting.status = "processing"
            await db.commit()
            logger.info("[BG] 처리 시작: meeting_id=%d", meeting_id)

            # ASR
            try:
                loop = asyncio.get_running_loop()
                transcript: str = await loop.run_in_executor(None, transcribe_audio, audio_path)
            except Exception as e:
                raise RuntimeError(f"음성 인식 중 오류: {e}") from e

            if not transcript.strip():
                raise ValueError("음성에서 텍스트를 인식하지 못했습니다.")

            meeting.raw_transcript = transcript
            await db.commit()
            logger.info("[BG] ASR 완료: meeting_id=%d, %d자", meeting_id, len(transcript))

            # LLM (병렬)
            meeting_data, refined_text = await asyncio.gather(
                text_to_meeting_json(transcript, metadata),
                refine_transcript(transcript, metadata),
            )

            meeting.meeting_json = meeting_data
            meeting.refined_transcript = refined_text
            meeting.status = "completed"
            await db.commit()
            logger.info("[BG] 처리 완료: meeting_id=%d", meeting_id)

        except Exception as e:
            logger.exception("[BG] 처리 실패: meeting_id=%d", meeting_id)
            await db.rollback()
            try:
                meeting = await db.get(Meeting, meeting_id)
                if meeting:
                    meeting.status = "failed"
                    meeting.error_message = str(e)[:500]
                    await db.commit()
            except Exception:
                logger.exception("[BG] 오류 상태 저장 실패: meeting_id=%d", meeting_id)
        finally:
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass


# ── 서버 수명 주기 ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    device = os.getenv("ASR_DEVICE", "cuda")
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "float16")
    batch_size = int(os.getenv("ASR_BATCH_SIZE", "16"))

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        partial(asr_module.init_model, device, compute_type, batch_size),
    )

    yield


app = FastAPI(
    title="회의록 자동 생성 API",
    description="음성 파일을 업로드하면 AI가 회의록을 자동으로 생성합니다.",
    version="2.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 등록 ───────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(meetings_router)


# ── 엔드포인트 ────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.post(
    "/generate-minutes",
    summary="음성 → 회의록 생성 (비동기)",
    response_description="생성된 회의 ID를 즉시 반환하고, 처리는 백그라운드에서 진행됩니다.",
    tags=["meetings"],
)
async def generate_minutes(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="회의 녹음 파일"),
    title: str | None = Form(default=None, description="회의명"),
    date_time: str | None = Form(default=None, description="일시"),
    location: str | None = Form(default=None, description="장소"),
    participants: str | None = Form(default=None, description="참석자"),
    agenda: list[str] | None = Form(default=None, description="안건 (여러 번 입력 가능)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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

    # 오디오 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    # Meeting 레코드 생성
    meeting = Meeting(
        user_id=current_user.id,
        title=title or "(제목 없음)",
        date_time=date_time,
        location=location,
        participants=participants,
        agenda=agenda or [],
        status="pending",
    )
    db.add(meeting)
    await db.commit()
    await db.refresh(meeting)

    metadata = {
        "회의명": title or "",
        "일시": date_time or "",
        "장소": location or "",
        "참석자": participants or "",
        "안건": agenda,
    }

    background_tasks.add_task(_process_meeting_bg, meeting.id, audio_path, metadata)

    logger.info(
        "[API] 회의 생성: meeting_id=%d, user_id=%d, file=%s",
        meeting.id, current_user.id, audio.filename,
    )

    return {"meeting_id": meeting.id}
