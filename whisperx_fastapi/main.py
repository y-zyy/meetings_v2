# fastapi for WhisperX
# uvicorn으로 별도로 실행하여 "meetings_v2" 서비스에 엔드포인트 제공

import asyncio
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import whisperx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whisperx_fastapi")

# 설정
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16

# meetings_v2 backend의 ASR_API_KEY와 동일한 값으로 설정해야 인증이 맞물린다.
# (backend/app/services/asr.py의 _transcribe_http가 이 값을 Authorization: Bearer
# 헤더로 보낸다.) 비워두면(기본값) 인증 없이 열려있으므로 운영 환경에서는 반드시 설정할 것.
API_KEY = os.environ.get("WHISPERX_API_KEY", "")

models = {}

# GPU 모델(models["whisperx"])은 프로세스 전체에서 인스턴스 하나를 공유한다.
# 기본 executor(스레드 풀, 최대 수십 개)에 맡기면 동시에 여러 요청이 같은 GPU
# 모델을 두드리게 되어 CUDA OOM/결과 뒤섞임 위험이 크다. Celery의 worker-asr가
# concurrency=4로 동시에 여러 회의를 이 엔드포인트로 보낼 수 있으므로, 추론은
# 전용 단일 스레드 executor로 직렬화한다.
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisperx-infer")


# 라이프사이클
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading WhisperX model ...")
    models["whisperx"] = whisperx.load_model(
        "large-v3",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        vad_method="pyannote",
        language="ko",
    )
    logger.info("Model loaded successfully!")
    if not API_KEY:
        logger.warning(
            "WHISPERX_API_KEY가 설정되지 않았습니다 - /transcribe가 인증 없이 "
            "열려 있습니다. 운영 환경에서는 backend의 ASR_API_KEY와 동일한 값으로 설정하세요."
        )
    yield
    models.clear()
    inference_executor.shutdown(wait=False)
    logger.info("Model unloaded")


# 앱 초기화
app = FastAPI(
    title="WhisperX Transcription API",
    description="Audio transcription powered by WhisperX",
    version="1.0.0",
    lifespan=lifespan,
)


def _check_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# 헬스 체크
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": "whisperx" in models}


# 전사 엔드포인트
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    if "whisperx" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    allowed_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    ext = os.path.splitext(file.filename or "")[-1].lower()

    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())
    except Exception:
        logger.exception("업로드 파일 저장 실패: %s", file.filename)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    start = time.monotonic()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            inference_executor,
            lambda: _run_transcription(tmp_path, language, batch_size),
        )
    except ValueError as exc:
        # 오디오 파일 자체가 문제인 경우 (손상/디코딩 불가 등) - 클라이언트 오류
        logger.warning("전사 실패(잘못된 오디오) %s: %s", file.filename, exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        # 모델 추론/GPU 관련 오류 - 서버 오류
        logger.exception("전사 실패(서버 오류) %s", file.filename)
        raise HTTPException(status_code=500, detail="Transcription failed")
    else:
        elapsed = time.monotonic() - start
        logger.info("전사 완료: %s (%.1fs)", file.filename, elapsed)
        return JSONResponse(content=result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _run_transcription(audio_path: str, language: Optional[str], batch_size: int) -> dict:
    """동기 전사 로직 (전용 단일 스레드 executor에서 직렬로 실행됨)"""

    model = models["whisperx"]
    try:
        audio = whisperx.load_audio(audio_path)
    except Exception as exc:
        raise ValueError(f"Failed to load audio: {exc}") from exc

    result = model.transcribe(audio, batch_size=batch_size, language=language)

    texts = [segment["text"] for segment in result["segments"]]

    return {"text": "".join(texts).strip()}
