# fastapi for DiariZen
# uvicorn으로 별도로 실행하여 whisperX FastAPI 서비스가 호출하는 화자 분리 엔드포인트 제공

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from diarizen.pipelines.inference import DiariZenPipeline

# 설정
MODEL_NAME = os.getenv("DIARIZEN_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md")
DEVICE = os.getenv("DIARIZEN_DEVICE", "cuda")

models = {}


# 라이프사이클
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading DiariZen model ({MODEL_NAME}) ...")
    pipeline = DiariZenPipeline.from_pretrained(MODEL_NAME)
    # DiariZen은 pyannote Pipeline을 감싸고 있어 .to(device)로 GPU 이동을 지원할 수 있음.
    # 버전에 따라 지원하지 않을 수 있으므로 실패해도 무시하고 계속 진행한다.
    try:
        pipeline = pipeline.to(DEVICE)
    except Exception as exc:
        print(f"[warn] pipeline.to({DEVICE!r}) 실패, 기본 디바이스로 진행: {exc}")
    models["diarizen"] = pipeline
    print("DiariZen model loaded successfully!")
    yield
    models.clear()
    print("DiariZen model unloaded")


# 앱 초기화
app = FastAPI(
    title="DiariZen Diarization API",
    description="Speaker diarization powered by DiariZen",
    version="1.0.0",
    lifespan=lifespan,
)


# 헬스 체크
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": "diarizen" in models}


# 화자 분리 엔드포인트
@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Query(default=None),
    min_speakers: Optional[int] = Query(default=None),
    max_speakers: Optional[int] = Query(default=None),
):
    if "diarizen" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    allowed_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    ext = os.path.splitext(file.filename or "")[-1].lower()

    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None,
            lambda: _run_diarization(tmp_path, num_speakers, min_speakers, max_speakers),
        )
        return JSONResponse(content={"segments": segments})

    finally:
        os.unlink(tmp_path)


def _run_diarization(
    audio_path: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> list[dict]:
    """동기 화자 분리 로직 (ThreadPoolExecutor에서 실행)"""

    pipeline = models["diarizen"]

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        diar_result = pipeline(audio_path, **kwargs)
    except TypeError:
        # 화자 수 힌트를 지원하지 않는 버전의 DiariZen
        diar_result = pipeline(audio_path)

    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in diar_result.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s["start"])
    return segments
