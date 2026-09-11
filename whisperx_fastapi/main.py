# fastapi for WhisperX
# uvicorn으로 별도로 실행하여 "meetings_v2" 서비스에 엔드포인트 제공

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisperx
import tempfile
import os
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

# 설정
DEVICE="cuda"
COMPUTE_TYPE="float16"
BATCH_SIZE=16

models = {}

# 라이프사이클
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading WhisperX model ...")
    models["whisperx"] = whisperx.load_model(
        "large-v3",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        vad_method="pyannote",
        language="ko"
    )
    print("Model loaded successfully!")
    yield
    models.clear()
    print("Model unloaded")
    
# 앱 초기화
app = FastAPI(
    title="WhisperX Transcription API",
    description="Audio transcription powered by WhisperX",
    version="1.0.0",
    lifespan=lifespan,
)

# 헬스 체크
@app.get("/health")
async def health_check():
    return {"status":"ok", "model_loaded": "whisperx" in models}
    
# 전사 엔드포인트
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
):
    if "whisperx" not in models:
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
        result = await loop.run_in_executor(
            None,
            lambda: _run_transcription(tmp_path, language, batch_size),
        )
        return JSONResponse(content=result)
        
    finally:
        os.unlink(tmp_path)

def _run_transcription(audio_path: str, language: Optional[str], batch_size: int) -> dict:
    """동기 전사 로직 (ThreadPoolExecutor에서 실행)"""
    
    model = models["whisperx"]
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    
    final_results = dict()
    
    texts = []
    
    for segment in result["segments"]:
        texts.extend(segment["text"])
        
    final_results["text"] = ''.join(texts).strip()
    
    return final_results
