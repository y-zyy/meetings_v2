"""FastAPI service for bounded, GPU-backed WhisperX transcription."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import whisperx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse


DEVICE = os.getenv("WHISPERX_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16")
BATCH_SIZE = max(1, int(os.getenv("WHISPERX_BATCH_SIZE", "16")))
MAX_BATCH_SIZE = max(1, int(os.getenv("WHISPERX_MAX_BATCH_SIZE", str(BATCH_SIZE))))
MAX_CONCURRENCY = max(1, int(os.getenv("WHISPERX_MAX_CONCURRENCY", "1")))
UPLOAD_CHUNK_BYTES = max(
    1, int(os.getenv("WHISPERX_UPLOAD_CHUNK_SIZE_MB", "4"))
) * 1024 * 1024
MAX_UPLOAD_BYTES = max(
    1, int(os.getenv("WHISPERX_MAX_UPLOAD_SIZE_MB", "500"))
) * 1024 * 1024

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run one server process per GPU. The semaphore bounds concurrent inference
    # inside that process so requests cannot overcommit the loaded model.
    print(f"Loading WhisperX model on {DEVICE} ...")
    models["whisperx"] = whisperx.load_model(
        "large-v3",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        vad_method="pyannote",
        language="ko",
    )
    app.state.inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    print("Model loaded successfully!")
    yield
    models.clear()
    print("Model unloaded")


app = FastAPI(
    title="WhisperX Transcription API",
    description="Audio transcription powered by WhisperX",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": "whisperx" in models,
        "device": DEVICE,
        "max_concurrency": MAX_CONCURRENCY,
    }


@app.post("/transcribe")
async def transcribe(
    request: Request,
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

    tmp_path = None
    total_size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file is too large",
                    )
                await asyncio.to_thread(tmp.write, chunk)

        effective_batch_size = min(max(1, batch_size), MAX_BATCH_SIZE)
        async with request.app.state.inference_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                _run_transcription,
                tmp_path,
                language,
                effective_batch_size,
            )
        return JSONResponse(content=result)
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _run_transcription(
    audio_path: str,
    language: Optional[str],
    batch_size: int,
) -> dict:
    """Synchronous GPU inference executed outside the event loop."""
    model = models["whisperx"]
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    text = "".join(segment["text"] for segment in result["segments"]).strip()
    return {"text": text}

