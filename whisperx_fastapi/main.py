"""FastAPI service for bounded, GPU-backed WhisperX transcription."""

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import whisperx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from whisperx.diarize import DiarizationPipeline


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
# 화자 분리(pyannote)는 HuggingFace 토큰이 필요하다. 없으면 화자 분리를 건너뛴다.
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or ""
DEFAULT_DIARIZE = os.getenv("WHISPERX_DIARIZE", "true").lower() in ("1", "true", "yes")
UNKNOWN_SPEAKER = "UNKNOWN"

models = {}
align_models = {}  # language -> (model, metadata) | None (정렬 모델 없음)


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
    if HF_TOKEN:
        print("Loading diarization pipeline ...")
        models["diarize"] = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    else:
        print("HF_TOKEN not set - speaker diarization disabled")
    app.state.inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    print("Model loaded successfully!")
    yield
    models.clear()
    align_models.clear()
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
        "diarization_available": "diarize" in models,
        "device": DEVICE,
        "max_concurrency": MAX_CONCURRENCY,
    }


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    """업로드 파일을 임시 파일로 저장하고 (경로, 확장자)를 돌려준다."""
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
    except BaseException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return tmp_path, ext


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
):
    """1단계: 전사만 수행한다. (정렬/화자 분리는 후처리 이후 /align_diarize 에서 수행)

    응답: {"text", "language", "segments": [{"start", "end", "text"}]}
    """
    if "whisperx" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    tmp_path = None
    try:
        tmp_path, _ = await _save_upload(file)
        effective_batch_size = min(max(1, batch_size), MAX_BATCH_SIZE)
        async with request.app.state.inference_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, _run_transcription, tmp_path, language, effective_batch_size
            )
        return JSONResponse(content=result)
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/align_diarize")
async def align_diarize(
    request: Request,
    file: UploadFile = File(...),
    segments: str = Form(..., description='후처리된 세그먼트 JSON: [{"start","end","text"}]'),
    language: Optional[str] = Form(None),
    diarize: bool = Form(DEFAULT_DIARIZE),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    """2단계: 후처리를 거친 텍스트를 오디오에 강제 정렬(align)하고 화자(diarize)를 부여한다.

    응답: {"language", "diarized", "segments": [{"start", "end", "speaker", "text"}]}
    """
    if "whisperx" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        parsed = json.loads(segments)
        input_segments = [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg.get("text") or "").strip(),
            }
            for seg in parsed
            if str(seg.get("text") or "").strip()
        ]
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid segments JSON: {exc}")
    if not input_segments:
        return JSONResponse(content={"language": language, "diarized": False, "segments": []})

    tmp_path = None
    try:
        tmp_path, _ = await _save_upload(file)
        async with request.app.state.inference_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                _run_align_diarize,
                tmp_path,
                input_segments,
                language or "ko",
                diarize,
                min_speakers,
                max_speakers,
            )
        return JSONResponse(content=result)
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _get_align_model(language: str):
    if language not in align_models:
        try:
            align_models[language] = whisperx.load_align_model(
                language_code=language, device=DEVICE
            )
        except Exception as exc:  # 해당 언어의 정렬 모델이 없는 경우
            print(f"No alignment model for '{language}': {exc}")
            align_models[language] = None
    return align_models[language]


def _split_by_speaker(segment: dict) -> list[dict]:
    """단어별 화자가 바뀌는 지점에서 세그먼트를 쪼갠다 (누가/언제부터/언제까지)."""
    words = segment.get("words") or []
    speaker = segment.get("speaker")
    if not any("start" in w and "end" in w for w in words):
        text = (segment.get("text") or "").strip()
        if not text:
            return []
        return [{
            "start": round(float(segment.get("start", 0.0)), 3),
            "end": round(float(segment.get("end", segment.get("start", 0.0))), 3),
            "speaker": speaker or UNKNOWN_SPEAKER,
            "text": text,
        }]

    runs: list[dict] = []
    current = None
    for w in words:
        w_speaker = w.get("speaker") or (current["speaker"] if current else speaker) or UNKNOWN_SPEAKER
        if current is None or w_speaker != current["speaker"]:
            current = {"speaker": w_speaker, "words": []}
            runs.append(current)
        current["words"].append(w)

    out = []
    for run in runs:
        ws = run["words"]
        # 화자가 하나뿐이면 후처리된 원문 텍스트를 그대로 유지한다.
        text = (segment.get("text") or "").strip() if len(runs) == 1 else " ".join((w.get("word") or "").strip() for w in ws).strip()
        if not text:
            continue
        starts = [w["start"] for w in ws if "start" in w]
        ends = [w["end"] for w in ws if "end" in w]
        out.append({
            "start": round(float(min(starts) if starts else segment.get("start", 0.0)), 3),
            "end": round(float(max(ends) if ends else segment.get("end", 0.0)), 3),
            "speaker": run["speaker"],
            "text": text,
        })
    return out


def _run_transcription(
    audio_path: str,
    language: Optional[str],
    batch_size: int,
) -> dict:
    """Synchronous GPU inference executed outside the event loop."""
    model = models["whisperx"]
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    segments = [
        {
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
        if seg["text"].strip()
    ]
    text = "".join(segment["text"] for segment in result["segments"]).strip()
    return {
        "text": text,
        "language": result.get("language") or language,
        "segments": segments,
    }


def _run_align_diarize(
    audio_path: str,
    input_segments: list[dict],
    language: str,
    diarize: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict:
    """후처리된 세그먼트 텍스트를 정렬하고 화자 라벨을 붙인다."""
    audio = whisperx.load_audio(audio_path)
    result = {"segments": input_segments, "language": language}

    # 2. Align whisper output (후처리된 텍스트 기준 단어 타임스탬프)
    aligned = _get_align_model(language)
    if aligned is not None:
        model_a, metadata = aligned
        result = whisperx.align(
            input_segments, model_a, metadata, audio, DEVICE,
            return_char_alignments=False,
        )

    # 3. Assign speaker labels
    diarized = False
    if diarize and "diarize" in models:
        diarize_kwargs = {}
        if min_speakers:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = models["diarize"](audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        diarized = True

    segments = []
    for seg in result["segments"]:
        segments.extend(_split_by_speaker(seg))

    return {"language": language, "diarized": diarized, "segments": segments}
