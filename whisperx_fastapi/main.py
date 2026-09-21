# fastapi for WhisperX
# uvicorn으로 별도로 실행하여 "meetings_v2" 서비스에 엔드포인트 제공
#
# 전사(transcribe) -> 정렬(align) -> 화자 분리(diarize, DiariZen FastAPI 호출) ->
# 화자 레이블 병합(assign_word_speakers) 순서로 처리해 "누가 언제 무엇을 말했는지"를
# 함께 반환한다.

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import pandas as pd
import whisperx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# 설정
DEVICE = os.getenv("WHISPERX_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
WHISPER_MODEL = os.getenv("WHISPERX_MODEL", "large-v3")
DEFAULT_LANGUAGE = os.getenv("WHISPERX_LANGUAGE", "ko")

# DiariZen FastAPI 서비스 (diarizen_fastapi/main.py)
DIARIZE_SERVICE_URL = os.getenv("DIARIZE_SERVICE_URL", "http://localhost:9001/diarize")
DIARIZE_TIMEOUT = float(os.getenv("DIARIZE_TIMEOUT", "1800"))

models = {}
align_models: dict[str, tuple] = {}  # language_code -> (model, metadata)


# 라이프사이클
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading WhisperX model ({WHISPER_MODEL}) ...")
    models["whisperx"] = whisperx.load_model(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        vad_method="pyannote",
        language=DEFAULT_LANGUAGE,
    )
    print("Model loaded successfully!")
    yield
    models.clear()
    align_models.clear()
    print("Model unloaded")


# 앱 초기화
app = FastAPI(
    title="WhisperX Transcription API",
    description="Audio transcription + speaker diarization (WhisperX + DiariZen)",
    version="1.1.0",
    lifespan=lifespan,
)


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
    diarize: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
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

        diarize_error: Optional[str] = None
        if diarize:
            try:
                diarize_df = await _run_diarization(tmp_path, num_speakers, min_speakers, max_speakers)
                result = await loop.run_in_executor(
                    None,
                    lambda: whisperx.assign_word_speakers(diarize_df, result),
                )
            except Exception as exc:
                # 화자 분리 서비스가 죽어있어도 전사 결과는 반환한다.
                diarize_error = str(exc)

        response = _serialize_result(result)
        if diarize_error:
            response["diarize_error"] = diarize_error
        return JSONResponse(content=response)

    finally:
        os.unlink(tmp_path)


def _run_transcription(audio_path: str, language: Optional[str], batch_size: int) -> dict:
    """동기 전사 + 정렬 로직 (ThreadPoolExecutor에서 실행)

    화자 레이블을 단어 단위로 붙이려면 word-level timestamp가 필요하므로
    원래 스크립트의 2단계(align)까지 항상 수행한다.
    """
    model = models["whisperx"]
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    lang = result.get("language") or language or DEFAULT_LANGUAGE
    if lang not in align_models:
        align_models[lang] = whisperx.load_align_model(language_code=lang, device=DEVICE)
    model_a, metadata = align_models[lang]

    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False
    )
    result["language"] = lang
    return result


async def _run_diarization(
    audio_path: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> pd.DataFrame:
    """DiariZen FastAPI 서비스를 호출해 화자 구간을 받아온다.

    whisperx.assign_word_speakers()가 기대하는 start/end/speaker 컬럼을 가진
    DataFrame으로 변환해서 반환한다 (whisperx.diarize.DiarizationPipeline과 동일한 형태).
    """
    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    async with httpx.AsyncClient(timeout=DIARIZE_TIMEOUT) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                DIARIZE_SERVICE_URL,
                params=params,
                files={"file": (os.path.basename(audio_path), f, "application/octet-stream")},
            )
        response.raise_for_status()

    segments = response.json().get("segments", [])
    if not segments:
        return pd.DataFrame(columns=["start", "end", "speaker"])
    return pd.DataFrame(segments)


def _format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "00:00:00"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _serialize_result(result: dict) -> dict:
    """JSON 직렬화 + 화자/시간 정보를 포함한 응답 구성."""
    segments = []
    diarized_lines = []
    plain_texts = []

    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker")
        start = seg.get("start")
        end = seg.get("end")

        segments.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text,
            "words": [
                {
                    "word": w.get("word"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "speaker": w.get("speaker"),
                }
                for w in seg.get("words", [])
            ],
        })

        if not text:
            continue
        plain_texts.append(text)
        label = speaker or "SPEAKER_UNKNOWN"
        diarized_lines.append(f"[{_format_timestamp(start)} - {_format_timestamp(end)}] {label}: {text}")

    return {
        "language": result.get("language"),
        "segments": segments,
        # 기존 클라이언트(backend/app/services/asr.py) 호환을 위해 평문 텍스트를 그대로 유지.
        "text": " ".join(plain_texts).strip(),
        # 화자/시간이 포함된 표시용 텍스트 ("누가 언제 무엇을 말했는지").
        "diarized_text": "\n".join(diarized_lines).strip(),
    }
