"""Run with uvicorn diarizen.api.app:app --host 0.0.0.0 --port 8000 --workers 1."""

import json
import logging
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import pandas as pd
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    model_id: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    model_dir: Optional[str] = None
    embedding_model: Optional[str] = None
    # 500 MiB and 6 hours. Environment variables can override these defaults.
    max_upload_bytes: int = 524_288_000
    max_audio_seconds: float = 21_600.0

    def __post_init__(self):
        if bool(self.model_dir) != bool(self.embedding_model):
            raise ValueError("Set both DIARIZEN_MODEL_DIR and DIARIZEN_EMBEDDING_MODEL.")
        if self.max_upload_bytes <= 0 or not 0 < self.max_audio_seconds < float("inf"):
            raise ValueError("Upload and duration limits must be positive and finite.")

    @classmethod
    def from_env(cls):
        return cls(
            model_id=os.getenv("DIARIZEN_MODEL_ID", cls.model_id),
            model_dir=os.getenv("DIARIZEN_MODEL_DIR") or None,
            embedding_model=os.getenv("DIARIZEN_EMBEDDING_MODEL") or None,
            max_upload_bytes=int(
                os.getenv("DIARIZEN_MAX_UPLOAD_BYTES", str(cls.max_upload_bytes))
            ),
            max_audio_seconds=float(
                os.getenv("DIARIZEN_MAX_AUDIO_SECONDS", str(cls.max_audio_seconds))
            ),
        )


class SegmentResult(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResult(BaseModel):
    session_id: str
    exclusive: bool = True
    num_speakers: int
    segments: list[SegmentResult]
    # Present only when transcript_json is supplied.
    whisperx_segments: Optional[list[dict[str, Any]]] = None


class InvalidAudio(ValueError):
    pass


class AudioTooLong(ValueError):
    pass


def _whisperx_speaker_name(speaker: object) -> str:
    """Normalize a DiariZen label to the speaker_* form used by this API."""
    value = str(speaker)
    return value if value.startswith("speaker_") else f"speaker_{value}"


def diarizen_to_whisperx_df(diar_results) -> pd.DataFrame:
    """Convert a DiariZen pyannote Annotation to a WhisperX diarization frame."""
    diarize_segments = pd.DataFrame(
        diar_results.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )

    if diarize_segments.empty:
        # Ensure WhisperX still receives every expected column for empty audio.
        diarize_segments["start"] = pd.Series(dtype="float64")
        diarize_segments["end"] = pd.Series(dtype="float64")
        diarize_segments["speaker"] = pd.Series(dtype="object")
        return diarize_segments

    diarize_segments["start"] = diarize_segments["segment"].apply(
        lambda segment: float(segment.start)
    )
    diarize_segments["end"] = diarize_segments["segment"].apply(
        lambda segment: float(segment.end)
    )
    diarize_segments["speaker"] = diarize_segments["speaker"].apply(
        _whisperx_speaker_name
    )
    return diarize_segments


def load_predictor(settings: Settings):
    # Heavy imports and model downloads happen only during application startup.
    import torch
    import torchaudio
    from diarizen.pipelines.inference import DiariZenPipeline

    if settings.model_dir:
        pipeline = DiariZenPipeline(
            diarizen_hub=Path(settings.model_dir),
            embedding_model=settings.embedding_model,
            exclusive=True,
        )
    else:
        pipeline = DiariZenPipeline.from_pretrained(settings.model_id, exclusive=True)

    def predict(path: str, session_id: str):
        try:
            metadata = torchaudio.info(path)
        except (RuntimeError, ValueError, OSError) as exc:
            raise InvalidAudio("Cannot decode this audio file.") from exc
        if metadata.sample_rate <= 0 or metadata.num_frames <= 0:
            raise InvalidAudio("Audio is empty or has no valid duration.")
        if metadata.num_frames / metadata.sample_rate > settings.max_audio_seconds:
            raise AudioTooLong("Audio exceeds the configured duration limit.")
        with torch.inference_mode():
            return pipeline(path, sess_name=session_id)

    return predict


def _parse_transcript_json(transcript_json: Optional[str]) -> Optional[dict[str, Any]]:
    if transcript_json is None:
        return None
    try:
        transcript = json.loads(transcript_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "transcript_json must be valid JSON.") from exc
    if not isinstance(transcript, dict) or not isinstance(transcript.get("segments"), list):
        raise HTTPException(
            422,
            "transcript_json must be an object containing a segments array.",
        )
    return transcript


def create_app(
    settings: Optional[Settings] = None,
    predictor_loader: Callable = load_predictor,
):
    """Create the application; injectable loader allows tests without model weights."""
    settings = settings or Settings.from_env()
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = await run_in_threadpool(predictor_loader, settings)
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(title="DiariZen API", version="1.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        if getattr(app.state, "predictor", None) is None:
            raise HTTPException(503, "Model is not ready.")
        return {
            "status": "ok",
            "exclusive": True,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_audio_seconds": settings.max_audio_seconds,
        }

    @app.post("/diarize", response_model=DiarizationResult)
    def diarize(
        file: UploadFile = File(...),
        session_id: Optional[str] = Form(None),
        transcript_json: Optional[str] = Form(None),
    ):
        # A synchronous endpoint runs in FastAPI's thread pool, keeping the
        # event loop responsive. The lock is held until inference truly ends.
        acquired = False
        try:
            session_id = session_id or uuid4().hex
            if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id) is None:
                raise HTTPException(
                    422,
                    "session_id must contain 1-128 letters, digits, _ or -.",
                )
            transcript = _parse_transcript_json(transcript_json)
            suffix = Path(file.filename or "").suffix.lower()
            predictor = getattr(app.state, "predictor", None)
            if predictor is None:
                raise HTTPException(503, "Model is not ready.")
            acquired = inference_lock.acquire(blocking=False)
            if not acquired:
                raise HTTPException(
                    503,
                    "Inference is busy. Retry later.",
                    headers={"Retry-After": "5"},
                )
            with tempfile.TemporaryDirectory(prefix="diarizen-") as directory:
                path = Path(directory) / ("audio" + suffix)
                size = 0
                with path.open("wb") as output:
                    while chunk := file.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > settings.max_upload_bytes:
                            raise HTTPException(
                                413,
                                "File exceeds the configured upload limit.",
                            )
                        output.write(chunk)
                if size == 0:
                    raise HTTPException(400, "Uploaded file is empty.")

                annotation = predictor(str(path), session_id)
                diarize_segments = diarizen_to_whisperx_df(annotation)
                segments = sorted(
                    [
                        SegmentResult(
                            start=float(row.start),
                            end=float(row.end),
                            speaker=str(row.speaker),
                        )
                        for row in diarize_segments.itertuples(index=False)
                    ],
                    key=lambda item: (item.start, item.end, item.speaker),
                )

                whisperx_segments = None
                if transcript is not None:
                    assigned = whisperx.assign_word_speakers(
                        diarize_segments,
                        transcript,
                    )
                    whisperx_segments = assigned["segments"]

                return DiarizationResult(
                    session_id=session_id,
                    num_speakers=len({segment.speaker for segment in segments}),
                    segments=segments,
                    whisperx_segments=whisperx_segments,
                )
        except InvalidAudio as exc:
            raise HTTPException(400, str(exc)) from exc
        except AudioTooLong as exc:
            raise HTTPException(413, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Diarization failed for session %s", session_id)
            raise HTTPException(500, "Diarization failed. See server logs.") from exc
        finally:
            if acquired:
                inference_lock.release()
            file.file.close()

    return app


app = create_app()
