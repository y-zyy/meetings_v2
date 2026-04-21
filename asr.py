import whisperx

_model = None
_batch_size: int = 16


def init_model(
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
) -> None:
    """서버 시작 시 한 번만 호출. 모델을 GPU에 올리고 모듈 전역에 유지합니다."""
    global _model, _batch_size
    _model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    _batch_size = batch_size


def transcribe_audio(audio_path: str) -> str:
    """전역 모델로 오디오를 전사하고 전체 텍스트를 반환합니다."""
    if _model is None:
        raise RuntimeError("모델이 초기화되지 않았습니다. init_model()을 먼저 호출하세요.")
    audio = whisperx.load_audio(audio_path)
    result = _model.transcribe(audio, batch_size=_batch_size)
    segments = result.get("segments", [])
    return " ".join(seg["text"].strip() for seg in segments if seg.get("text"))
