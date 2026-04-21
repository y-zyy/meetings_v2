import gc
import whisperx


def transcribe_audio(
    audio_path: str,
    device: str = "cuda",
    batch_size: int = 16,
    compute_type: str = "float16",
) -> str:
    """Run WhisperX on audio_path and return the full transcript as a string."""
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    del model
    gc.collect()

    segments = result.get("segments", [])
    return " ".join(seg["text"].strip() for seg in segments if seg.get("text"))
