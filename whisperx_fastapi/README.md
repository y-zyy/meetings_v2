# WhisperX FastAPI

WhisperX(전사 + 정렬)와 [DiariZen](../diarizen_fastapi) 화자 분리를 연결해
"누가(speaker) / 언제(start~end) / 무엇을(text) 말했는지" 알 수 있는 세그먼트를
반환하는 독립 서비스입니다.

## 처리 순서

1. `whisperx.load_model(...).transcribe()` — 배치 전사
2. `whisperx.load_align_model()` + `whisperx.align()` — 단어 단위 타임스탬프 정렬
3. `diarizen_fastapi`(`POST /diarize`) 호출 — 화자 구간(start/end/speaker) 조회
4. `whisperx.assign_word_speakers()` — 세그먼트/단어에 화자 레이블 병합

DiariZen 서비스가 응답하지 않아도 전사 자체는 실패하지 않고, `diarize_error`
필드에 원인을 담아 반환합니다 (화자 레이블 없이).

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `WHISPERX_MODEL` | `large-v3` | Whisper 모델 크기 |
| `WHISPERX_DEVICE` | `cuda` | `cuda` / `cpu` |
| `WHISPERX_COMPUTE_TYPE` | `float16` | GPU 메모리 부족 시 `int8` |
| `WHISPERX_BATCH_SIZE` | `16` | 배치 크기 |
| `WHISPERX_LANGUAGE` | `ko` | 모델 로딩 시 기본 언어 |
| `DIARIZE_SERVICE_URL` | `http://localhost:9001/diarize` | `diarizen_fastapi` 엔드포인트 |
| `DIARIZE_TIMEOUT` | `1800` | 화자 분리 호출 타임아웃(초) |

## 실행

```bash
pip install -r whisperx_fastapi/requirements.txt
uvicorn whisperx_fastapi.main:app --host 0.0.0.0 --port 9000
```

`diarizen_fastapi`는 별도 프로세스(필요 시 별도 GPU/서버)로 실행합니다:

```bash
uvicorn diarizen_fastapi.main:app --host 0.0.0.0 --port 9001
```

## API

### `GET /health`

### `POST /transcribe`
- multipart `file`: 오디오 파일 (`.mp3 .wav .m4a .flac .ogg .webm`)
- query params (선택): `language`, `batch_size`, `diarize` (기본 `true`),
  `num_speakers`, `min_speakers`, `max_speakers`

응답 예시:

```json
{
  "language": "ko",
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "speaker": "SPEAKER_00",
      "text": "안녕하세요, 오늘 회의 시작하겠습니다.",
      "words": [
        {"word": "안녕하세요,", "start": 0.0, "end": 0.6, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "text": "안녕하세요, 오늘 회의 시작하겠습니다. ...",
  "diarized_text": "[00:00:00 - 00:00:03] SPEAKER_00: 안녕하세요, 오늘 회의 시작하겠습니다.\n..."
}
```

- `text`: 기존 `backend/app/services/asr.py`(`ASR_RESPONSE_FIELD=text`)와의
  호환을 위한 평문 전사 결과 (화자 레이블 없음, 기존 동작과 동일).
- `diarized_text` / `segments[].speaker` / `segments[].words[].speaker`:
  화자·시간 정보가 포함된 새 필드. 회의록 화면에 "누가 언제 무엇을 말했는지"를
  보여주려면 backend에서 이 필드를 사용하도록 확장하면 됩니다.
