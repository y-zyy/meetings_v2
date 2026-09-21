# DiariZen FastAPI

[DiariZen](https://github.com/BUTSpeechFIT/DiariZen) 화자 분리(diarization) 모델을
FastAPI로 감싼 독립 서비스입니다. `whisperx_fastapi` 서비스가 전사(transcribe)
결과에 화자 레이블을 붙이기 위해 HTTP로 이 서버를 호출합니다.

## 설치

DiariZen은 PyPI 패키지가 아니라 소스 설치가 필요합니다 (pyannote-audio 서브모듈 포함).

```bash
git clone https://github.com/BUTSpeechFIT/DiariZen.git
cd DiariZen
conda create --name diarizen python=3.10 -y
conda activate diarizen
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e .
cd pyannote-audio && pip install -e .[dev,testing] -c ../constraints.txt && cd ..

# 이 저장소의 FastAPI 래퍼 의존성 설치
pip install -r <이 저장소>/diarizen_fastapi/requirements.txt
```

모델 가중치(`BUT-FIT/diarizen-wavlm-large-s80-md`)는 최초 실행 시 Hugging Face에서
자동으로 다운로드됩니다.

## 실행

```bash
export DIARIZEN_MODEL=BUT-FIT/diarizen-wavlm-large-s80-md   # 기본값
export DIARIZEN_DEVICE=cuda                                  # 기본값

uvicorn diarizen_fastapi.main:app --host 0.0.0.0 --port 9001
```

## API

### `GET /health`
모델 로딩 여부 확인.

### `POST /diarize`
- multipart `file`: 오디오 파일 (`.mp3 .wav .m4a .flac .ogg .webm`)
- query params (선택): `num_speakers`, `min_speakers`, `max_speakers`
  (설치된 DiariZen 버전이 지원하지 않으면 무시됩니다)

응답:

```json
{
  "segments": [
    {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"},
    {"start": 3.2, "end": 7.8, "speaker": "SPEAKER_01"}
  ]
}
```

이 형식은 `whisperx.assign_word_speakers()`가 기대하는
`start` / `end` / `speaker` 컬럼과 동일하게 맞춰져 있습니다.
