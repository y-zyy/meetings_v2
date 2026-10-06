# meetings_v2

## HTTPS 설정 (사내망, 자체 서명 인증서)

마이크 녹음(`getUserMedia`)은 브라우저 보안 컨텍스트(HTTPS 또는 `localhost`)에서만
동작합니다. 사설 IP로 접속하는 사내망 환경에서는 아래 절차로 자체 서명 인증서를
발급해 nginx에 HTTPS를 붙이세요.

### 1. 인증서 발급

```bash
cd infra/nginx
./gen-selfsigned-cert.sh <서버 IP 또는 도메인>
# 예: ./gen-selfsigned-cert.sh 192.168.0.10
```

여러 IP/도메인으로 접속한다면 인자를 이어서 추가하면 됩니다.

```bash
./gen-selfsigned-cert.sh meetings.local 192.168.0.10
```

`infra/nginx/certs/selfsigned.{crt,key}` 파일이 생성됩니다. 이 파일들은
`.gitignore`에 등록되어 있어 커밋되지 않습니다.

### 2. nginx 적용

```bash
cd infra
docker compose up -d --build
# 이미 떠 있다면
docker compose restart nginx
```

### 3. 접속

- `https://<서버 IP>:8443` 으로 접속하세요.
- `http://<서버 IP>:8080` 으로 접속하면 자동으로 `https://<서버 IP>:8443` 으로 리다이렉트됩니다.
- 자체 서명 인증서이므로 최초 접속 시 브라우저가 "안전하지 않음" 경고를 띄웁니다.
  Chrome 기준 "고급" → "**(주소)**(안전하지 않음)로 이동" 을 클릭하면 정상적으로
  접속되고, 이후 마이크 녹음 기능도 정상 동작합니다.
- 경고 자체를 없애려면 `infra/nginx/certs/selfsigned.crt` 를 사내 PC들의 신뢰된
  루트 인증기관 저장소에 배포하면 됩니다.

### 실제 도메인이 있는 경우

외부에서 접근 가능한 실제 도메인이 있다면 자체 서명 인증서 대신 Let's Encrypt
(certbot) 등 정식 인증서 발급으로 교체하는 것을 권장합니다.

## 화자 분리 (WhisperX + pyannote)

`whisperx_fastapi/` 는 별도 GPU 도커로 구동하는 ASR 서버입니다.

```bash
cd whisperx_fastapi
docker build -t whisperx-asr:261006 .
docker run --gpus all -p 9000:9000 -e HF_TOKEN=<HuggingFace 토큰> whisperx-asr:261006
```

- 처리 순서: **ASR → STT 후처리(규칙/LLM 교정) → 정렬·화자 분리 → 회의록 생성**
  - `POST /transcribe`: 전사만 수행. 응답 `{"text", "language", "segments": [{"start", "end", "text"}]}`
  - `POST /align_diarize`: 오디오 + 후처리된 `segments`(JSON, form 필드)를 받아 후처리 텍스트를 강제 정렬(align)하고
    화자를 부여. 응답 `{"language", "diarized", "segments": [{"start", "end", "speaker", "text"}]}`
    (form: `language`, `diarize`, `min_speakers`, `max_speakers`)
  - `HF_TOKEN` 이 없으면 화자 분리는 건너뛰고 `diarized=false`, `speaker=UNKNOWN`을 반환합니다. 백엔드는 이 라벨을 제거하고 화면은 시간 구간과 “화자 미상”을 표시합니다.
  - 백엔드는 `ASR_API_URL` 의 `/transcribe` 를 `/align_diarize` 로 치환해 호출합니다
    (`ASR_DIARIZE_API_URL` 로 지정 가능, `ASR_DIARIZE_ENABLED=false` 로 끌 수 있음).
  - 화자 분리 단계가 실패하면 화자 정보 없이 후처리된 텍스트로 회의록 생성을 계속합니다.
- 메인 백엔드는 `segments` 를 `meetings.segments` (JSON)에 저장하고, 화면의 "발화 기록"에
  Speaker / 발화 시간(시작–종료) / 발화 내용 순서의 표를 표시합니다. 화자 이름 클릭 시 표시 이름을 변경할 수 있습니다(`speaker_names`).
- 기존 DB는 백엔드 기동 시 `segments`, `speaker_names` 컬럼이 자동 추가됩니다
  (alembic `005_diarization` 과 동일).
- 기존 텍스트만 있는 기록(OpenAI Whisper 포함)도 표에 표시하며 화자는 “화자 미상”, 시간은 “—”로 표시합니다. 기존 기록을 재처리하거나 수정하지 않습니다.

## 261006 이미지 빌드 및 기존 DB 유지

직접 빌드하는 이미지는 `meetings-backend:261006`(API 및 4개 worker 공용),
`meetings-frontend:261006`, `whisperx-asr:261006`입니다.
PostgreSQL(`postgres:17-alpine`)과 Redis(`redis:7-alpine`)는 기존 공식 이미지 태그를 유지합니다.

기존 서버의 동일한 체크아웃 경로에서 기존 `.env`, 업로드 폴더, 인증서와 Compose 프로젝트 이름을 유지한 채 실행하세요.
기존 실행에 `-p` 또는 `COMPOSE_PROJECT_NAME`을 사용했다면 아래 명령에도 같은 값을 적용해야 기존 볼륨을 사용합니다.

```bash
cd infra
docker compose build api nginx
# 실행 중인 postgres/redis는 재생성하지 않고 애플리케이션만 교체합니다.
docker compose up -d --no-deps api worker-ingest worker-asr worker-postprocess worker-minutes nginx
cd ..
docker build -t whisperx-asr:261006 ./whisperx_fastapi
```

WhisperX 컨테이너 교체 시 기존 GPU/포트/네트워크/환경 변수 설정을 유지하고 이미지 이름만
`whisperx-asr:261006`으로 지정하세요. `.env`의 ASR 주소는 해당 서버 주소를 유지합니다.
기본 Compose는 프런트엔드 폴더와 nginx 설정을 바인드 마운트하므로 서버의 소스도 함께 갱신해야 합니다.
`docker-compose.override.yml`은 기존과 동일하게 개발용 소스 마운트를 적용합니다.

`postgres_data` 볼륨 선언과 마운트, DB 설정 및 마이그레이션은 변경하지 않습니다.
`docker compose down -v` 또는 볼륨 삭제/DB 초기화 명령을 실행하지 마세요.
기존 백엔드의 시작 동작(누락된 화자 관련 컬럼 추가 등)은 그대로 유지됩니다.

프런트엔드 회귀 검사: 저장소 루트에서 `node --test tests/transcript.test.cjs`.
