# meetings_v2

## 실행 / 업데이트 (Docker Compose)

### 최초 실행

```bash
cd infra
docker compose up -d --build
```

### 코드를 새로 받은 뒤 (브랜치 전환, `git pull` 등)

`backend/Dockerfile`은 애플리케이션 코드를 `COPY . .`로 이미지 안에 박아넣기
때문에, 코드가 바뀌면(의존성 변경 여부와 무관하게) 이미지를 다시 빌드해야
컨테이너에 반영됩니다.

```bash
cd infra
git pull                      # 또는 원하는 브랜치로 checkout
docker compose up -d --build
```

- `--build`는 변경된 레이어만 다시 빌드하고, 새 이미지로 컨테이너를 재생성합니다.
- **개발 환경**(`docker-compose.override.yml`이 같이 적용되는 경우)은 `backend/`
  디렉터리를 컨테이너에 통째로 bind mount하므로, `api`/`worker-*` 컨테이너의 코드
  변경은 재빌드 없이 자동 반영됩니다 (`uvicorn --reload`, `watchmedo auto-restart`).
  이 경우엔 `--build` 없이 `docker compose up -d`만 해도 됩니다. 단
  `requirements.txt` 등 의존성이 바뀔 때는 개발 환경에서도 `--build`가 필요합니다.

### Celery 워커 서비스 분리 (`worker` → `worker-asr` / `worker-llm`)

ASR과 LLM 처리를 별도 Celery 큐/워커로 분리하면서, 기존 `worker` 서비스 하나가
`worker-asr`(ASR 전용, concurrency=4)와 `worker-llm`(LLM 전용, concurrency=12)
두 개로 나뉘었습니다. 이전에 떠 있던 `worker` 컨테이너는 새 compose 파일에 더
이상 정의돼 있지 않아 고아 컨테이너로 남으므로, 업데이트 시 `--remove-orphans`를
붙여서 정리하세요.

```bash
cd infra
docker compose up -d --build --remove-orphans
```

정상적으로 반영됐다면 `worker` 컨테이너는 사라지고, `worker-asr`/`worker-llm`
두 개가 대신 떠 있어야 합니다.

```bash
docker compose ps
docker compose logs -f worker-asr
docker compose logs -f worker-llm
```

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
