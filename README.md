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
