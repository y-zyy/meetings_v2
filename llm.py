"""
LLM 모듈 (OpenAI 호환 API, 스트리밍)

환경 변수:
  LLM_ENDPOINT                  - API 엔드포인트 (예: http://10.0.0.1:8080/v1)
  LLM_API_TOKEN                 - API 인증 토큰
  LLM_MODEL                     - 모델 이름 (기본: default)
  TRANSCRIPT_REFINE_PROMPT      - 전사본 정제용 시스템 프롬프트 (직접 입력)
  TRANSCRIPT_REFINE_PROMPT_FILE - 전사본 정제용 시스템 프롬프트 파일 경로
                                  (TRANSCRIPT_REFINE_PROMPT 미설정 시 사용)
"""

import json
import logging
import os
import re
import time

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ── 회의록 JSON 생성 프롬프트 ─────────────────────────────────────────

_MINUTES_SYSTEM_PROMPT = """\
당신은 회의 내용을 분석하여 정형화된 회의록 JSON을 생성하는 전문가입니다.

사용자가 제공하는 텍스트는 회의 음성을 자동 인식한 전사본입니다.
[사전 제공 정보]가 있으면 해당 정보를 우선적으로 사용하세요.
특히 "안건"이 제공된 경우 반드시 해당 안건을 "회의 안건" 필드에 반영하고,
각 안건별로 전사본에서 관련 내용을 정리하여 "주요 회의 내용"을 작성하세요.

## 출력 형식 (JSON only, 다른 텍스트 없이)

{
  "회의명": "회의 제목",
  "일시 및 장소": "날짜 및 시간 / 장소",
  "참석자": "이름 (역할), 이름 (역할), ...",
  "회의 안건": [
    "1. 안건1",
    "2. 안건2"
  ],
  "주요 회의 내용": [
    "▶ 안건1 관련 주요 내용",
    "  - 세부 내용",
    "",
    "▶ 안건2 관련 주요 내용",
    "  - 세부 내용"
  ],
  "Action Item": [
    "담당자 – 업무 내용 (기한)",
    "담당자 – 업무 내용 (기한)"
  ]
}

## 규칙

- 반드시 유효한 JSON만 출력하세요. 마크다운 코드 블록(```)은 포함하지 마세요.
- 사전 제공된 일시·장소·참석자 정보는 그대로 JSON 필드에 사용하세요.
- 전사본에서 명확히 파악할 수 없는 정보는 "미정" 또는 빈 문자열로 채우세요.
- 주요 회의 내용은 안건 순서에 맞춰 구분하고, 핵심 내용을 불릿 포인트로 정리하세요.
- Action Item이 없으면 빈 배열 []을 사용하세요.
- 모든 텍스트는 한국어로 작성하세요.
"""

# ── 전사본 정제 프롬프트 (기본값 / 환경 변수로 교체 가능) ─────────────

_DEFAULT_TRANSCRIPT_REFINE_PROMPT = """\
당신은 회의 음성 전사본을 교정하는 전문가입니다.

아래 지침에 따라 전사본을 정제하세요:
1. 음성인식 오류(잘못 인식된 단어, 불필요한 반복)를 수정하세요.
2. 문장 부호(마침표, 쉼표 등)를 자연스럽게 추가하세요.
3. 주제 전환이 있는 부분에서 단락을 나누세요.
4. 원문 내용을 요약하거나 삭제하지 말고 최대한 보존하세요.
5. 마크다운 없이 순수 텍스트로 출력하세요.
"""


def _get_transcript_refine_prompt() -> str:
    """환경 변수 또는 파일에서 전사본 정제 프롬프트를 로드합니다."""
    # 1순위: 환경 변수에 직접 입력된 프롬프트
    prompt = os.environ.get("TRANSCRIPT_REFINE_PROMPT", "").strip()
    if prompt:
        logger.info("[LLM] 전사본 정제 프롬프트: 환경 변수(TRANSCRIPT_REFINE_PROMPT) 사용")
        return prompt
    # 2순위: 파일 경로로 지정된 프롬프트
    prompt_file = os.environ.get("TRANSCRIPT_REFINE_PROMPT_FILE", "").strip()
    if prompt_file:
        if os.path.exists(prompt_file):
            with open(prompt_file, encoding="utf-8") as f:
                content = f.read().strip()
            logger.info("[LLM] 전사본 정제 프롬프트: 파일(%s) 사용", prompt_file)
            return content
        logger.warning("[LLM] TRANSCRIPT_REFINE_PROMPT_FILE 경로를 찾을 수 없습니다: %s", prompt_file)
    # 기본값
    logger.info("[LLM] 전사본 정제 프롬프트: 기본값 사용")
    return _DEFAULT_TRANSCRIPT_REFINE_PROMPT


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────

def _get_client() -> AsyncOpenAI:
    endpoint = os.environ.get("LLM_ENDPOINT", "").strip()
    token = os.environ.get("LLM_API_TOKEN", "none").strip()
    if not endpoint:
        raise EnvironmentError(
            "LLM_ENDPOINT 환경 변수가 설정되지 않았습니다. "
            "예: export LLM_ENDPOINT=http://10.0.0.1:8080/v1"
        )
    return AsyncOpenAI(base_url=endpoint, api_key=token)


def _build_meta_block(metadata: dict) -> str:
    """사전 제공 정보를 프롬프트에 삽입할 텍스트 블록으로 변환합니다."""
    parts: list[str] = []
    if metadata.get("회의명"):
        parts.append(f"회의명: {metadata['회의명']}")
    if metadata.get("일시"):
        parts.append(f"일시: {metadata['일시']}")
    if metadata.get("장소"):
        parts.append(f"장소: {metadata['장소']}")
    if metadata.get("참석자"):
        parts.append(f"참석자: {metadata['참석자']}")
    if metadata.get("안건"):
        agenda_lines = "\n".join(
            f"  {i+1}. {a}" for i, a in enumerate(metadata["안건"])
        )
        parts.append(f"안건:\n{agenda_lines}")
    if not parts:
        return ""
    return "\n[사전 제공 정보]\n" + "\n".join(parts) + "\n"


def _extract_json(text: str) -> str:
    """마크다운 코드 펜스를 제거하고 순수 JSON 문자열을 반환합니다."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def _stream_completion(client: AsyncOpenAI, model: str, messages: list) -> str:
    """스트리밍 완성 요청을 보내고 전체 텍스트를 반환합니다."""
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=32768,
        stream=True,
    )
    full_text = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_text += delta
    return full_text


# ── 공개 API ──────────────────────────────────────────────────────────

async def text_to_meeting_json(transcript: str, metadata: dict | None = None) -> dict:
    """ASR 전사본과 메타데이터를 LLM에 전달하고 회의록 JSON dict를 반환합니다."""
    client = _get_client()
    model = os.environ.get("LLM_MODEL", "default")

    meta_block = _build_meta_block(metadata or {})
    user_content = (
        f"다음 회의 전사본을 분석하여 회의록 JSON을 생성해주세요."
        f"{meta_block}\n"
        f"[전사본]\n{transcript}"
    )

    logger.info("[LLM] 회의록 JSON 생성 시작 (model=%s)", model)
    t0 = time.time()

    full_text = await _stream_completion(
        client,
        model,
        [
            {"role": "system", "content": _MINUTES_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    logger.info("[LLM] 회의록 JSON 생성 완료 (%.1f초, %d자)", time.time() - t0, len(full_text))

    json_str = _extract_json(full_text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM이 유효한 JSON을 반환하지 않았습니다: {e}\n응답:\n{full_text}"
        ) from e


async def refine_transcript(transcript: str, metadata: dict | None = None) -> str:
    """ASR 원문을 LLM으로 정제하여 읽기 좋은 텍스트를 반환합니다."""
    client = _get_client()
    model = os.environ.get("LLM_MODEL", "default")
    system_prompt = _get_transcript_refine_prompt()

    meta_block = _build_meta_block(metadata or {})
    user_content = (
        f"다음 회의 전사본을 정제해주세요."
        f"{meta_block}\n"
        f"[전사본]\n{transcript}"
    )

    logger.info("[LLM] 전사본 정제 시작 (model=%s)", model)
    t0 = time.time()

    refined = await _stream_completion(
        client,
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    logger.info("[LLM] 전사본 정제 완료 (%.1f초, %d자)", time.time() - t0, len(refined))
    return refined
