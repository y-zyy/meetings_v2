"""
LLM 모듈 (OpenAI 호환 API, 스트리밍)

환경 변수:
  LLM_ENDPOINT   - API 엔드포인트 (예: http://10.0.0.1:8080/v1)
  LLM_API_TOKEN  - API 인증 토큰
  LLM_MODEL      - 모델 이름 (기본: default)
"""

import json
import os
import re

from openai import AsyncOpenAI

_SYSTEM_PROMPT = """\
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


async def text_to_meeting_json(transcript: str, metadata: dict | None = None) -> dict:
    """ASR 전사본과 선택적 메타데이터를 LLM에 전달하고 회의록 JSON dict를 반환합니다."""
    client = _get_client()
    model = os.environ.get("LLM_MODEL", "default")

    meta_block = _build_meta_block(metadata or {})
    user_content = (
        f"다음 회의 전사본을 분석하여 회의록 JSON을 생성해주세요."
        f"{meta_block}\n"
        f"[전사본]\n{transcript}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

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

    json_str = _extract_json(full_text)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM이 유효한 JSON을 반환하지 않았습니다: {e}\n응답:\n{full_text}"
        ) from e
