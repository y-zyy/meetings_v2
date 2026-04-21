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
이 전사본을 분석하여 아래 JSON 형식으로 반환하세요.

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
    "▶ 주제1",
    "  - 내용",
    "",
    "▶ 주제2",
    "  - 내용"
  ],
  "Action Item": [
    "담당자 – 업무 내용 (기한)",
    "담당자 – 업무 내용 (기한)"
  ]
}

## 규칙

- 반드시 유효한 JSON만 출력하세요. 마크다운 코드 블록(```)은 포함하지 마세요.
- 전사본에서 명확히 파악할 수 없는 정보(날짜, 장소 등)는 "미정" 또는 빈 문자열로 채우세요.
- 주요 회의 내용은 주제별로 구분하고, 핵심 내용을 불릿 포인트로 정리하세요.
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


def _extract_json(text: str) -> str:
    """마크다운 코드 펜스를 제거하고 순수 JSON 문자열을 반환합니다."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def text_to_meeting_json(transcript: str) -> dict:
    """ASR 전사본을 LLM에 전달하고 회의록 JSON dict를 반환합니다."""
    client = _get_client()
    model = os.environ.get("LLM_MODEL", "default")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"다음 회의 전사본을 분석하여 회의록 JSON을 생성해주세요:\n\n{transcript}",
        },
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
