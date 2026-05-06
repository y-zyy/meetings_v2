import json
import re

from app.config import settings

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(
            base_url=settings.LLM_API_BASE_URL,
            api_key=settings.LLM_API_KEY or "none",
            timeout=settings.LLM_TIMEOUT,
        )
    return _openai_client


SYSTEM_PROMPT = (
    "당신은 전문 회의록 작성 AI입니다. "
    "주어진 발화 기록을 분석하여 구조화된 회의록을 작성합니다. "
    "반드시 JSON 형식으로만 응답하고, 다른 텍스트는 포함하지 마세요."
)

USER_PROMPT_TEMPLATE = """\
다음 회의 정보와 발화 기록을 바탕으로 회의록을 작성해 주세요.

회의 제목: {title}
회의 일시: {meeting_date}
회의 장소: {location}
참석자: {attendees}
회의 안건:
{agenda}

발화 기록:
{transcript}

아래 JSON 형식으로만 응답하세요:
{{
  "summary": "주요 내용 요약 (3~5문장)",
  "decisions": [
    "결정사항 1",
    "결정사항 2"
  ],
  "action_items": [
    {{
      "content": "액션 아이템 내용",
      "assignee": "담당자 이름 또는 null",
      "due_date": "YYYY-MM-DD 또는 null"
    }}
  ]
}}"""


def generate_minutes(
    title: str,
    meeting_date: str,
    location: str,
    attendees: str,
    agenda: str,
    transcript: str,
) -> dict:
    """Call the LLM API and return parsed meeting minutes."""
    prompt = USER_PROMPT_TEMPLATE.format(
        title=title,
        meeting_date=meeting_date,
        location=location,
        attendees=attendees,
        agenda=agenda,
        transcript=transcript[:12000],  # guard against token overflow
    )

    if settings.ANTHROPIC_API_KEY:
        raw = _call_anthropic(prompt)
    else:
        raw = _call_openai(prompt)

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Return minimal structure so the meeting isn't left broken
        return {"summary": raw, "decisions": [], "action_items": []}


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=settings.LLM_TIMEOUT)
    message = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
