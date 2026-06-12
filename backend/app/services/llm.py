import json
import re

from app.config import settings


SYSTEM_PROMPT = (
    "당신은 전문 회의록 작성 AI입니다. "
    "주어진 발화 기록을 분석하여 구조화된 회의록을 작성합니다. "
    "회의록 요약은 반드시 상세하게 작성합니다. 분량을 최대한 길고 상세하게 작성합니다."
    "회의록 요약은 회의 모든 내용을 최대한 반영합니다."
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
  "summary": "주요 내용 요약",
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
    effective: dict | None = None,
) -> dict:
    """Call the LLM API and return parsed meeting minutes.

    effective: dict returned by get_effective_settings_sync(); when None the
    module-level settings object is used directly (backward-compat).
    """
    cfg = effective or {k: getattr(settings, k, "") for k in (
        "LLM_API_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT", "LLM_MAX_TOKENS",
    )}

    prompt = USER_PROMPT_TEMPLATE.format(
        title=title,
        meeting_date=meeting_date,
        location=location,
        attendees=attendees,
        agenda=agenda,
        transcript=transcript,
    )

    raw = _call_openai(prompt, cfg)

    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw, "decisions": [], "action_items": []}


def _call_openai(prompt: str, cfg: dict) -> str:
    from openai import OpenAI
    base_url = cfg.get("LLM_API_BASE_URL") or settings.LLM_API_BASE_URL
    api_key = cfg.get("LLM_API_KEY") or "none"
    model = cfg.get("LLM_MODEL") or settings.LLM_MODEL
    timeout = int(cfg.get("LLM_TIMEOUT") or settings.LLM_TIMEOUT)
    max_tokens = int(cfg.get("LLM_MAX_TOKENS") or settings.LLM_MAX_TOKENS)

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
