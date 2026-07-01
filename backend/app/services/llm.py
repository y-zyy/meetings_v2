from app.config import settings

USER_PROMPT_TEMPLATE = """\
회의 제목: {title}
참석자: {attendees}
회의 안건: {agenda}

회의록 스크립트: {transcript}

1. 주요 내용 예약
회의록 스크립트에 대해서 상세하게 요약해줘.
회의 정보(회의 제목, 참석자, 회의 안건)을 적극적으로 활용해서 상세 요약에 활용해줘.

2. Action Item 추출
Action Item도 함께 상세하게 요약해줘.
주어진 참석자 정보를 활용하여 다음과 같은 markdown 표 형태로 추출해줘.
|Action Item|담당자|일정|비고|
|------|-----|-----|-----|
"""


def generate_minutes(
    title: str,
    meeting_date: str,
    location: str,
    attendees: str,
    agenda: str,
    transcript: str,
    effective: dict | None = None,
) -> dict:
    """Call the LLM API and return meeting minutes as free-form text."""
    cfg = effective or {k: getattr(settings, k, "") for k in (
        "LLM_API_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT", "LLM_MAX_TOKENS",
    )}

    prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript)

    raw = _call_openai(prompt, cfg)

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
            {"role": "user", "content": prompt},
        ],
        temperature=0.9, # follows the default settings
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
