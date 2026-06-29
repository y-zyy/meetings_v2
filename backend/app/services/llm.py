from app.config import settings


USER_PROMPT_TEMPLATE = """\
{transcript}

회의록 스크립트에 대해서 상세하게 요약해줘.
Action Item도 함께 상세하게 요약해줘."""


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
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
