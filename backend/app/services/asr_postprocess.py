"""STT 후처리: LLM을 사용해 음성인식 오류를 보정합니다."""

SYSTEM_PROMPT = (
    "당신은 음성인식(STT) 결과를 후처리하는 전문가입니다. "
    "음성인식 출력에서 발생한 오인식 용어, 인명, 장소, 전문용어 등의 오류를 수정합니다. "
    "다음 원칙을 반드시 준수하세요:\n"
    "1. 원문의 표현, 말투, 문체, 구조를 그대로 유지합니다.\n"
    "2. 확실한 오인식(잘못 인식된 고유명사, 전문용어, 인명, 장소 등)만 수정합니다.\n"
    "3. 내용 요약, 재구성, 문장 다듬기는 하지 않습니다.\n"
    "4. 제공된 용어사전의 단어들이 정확하게 표기되도록 수정합니다.\n"
    "5. 수정된 텍스트만 출력하고, 설명이나 다른 텍스트는 포함하지 마세요."
)

POSTPROCESS_PROMPT_TEMPLATE = """\
아래 회의 정보와 용어사전을 참고하여 음성인식 결과를 후처리하세요.

=== 회의 정보 ===
회의 제목: {title}
회의 일시: {meeting_date}
회의 장소: {location}
참석자: {attendees}
회의 안건:
{agenda}
참고사항: {notes}

=== 용어사전 ===
{glossary_section}

=== 음성인식 원문 ===
{transcript}

위 원문에서 회의 정보 및 용어사전을 참고하여 확실한 오인식만 수정한 후처리 결과를 출력하세요.
원문의 표현과 구조를 그대로 유지하고, 수정된 텍스트만 출력하세요."""


def postprocess_transcript(
    transcript: str,
    title: str,
    meeting_date: str,
    location: str,
    attendees: str,
    agenda: str,
    notes: str,
    admin_terms: list[str],
    user_terms: list[str],
    effective: dict | None = None,
) -> str:
    """STT 결과를 LLM으로 후처리하여 반환합니다."""
    if not transcript or not transcript.strip():
        return transcript

    glossary_parts = []
    if admin_terms:
        glossary_parts.append("【전역 용어사전】\n" + "\n".join(f"- {t}" for t in admin_terms))
    if user_terms:
        glossary_parts.append("【사용자 용어사전】\n" + "\n".join(f"- {t}" for t in user_terms))

    glossary_section = "\n\n".join(glossary_parts) if glossary_parts else "(용어사전 없음)"

    prompt = POSTPROCESS_PROMPT_TEMPLATE.format(
        title=title or "",
        meeting_date=meeting_date or "",
        location=location or "",
        attendees=attendees or "",
        agenda=agenda or "",
        notes=notes or "",
        glossary_section=glossary_section,
        transcript=transcript,
    )

    try:
        result = _call_llm(prompt, effective)
        return result.strip() if result.strip() else transcript
    except Exception:
        return transcript


def _call_llm(prompt: str, effective: dict | None) -> str:
    from app.config import settings
    from openai import OpenAI

    cfg = effective or {}
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
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
