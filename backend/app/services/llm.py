import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 기업 회의의 공식 회의록을 작성하는 전문 서기입니다.
회의에 참석하지 않은 사람이 회의록만 읽고도 논의의 흐름, 각자의 입장, 결정과 그 근거, 후속 업무를 \
완전히 파악할 수 있도록 작성하는 것이 목표입니다.

# 작성 원칙
- 요약하지 말고 기록하세요. 분량을 줄이는 것보다 빠짐없이 담는 것이 우선입니다.
- 스크립트에 나온 수치, 날짜, 금액, 고유명사(제품·시스템·조직명), 예시는 원문 그대로 보존하세요.
- 누가 어떤 의견을 냈는지 발언자를 밝혀 적고, 반대 의견·우려·대안도 생략하지 마세요.
- 스크립트에 근거가 없는 내용은 추측해서 쓰지 마세요. 불명확한 정보는 "미정" 또는 "확인 필요"로 표기하세요.
- 음성 인식 오류로 보이는 단어는 회의 정보(제목, 참석자, 안건)와 문맥을 참고해 바로잡되, 의미를 바꾸지 마세요.

# 작업 절차
반드시 아래 두 단계를 순서대로 수행하고, 각 단계의 결과를 지정된 태그 안에 출력하세요.

## 1단계: 분석 — <analysis> 태그 안에 작성
회의록을 쓰기 전에 스크립트 전체를 처음부터 끝까지 검토하고 다음을 정리하세요.
1. 논의 주제 목록: 스크립트에 등장한 주제를 시간 순서대로 빠짐없이 나열하고, 각 주제가 어느 안건에 해당하는지 연결하세요. 안건에 없던 주제도 포함하세요.
2. 주제별 핵심 사실: 각 주제에서 나온 발언자별 주장, 근거, 수치, 반론을 메모하세요.
3. 결정 사항: 확정된 것과 보류·미결된 것을 구분하세요. 논의 중 의견이 바뀐 경우 최종 결론을 기준으로 하세요.
4. Action Item 후보: 해야 할 일, 담당자(참석자 명단과 대조), 기한, 관련 조건을 정리하세요.
5. 누락 점검: 위 목록에 빠진 발언이나 주제가 없는지 스크립트를 다시 확인하세요.

## 2단계: 회의록 작성 — <minutes> 태그 안에 작성
1단계 분석을 바탕으로 아래 형식에 맞춰 최종 회의록을 작성하세요. 1단계에서 정리한 주제는 모두 회의록에 반영되어야 합니다.

<minutes>
## 1. 회의 개요
- 회의명 / 일시 / 장소 / 참석자 / 안건

## 2. 안건별 논의 내용
### 2-1. (주제명)
- **배경**: 이 주제가 논의된 이유와 현재 상황
- **논의 내용**: 발언자별 주요 의견, 근거, 수치, 반대 의견을 논의 흐름에 따라 서술
- **결정 사항**: 결론과 그 근거 (결정되지 않았다면 "결정 보류"와 사유)
- **추가 검토 사항**: 남은 쟁점, 확인이 필요한 사항

(주제 수만큼 2-2, 2-3 ... 반복)

## 3. 주요 결정 사항
- 확정된 결정을 번호 목록으로 정리

## 4. Action Item
|Action Item|담당자|일정|비고|
|------|-----|-----|-----|
</minutes>

# 작성 형식
- 일반 텍스트(plain markdown)로만 작성하고, LaTeX/수식 표기(예: $...$, \\rightarrow)는 사용하지 마세요.
- 화살표는 반드시 유니코드 문자 → 를 그대로 사용하세요.
- <analysis>와 <minutes> 태그 밖에는 아무것도 쓰지 마세요.
"""

USER_PROMPT_TEMPLATE = """\
[회의 정보]
회의 제목: {title}
일시: {meeting_date}
장소: {location}
참석자: {attendees}
회의 안건: {agenda}

[회의록 스크립트]
<transcript>
{transcript}
</transcript>

위 스크립트로 작업 절차에 따라 1단계 분석(<analysis>)을 먼저 수행한 뒤, 2단계 최종 회의록(<minutes>)을 작성해주세요.
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANALYSIS_RE = re.compile(r"<analysis>.*?(?:</analysis>|(?=<minutes>))", re.DOTALL)
_MINUTES_RE = re.compile(r"<minutes>(.*?)(?:</minutes>|$)", re.DOTALL)


def _extract_minutes(text: str) -> str:
    """Return only the <minutes> section, dropping the model's analysis/reasoning."""
    text = _THINK_RE.sub("", text)
    match = _MINUTES_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback when the model skipped the tags: drop any analysis block and keep the rest.
    return _ANALYSIS_RE.sub("", text).strip()


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

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=title,
        meeting_date=meeting_date,
        location=location,
        attendees=attendees,
        agenda=agenda,
        transcript=transcript,
    )
    raw = _call_openai(user_prompt, cfg, system_prompt=SYSTEM_PROMPT)
    minutes = _extract_minutes(raw)

    return {"summary": _sanitize_latex_arrows(minutes), "decisions": [], "action_items": []}


_LATEX_ARROW_MAP = {
    r"\rightarrow": "→",
    r"\Rightarrow": "⇒",
    r"\leftarrow": "←",
    r"\Leftarrow": "⇐",
    r"\leftrightarrow": "↔",
    r"\Leftrightarrow": "⇔",
}

# Matches a LaTeX math span like "$...\rightarrow...$" and unwraps it,
# since the frontend's markdown renderer doesn't evaluate LaTeX.
_LATEX_MATH_SPAN_RE = re.compile(r"\$([^$\n]*(?:\\\w+)[^$\n]*)\$")


def _sanitize_latex_arrows(text: str) -> str:
    def _unwrap(match: re.Match) -> str:
        inner = match.group(1)
        for latex, unicode_char in _LATEX_ARROW_MAP.items():
            inner = inner.replace(latex, unicode_char)
        return inner

    text = _LATEX_MATH_SPAN_RE.sub(_unwrap, text)
    for latex, unicode_char in _LATEX_ARROW_MAP.items():
        text = text.replace(latex, unicode_char)
    return text


def _call_openai(prompt: str, cfg: dict, system_prompt: str | None = None) -> str:
    from openai import OpenAI
    base_url = cfg.get("LLM_API_BASE_URL") or settings.LLM_API_BASE_URL
    api_key = cfg.get("LLM_API_KEY") or "none"
    model = cfg.get("LLM_MODEL") or settings.LLM_MODEL
    timeout = int(cfg.get("LLM_TIMEOUT") or settings.LLM_TIMEOUT)
    max_tokens = int(cfg.get("LLM_MAX_TOKENS") or settings.LLM_MAX_TOKENS)

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # low temperature keeps the minutes faithful to the transcript
        max_tokens=max_tokens,
    )
    if response.choices[0].finish_reason == "length":
        logger.warning("LLM output truncated at max_tokens=%s; consider raising LLM_MAX_TOKENS", max_tokens)
    return response.choices[0].message.content or ""
