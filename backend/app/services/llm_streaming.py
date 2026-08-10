"""OpenAI 호환 LLM 스트리밍 호출 + 반복(hallucination) 감지/재생성 공용 헬퍼.

self-hosted LLM 서버(vLLM 등)는 종종 동일한 토큰/문구를 max_tokens까지 계속
반복 출력하는 degenerate loop(hallucination) 현상을 보인다. repetition_penalty를
서버에 고정해둬도 특정 입력에서는 여전히 재발할 수 있으므로, 클라이언트 쪽에서도
아래와 같은 안전장치를 둔다.

1. 응답을 스트리밍(stream=True)으로 받아 토큰이 도착할 때마다 검사한다.
2. 누적된 텍스트의 "끝부분"에서 같은 문자열 단위(1글자 ~ 짧은 문구)가
   연속으로 반복되는지 확인한다. 단위가 짧을수록(예: 글자 1개) 우연히
   반복될 수 있으므로 더 많은 반복을, 단위가 길수록(문구 단위)는 적은
   반복만으로도 반복으로 판단한다. 단일 토큰 반복뿐 아니라 "감사합니다.
   감사합니다. 감사합니다." 같은 짧은 문구 단위 반복도 이 방식으로 잡힌다.
3. 반복이 감지되면 응답을 끝까지 기다리지 않고(= max_tokens까지 낭비하지
   않고) 즉시 스트림을 끊은 뒤 처음부터 다시 생성을 요청한다(`max_retries`
   회까지). 재시도할 때마다
     - repetition_penalty(vLLM extra_body)를 점진적으로 올리고
     - temperature를 소폭 올려 결정적인 반복 경로에서 벗어날 확률을 높인다.
4. 재시도를 모두 소진했는데도 계속 반복되면, 각 시도에서 "반복이 시작되기
   직전까지" 확보한 텍스트 중 가장 긴 것을 대체 결과로 사용한다. 그마저도
   없으면 예외를 발생시켜 호출자가 처리(재시도/실패 처리/원문 fallback)하게
   한다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 반복 검사용 꼬리 버퍼는 이 길이만 유지한다 (문구 단위 반복을 판단하기에
# 충분하면서, 매 토큰마다 비교하는 비용을 상수로 묶어두기 위함).
_TAIL_BUFFER_CAP = 400


class LLMRepetitionError(RuntimeError):
    """모든 시도에서 반복(hallucination) 루프가 감지되어 사용할 텍스트를
    하나도 확보하지 못했을 때 발생한다."""


def _find_tail_repeat(tail: str, max_unit_chars: int, base_threshold: int) -> tuple[str, int] | None:
    """`tail` 문자열 끝에서 길이 1~max_unit_chars자인 단위가 연속으로
    반복되고 있는지 찾는다. 발견하면 (반복 단위, 반복 횟수)를, 없으면
    None을 반환한다.

    단위가 짧을수록(예: 글자 1개) 우연히 반복될 수 있으므로 더 많은 반복
    (base_threshold)을 요구하고, 단위가 길수록(문구 단위) 그보다 적은
    반복만으로도 이상 신호로 본다.
    """
    n = len(tail)
    max_unit = min(max_unit_chars, n // 2)
    for unit_len in range(1, max_unit + 1):
        unit = tail[n - unit_len:]
        if not unit.strip():
            continue  # 공백만 반복되는 것은 무시

        count = 1
        pos = n - unit_len
        while pos - unit_len >= 0 and tail[pos - unit_len:pos] == unit:
            count += 1
            pos -= unit_len

        threshold = max(3, base_threshold - unit_len + 1)
        if count > threshold:
            return unit, count
    return None


def stream_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_tokens: int,
    messages: list[dict],
    temperature: float = 0.7,
    repeat_max: int = 8,
    ngram_max_chars: int = 12,
    max_retries: int = 2,
    repetition_penalty: float = 0.0,
    log_prefix: str = "LLM",
) -> str:
    """스트리밍으로 chat completion을 호출하고 토큰/문구 반복(hallucination)을
    감지하면 재생성한다.

    Args:
        repeat_max: 한 글자 단위 반복을 몇 번까지 허용할지 (짧은 단위일수록
            엄격하게, 긴 문구 단위일수록 완화된 기준이 내부에서 자동 계산됨).
        ngram_max_chars: 반복 여부를 검사할 최대 단위 길이(글자 수).
        max_retries: 반복 감지 시 처음부터 다시 생성을 요청할 최대 횟수.
            (총 시도 횟수 = max_retries + 1)
        repetition_penalty: vLLM 등이 지원하는 repetition_penalty를
            extra_body로 전달한다 (1.0 이하면 전달하지 않음). 재시도마다
            점진적으로 올려서 같은 loop이 재발할 확률을 낮춘다.

    Returns:
        생성된 텍스트. 모든 시도가 반복으로 끝났다면, 각 시도에서 반복
        직전까지 확보한 텍스트 중 가장 긴 것을 반환한다.

    Raises:
        LLMRepetitionError: 어떤 시도에서도 사용 가능한 텍스트를 전혀
            확보하지 못한 경우.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    best_partial = ""
    last_error: Exception | None = None
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        # 재시도마다 repetition_penalty / temperature를 조금씩 올려서 같은
        # 반복 loop에 다시 빠질 확률을 낮춘다. (repetition_penalty를 서버에
        # 고정해둬도 재발하는 경우가 있어 클라이언트에서 한 번 더 강화한다.)
        attempt_temperature = min(temperature + 0.1 * (attempt - 1), 1.2)
        extra_body: dict = {}
        if repetition_penalty > 1.0:
            extra_body["repetition_penalty"] = min(repetition_penalty + 0.12 * (attempt - 1), 1.8)

        tokens: list[str] = []
        tail_buf = ""
        repetition_detected = False
        detected_unit = ""
        detected_count = 0
        stream = None

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=attempt_temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if not token:
                    continue

                tokens.append(token)
                tail_buf = (tail_buf + token)[-_TAIL_BUFFER_CAP:]

                hit = _find_tail_repeat(tail_buf, ngram_max_chars, repeat_max)
                if hit:
                    detected_unit, detected_count = hit
                    logger.warning(
                        "[%s] 반복 패턴 감지: %r 단위가 %d회 연속 반복 "
                        "(attempt %d/%d) - 스트림 중단 후 재생성",
                        log_prefix, detected_unit, detected_count, attempt, total_attempts,
                    )
                    repetition_detected = True
                    break

            if not repetition_detected:
                return "".join(tokens)

            # 반복이 시작되기 직전까지만 남기고(반복 단위는 1개만 유지) 후보로 저장
            full_text = "".join(tokens)
            trim_len = len(detected_unit) * (detected_count - 1)
            trimmed = full_text[: max(len(full_text) - trim_len, 0)]
            if len(trimmed) > len(best_partial):
                best_partial = trimmed
            last_error = LLMRepetitionError(
                f"{log_prefix}: {detected_unit!r} 단위가 {detected_count}회 연속 "
                f"반복되어 생성을 중단함 (attempt {attempt}/{total_attempts})"
            )

        except Exception as exc:  # 네트워크/타임아웃 등 - 다음 시도로 넘어감
            logger.warning(
                "[%s] 스트리밍 호출 실패 (attempt %d/%d): %s",
                log_prefix, attempt, total_attempts, exc,
            )
            last_error = exc
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    if best_partial:
        logger.warning(
            "[%s] %d회 시도 모두 반복으로 종료 - 반복 직전까지의 텍스트(%d자)로 대체",
            log_prefix, total_attempts, len(best_partial),
        )
        return best_partial

    raise last_error or LLMRepetitionError(f"{log_prefix}: 알 수 없는 오류로 생성 실패")
