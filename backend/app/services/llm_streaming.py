"""OpenAI 호환 LLM 스트리밍 호출 + 반복(hallucination) 감지/재생성 공용 헬퍼.

self-hosted LLM 서버(vLLM 등)는 종종 동일한 토큰을 max_tokens까지 계속
반복 출력하는 degenerate loop(hallucination) 현상을 보인다. 이를 완화하기
위해 다음과 같이 동작한다.

1. 응답을 스트리밍(stream=True)으로 받아 토큰이 도착할 때마다 검사한다.
2. 같은 토큰이 `repeat_max`회를 초과해 연속으로 나오면, 응답을 끝까지
   기다리지 않고 즉시 스트림을 끊는다(= max_tokens까지 낭비하지 않음).
3. 스트림을 끊은 뒤에는 처음부터 다시 생성을 요청한다(`max_retries`회까지).
   재시도할 때마다 frequency_penalty를 조금씩 올려 동일한 loop에 다시
   빠질 확률을 낮춘다.
4. 재시도를 모두 소진했는데도 계속 반복되면, 각 시도에서 "반복이 시작되기
   직전까지" 확보한 텍스트 중 가장 긴 것을 대체 결과로 사용한다. 그마저도
   없으면 예외를 발생시켜 호출자가 처리(재시도/실패 처리/원문 fallback)하게
   한다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LLMRepetitionError(RuntimeError):
    """모든 시도에서 반복(hallucination) 루프가 감지되어 사용할 텍스트를
    하나도 확보하지 못했을 때 발생한다."""


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
    max_retries: int = 2,
    log_prefix: str = "LLM",
) -> str:
    """스트리밍으로 chat completion을 호출하고 토큰 반복(hallucination)을
    감지하면 재생성한다.

    Args:
        repeat_max: 동일 토큰이 연속으로 나와도 되는 최대 횟수. 이를 초과하는
            (repeat_max + 1)번째 동일 토큰이 오면 반복으로 판단한다.
        max_retries: 반복 감지 시 처음부터 다시 생성을 요청할 최대 횟수.
            (총 시도 횟수 = max_retries + 1)

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
        # 재시도마다 frequency_penalty를 소폭 올려 같은 반복 loop에
        # 다시 빠질 확률을 낮춘다.
        frequency_penalty = min(0.4 * (attempt - 1), 1.6)

        tokens: list[str] = []
        last_token: str | None = None
        repeat_count = 0
        repetition_detected = False
        stream = None

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if not token:
                    continue

                tokens.append(token)

                if token == last_token:
                    repeat_count += 1
                else:
                    last_token = token
                    repeat_count = 1

                if repeat_count > repeat_max:
                    logger.warning(
                        "[%s] 동일 토큰 %r 이(가) %d회 연속 반복 감지 "
                        "(attempt %d/%d) - 스트림 중단 후 재생성",
                        log_prefix, token, repeat_count, attempt, total_attempts,
                    )
                    repetition_detected = True
                    break

            if not repetition_detected:
                return "".join(tokens)

            # 반복이 시작되기 직전까지만 남기고(반복 토큰은 1개만 유지) 후보로 저장
            keep = max(len(tokens) - repeat_count + 1, 0)
            trimmed = "".join(tokens[:keep])
            if len(trimmed) > len(best_partial):
                best_partial = trimmed
            last_error = LLMRepetitionError(
                f"{log_prefix}: 동일 토큰이 {repeat_count}회 연속 반복되어 "
                f"생성을 중단함 (attempt {attempt}/{total_attempts})"
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
