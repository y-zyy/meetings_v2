"""Shared OpenAI-compatible client factory for LLM-calling services.

`_call_openai`/`_call_llm` used to construct a brand-new `OpenAI(...)` client
(and therefore a brand-new HTTP connection pool) on every single call. Since
admin-configured overrides mean the effective base_url/api_key/timeout can
change at runtime, the client can't just be a fixed module-level singleton —
instead it's cached per resolved config, so repeated calls with the same
settings reuse one client (and its connections) instead of paying setup cost
every time.
"""

from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=8)
def get_openai_client(base_url: str, api_key: str, timeout: int) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
