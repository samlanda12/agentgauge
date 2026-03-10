from __future__ import annotations

import time
from typing import Any

from .metrics import (
    LLM_ACTIVE_REQUESTS,
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
)

class InstrumentedMessages:
    """Proxy around an Anthropic "messages" resource.

    Delegates every call to the real resource while recording
    request count, latency, and token usage as Prometheus metrics.
    """

    def __init__(self, messages: Any) -> None:
        self._messages = messages

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        start = time.monotonic()
        status = "ok"

        LLM_ACTIVE_REQUESTS.labels(model=model).inc()
        try:
            response = self._messages.create(**kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.monotonic() - start
            LLM_ACTIVE_REQUESTS.labels(model=model).dec()
            LLM_REQUESTS_TOTAL.labels(model=model, method="create", status=status).inc()
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method="create").observe(duration)

        if hasattr(response, "usage") and response.usage is not None:
            LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(
                response.usage.input_tokens
            )
            LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(
                response.usage.output_tokens
            )

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)
