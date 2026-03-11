from __future__ import annotations

import time
from typing import Any, Iterator

from .metrics import (
    LLM_ACTIVE_REQUESTS,
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    LLM_TOOL_CALLS_TOTAL,
)

def _extract_tool_calls_anthropic(response: Any) -> list[str]:
    """Extract tool names from response content blocks.

    Args:
        response: The API response object.

    Returns:
        A list of tool names that were called.
    """
    tool_names = []
    if hasattr(response, "content") and response.content is not None:
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                if hasattr(block, "name"):
                    tool_names.append(block.name)
    return tool_names


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
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(input_tokens)
            LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(output_tokens)

        for tool_name in _extract_tool_calls_anthropic(response):
            LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()

        return response

    def stream(self, **kwargs: Any) -> Iterator[Any]:
        """Stream messages with duration and token tracking.

        Wraps the messages.stream() method to track request duration,
        token usage, and tool calls while streaming the response.

        Args:
            **kwargs: Arguments passed to messages.stream()

        Yields:
            Stream events from the underlying messages.stream() call
        """
        model = kwargs.get("model", "unknown")
        start = time.monotonic()
        status = "ok"

        LLM_ACTIVE_REQUESTS.labels(model=model).inc()

        try:
            stream = self._messages.stream(**kwargs)
        except Exception:
            status = "error"
            duration = time.monotonic() - start
            LLM_ACTIVE_REQUESTS.labels(model=model).dec()
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status=status).inc()
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method="stream").observe(duration)
            raise

        try:
            for event in stream:
                yield event
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.monotonic() - start
            LLM_ACTIVE_REQUESTS.labels(model=model).dec()
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status=status).inc()
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method="stream").observe(duration)

            # Try to extract final message for token usage and tool calls
            try:
                if hasattr(stream, "get_final_message"):
                    final_message = stream.get_final_message()
                    if hasattr(final_message, "usage") and final_message.usage is not None:
                        input_tokens = final_message.usage.input_tokens
                        output_tokens = final_message.usage.output_tokens

                        LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(input_tokens)
                        LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(output_tokens)

                    for tool_name in _extract_tool_calls_anthropic(final_message):
                        LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()
            except Exception:
                # If we can't get the final message, just skip token tracking
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)
