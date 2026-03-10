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

def _extract_tool_calls_openai(response: Any) -> list[str]:
    """Extract tool names from OpenAI or OpenAI-compatible API response.

    Args:
        response: The API response object.

    Returns:
        A list of tool names that were called.
    """
    tool_names = []
    if hasattr(response, "choices") and response.choices:
        for choice in response.choices:
            if hasattr(choice, "message") and choice.message:
                message = choice.message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, "function") and tool_call.function:
                            if hasattr(tool_call.function, "name"):
                                tool_names.append(tool_call.function.name)
    return tool_names


class InstrumentedChatCompletion:
    """Proxy around an OpenAI or OpenAI-compatible "chat.completions" resource.

    Delegates every call to the real resource while recording
    request count, latency, and token usage as Prometheus metrics.
    """

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        start = time.monotonic()
        status = "ok"

        LLM_ACTIVE_REQUESTS.labels(model=model).inc()
        try:
            response = self._completions.create(**kwargs)
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
                response.usage.prompt_tokens
            )
            LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(
                response.usage.completion_tokens
            )

        for tool_name in _extract_tool_calls_openai(response):
            LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()

        return response

    def stream(self, **kwargs: Any) -> Iterator[Any]:
        """Stream chat completions with duration and token tracking.
        
        Wraps the chat.completions.create(stream=True) method to track request duration,
        token usage, and tool calls while streaming the response.
        
        Args:
            **kwargs: Arguments passed to chat.completions.create(stream=True)
            
        Yields:
            Stream chunks from the underlying chat.completions.create() call
        """
        model = kwargs.get("model", "unknown")
        start = time.monotonic()
        status = "ok"

        LLM_ACTIVE_REQUESTS.labels(model=model).inc()
        
        try:
            stream = self._completions.create(**kwargs)
        except Exception:
            status = "error"
            duration = time.monotonic() - start
            LLM_ACTIVE_REQUESTS.labels(model=model).dec()
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status=status).inc()
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method="stream").observe(duration)
            raise

        try:
            for chunk in stream:
                yield chunk
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.monotonic() - start
            LLM_ACTIVE_REQUESTS.labels(model=model).dec()
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status=status).inc()
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method="stream").observe(duration)
            
            # Try to extract final response for token usage and tool calls
            try:
                if hasattr(stream, "get_final_response"):
                    final_response = stream.get_final_response()
                    if hasattr(final_response, "usage") and final_response.usage is not None:
                        LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(
                            final_response.usage.prompt_tokens
                        )
                        LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(
                            final_response.usage.completion_tokens
                        )
                    
                    for tool_name in _extract_tool_calls_openai(final_response):
                        LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()
            except Exception:
                # If we can't get the final response, just skip token tracking
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)
