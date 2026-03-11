from __future__ import annotations

import time
from typing import Any, Iterator, Optional

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
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(input_tokens)
            LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(output_tokens)

        for tool_name in _extract_tool_calls_openai(response):
            LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()

        return response

    def stream(self, **kwargs: Any) -> "InstrumentedOpenAIStream":
        """Stream chat completions with duration and token tracking.

        Wraps the chat.completions.create(stream=True) method to track request duration,
        token usage, and tool calls while streaming the response.
        Returns a context manager that ensures proper cleanup.

        Args:
            **kwargs: Arguments passed to chat.completions.create()
                     (stream=True is enforced and will override if set to False)

        Returns:
            An InstrumentedOpenAIStream context manager that yields stream chunks

        Usage:
            with client.chat.completions.stream(model="gpt-4", messages=[]) as stream:
                for chunk in stream:
                    print(chunk)
        """
        # Enforce stream=True
        kwargs["stream"] = True

        try:
            stream = self._completions.create(**kwargs)
        except Exception:
            # If stream creation fails, only record error counter
            model = kwargs.get("model", "unknown")
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status="error").inc()
            raise
        return InstrumentedOpenAIStream(stream, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class InstrumentedOpenAIStream:
    """Context manager wrapper for OpenAI chat completion streams.

    Ensures proper resource cleanup and tracks metrics for streaming requests.
    """

    def __init__(self, stream: Any, kwargs: dict[str, Any]) -> None:
        self._stream = stream
        self._kwargs = kwargs
        self._model = kwargs.get("model", "unknown")
        self._start: Optional[float] = None
        self._status = "ok"
        self._entered = False

    def __enter__(self) -> "InstrumentedOpenAIStream":
        """Enter the stream context and start tracking metrics."""
        self._start = time.monotonic()
        LLM_ACTIVE_REQUESTS.labels(model=self._model).inc()
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the stream context and record final metrics."""
        if exc_type is not None:
            self._status = "error"

        self._record_metrics()
        return False

    def __iter__(self) -> Iterator[Any]:
        """Iterate over stream chunks."""
        if not self._entered:
            raise RuntimeError("Stream must be used as a context manager")

        try:
            for chunk in self._stream:
                yield chunk
        except Exception:
            self._status = "error"
            raise

    def _record_metrics(self) -> None:
        """Record all metrics for this stream request."""
        if self._start is None:
            return

        duration = time.monotonic() - self._start
        LLM_ACTIVE_REQUESTS.labels(model=self._model).dec()
        LLM_REQUESTS_TOTAL.labels(
            model=self._model, method="stream", status=self._status
        ).inc()
        LLM_REQUEST_DURATION_SECONDS.labels(
            model=self._model, method="stream"
        ).observe(duration)

        # Try to extract token usage and tool calls from the stream object
        if self._status == "ok":
            try:
                # OpenAI streams may include usage in the stream object with stream_options
                if hasattr(self._stream, "usage") and self._stream.usage is not None:
                    input_tokens = self._stream.usage.prompt_tokens
                    output_tokens = self._stream.usage.completion_tokens

                    LLM_TOKENS_TOTAL.labels(
                        model=self._model, token_type="input"
                    ).inc(input_tokens)
                    LLM_TOKENS_TOTAL.labels(
                        model=self._model, token_type="output"
                    ).inc(output_tokens)

                # Check for tool calls in the stream object
                for tool_name in _extract_tool_calls_openai(self._stream):
                    LLM_TOOL_CALLS_TOTAL.labels(
                        model=self._model, tool_name=tool_name
                    ).inc()
            except Exception:
                # If we can't get token usage or tool calls, just skip it
                pass
