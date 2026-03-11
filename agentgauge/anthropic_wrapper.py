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

    def stream(self, **kwargs: Any) -> "InstrumentedStream":
        """Stream messages with duration and token tracking.

        Wraps the messages.stream() method to track request duration,
        token usage, and tool calls while streaming the response.
        Returns a context manager that ensures proper cleanup.

        Args:
            **kwargs: Arguments passed to messages.stream()

        Returns:
            An InstrumentedStream context manager that yields stream events

        Usage:
            with client.messages.stream(model="claude-3", messages=[]) as stream:
                for event in stream:
                    print(event)
        """
        try:
            stream_cm = self._messages.stream(**kwargs)
        except Exception:
            # If stream creation fails, record error counter
            model = kwargs.get("model", "unknown")
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status="error").inc()
            raise
        return InstrumentedStream(stream_cm, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class InstrumentedStream:
    """Context manager wrapper for Anthropic message streams.

    Ensures proper resource cleanup and tracks metrics for streaming requests.
    """

    def __init__(self, stream_cm: Any, kwargs: dict[str, Any]) -> None:
        self._stream_cm = stream_cm
        self._kwargs = kwargs
        self._model = kwargs.get("model", "unknown")
        self._start: Optional[float] = None
        self._status = "ok"
        self._entered = False

    def __enter__(self) -> "InstrumentedStream":
        """Enter the stream context and start tracking metrics."""
        self._start = time.monotonic()
        LLM_ACTIVE_REQUESTS.labels(model=self._model).inc()
        self._entered = True

        try:
            self._stream_cm.__enter__()
            return self
        except Exception:
            self._status = "error"
            self._record_metrics()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the stream context and record final metrics."""
        if exc_type is not None:
            self._status = "error"

        self._record_metrics()

        # Exit the underlying context manager
        try:
            self._stream_cm.__exit__(exc_type, exc_val, exc_tb)
        except Exception:
            pass

        return False

    def __iter__(self) -> Iterator[Any]:
        """Iterate over stream events."""
        if not self._entered:
            raise RuntimeError("Stream must be used as a context manager")

        try:
            for event in self._stream_cm:
                yield event
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

        # Try to extract final message for token usage and tool calls
        if self._status == "ok":
            try:
                if hasattr(self._stream_cm, "get_final_message"):
                    final_message = self._stream_cm.get_final_message()
                    if (
                        hasattr(final_message, "usage")
                        and final_message.usage is not None
                    ):
                        input_tokens = final_message.usage.input_tokens
                        output_tokens = final_message.usage.output_tokens

                        LLM_TOKENS_TOTAL.labels(
                            model=self._model, token_type="input"
                        ).inc(input_tokens)
                        LLM_TOKENS_TOTAL.labels(
                            model=self._model, token_type="output"
                        ).inc(output_tokens)

                    for tool_name in _extract_tool_calls_anthropic(final_message):
                        LLM_TOOL_CALLS_TOTAL.labels(
                            model=self._model, tool_name=tool_name
                        ).inc()
            except Exception:
                # If we can't get the final message, just skip token tracking
                pass

    def get_final_message(self) -> Any:
        """Get the final message from the stream (delegates to underlying stream)."""
        return self._stream_cm.get_final_message()
