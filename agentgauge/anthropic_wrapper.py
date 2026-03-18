from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Iterator, Optional

from .metrics import (
    LLM_ACTIVE_REQUESTS,
    LLM_CACHE_TOKENS_TOTAL,
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    LLM_TOOL_CALLS_TOTAL,
)

logger = logging.getLogger(__name__)

def _record_anthropic_cache_tokens(usage: Any, model: str) -> None:
    """Record Anthropic cache token metrics from a usage object.

    Handles both cache_creation_input_tokens and cache_read_input_tokens
    from Anthropic's usage response, with proper existence and type checking.

    Args:
        usage: The usage object from an Anthropic API response.
        model: The model name for metric labeling.
    """
    if hasattr(usage, "cache_creation_input_tokens"):
        cache_creation_tokens = usage.cache_creation_input_tokens
        if isinstance(cache_creation_tokens, int):
            LLM_CACHE_TOKENS_TOTAL.labels(model=model, cache_type="creation").inc(
                cache_creation_tokens
            )
    if hasattr(usage, "cache_read_input_tokens"):
        cache_read_tokens = usage.cache_read_input_tokens
        if isinstance(cache_read_tokens, int):
            LLM_CACHE_TOKENS_TOTAL.labels(model=model, cache_type="read").inc(
                cache_read_tokens
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

            _record_anthropic_cache_tokens(response.usage, model)

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


class InstrumentedAsyncMessages:
    """Proxy around an AsyncAnthropic "messages" resource.

    Delegates every call to the real resource while recording
    request count, latency, and token usage as Prometheus metrics.
    """

    def __init__(self, messages: Any) -> None:
        self._messages = messages

    async def create(self, **kwargs: Any) -> Any:
        """Async version of create for AsyncAnthropic clients."""
        model = kwargs.get("model", "unknown")
        start = time.monotonic()
        status = "ok"

        LLM_ACTIVE_REQUESTS.labels(model=model).inc()
        try:
            response = await self._messages.create(**kwargs)
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

            _record_anthropic_cache_tokens(response.usage, model)

        for tool_name in _extract_tool_calls_anthropic(response):
            LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()

        return response

    def stream(self, **kwargs: Any) -> "InstrumentedAsyncStream":
        """Async stream messages with duration and token tracking.

        Wraps the messages.stream() method to track request duration,
        token usage, and tool calls while streaming the response.
        Returns an async context manager that ensures proper cleanup.

        Args:
            **kwargs: Arguments passed to messages.stream()

        Returns:
            An InstrumentedAsyncStream context manager that yields stream events

        Usage:
            async with client.messages.stream(model="claude-3", messages=[]) as stream:
                async for event in stream:
                    print(event)
        """
        try:
            stream_cm = self._messages.stream(**kwargs)
        except Exception:
            # If stream creation fails, record error counter
            model = kwargs.get("model", "unknown")
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status="error").inc()
            raise
        return InstrumentedAsyncStream(stream_cm, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class InstrumentedStream:
    """Context manager wrapper for Anthropic message streams.

    Ensures proper resource cleanup and tracks metrics for streaming requests.
    """

    def __init__(self, stream_cm: Any, kwargs: dict[str, Any]) -> None:
        self._stream_cm = stream_cm
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

        # Exit the underlying context manager
        try:
            self._stream_cm.__exit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.exception(
                "Exception during stream cleanup for model %s: %s",
                self._model,
                e,
            )

        self._record_metrics()

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

                        _record_anthropic_cache_tokens(final_message.usage, self._model)

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

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying stream context manager.

        This exposes the full MessageStream API including:
        - get_final_text()
        - text_stream
        - accumulators
        - And any other MessageStream attributes
        """
        return getattr(self._stream_cm, name)


class InstrumentedAsyncStream:
    """Async context manager wrapper for Anthropic message streams.

    Ensures proper resource cleanup and tracks metrics for streaming requests.
    """

    def __init__(self, stream_cm: Any, kwargs: dict[str, Any]) -> None:
        self._stream_cm = stream_cm
        self._model = kwargs.get("model", "unknown")
        self._start: Optional[float] = None
        self._status = "ok"
        self._entered = False
        self._stream: Any = None  # the AsyncMessageStream returned by __aenter__

    async def __aenter__(self) -> "InstrumentedAsyncStream":
        """Enter the async stream context and start tracking metrics."""
        self._start = time.monotonic()
        LLM_ACTIVE_REQUESTS.labels(model=self._model).inc()
        self._entered = True

        try:
            # __aenter__ returns the real AsyncMessageStream, not the manager
            self._stream = await self._stream_cm.__aenter__()
            return self
        except Exception:
            self._status = "error"
            await self._record_metrics()
            raise

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the async stream context and record final metrics."""
        if exc_type is not None:
            self._status = "error"

        # Exit the underlying async context manager
        try:
            await self._stream_cm.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.exception(
                "Exception during async stream cleanup for model %s: %s",
                self._model,
                e,
            )

        await self._record_metrics()

        return False

    async def __aiter__(self) -> AsyncIterator[Any]:
        """Iterate over stream events asynchronously."""
        if not self._entered:
            raise RuntimeError("Async stream must be used as a context manager")

        try:
            async for event in self._stream:
                yield event
        except Exception:
            self._status = "error"
            raise

    async def _record_metrics(self) -> None:
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
                # Use the entered stream (AsyncMessageStream), not the manager
                stream = self._stream if self._stream is not None else self._stream_cm
                if hasattr(stream, "get_final_message"):
                    final_message = await stream.get_final_message()
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

                        _record_anthropic_cache_tokens(final_message.usage, self._model)

                    for tool_name in _extract_tool_calls_anthropic(final_message):
                        LLM_TOOL_CALLS_TOTAL.labels(
                            model=self._model, tool_name=tool_name
                        ).inc()
            except Exception:
                # If we can't get the final message, just skip token tracking
                pass

    async def get_final_message(self) -> Any:
        """Get the final message from the stream (delegates to the entered stream)."""
        stream = self._stream if self._stream is not None else self._stream_cm
        return await stream.get_final_message()

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the entered stream.

        This exposes the full MessageStream API including:
        - get_final_text()
        - text_stream
        - accumulators
        - And any other MessageStream attributes
        """
        stream = self._stream if self._stream is not None else self._stream_cm
        return getattr(stream, name)