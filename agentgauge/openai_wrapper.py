from __future__ import annotations

import logging
import time
import inspect
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

def _record_openai_cache_tokens(usage: Any, model: str) -> None:
    """Record OpenAI cache token metrics from a usage object.

    Handles cached_tokens from prompt_tokens_details in OpenAI's usage response,
    with proper existence, None-checking, and type checking.

    Args:
        usage: The usage object from an OpenAI API response.
        model: The model name for metric labeling.
    """
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details is not None:
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", None)
        if isinstance(cached_tokens, int):
            LLM_CACHE_TOKENS_TOTAL.labels(model=model, cache_type="read").inc(
                cached_tokens
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


class InstrumentedAsyncChatCompletion:
    """Proxy around an AsyncOpenAI or async OpenAI-compatible "chat.completions" resource.

    Delegates every call to the real resource while recording
    request count, latency, and token usage as Prometheus metrics.
    """

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    async def create(self, **kwargs: Any) -> Any:
        """Async version of create for AsyncOpenAI clients."""
        model = kwargs.get("model", "unknown")
        start = time.monotonic()
        status = "ok"

        LLM_ACTIVE_REQUESTS.labels(model=model).inc()
        try:
            response = await self._completions.create(**kwargs)
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

            _record_openai_cache_tokens(response.usage, model)

        for tool_name in _extract_tool_calls_openai(response):
            LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()

        return response

    async def stream(self, **kwargs: Any) -> "InstrumentedAsyncOpenAIStream":
        """Async stream chat completions with duration and token tracking.

        Wraps the chat.completions.create(stream=True) async method to track request duration,
        token usage, and tool calls while streaming the response.
        Returns an async context manager that ensures proper cleanup.

        Args:
            **kwargs: Arguments passed to chat.completions.create()
                     (stream=True is enforced and will override if set to False)
                     (stream_options={"include_usage": True} is automatically injected
                      for token tracking and will be merged with any existing stream_options)

        Returns:
            An InstrumentedAsyncOpenAIStream context manager that yields stream chunks

        Usage:
            stream = await client.chat.completions.stream(model="gpt-4", messages=[])
            async with stream as s:
                async for chunk in s:
                    print(chunk)
        """
        # Enforce stream=True
        kwargs["stream"] = True

        # Automatically inject stream_options to enable usage tracking
        existing_options = kwargs.get("stream_options")
        if existing_options is None or not isinstance(existing_options, dict):
            existing_options = {}
        kwargs["stream_options"] = {**existing_options, "include_usage": True}

        try:
            stream = await self._completions.create(**kwargs)
        except Exception:
            # If stream creation fails, only record error counter
            model = kwargs.get("model", "unknown")
            LLM_REQUESTS_TOTAL.labels(model=model, method="stream", status="error").inc()
            raise
        return InstrumentedAsyncOpenAIStream(stream, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


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

            _record_openai_cache_tokens(response.usage, model)

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
                     (stream_options={"include_usage": True} is automatically injected
                      for token tracking and will be merged with any existing stream_options)

        Returns:
            An InstrumentedOpenAIStream context manager that yields stream chunks

        Usage:
            with client.chat.completions.stream(model="gpt-4", messages=[]) as stream:
                for chunk in stream:
                    print(chunk)
        """
        # Enforce stream=True
        kwargs["stream"] = True

        # Automatically inject stream_options to enable usage tracking
        existing_options = kwargs.get("stream_options", {})
        kwargs["stream_options"] = {**existing_options, "include_usage": True}

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
        self._model = kwargs.get("model", "unknown")
        self._start: Optional[float] = None
        self._status = "ok"
        self._entered = False
        self._usage: Any = None  # captured from the final usage chunk during iteration

    def __enter__(self) -> "InstrumentedOpenAIStream":
        """Enter the stream context and start tracking metrics."""
        self._start = time.monotonic()
        LLM_ACTIVE_REQUESTS.labels(model=self._model).inc()
        self._entered = True
        try:
            entered_stream = self._stream.__enter__()
            if entered_stream is not None:
                self._stream = entered_stream
            return self
        except Exception:
            self._status = "error"
            self._record_metrics()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the stream context and record final metrics."""
        if exc_type is not None:
            self._status = "error"

        try:
            self._stream.__exit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.exception(
                "Exception during stream cleanup for model %s: %s",
                self._model,
                e,
            )

        # Fallback: close if available
        if hasattr(self._stream, "close"):
            try:
                self._stream.close()
            except Exception as e:
                logger.exception(
                    "Exception closing stream for model %s: %s",
                    self._model,
                    e,
                )

        self._record_metrics()

        return False

    def __iter__(self) -> Iterator[Any]:
        """Iterate over stream chunks."""
        if not self._entered:
            raise RuntimeError("Stream must be used as a context manager")

        try:
            for chunk in self._stream:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    self._usage = chunk.usage
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

        if self._status == "ok":
            try:
                if self._usage is not None:
                    LLM_TOKENS_TOTAL.labels(
                        model=self._model, token_type="input"
                    ).inc(self._usage.prompt_tokens)
                    LLM_TOKENS_TOTAL.labels(
                        model=self._model, token_type="output"
                    ).inc(self._usage.completion_tokens)

                    _record_openai_cache_tokens(self._usage, self._model)

                for tool_name in _extract_tool_calls_openai(self._stream):
                    LLM_TOOL_CALLS_TOTAL.labels(
                        model=self._model, tool_name=tool_name
                    ).inc()
            except Exception:
                logger.warning(
                    "Failed to record token/tool metrics for model %s",
                    self._model,
                )


class InstrumentedAsyncOpenAIStream:
    """Async context manager wrapper for OpenAI chat completion streams.

    Ensures proper resource cleanup and tracks metrics for streaming requests.
    """

    def __init__(self, stream: Any, kwargs: dict[str, Any]) -> None:
        self._stream = stream
        self._model = kwargs.get("model", "unknown")
        self._start: Optional[float] = None
        self._status = "ok"
        self._entered = False
        self._usage: Any = None  # captured from the final usage chunk during iteration

    async def __aenter__(self) -> "InstrumentedAsyncOpenAIStream":
        """Enter the async stream context and start tracking metrics."""
        self._start = time.monotonic()
        LLM_ACTIVE_REQUESTS.labels(model=self._model).inc()
        self._entered = True
        try:
            entered_stream = await self._stream.__aenter__()
            if entered_stream is not None:
                self._stream = entered_stream
            return self
        except Exception:
            self._status = "error"
            self._record_metrics()
            raise

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the async stream context and record final metrics."""
        if exc_type is not None:
            self._status = "error"

        try:
            await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.exception(
                "Exception during stream cleanup for model %s: %s",
                self._model,
                e,
            )

        if hasattr(self._stream, "close"):
            try:
                close_result = self._stream.close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as e:
                logger.exception(
                    "Exception closing stream for model %s: %s",
                    self._model,
                    e,
                )

        self._record_metrics()
        return False

    async def __aiter__(self) -> AsyncIterator[Any]:
        """Iterate over stream chunks asynchronously.

        Captures the final usage chunk emitted by the API when
        stream_options={"include_usage": True} is set.
        """
        if not self._entered:
            raise RuntimeError("Async stream must be used as a context manager")

        try:
            async for chunk in self._stream:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    self._usage = chunk.usage
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

        # Try to extract token usage and tool calls from chunks seen during iteration
        if self._status == "ok":
            try:
                if self._usage is not None:
                    LLM_TOKENS_TOTAL.labels(
                        model=self._model, token_type="input"
                    ).inc(self._usage.prompt_tokens)
                    LLM_TOKENS_TOTAL.labels(
                        model=self._model, token_type="output"
                    ).inc(self._usage.completion_tokens)

                    _record_openai_cache_tokens(self._usage, self._model)

                # Check for tool calls in the stream object
                for tool_name in _extract_tool_calls_openai(self._stream):
                    LLM_TOOL_CALLS_TOTAL.labels(
                        model=self._model, tool_name=tool_name
                    ).inc()
            except Exception:
                logger.warning(
                    "Failed to record token/tool metrics for model %s",
                    self._model,
                )

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying stream."""
        return getattr(self._stream, name)