"""Tests for the InstrumentedAsyncChatCompletion wrapper."""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, AsyncMock

import pytest
from prometheus_client import REGISTRY

from agentgauge.openai_wrapper import InstrumentedAsyncChatCompletion

MODEL = "gpt-4"


@dataclass
class FakePromptTokensDetails:
    cached_tokens: int = 0


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 25
    prompt_tokens_details: Optional[FakePromptTokensDetails] = field(default_factory=FakePromptTokensDetails)


@dataclass
class FakeChunk:
    """A stream chunk, optionally carrying usage (mirrors the real OpenAI final usage chunk)."""
    usage: Optional[FakeUsage] = None
    choices: Optional[list] = None


class FakeAsyncStream:
    """Mock async stream that yields chunks and emits a final usage chunk when usage is given.

    This mirrors the real OpenAI behaviour when stream_options={"include_usage": True}:
    the API appends a final chunk with usage before the [DONE] sentinel.
    """

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __init__(self, chunks=None, usage=None, final_response=None):
        content_chunks = [FakeChunk() if not isinstance(c, FakeChunk) else c for c in (chunks or [])]
        # Append the usage chunk last, as the real API does
        if usage is not None:
            content_chunks.append(FakeChunk(usage=usage))
        self._chunks = content_chunks
        self._iter_chunks = []
        self._final_response = final_response
        # Expose choices on the stream for tool-call tracking
        self.choices = final_response.choices if final_response else None
        self._entered = False

    def __aiter__(self):
        self._iter_chunks = list(self._chunks)
        return self

    async def __anext__(self):
        if not self._iter_chunks:
            raise StopAsyncIteration
        return self._iter_chunks.pop(0)

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def close(self):
        pass


@dataclass
class FakeFunction:
    name: str = "search_web"


@dataclass
class FakeToolCall:
    function: FakeFunction = field(default_factory=FakeFunction)


@dataclass
class FakeMessage:
    tool_calls: Optional[List[FakeToolCall]] = None


@dataclass
class FakeChoice:
    message: FakeMessage = field(default_factory=FakeMessage)


@dataclass
class FakeChatCompletion:
    id: str = "chatcmpl_fake"
    model: str = MODEL
    usage: FakeUsage = field(default_factory=FakeUsage)
    choices: List[FakeChoice] = field(default_factory=lambda: [FakeChoice()])


@pytest.fixture
def inner():
    mock = MagicMock()
    mock.create = AsyncMock(return_value=FakeChatCompletion())
    return mock


@pytest.fixture
def wrapped(inner):
    return InstrumentedAsyncChatCompletion(inner)


@pytest.fixture
def wrapped_with_error():
    mock = MagicMock()
    mock.create = AsyncMock(side_effect=RuntimeError("API down"))
    return InstrumentedAsyncChatCompletion(mock)


def _sample(metric_name, **labels):
    return REGISTRY.get_sample_value(metric_name, labels)


class TestAsyncCreateMetrics:
    async def test_records_request_count(self, wrapped):
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 1.0

    async def test_records_duration(self, wrapped):
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    async def test_records_prompt_tokens(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            usage=FakeUsage(prompt_tokens=200, completion_tokens=50)
        ))
        wrapped = InstrumentedAsyncChatCompletion(inner)
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 200.0

    async def test_records_completion_tokens(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            usage=FakeUsage(prompt_tokens=200, completion_tokens=50)
        ))
        wrapped = InstrumentedAsyncChatCompletion(inner)
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 50.0

    async def test_multiple_calls_accumulate(self, wrapped):
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 2.0

    async def test_records_cached_tokens(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            usage=FakeUsage(
                prompt_tokens=200,
                completion_tokens=50,
                prompt_tokens_details=FakePromptTokensDetails(cached_tokens=150)
            )
        ))
        wrapped = InstrumentedAsyncChatCompletion(inner)
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_cache_tokens_total", model=MODEL, cache_type="read") == 150.0

    async def test_cached_tokens_zero_recorded_as_zero(self, inner):
        """When cached_tokens is 0, the metric IS recorded with value 0 (via .inc(0))."""
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            usage=FakeUsage(prompt_tokens=200, completion_tokens=50)
        ))
        wrapped = InstrumentedAsyncChatCompletion(inner)
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        # .inc(0) still creates the label combination and sets it to 0
        assert _sample("llm_cache_tokens_total", model=MODEL, cache_type="read") == 0.0

    async def test_cached_tokens_metric_not_recorded_when_prompt_tokens_details_missing(self, inner):
        """When prompt_tokens_details is None, the metric is NOT recorded at all."""
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            usage=FakeUsage(prompt_tokens=200, completion_tokens=50, prompt_tokens_details=None)
        ))
        wrapped = InstrumentedAsyncChatCompletion(inner)
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        # When prompt_tokens_details is None, the metric should not exist (returns None)
        assert _sample("llm_cache_tokens_total", model=MODEL, cache_type="read") is None


class TestAsyncCreateErrorHandling:
    async def test_error_is_reraised(self, wrapped_with_error):
        with pytest.raises(RuntimeError, match="API down"):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])

    async def test_error_records_error_status(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="error") == 1.0

    async def test_error_still_records_duration(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    async def test_active_requests_returns_to_zero_on_error(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_active_requests", model=MODEL) == 0.0


class TestAsyncCreateDelegation:
    async def test_passes_kwargs_to_inner_client(self, wrapped, inner):
        kwargs = {
            "model": MODEL,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hi"}],
        }
        await wrapped.create(**kwargs)
        inner.create.assert_called_once_with(**kwargs)

    async def test_returns_original_response(self, inner):
        expected = FakeChatCompletion(id="chatcmpl_123")
        inner.create = AsyncMock(return_value=expected)
        result = await InstrumentedAsyncChatCompletion(inner).create(
            model=MODEL, max_tokens=1024, messages=[]
        )
        assert result.id == "chatcmpl_123"

    def test_proxies_other_attributes(self, wrapped, inner):
        inner.some_other_method.return_value = "hello"
        assert wrapped.some_other_method() == "hello"


class TestAsyncToolCallTracking:
    async def test_no_tool_calls_when_choices_is_empty(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(choices=[]))
        await InstrumentedAsyncChatCompletion(inner).create(
            model=MODEL, max_tokens=1024, messages=[]
        )
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    async def test_no_tool_calls_when_message_has_no_tool_calls(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(tool_calls=None))]
        ))
        await InstrumentedAsyncChatCompletion(inner).create(
            model=MODEL, max_tokens=1024, messages=[]
        )
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    async def test_single_tool_call_is_recorded(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="search_web"))]
                    )
                )
            ]
        ))
        await InstrumentedAsyncChatCompletion(inner).create(
            model=MODEL, max_tokens=1024, messages=[]
        )
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0

    async def test_multiple_distinct_tool_calls_are_recorded(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(function=FakeFunction(name="search_web")),
                            FakeToolCall(function=FakeFunction(name="read_file")),
                        ]
                    )
                )
            ]
        ))
        await InstrumentedAsyncChatCompletion(inner).create(
            model=MODEL, max_tokens=1024, messages=[]
        )
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="read_file") == 1.0

    async def test_repeated_same_tool_accumulates(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(function=FakeFunction(name="search_web")),
                            FakeToolCall(function=FakeFunction(name="search_web")),
                        ]
                    )
                )
            ]
        ))
        await InstrumentedAsyncChatCompletion(inner).create(
            model=MODEL, max_tokens=1024, messages=[]
        )
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 2.0

    async def test_multiple_choices_with_tool_calls(self, inner):
        inner.create = AsyncMock(return_value=FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="search_web"))]
                    )
                ),
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="read_file"))]
                    )
                ),
            ]
        ))
        await InstrumentedAsyncChatCompletion(inner).create(
            model=MODEL, max_tokens=1024, messages=[]
        )
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="read_file") == 1.0

    async def test_tool_calls_not_recorded_on_error(self, wrapped_with_error):
        inner = MagicMock()
        inner.create = AsyncMock(side_effect=RuntimeError("API down"))
        wrapped = InstrumentedAsyncChatCompletion(inner)
        with pytest.raises(RuntimeError):
            await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None


class TestAsyncStreamMetrics:
    async def test_records_stream_request_count(self):
        stream = FakeAsyncStream(chunks=["chunk1", "chunk2"])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        wrapped = InstrumentedAsyncChatCompletion(inner)
        chunks = []
        stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with stream as s:
            async for chunk in s:
                chunks.append(chunk)

        assert len(chunks) == 2
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 1.0

    async def test_records_stream_duration(self):
        stream = FakeAsyncStream(chunks=["chunk1", "chunk2"])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        wrapped = InstrumentedAsyncChatCompletion(inner)
        stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with stream as s:
            async for chunk in s:
                pass

        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") == 1.0

    async def test_yields_all_stream_chunks(self):
        c1, c2, c3 = FakeChunk(), FakeChunk(), FakeChunk()
        stream = FakeAsyncStream(chunks=[c1, c2, c3])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        wrapped = InstrumentedAsyncChatCompletion(inner)
        result = []
        stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with stream as s:
            async for chunk in s:
                result.append(chunk)

        assert result == [c1, c2, c3]

    async def test_records_tokens_from_final_response(self, inner):
        stream = FakeAsyncStream(
            chunks=["chunk1"],
            usage=FakeUsage(prompt_tokens=150, completion_tokens=30),
        )
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        wrapped = InstrumentedAsyncChatCompletion(inner)
        stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with stream as s:
            async for chunk in s:
                pass

        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 150.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 30.0

    async def test_records_tool_calls_from_streaming_deltas(self, inner):
        """Test tool call tracking from streaming chunks with delta.tool_calls.

        This simulates how OpenAI actually streams tool calls - as incremental deltas
        across multiple chunks. This provider-agnostic approach works with vLLM,
        Ollama, and other OpenAI-compatible providers.
        """
        # Helper dataclasses for streaming delta format
        @dataclass
        class FakeDeltaFunction:
            name: Optional[str] = None

        @dataclass
        class FakeDeltaToolCall:
            index: int
            function: Optional[FakeDeltaFunction] = None

        @dataclass
        class FakeDelta:
            tool_calls: Optional[List[FakeDeltaToolCall]] = None

        @dataclass
        class FakeChoiceWithDelta:
            delta: FakeDelta

        # Streaming chunks with tool call deltas
        chunk1 = FakeChunk(
            choices=[FakeChoiceWithDelta(
                delta=FakeDelta(
                    tool_calls=[FakeDeltaToolCall(index=0, function=FakeDeltaFunction(name="search_web"))]
                )
            )]
        )
        chunk2 = FakeChunk(
            choices=[FakeChoiceWithDelta(
                delta=FakeDelta(
                    tool_calls=[FakeDeltaToolCall(index=1, function=FakeDeltaFunction(name="read_file"))]
                )
            )]
        )

        stream = FakeAsyncStream(chunks=[chunk1, chunk2])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        wrapped = InstrumentedAsyncChatCompletion(inner)
        stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with stream as s:
            async for chunk in s:
                pass

        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="read_file") == 1.0

    async def test_multiple_stream_calls_accumulate(self):
        stream1 = FakeAsyncStream(chunks=["c1"])
        stream2 = FakeAsyncStream(chunks=["c2"])
        inner = MagicMock()
        inner.create = AsyncMock(side_effect=[stream1, stream2])

        wrapped = InstrumentedAsyncChatCompletion(inner)
        # First stream call
        stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with stream as s:
            async for chunk in s:
                pass
        # Second stream call
        stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with stream as s:
            async for chunk in s:
                pass

        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 2.0


class TestAsyncStreamErrorHandling:
    async def test_error_during_stream_initialization(self):
        inner = MagicMock()
        inner.create.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedAsyncChatCompletion(inner)

        with pytest.raises(RuntimeError, match="Stream init failed"):
            await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])

        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) is None

    async def test_error_during_stream_iteration(self, inner):
        class FailingStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("Stream error")

            async def close(self):
                pass

        inner = MagicMock()
        inner.create = AsyncMock(return_value=FailingStream())
        wrapped = InstrumentedAsyncChatCompletion(inner)

        with pytest.raises(RuntimeError, match="Stream error"):
            stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
            async with stream as s:
                async for chunk in s:
                    pass

        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    async def test_tokens_not_recorded_on_error(self, inner):
        class FailingStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("Stream error")

            async def close(self):
                pass

        inner = MagicMock()
        inner.create = AsyncMock(return_value=FailingStream())
        wrapped = InstrumentedAsyncChatCompletion(inner)
        with pytest.raises(RuntimeError):
            stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
            async with stream as s:
                async for chunk in s:
                    pass

        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") is None

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_abandoned_stream_releases_active_request_gauge(self):
        stream = FakeAsyncStream(chunks=["chunk1"])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        wrapped = InstrumentedAsyncChatCompletion(inner)
        stream_coroutine = wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        # Never enter the context manager - simulating abandoned stream
        _ = stream_coroutine
        assert _sample("llm_active_requests", model=MODEL) is None


class TestAsyncStreamDelegation:
    async def test_passes_kwargs_to_inner_create(self):
        stream = FakeAsyncStream(chunks=[])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)
        kwargs = {
            "model": MODEL,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hi"}],
        }

        wrapped = InstrumentedAsyncChatCompletion(inner)
        result_stream = await wrapped.stream(**kwargs)
        async with result_stream as s:
            async for chunk in s:
                pass

        # Verify stream=True was added
        expected_kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        inner.create.assert_called_once_with(**expected_kwargs)

    async def test_merges_existing_stream_options(self):
        stream = FakeAsyncStream(chunks=[])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)
        kwargs = {
            "model": MODEL,
            "messages": [],
            "stream_options": {"custom_option": True},
        }

        wrapped = InstrumentedAsyncChatCompletion(inner)
        result_stream = await wrapped.stream(**kwargs)
        async with result_stream as s:
            async for chunk in s:
                pass

        call_kwargs = inner.create.call_args[1]

        # Ensure streaming is enforced
        assert call_kwargs.get("stream") is True

        # Ensure stream_options are merged correctly:
        # - original custom_option is preserved
        # - include_usage is injected
        stream_options = call_kwargs.get("stream_options") or {}
        assert stream_options.get("custom_option") is True
        assert stream_options.get("include_usage") is True


class TestAsyncStreamContextManagerDelegation:
    """Verify that InstrumentedAsyncOpenAIStream properly delegates __aenter__/__aexit__
    to the inner stream, mirroring the behaviour of the sync InstrumentedOpenAIStream."""

    async def test_aenter_is_called_on_inner_stream(self):
        stream = FakeAsyncStream(chunks=[])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        assert stream._entered is False

        wrapped = InstrumentedAsyncChatCompletion(inner)
        result_stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with result_stream:
            assert stream._entered is True

    async def test_aexit_is_called_on_inner_stream(self):
        aexit_args: list = []

        class TrackingStream(FakeAsyncStream):
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                aexit_args.append((exc_type, exc_val, exc_tb))
                return False

        stream = TrackingStream(chunks=[])
        inner = MagicMock()
        inner.create = AsyncMock(return_value=stream)

        wrapped = InstrumentedAsyncChatCompletion(inner)
        result_stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])
        async with result_stream:
            pass

        assert len(aexit_args) == 1
        assert aexit_args[0] == (None, None, None)

    async def test_aenter_error_records_error_status_and_reraises(self):
        class FailingEnterStream:
            async def __aenter__(self):
                raise RuntimeError("enter failed")

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def close(self):
                pass

        inner = MagicMock()
        inner.create = AsyncMock(return_value=FailingEnterStream())

        wrapped = InstrumentedAsyncChatCompletion(inner)
        result_stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])

        with pytest.raises(RuntimeError, match="enter failed"):
            async with result_stream:
                pass

        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0

    async def test_aenter_error_decrements_active_requests(self):
        class FailingEnterStream:
            async def __aenter__(self):
                raise RuntimeError("enter failed")

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            async def close(self):
                pass

        inner = MagicMock()
        inner.create = AsyncMock(return_value=FailingEnterStream())

        wrapped = InstrumentedAsyncChatCompletion(inner)
        result_stream = await wrapped.stream(model=MODEL, max_tokens=1024, messages=[])

        with pytest.raises(RuntimeError):
            async with result_stream:
                pass

        assert _sample("llm_active_requests", model=MODEL) == 0.0
