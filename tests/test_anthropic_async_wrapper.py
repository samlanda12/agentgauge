"""Tests for the InstrumentedAsyncMessages wrapper."""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, AsyncMock

import pytest
from prometheus_client import REGISTRY

from agentgauge.anthropic_wrapper import InstrumentedAsyncMessages

MODEL = "claude-sonnet-4-5-20250929"


class FakeAsyncStream:
    """Mock async stream object that yields events and optionally provides final message."""

    def __setattr__(self, name, value):
        # Allow arbitrary attribute assignment for testing
        object.__setattr__(self, name, value)

    def __init__(self, events=None, final_message=None):
        self._events = events or []
        self._final_message = final_message
        self._entered = False

    def __aiter__(self):
        self._iter_events = list(self._events)
        return self

    async def __anext__(self):
        if not self._iter_events:
            raise StopAsyncIteration
        return self._iter_events.pop(0)

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def get_final_message(self):
        if self._final_message is None:
            raise RuntimeError("No final message available")
        return self._final_message


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 25


@dataclass
class FakeContentBlock:
    type: str
    name: Optional[str] = None


@dataclass
class FakeMessage:
    id: str = "msg_fake"
    model: str = MODEL
    usage: FakeUsage = field(default_factory=FakeUsage)
    content: Optional[List[FakeContentBlock]] = None


@pytest.fixture
def inner():
    mock = MagicMock()
    mock.create = AsyncMock(return_value=FakeMessage())
    return mock


@pytest.fixture
def wrapped(inner):
    return InstrumentedAsyncMessages(inner)


@pytest.fixture
def wrapped_with_error():
    mock = MagicMock()
    mock.create = AsyncMock(side_effect=RuntimeError("API down"))
    return InstrumentedAsyncMessages(mock)


def _sample(metric_name, **labels):
    return REGISTRY.get_sample_value(metric_name, labels)


class TestAsyncCreateMetrics:
    @pytest.mark.asyncio
    async def test_records_request_count(self, wrapped):
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 1.0

    @pytest.mark.asyncio
    async def test_records_duration(self, wrapped):
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    @pytest.mark.asyncio
    async def test_records_input_tokens(self, inner):
        inner.create.return_value = FakeMessage(usage=FakeUsage(input_tokens=200, output_tokens=50))
        wrapped = InstrumentedAsyncMessages(inner)
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 200.0

    @pytest.mark.asyncio
    async def test_records_output_tokens(self, inner):
        inner.create.return_value = FakeMessage(usage=FakeUsage(input_tokens=200, output_tokens=50))
        wrapped = InstrumentedAsyncMessages(inner)
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 50.0

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate(self, wrapped):
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 2.0


class TestAsyncCreateErrorHandling:
    @pytest.mark.asyncio
    async def test_error_is_reraised(self, wrapped_with_error):
        with pytest.raises(RuntimeError, match="API down"):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])

    @pytest.mark.asyncio
    async def test_error_records_error_status(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="error") == 1.0

    @pytest.mark.asyncio
    async def test_error_still_records_duration(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    @pytest.mark.asyncio
    async def test_active_requests_returns_to_zero_on_error(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            await wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_active_requests", model=MODEL) == 0.0


class TestAsyncCreateDelegation:
    @pytest.mark.asyncio
    async def test_passes_kwargs_to_inner_client(self, wrapped, inner):
        kwargs = {"model": MODEL, "max_tokens": 512, "messages": [{"role": "user", "content": "hi"}]}
        await wrapped.create(**kwargs)
        inner.create.assert_called_once_with(**kwargs)

    @pytest.mark.asyncio
    async def test_returns_original_response(self, inner):
        expected = FakeMessage(id="msg_123")
        inner.create.return_value = expected
        result = await InstrumentedAsyncMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert result.id == "msg_123"

    def test_proxies_other_attributes(self, wrapped, inner):
        inner.some_other_method.return_value = "hello"
        assert wrapped.some_other_method() == "hello"


class TestAsyncToolCallTracking:
    @pytest.mark.asyncio
    async def test_no_tool_calls_when_content_is_none(self, inner):
        inner.create.return_value = FakeMessage(content=None)
        await InstrumentedAsyncMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    @pytest.mark.asyncio
    async def test_no_tool_calls_when_content_has_no_tool_use_blocks(self, inner):
        inner.create.return_value = FakeMessage(
            content=[FakeContentBlock(type="text")]
        )
        await InstrumentedAsyncMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    @pytest.mark.asyncio
    async def test_single_tool_call_is_recorded(self, inner):
        inner.create.return_value = FakeMessage(
            content=[FakeContentBlock(type="tool_use", name="search_web")]
        )
        await InstrumentedAsyncMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0

    @pytest.mark.asyncio
    async def test_multiple_distinct_tool_calls_are_recorded(self, inner):
        inner.create.return_value = FakeMessage(
            content=[
                FakeContentBlock(type="tool_use", name="search_web"),
                FakeContentBlock(type="tool_use", name="read_file"),
            ]
        )
        await InstrumentedAsyncMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="read_file") == 1.0

    @pytest.mark.asyncio
    async def test_repeated_same_tool_accumulates(self, inner):
        inner.create.return_value = FakeMessage(
            content=[
                FakeContentBlock(type="tool_use", name="search_web"),
                FakeContentBlock(type="tool_use", name="search_web"),
            ]
        )
        await InstrumentedAsyncMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 2.0

    @pytest.mark.asyncio
    async def test_mixed_content_blocks_only_counts_tool_use(self, inner):
        inner.create.return_value = FakeMessage(
            content=[
                FakeContentBlock(type="text"),
                FakeContentBlock(type="tool_use", name="search_web"),
                FakeContentBlock(type="text"),
                FakeContentBlock(type="tool_use", name="read_file"),
            ]
        )
        await InstrumentedAsyncMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="read_file") == 1.0

    @pytest.mark.asyncio
    async def test_tool_calls_not_recorded_on_error(self, wrapped_with_error):
        inner = MagicMock()
        inner.create = AsyncMock(side_effect=RuntimeError("API down"))
        wrapped = InstrumentedAsyncMessages(inner)
        with pytest.raises(RuntimeError):
            await wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None


class TestAsyncStreamMetrics:
    @pytest.mark.asyncio
    async def test_records_stream_request_count(self, inner):
        final_message = FakeMessage(usage=FakeUsage(input_tokens=150, output_tokens=30))
        stream = FakeAsyncStream(events=["event1", "event2"], final_message=final_message)
        inner.stream.return_value = stream

        wrapped = InstrumentedAsyncMessages(inner)
        events = []
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                events.append(event)

        assert len(events) == 2
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 1.0

    @pytest.mark.asyncio
    async def test_records_stream_duration(self, inner):
        final_message = FakeMessage(usage=FakeUsage(input_tokens=150, output_tokens=30))
        stream = FakeAsyncStream(events=["event1", "event2"], final_message=final_message)
        inner.stream.return_value = stream

        wrapped = InstrumentedAsyncMessages(inner)
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                pass

        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") == 1.0

    @pytest.mark.asyncio
    async def test_yields_all_stream_events(self, inner):
        events = ["event1", "event2", "event3"]
        stream = FakeAsyncStream(events=events, final_message=FakeMessage())
        inner.stream.return_value = stream

        wrapped = InstrumentedAsyncMessages(inner)
        result = []
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                result.append(event)

        assert result == ["event1", "event2", "event3"]

    @pytest.mark.asyncio
    async def test_records_tokens_from_final_message(self, inner):
        final_message = FakeMessage(usage=FakeUsage(input_tokens=150, output_tokens=30))
        stream = FakeAsyncStream(events=["event1"], final_message=final_message)
        inner.stream.return_value = stream

        wrapped = InstrumentedAsyncMessages(inner)
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                pass

        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 150.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 30.0

    @pytest.mark.asyncio
    async def test_records_tool_calls_from_final_message(self, inner):
        final_message = FakeMessage(
            content=[
                FakeContentBlock(type="tool_use", name="search_web"),
                FakeContentBlock(type="tool_use", name="read_file"),
            ]
        )
        stream = FakeAsyncStream(events=["event1"], final_message=final_message)
        inner.stream.return_value = stream

        wrapped = InstrumentedAsyncMessages(inner)
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                pass

        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="search_web") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="read_file") == 1.0

    @pytest.mark.asyncio
    async def test_multiple_stream_calls_accumulate(self, inner):
        stream1 = FakeAsyncStream(events=["e1"], final_message=FakeMessage())
        stream2 = FakeAsyncStream(events=["e2"], final_message=FakeMessage())
        inner.stream.side_effect = [stream1, stream2]

        wrapped = InstrumentedAsyncMessages(inner)
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                pass
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                pass

        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 2.0


class TestAsyncStreamErrorHandling:
    def test_error_during_stream_initialization(self, inner):
        inner.stream.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedAsyncMessages(inner)

        with pytest.raises(RuntimeError, match="Stream init failed"):
            wrapped.stream(model=MODEL, max_tokens=1024, messages=[])

        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) is None

    @pytest.mark.asyncio
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

        inner.stream.return_value = FailingStream()
        wrapped = InstrumentedAsyncMessages(inner)

        with pytest.raises(RuntimeError, match="Stream error"):
            async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
                async for event in s:
                    pass

        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    @pytest.mark.asyncio
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

        inner.stream.return_value = FailingStream()
        wrapped = InstrumentedAsyncMessages(inner)

        with pytest.raises(RuntimeError):
            async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
                async for event in s:
                    pass

        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") is None

    @pytest.mark.asyncio
    async def test_abandoned_stream_releases_active_request_gauge(self, inner):
        stream = FakeAsyncStream(events=["event1"], final_message=FakeMessage())
        inner.stream.return_value = stream

        wrapped = InstrumentedAsyncMessages(inner)
        stream_cm = wrapped.stream(model=MODEL, max_tokens=1024, messages=[])

        # Never enter the context manager - simulating abandoned stream
        _ = stream_cm
        assert _sample("llm_active_requests", model=MODEL) is None


class TestAsyncStreamDelegation:
    @pytest.mark.asyncio
    async def test_passes_kwargs_to_inner_stream(self, inner):
        stream = FakeAsyncStream(events=[], final_message=FakeMessage())
        inner.stream.return_value = stream
        kwargs = {"model": MODEL, "max_tokens": 512, "messages": [{"role": "user", "content": "hi"}]}

        wrapped = InstrumentedAsyncMessages(inner)
        async with wrapped.stream(**kwargs) as s:
            async for event in s:
                pass

        inner.stream.assert_called_once_with(**kwargs)

    @pytest.mark.asyncio
    async def test_gracefully_handles_get_final_message_error(self, inner):
        class MinimalStream:
            def __init__(self):
                self.entered = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.entered:
                    raise StopAsyncIteration
                raise StopAsyncIteration

            async def get_final_message(self):
                raise RuntimeError("No final message")

        inner.stream.return_value = MinimalStream()
        wrapped = InstrumentedAsyncMessages(inner)

        # Should not raise, just skip token tracking
        async with wrapped.stream(model=MODEL, max_tokens=1024, messages=[]) as s:
            async for event in s:
                pass

        # Metrics should still be recorded
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 1.0

    def test_proxies_attributes_to_underlying_stream(self, inner):
        stream = FakeAsyncStream(events=[], final_message=FakeMessage())
        stream.custom_attribute = "custom_value"
        inner.stream.return_value = stream

        wrapped = InstrumentedAsyncMessages(inner)
        result = wrapped.stream(model=MODEL, max_tokens=1024, messages=[])

        # Should proxy attributes to the underlying stream
        assert hasattr(result, 'custom_attribute')
        assert result.custom_attribute == "custom_value"
