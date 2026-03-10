"""Tests for the InstrumentedMessages wrapper."""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

from agentgauge.anthropic_wrapper import InstrumentedMessages

MODEL = "claude-sonnet-4-5-20250929"


class FakeStream:
    """Mock stream object that yields events and optionally provides final message."""

    def __init__(self, events=None, final_message=None):
        self._events = events or []
        self._final_message = final_message

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
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
    mock.create.return_value = FakeMessage()
    return mock


@pytest.fixture
def wrapped(inner):
    return InstrumentedMessages(inner)


@pytest.fixture
def wrapped_with_error():
    mock = MagicMock()
    mock.create.side_effect = RuntimeError("API down")
    return InstrumentedMessages(mock)


def _sample(metric_name, **labels):
    return REGISTRY.get_sample_value(metric_name, labels)


class TestCreateMetrics:
    def test_records_request_count(self, wrapped):
        wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 1.0

    def test_records_duration(self, wrapped):
        wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    def test_records_input_tokens(self, inner):
        inner.create.return_value = FakeMessage(usage=FakeUsage(input_tokens=200, output_tokens=50))
        wrapped = InstrumentedMessages(inner)
        wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 200.0

    def test_records_output_tokens(self, inner):
        inner.create.return_value = FakeMessage(usage=FakeUsage(input_tokens=200, output_tokens=50))
        wrapped = InstrumentedMessages(inner)
        wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 50.0

    def test_multiple_calls_accumulate(self, wrapped):
        wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        wrapped.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 2.0


class TestCreateErrorHandling:
    def test_error_is_reraised(self, wrapped_with_error):
        with pytest.raises(RuntimeError, match="API down"):
            wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])

    def test_error_records_error_status(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="error") == 1.0

    def test_error_still_records_duration(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    def test_active_requests_returns_to_zero_on_error(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            wrapped_with_error.create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_active_requests", model=MODEL) == 0.0


class TestCreateDelegation:
    def test_passes_kwargs_to_inner_client(self, wrapped, inner):
        kwargs = {"model": MODEL, "max_tokens": 512, "messages": [{"role": "user", "content": "hi"}]}
        wrapped.create(**kwargs)
        inner.create.assert_called_once_with(**kwargs)

    def test_returns_original_response(self, inner):
        expected = FakeMessage(id="msg_123")
        inner.create.return_value = expected
        result = InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert result.id == "msg_123"

    def test_proxies_other_attributes(self, wrapped, inner):
        inner.some_other_method.return_value = "hello"
        assert wrapped.some_other_method() == "hello"


class TestToolCallTracking:
    def test_no_tool_calls_when_content_is_none(self, inner):
        inner.create.return_value = FakeMessage(content=None)
        InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    def test_no_tool_calls_when_content_has_no_tool_use_blocks(self, inner):
        inner.create.return_value = FakeMessage(
            content=[FakeContentBlock(type="text")]
        )
        InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    def test_single_tool_call_is_recorded(self, inner):
        inner.create.return_value = FakeMessage(
            content=[FakeContentBlock(type="tool_use", name="web_search")]
        )
        InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0

    def test_multiple_distinct_tool_calls_are_recorded(self, inner):
        inner.create.return_value = FakeMessage(
            content=[
                FakeContentBlock(type="tool_use", name="web_search"),
                FakeContentBlock(type="tool_use", name="calculator"),
            ]
        )
        InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="calculator") == 1.0

    def test_repeated_same_tool_accumulates(self, inner):
        inner.create.return_value = FakeMessage(
            content=[
                FakeContentBlock(type="tool_use", name="web_search"),
                FakeContentBlock(type="tool_use", name="web_search"),
            ]
        )
        InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 2.0

    def test_mixed_content_blocks_only_counts_tool_use(self, inner):
        inner.create.return_value = FakeMessage(
            content=[
                FakeContentBlock(type="text"),
                FakeContentBlock(type="tool_use", name="calculator"),
                FakeContentBlock(type="text"),
            ]
        )
        InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="calculator") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="text") is None

    def test_tool_calls_not_recorded_on_error(self, inner):
        inner.create.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError):
            InstrumentedMessages(inner).create(model=MODEL, max_tokens=1024, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None


class TestStreamMetrics:
    def test_records_stream_request_count(self, inner):
        inner.stream.return_value = FakeStream(events=["chunk1", "chunk2"])
        wrapped = InstrumentedMessages(inner)
        list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 1.0

    def test_records_stream_duration(self, inner):
        inner.stream.return_value = FakeStream(events=["chunk1", "chunk2"])
        wrapped = InstrumentedMessages(inner)
        list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") == 1.0

    def test_yields_all_stream_events(self, inner):
        events = ["event1", "event2", "event3"]
        inner.stream.return_value = FakeStream(events=events)
        wrapped = InstrumentedMessages(inner)
        result = list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert result == events

    def test_records_tokens_from_final_message(self, inner):
        final_message = FakeMessage(usage=FakeUsage(input_tokens=150, output_tokens=30))
        inner.stream.return_value = FakeStream(events=["chunk"], final_message=final_message)
        wrapped = InstrumentedMessages(inner)
        list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 150.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 30.0

    def test_records_tool_calls_from_final_message(self, inner):
        final_message = FakeMessage(
            content=[FakeContentBlock(type="tool_use", name="web_search")]
        )
        inner.stream.return_value = FakeStream(events=["chunk"], final_message=final_message)
        wrapped = InstrumentedMessages(inner)
        list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0

    def test_multiple_stream_calls_accumulate(self, inner):
        inner.stream.return_value = FakeStream(events=["chunk"])
        wrapped = InstrumentedMessages(inner)
        list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 2.0


class TestStreamErrorHandling:
    def test_error_during_stream_initialization(self, inner):
        inner.stream.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedMessages(inner)
        with pytest.raises(RuntimeError, match="Stream init failed"):
            list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    def test_error_during_stream_iteration(self, inner):
        def failing_generator():
            yield "chunk1"
            raise RuntimeError("Stream failed")

        inner.stream.return_value = failing_generator()
        wrapped = InstrumentedMessages(inner)
        with pytest.raises(RuntimeError, match="Stream failed"):
            list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    def test_error_still_records_duration(self, inner):
        inner.stream.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedMessages(inner)
        with pytest.raises(RuntimeError):
            list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") == 1.0

    def test_tokens_not_recorded_on_error(self, inner):
        inner.stream.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedMessages(inner)
        with pytest.raises(RuntimeError):
            list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") is None


class TestStreamDelegation:
    def test_passes_kwargs_to_inner_stream(self, inner):
        inner.stream.return_value = FakeStream(events=["chunk"])
        wrapped = InstrumentedMessages(inner)
        kwargs = {"model": MODEL, "max_tokens": 512, "messages": [{"role": "user", "content": "hi"}]}
        list(wrapped.stream(**kwargs))
        inner.stream.assert_called_once_with(**kwargs)

    def test_handles_stream_without_final_message(self, inner):
        # Stream without get_final_message method
        inner.stream.return_value = iter(["chunk1", "chunk2"])
        wrapped = InstrumentedMessages(inner)
        result = list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert result == ["chunk1", "chunk2"]
        # Should not record tokens since no final message
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None

    def test_gracefully_handles_get_final_message_error(self, inner):
        def broken_get_final_message():
            raise RuntimeError("Not available")

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter(["chunk"])
        mock_stream.get_final_message = broken_get_final_message
        inner.stream.return_value = mock_stream

        wrapped = InstrumentedMessages(inner)
        # Should not raise, just skip token tracking
        result = list(wrapped.stream(model=MODEL, max_tokens=1024, messages=[]))
        assert result == ["chunk"]
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
