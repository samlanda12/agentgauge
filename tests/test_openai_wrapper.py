"""Tests for the OpenAI and OpenAI-compatible InstrumentedChatCompletion wrapper."""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

from agentgauge.openai_wrapper import InstrumentedChatCompletion

MODEL = "gpt-4-turbo"


class FakeStream:
    """Mock stream object that yields chunks and optionally provides final response."""

    def __init__(self, chunks=None, final_response=None, usage=None, choices=None):
        self._chunks = chunks or []
        self._final_response = final_response
        self.usage = usage
        self.choices = choices

    def __iter__(self):
        return iter(self._chunks)

    def get_final_response(self):
        if self._final_response is None:
            raise RuntimeError("No final response available")
        return self._final_response


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 25


@dataclass
class FakeFunction:
    name: str


@dataclass
class FakeToolCall:
    function: FakeFunction


@dataclass
class FakeMessage:
    content: Optional[str] = None
    tool_calls: Optional[List[FakeToolCall]] = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeChatCompletion:
    id: str = "chatcmpl_fake"
    model: str = MODEL
    usage: FakeUsage = field(default_factory=FakeUsage)
    choices: List[FakeChoice] = field(default_factory=lambda: [FakeChoice(message=FakeMessage())])


@pytest.fixture
def inner():
    mock = MagicMock()
    mock.create.return_value = FakeChatCompletion()
    return mock


@pytest.fixture
def wrapped(inner):
    return InstrumentedChatCompletion(inner)


@pytest.fixture
def wrapped_with_error():
    mock = MagicMock()
    mock.create.side_effect = RuntimeError("API down")
    return InstrumentedChatCompletion(mock)


def _sample(metric_name, **labels):
    return REGISTRY.get_sample_value(metric_name, labels)


class TestCreateMetrics:
    def test_records_request_count(self, wrapped):
        wrapped.create(model=MODEL, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 1.0

    def test_records_duration(self, wrapped):
        wrapped.create(model=MODEL, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    def test_records_prompt_tokens(self, inner):
        inner.create.return_value = FakeChatCompletion(usage=FakeUsage(prompt_tokens=200, completion_tokens=50))
        wrapped = InstrumentedChatCompletion(inner)
        wrapped.create(model=MODEL, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 200.0

    def test_records_completion_tokens(self, inner):
        inner.create.return_value = FakeChatCompletion(usage=FakeUsage(prompt_tokens=200, completion_tokens=50))
        wrapped = InstrumentedChatCompletion(inner)
        wrapped.create(model=MODEL, messages=[])
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 50.0

    def test_multiple_calls_accumulate(self, wrapped):
        wrapped.create(model=MODEL, messages=[])
        wrapped.create(model=MODEL, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="ok") == 2.0


class TestCreateErrorHandling:
    def test_error_is_reraised(self, wrapped_with_error):
        with pytest.raises(RuntimeError, match="API down"):
            wrapped_with_error.create(model=MODEL, messages=[])

    def test_error_records_error_status(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            wrapped_with_error.create(model=MODEL, messages=[])
        assert _sample("llm_requests_total", model=MODEL, method="create", status="error") == 1.0

    def test_error_still_records_duration(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            wrapped_with_error.create(model=MODEL, messages=[])
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="create") == 1.0

    def test_active_requests_returns_to_zero_on_error(self, wrapped_with_error):
        with pytest.raises(RuntimeError):
            wrapped_with_error.create(model=MODEL, messages=[])
        assert _sample("llm_active_requests", model=MODEL) == 0.0


class TestCreateDelegation:
    def test_passes_kwargs_to_inner_client(self, wrapped, inner):
        kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        wrapped.create(**kwargs)
        inner.create.assert_called_once_with(**kwargs)

    def test_returns_original_response(self, inner):
        expected = FakeChatCompletion(id="chatcmpl_123")
        inner.create.return_value = expected
        result = InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert result.id == "chatcmpl_123"

    def test_proxies_other_attributes(self, wrapped, inner):
        inner.some_other_method.return_value = "hello"
        assert wrapped.some_other_method() == "hello"


class TestToolCallTracking:
    def test_no_tool_calls_when_choices_is_empty(self, inner):
        inner.create.return_value = FakeChatCompletion(choices=[])
        InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    def test_no_tool_calls_when_message_has_no_tool_calls(self, inner):
        inner.create.return_value = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="Hello", tool_calls=None))]
        )
        InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None

    def test_single_tool_call_is_recorded(self, inner):
        inner.create.return_value = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="web_search"))]
                    )
                )
            ]
        )
        InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0

    def test_multiple_distinct_tool_calls_are_recorded(self, inner):
        inner.create.return_value = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(function=FakeFunction(name="web_search")),
                            FakeToolCall(function=FakeFunction(name="calculator")),
                        ]
                    )
                )
            ]
        )
        InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="calculator") == 1.0

    def test_repeated_same_tool_accumulates(self, inner):
        inner.create.return_value = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(function=FakeFunction(name="web_search")),
                            FakeToolCall(function=FakeFunction(name="web_search")),
                        ]
                    )
                )
            ]
        )
        InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 2.0

    def test_multiple_choices_with_tool_calls(self, inner):
        inner.create.return_value = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="web_search"))]
                    )
                ),
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="calculator"))]
                    )
                ),
            ]
        )
        InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="calculator") == 1.0

    def test_tool_calls_not_recorded_on_error(self, inner):
        inner.create.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError):
            InstrumentedChatCompletion(inner).create(model=MODEL, messages=[])
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="any_tool") is None


class TestStreamMetrics:
    def test_records_stream_request_count(self, inner):
        inner.create.return_value = FakeStream(chunks=["chunk1", "chunk2"])
        wrapped = InstrumentedChatCompletion(inner)
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            list(stream)
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 1.0

    def test_records_stream_duration(self, inner):
        inner.create.return_value = FakeStream(chunks=["chunk1", "chunk2"])
        wrapped = InstrumentedChatCompletion(inner)
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            list(stream)
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") == 1.0

    def test_yields_all_stream_chunks(self, inner):
        chunks = ["chunk1", "chunk2", "chunk3"]
        inner.create.return_value = FakeStream(chunks=chunks)
        wrapped = InstrumentedChatCompletion(inner)
        result = None
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            result = list(stream)
        assert result == chunks

    def test_records_tokens_from_final_response(self, inner):
        # OpenAI streams with stream_options can include usage in the stream object itself
        stream_with_usage = FakeStream(
            chunks=["chunk"],
            usage=FakeUsage(prompt_tokens=175, completion_tokens=40)
        )
        inner.create.return_value = stream_with_usage
        wrapped = InstrumentedChatCompletion(inner)
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            list(stream)
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 175.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 40.0

    def test_records_tool_calls_from_final_response(self, inner):
        # OpenAI streams can accumulate tool calls in the stream object
        stream_with_tool = FakeStream(
            chunks=["chunk"],
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="web_search"))]
                    )
                )
            ]
        )
        inner.create.return_value = stream_with_tool
        wrapped = InstrumentedChatCompletion(inner)
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            list(stream)
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0

    def test_multiple_stream_calls_accumulate(self, inner):
        inner.create.return_value = FakeStream(chunks=["chunk"])
        wrapped = InstrumentedChatCompletion(inner)
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            list(stream)
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            list(stream)
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 2.0


class TestStreamErrorHandling:
    def test_error_during_stream_initialization(self, inner):
        """When stream() call fails, only error counter is incremented - gauge and timer are untouched."""
        inner.create.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError, match="Stream init failed"):
            with wrapped.stream(model=MODEL, messages=[]) as stream:
                list(stream)
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        # Gauge should never have been touched (None means no samples, which is correct)
        assert _sample("llm_active_requests", model=MODEL) is None

    def test_error_during_stream_iteration(self, inner):
        def failing_generator():
            yield "chunk1"
            raise RuntimeError("Stream failed")

        inner.create.return_value = failing_generator()
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError, match="Stream failed"):
            with wrapped.stream(model=MODEL, messages=[]) as stream:
                list(stream)
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    def test_error_no_duration_for_init_failure(self, inner):
        """Duration should NOT be recorded when stream initialization fails - the request barely existed."""
        inner.create.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError):
            with wrapped.stream(model=MODEL, messages=[]) as stream:
                list(stream)
        # Duration histogram should have no samples for init failures
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") is None

    def test_tokens_not_recorded_on_error(self, inner):
        inner.create.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError):
            with wrapped.stream(model=MODEL, messages=[]) as stream:
                list(stream)
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") is None

    def test_abandoned_stream_releases_active_request_gauge(self, inner):
        """Test that active requests gauge is decremented even if stream is not fully consumed."""
        inner.create.return_value = FakeStream(chunks=["chunk1", "chunk2", "chunk3"])
        wrapped = InstrumentedChatCompletion(inner)
        
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            # Only consume first chunk, then abandon
            first = next(iter(stream))
            assert first == "chunk1"
        
        # After context exits, active requests should be back to zero
        assert _sample("llm_active_requests", model=MODEL) == 0.0
        # Request should still be counted
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 1.0


class TestStreamDelegation:
    def test_passes_kwargs_to_inner_create(self, inner):
        inner.create.return_value = FakeStream(chunks=["chunk"])
        wrapped = InstrumentedChatCompletion(inner)
        kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        with wrapped.stream(**kwargs) as stream:
            list(stream)
        # stream() should enforce stream=True and inject stream_options
        expected_kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        inner.create.assert_called_once_with(**expected_kwargs)

    def test_merges_existing_stream_options(self, inner):
        """Test that existing stream_options are preserved when injecting include_usage."""
        inner.create.return_value = FakeStream(chunks=["chunk"])
        wrapped = InstrumentedChatCompletion(inner)
        kwargs = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"continuous_usage": True}
        }
        with wrapped.stream(**kwargs) as stream:
            list(stream)
        # Should merge existing options with include_usage
        expected_kwargs = {
            **kwargs,
            "stream": True,
            "stream_options": {"continuous_usage": True, "include_usage": True}
        }
        inner.create.assert_called_once_with(**expected_kwargs)

    def test_handles_stream_without_final_response(self, inner):
        # Stream without get_final_response method
        inner.create.return_value = iter(["chunk1", "chunk2"])
        wrapped = InstrumentedChatCompletion(inner)
        result = None
        with wrapped.stream(model=MODEL, messages=[]) as stream:
            result = list(stream)
        assert result == ["chunk1", "chunk2"]
        # Should not record tokens since no final response
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None

