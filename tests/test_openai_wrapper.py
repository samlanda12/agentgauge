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

    def __init__(self, chunks=None, final_response=None):
        self._chunks = chunks or []
        self._final_response = final_response

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
        list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 1.0

    def test_records_stream_duration(self, inner):
        inner.create.return_value = FakeStream(chunks=["chunk1", "chunk2"])
        wrapped = InstrumentedChatCompletion(inner)
        list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") == 1.0

    def test_yields_all_stream_chunks(self, inner):
        chunks = ["chunk1", "chunk2", "chunk3"]
        inner.create.return_value = FakeStream(chunks=chunks)
        wrapped = InstrumentedChatCompletion(inner)
        result = list(wrapped.stream(model=MODEL, messages=[]))
        assert result == chunks

    def test_records_tokens_from_final_response(self, inner):
        final_response = FakeChatCompletion(usage=FakeUsage(prompt_tokens=175, completion_tokens=40))
        inner.create.return_value = FakeStream(chunks=["chunk"], final_response=final_response)
        wrapped = InstrumentedChatCompletion(inner)
        list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 175.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 40.0

    def test_records_tool_calls_from_final_response(self, inner):
        final_response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[FakeToolCall(function=FakeFunction(name="web_search"))]
                    )
                )
            ]
        )
        inner.create.return_value = FakeStream(chunks=["chunk"], final_response=final_response)
        wrapped = InstrumentedChatCompletion(inner)
        list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_tool_calls_total", model=MODEL, tool_name="web_search") == 1.0

    def test_multiple_stream_calls_accumulate(self, inner):
        inner.create.return_value = FakeStream(chunks=["chunk"])
        wrapped = InstrumentedChatCompletion(inner)
        list(wrapped.stream(model=MODEL, messages=[]))
        list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="ok") == 2.0


class TestStreamErrorHandling:
    def test_error_during_stream_initialization(self, inner):
        inner.create.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError, match="Stream init failed"):
            list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    def test_error_during_stream_iteration(self, inner):
        def failing_generator():
            yield "chunk1"
            raise RuntimeError("Stream failed")

        inner.create.return_value = failing_generator()
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError, match="Stream failed"):
            list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_requests_total", model=MODEL, method="stream", status="error") == 1.0
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    def test_error_still_records_duration(self, inner):
        inner.create.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError):
            list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="stream") == 1.0

    def test_tokens_not_recorded_on_error(self, inner):
        inner.create.side_effect = RuntimeError("Stream init failed")
        wrapped = InstrumentedChatCompletion(inner)
        with pytest.raises(RuntimeError):
            list(wrapped.stream(model=MODEL, messages=[]))
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") is None


class TestStreamDelegation:
    def test_passes_kwargs_to_inner_create(self, inner):
        inner.create.return_value = FakeStream(chunks=["chunk"])
        wrapped = InstrumentedChatCompletion(inner)
        kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        list(wrapped.stream(**kwargs))
        inner.create.assert_called_once_with(**kwargs)

    def test_handles_stream_without_final_response(self, inner):
        # Stream without get_final_response method
        inner.create.return_value = iter(["chunk1", "chunk2"])
        wrapped = InstrumentedChatCompletion(inner)
        result = list(wrapped.stream(model=MODEL, messages=[]))
        assert result == ["chunk1", "chunk2"]
        # Should not record tokens since no final response
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None

    def test_gracefully_handles_get_final_response_error(self, inner):
        def broken_get_final_response():
            raise RuntimeError("Not available")

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter(["chunk"])
        mock_stream.get_final_response = broken_get_final_response
        inner.create.return_value = mock_stream

        wrapped = InstrumentedChatCompletion(inner)
        # Should not raise, just skip token tracking
        result = list(wrapped.stream(model=MODEL, messages=[]))
        assert result == ["chunk"]
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
