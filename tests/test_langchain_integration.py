"""Integration tests for AgentGaugeCallbackHandler with LangChain.

These tests verify that the callback handler works correctly with
actual LangChain chat models (using mocks to avoid real API calls).
"""

import pytest

pytest.importorskip("langchain_core")

from unittest.mock import MagicMock, patch
from uuid import uuid4

from langchain_core.outputs import LLMResult, ChatGeneration
from prometheus_client import REGISTRY

from agentgauge.langchain_callback import AgentGaugeCallbackHandler


MODEL = "gpt-4o"


def _sample(metric_name, **labels):
    return REGISTRY.get_sample_value(metric_name, labels)


class TestLangChainChatOpenAIIntegration:
    """Integration tests for AgentGaugeCallbackHandler with ChatOpenAI-style models."""

    @pytest.fixture
    def handler(self):
        return AgentGaugeCallbackHandler()

    @pytest.fixture
    def mock_chat_result(self):
        """Create a mock LLMResult similar to what ChatOpenAI returns."""
        generation = MagicMock(spec=ChatGeneration)
        generation.text = "Hello, world!"
        generation.message = MagicMock()
        generation.message.content = "Hello, world!"

        result = LLMResult(
            generations=[[generation]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model_name": MODEL,
            },
        )
        return result

    def test_full_lifecycle_with_chat_openai(self, handler, mock_chat_result):
        """Simulate a complete ChatOpenAI request lifecycle."""
        run_id = uuid4()

        # on_chat_model_start is called when ChatOpenAI starts a request
        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI", "kwargs": {"model_name": MODEL}},
            messages=[[{"role": "user", "content": "Hello"}]],
            run_id=run_id,
            invocation_params={"model": MODEL, "_type": "openai-chat"},
        )

        # Verify active request was incremented
        assert _sample("llm_active_requests", model=MODEL) == 1.0

        # on_llm_end is called when the request completes
        handler.on_llm_end(mock_chat_result, run_id=run_id)

        # Verify all metrics were recorded
        assert _sample("llm_active_requests", model=MODEL) == 0.0
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="ok") == 1.0
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="invoke") == 1.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 10.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 5.0

    def test_error_lifecycle_with_chat_openai(self, handler):
        """Simulate an error scenario with ChatOpenAI."""
        run_id = uuid4()

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[{"role": "user", "content": "Hello"}]],
            run_id=run_id,
            invocation_params={"model": MODEL},
        )

        assert _sample("llm_active_requests", model=MODEL) == 1.0

        # Simulate an error
        handler.on_llm_error(RuntimeError("API timeout"), run_id=run_id)

        assert _sample("llm_active_requests", model=MODEL) == 0.0
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="error") == 1.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None


class TestLangChainAnthropicIntegration:
    """Integration tests for AgentGaugeCallbackHandler with ChatAnthropic-style models."""

    @pytest.fixture
    def handler(self):
        return AgentGaugeCallbackHandler()

    @pytest.fixture
    def mock_anthropic_result(self):
        """Create a mock LLMResult similar to what ChatAnthropic returns."""
        generation = MagicMock(spec=ChatGeneration)
        generation.text = "Hello!"
        generation.message = MagicMock()
        generation.message.content = "Hello!"

        result = LLMResult(
            generations=[[generation]],
            llm_output={
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 10,
                },
                "model": "claude-3-5-sonnet-20241022",
            },
        )
        return result

    def test_anthropic_style_token_usage(self, handler, mock_anthropic_result):
        """Test that Anthropic-style token usage (input_tokens/output_tokens) is handled."""
        run_id = uuid4()
        model = "claude-3-5-sonnet-20241022"

        handler.on_chat_model_start(
            serialized={"name": "ChatAnthropic", "kwargs": {"model_name": model}},
            messages=[[{"role": "user", "content": "Hello"}]],
            run_id=run_id,
            invocation_params={"model": model},
        )

        handler.on_llm_end(mock_anthropic_result, run_id=run_id)

        assert _sample("llm_tokens_total", model=model, token_type="input") == 25.0
        assert _sample("llm_tokens_total", model=model, token_type="output") == 10.0


class TestLangChainToolCalls:
    """Test tool call tracking through the callback handler."""

    @pytest.fixture
    def handler(self):
        return AgentGaugeCallbackHandler()

    def test_tool_start_records_tool_call(self, handler):
        """Test that on_tool_start records tool call metrics."""
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="search for Python tutorials",
            run_id=run_id,
        )

        assert _sample("llm_tool_calls_total", model="unknown", tool_name="web_search") == 1.0

    def test_multiple_same_tool_accumulates(self, handler):
        """Test that multiple calls to the same tool accumulate correctly."""
        for _ in range(3):
            handler.on_tool_start(
                serialized={"name": "calculator"},
                input_str="1 + 1",
                run_id=uuid4(),
            )

        assert _sample("llm_tool_calls_total", model="unknown", tool_name="calculator") == 3.0


class TestLangChainChainPropagation:
    """Test that callbacks propagate through LangChain chains."""

    @pytest.fixture
    def handler(self):
        return AgentGaugeCallbackHandler()

    def test_nested_runs_are_independently_tracked(self, handler):
        """Test that nested/parallel runs are tracked independently."""
        run1, run2 = uuid4(), uuid4()

        # Start two "parallel" requests
        handler.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id=run1,
            invocation_params={"model": "gpt-4o"},
        )
        handler.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id=run2,
            invocation_params={"model": "gpt-4-turbo"},
        )

        # Both should be active
        assert _sample("llm_active_requests", model="gpt-4o") == 1.0
        assert _sample("llm_active_requests", model="gpt-4-turbo") == 1.0

        # End first
        result = LLMResult(
            generations=[[]],
            llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )
        handler.on_llm_end(result, run_id=run1)

        # First should be inactive, second still active
        assert _sample("llm_active_requests", model="gpt-4o") == 0.0
        assert _sample("llm_active_requests", model="gpt-4-turbo") == 1.0

        # End second
        handler.on_llm_end(result, run_id=run2)

        assert _sample("llm_active_requests", model="gpt-4-turbo") == 0.0