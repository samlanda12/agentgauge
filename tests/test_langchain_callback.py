"""Tests for AgentGaugeCallbackHandler."""

from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.outputs import LLMResult  # noqa: E402

from agentgauge.langchain_callback import AgentGaugeCallbackHandler  # noqa: E402

from prometheus_client import REGISTRY  # noqa: E402

MODEL = "gpt-4o"


def _sample(metric_name, **labels):
    return REGISTRY.get_sample_value(metric_name, labels)


def _make_llm_result(
    prompt_tokens: int = 100,
    completion_tokens: int = 25,
    model_name: str = MODEL,
    style: str = "openai",
) -> LLMResult:
    """Build an LLMResult with token usage in either OpenAI or Anthropic format."""
    if style == "openai":
        llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": model_name,
        }
    else:
        llm_output = {
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
            "model": model_name,
        }
    return LLMResult(generations=[[]], llm_output=llm_output)


def _serialized(model: str = MODEL) -> dict:
    return {"name": "ChatOpenAI", "kwargs": {"model_name": model}}


def _invocation_kwargs(model: str = MODEL) -> dict:
    return {"invocation_params": {"model": model, "_type": "openai-chat"}}


@pytest.fixture
def handler() -> AgentGaugeCallbackHandler:
    return AgentGaugeCallbackHandler()

# on_llm_start

class TestOnLlmStart:
    def test_increments_active_requests(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hello"], run_id=run_id, **_invocation_kwargs())
        assert _sample("llm_active_requests", model=MODEL) == 1.0

    def test_tracks_run_id(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hello"], run_id=run_id, **_invocation_kwargs())
        assert str(run_id) in handler._model_names
        assert handler._model_names[str(run_id)] == MODEL

    def test_multiple_runs_tracked_independently(self, handler):
        run1, run2 = uuid4(), uuid4()
        handler.on_llm_start(_serialized("gpt-4o"), ["a"], run_id=run1, **_invocation_kwargs("gpt-4o"))
        handler.on_llm_start(_serialized("gpt-4-turbo"), ["b"], run_id=run2, **_invocation_kwargs("gpt-4-turbo"))
        assert handler._model_names[str(run1)] == "gpt-4o"
        assert handler._model_names[str(run2)] == "gpt-4-turbo"
        assert _sample("llm_active_requests", model="gpt-4o") == 1.0
        assert _sample("llm_active_requests", model="gpt-4-turbo") == 1.0

# on_chat_model_start

class TestOnChatModelStart:
    def test_increments_active_requests(self, handler):
        run_id = uuid4()
        handler.on_chat_model_start(_serialized(), [[]], run_id=run_id, **_invocation_kwargs())
        assert _sample("llm_active_requests", model=MODEL) == 1.0

    def test_tracks_run_id(self, handler):
        run_id = uuid4()
        handler.on_chat_model_start(_serialized(), [[]], run_id=run_id, **_invocation_kwargs())
        assert str(run_id) in handler._model_names

    def test_records_start_time(self, handler):
        run_id = uuid4()
        handler.on_chat_model_start(_serialized(), [[]], run_id=run_id, **_invocation_kwargs())
        assert str(run_id) in handler._request_starts

# on_llm_end

class TestOnLlmEnd:
    def test_records_request_count(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(_make_llm_result(), run_id=run_id)
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="ok") == 1.0

    def test_decrements_active_requests(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(_make_llm_result(), run_id=run_id)
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    def test_records_duration(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(_make_llm_result(), run_id=run_id)
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="invoke") == 1.0

    def test_cleans_up_tracking_state(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(_make_llm_result(), run_id=run_id)
        assert str(run_id) not in handler._model_names
        assert str(run_id) not in handler._request_starts

    def test_multiple_calls_accumulate(self, handler):
        for _ in range(3):
            run_id = uuid4()
            handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
            handler.on_llm_end(_make_llm_result(), run_id=run_id)
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="ok") == 3.0

    def test_orphaned_end_without_start_is_safe(self, handler):
        """on_llm_end with an unknown run_id must not raise or corrupt metrics."""
        handler.on_llm_end(_make_llm_result(), run_id=uuid4())
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="ok") is None

    def test_works_after_chat_model_start(self, handler):
        run_id = uuid4()
        handler.on_chat_model_start(_serialized(), [[]], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(_make_llm_result(), run_id=run_id)
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="ok") == 1.0

# on_llm_error

class TestOnLlmError:
    def test_records_error_status(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="error") == 1.0

    def test_decrements_active_requests(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)
        assert _sample("llm_active_requests", model=MODEL) == 0.0

    def test_records_duration(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)
        assert _sample("llm_request_duration_seconds_count", model=MODEL, method="invoke") == 1.0

    def test_does_not_record_ok_status(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="ok") is None

    def test_orphaned_error_without_start_is_safe(self, handler):
        handler.on_llm_error(RuntimeError("fail"), run_id=uuid4())
        assert _sample("llm_requests_total", model=MODEL, method="invoke", status="error") is None

    def test_cleans_up_tracking_state(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)
        assert str(run_id) not in handler._model_names
        assert str(run_id) not in handler._request_starts

# Token tracking

class TestTokenTracking:
    def test_openai_style_input_tokens(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(_make_llm_result(prompt_tokens=200, completion_tokens=50), run_id=run_id)
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 200.0

    def test_openai_style_output_tokens(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(_make_llm_result(prompt_tokens=200, completion_tokens=50), run_id=run_id)
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 50.0

    def test_anthropic_style_input_tokens(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(
            _make_llm_result(prompt_tokens=150, completion_tokens=30, style="anthropic"),
            run_id=run_id,
        )
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 150.0

    def test_anthropic_style_output_tokens(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(
            _make_llm_result(prompt_tokens=150, completion_tokens=30, style="anthropic"),
            run_id=run_id,
        )
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 30.0

    def test_missing_token_usage_does_not_record(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_end(LLMResult(generations=[[]], llm_output={}), run_id=run_id)
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") is None

    def test_no_token_usage_on_error(self, handler):
        run_id = uuid4()
        handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") is None

    def test_multiple_calls_accumulate_tokens(self, handler):
        for _ in range(3):
            run_id = uuid4()
            handler.on_llm_start(_serialized(), ["hi"], run_id=run_id, **_invocation_kwargs())
            handler.on_llm_end(_make_llm_result(prompt_tokens=100, completion_tokens=25), run_id=run_id)
        assert _sample("llm_tokens_total", model=MODEL, token_type="input") == 300.0
        assert _sample("llm_tokens_total", model=MODEL, token_type="output") == 75.0

# Tool call tracking

class TestToolCallTracking:
    def test_records_tool_call(self, handler):
        handler.on_tool_start({"name": "web_search"}, "query", run_id=uuid4())
        assert _sample("llm_tool_calls_total", model="unknown", tool_name="web_search") == 1.0

    def test_records_distinct_tool_names(self, handler):
        handler.on_tool_start({"name": "web_search"}, "query", run_id=uuid4())
        handler.on_tool_start({"name": "calculator"}, "1+1", run_id=uuid4())
        assert _sample("llm_tool_calls_total", model="unknown", tool_name="web_search") == 1.0
        assert _sample("llm_tool_calls_total", model="unknown", tool_name="calculator") == 1.0

    def test_same_tool_accumulates(self, handler):
        handler.on_tool_start({"name": "web_search"}, "q1", run_id=uuid4())
        handler.on_tool_start({"name": "web_search"}, "q2", run_id=uuid4())
        assert _sample("llm_tool_calls_total", model="unknown", tool_name="web_search") == 2.0

    def test_missing_tool_name_uses_unknown(self, handler):
        handler.on_tool_start({}, "input", run_id=uuid4())
        assert _sample("llm_tool_calls_total", model="unknown", tool_name="unknown") == 1.0

    def test_tracks_start_time(self, handler):
        run_id = uuid4()
        handler.on_tool_start({"name": "web_search"}, "query", run_id=run_id)
        assert str(run_id) in handler._tool_starts
        assert str(run_id) in handler._tool_names

    def test_tracks_tool_name(self, handler):
        run_id = uuid4()
        handler.on_tool_start({"name": "calculator"}, "1+1", run_id=run_id)
        assert handler._tool_names[str(run_id)] == "calculator"


# Tool duration tracking

class TestToolDurationTracking:
    def test_on_tool_end_records_duration(self, handler):
        run_id = uuid4()
        handler.on_tool_start({"name": "web_search"}, "query", run_id=run_id)
        handler.on_tool_end("search results", run_id=run_id)
        assert _sample("llm_tool_duration_seconds_count", tool_name="web_search") == 1.0

    def test_on_tool_error_records_duration(self, handler):
        run_id = uuid4()
        handler.on_tool_start({"name": "calculator"}, "1/0", run_id=run_id)
        handler.on_tool_error(RuntimeError("division by zero"), run_id=run_id)
        assert _sample("llm_tool_duration_seconds_count", tool_name="calculator") == 1.0

    def test_cleans_up_tracking_state_on_end(self, handler):
        run_id = uuid4()
        handler.on_tool_start({"name": "web_search"}, "query", run_id=run_id)
        handler.on_tool_end("results", run_id=run_id)
        assert str(run_id) not in handler._tool_starts
        assert str(run_id) not in handler._tool_names

    def test_cleans_up_tracking_state_on_error(self, handler):
        run_id = uuid4()
        handler.on_tool_start({"name": "web_search"}, "query", run_id=run_id)
        handler.on_tool_error(RuntimeError("fail"), run_id=run_id)
        assert str(run_id) not in handler._tool_starts
        assert str(run_id) not in handler._tool_names

    def test_orphaned_tool_end_is_safe(self, handler):
        handler.on_tool_end("result", run_id=uuid4())
        assert _sample("llm_tool_duration_seconds_count", tool_name="unknown") is None

    def test_orphaned_tool_error_is_safe(self, handler):
        handler.on_tool_error(RuntimeError("fail"), run_id=uuid4())
        assert _sample("llm_tool_duration_seconds_count", tool_name="unknown") is None

    def test_multiple_tool_calls_accumulate_duration(self, handler):
        for _ in range(3):
            run_id = uuid4()
            handler.on_tool_start({"name": "web_search"}, "query", run_id=run_id)
            handler.on_tool_end("results", run_id=run_id)
        assert _sample("llm_tool_duration_seconds_count", tool_name="web_search") == 3.0

# Model name extraction

class TestModelExtraction:
    def test_model_from_invocation_params_model_key(self, handler):
        run_id = uuid4()
        handler.on_llm_start(
            {},
            ["hi"],
            run_id=run_id,
            invocation_params={"model": "gpt-4-turbo"},
        )
        assert handler._model_names[str(run_id)] == "gpt-4-turbo"

    def test_model_from_invocation_params_model_name_key(self, handler):
        run_id = uuid4()
        handler.on_llm_start(
            {},
            ["hi"],
            run_id=run_id,
            invocation_params={"model_name": "gpt-4-turbo"},
        )
        assert handler._model_names[str(run_id)] == "gpt-4-turbo"

    def test_model_from_serialized_kwargs(self, handler):
        run_id = uuid4()
        handler.on_llm_start(
            {"kwargs": {"model_name": "claude-3-5-sonnet-20241022"}},
            ["hi"],
            run_id=run_id,
        )
        assert handler._model_names[str(run_id)] == "claude-3-5-sonnet-20241022"

    def test_invocation_params_takes_precedence_over_serialized(self, handler):
        run_id = uuid4()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-3.5"}},
            ["hi"],
            run_id=run_id,
            invocation_params={"model": "gpt-4o"},
        )
        assert handler._model_names[str(run_id)] == "gpt-4o"

    def test_fallback_to_unknown(self, handler):
        run_id = uuid4()
        handler.on_llm_start({}, ["hi"], run_id=run_id)
        assert handler._model_names[str(run_id)] == "unknown"
