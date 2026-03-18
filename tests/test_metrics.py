"""Tests for Prometheus metric definitions."""

import pytest
from prometheus_client import REGISTRY

from agentgauge.metrics import (
    LLM_ACTIVE_REQUESTS,
    LLM_CACHE_TOKENS_TOTAL,
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    LLM_TOOL_CALLS_TOTAL,
)

MODEL = "claude-sonnet-4-5-20250929"

@pytest.mark.parametrize("name", [
    "llm_requests",
    "llm_request_duration_seconds",
    "llm_tokens",
    "llm_active_requests",
    "llm_tool_calls",
    "llm_cache_tokens",
])
def test_metric_registered(name):
    metric_names = {m.name for m in REGISTRY.collect()}
    assert name in metric_names

@pytest.mark.parametrize("metric,expected_name,expected_labels", [
    (LLM_REQUESTS_TOTAL,          "llm_requests",              ("model", "method", "status")),
    (LLM_REQUEST_DURATION_SECONDS,"llm_request_duration_seconds", ("model", "method")),
    (LLM_TOKENS_TOTAL,            "llm_tokens",                ("model", "token_type")),
    (LLM_ACTIVE_REQUESTS,         "llm_active_requests",       ("model",)),
    (LLM_TOOL_CALLS_TOTAL,        "llm_tool_calls",            ("model", "tool_name")),
    (LLM_CACHE_TOKENS_TOTAL,      "llm_cache_tokens",          ("model", "cache_type")),
])
def test_metric_shape(metric, expected_name, expected_labels):
    assert metric._name == expected_name
    assert metric._labelnames == expected_labels


def test_increment_requests_counter():
    LLM_REQUESTS_TOTAL.labels(model=MODEL, method="create", status="ok").inc()
    assert REGISTRY.get_sample_value(
        "llm_requests_total", {"model": MODEL, "method": "create", "status": "ok"}
    ) == 1.0


def test_observe_duration():
    LLM_REQUEST_DURATION_SECONDS.labels(model=MODEL, method="create").observe(1.5)
    assert REGISTRY.get_sample_value(
        "llm_request_duration_seconds_count", {"model": MODEL, "method": "create"}
    ) == 1.0


def test_increment_tokens():
    LLM_TOKENS_TOTAL.labels(model=MODEL, token_type="input").inc(150)
    LLM_TOKENS_TOTAL.labels(model=MODEL, token_type="output").inc(42)

    assert REGISTRY.get_sample_value("llm_tokens_total", {"model": MODEL, "token_type": "input"}) == 150.0
    assert REGISTRY.get_sample_value("llm_tokens_total", {"model": MODEL, "token_type": "output"}) == 42.0


def test_active_requests_gauge():
    gauge = LLM_ACTIVE_REQUESTS.labels(model=MODEL)
    gauge.inc()
    gauge.inc()
    gauge.dec()

    assert REGISTRY.get_sample_value("llm_active_requests", {"model": MODEL}) == 1.0


def test_increment_tool_calls():
    LLM_TOOL_CALLS_TOTAL.labels(model=MODEL, tool_name="web_search").inc(3)
    LLM_TOOL_CALLS_TOTAL.labels(model=MODEL, tool_name="calculator").inc(1)

    assert REGISTRY.get_sample_value("llm_tool_calls_total", {"model": MODEL, "tool_name": "web_search"}) == 3.0
    assert REGISTRY.get_sample_value("llm_tool_calls_total", {"model": MODEL, "tool_name": "calculator"}) == 1.0


def test_increment_cache_tokens():
    """Test cache token counter for both creation and read cache types."""
    LLM_CACHE_TOKENS_TOTAL.labels(model=MODEL, cache_type="creation").inc(500)
    LLM_CACHE_TOKENS_TOTAL.labels(model=MODEL, cache_type="read").inc(250)

    assert REGISTRY.get_sample_value("llm_cache_tokens_total", {"model": MODEL, "cache_type": "creation"}) == 500.0
    assert REGISTRY.get_sample_value("llm_cache_tokens_total", {"model": MODEL, "cache_type": "read"}) == 250.0
