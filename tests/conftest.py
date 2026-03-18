"""Shared fixtures for the agentgauge test suite."""

import pytest

from agentgauge.metrics import (
    LLM_ACTIVE_REQUESTS,
    LLM_CACHE_TOKENS_TOTAL,
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    LLM_TOOL_CALLS_TOTAL,
)

_ALL_METRICS = (
    LLM_REQUESTS_TOTAL,
    LLM_REQUEST_DURATION_SECONDS,
    LLM_TOKENS_TOTAL,
    LLM_ACTIVE_REQUESTS,
    LLM_TOOL_CALLS_TOTAL,
    LLM_CACHE_TOKENS_TOTAL,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset all metric label-sets between tests.

    prometheus_client metrics are module-level singletons and cannot be
    re-registered.  Clearing "_metrics" removes accumulated label
    combinations so every test starts from a clean slate.
    """
    yield
    for metric in _ALL_METRICS:
        metric._metrics.clear()
