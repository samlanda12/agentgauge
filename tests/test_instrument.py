"""Tests for the instrument() public API."""

from unittest.mock import MagicMock, patch

import pytest

import agentgauge
from agentgauge import instrument
from agentgauge.wrapper import InstrumentedMessages

MODEL = "claude-sonnet-4-5-20250929"


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    return client


@pytest.fixture(autouse=True)
def _reset_server_flag():
    """Ensure the server-started flag is always clean before and after each test."""
    agentgauge._server_started = False
    yield
    agentgauge._server_started = False

# Wrapping behaviour

def test_returns_instrumented_messages(mock_client):
    wrapped = instrument(mock_client, start_server=False)
    assert isinstance(wrapped.messages, InstrumentedMessages)


def test_messages_create_delegates(mock_client):
    wrapped = instrument(mock_client, start_server=False)
    wrapped.messages.create(model=MODEL, max_tokens=1024, messages=[])
    mock_client.messages.create.assert_called_once()


def test_proxies_non_messages_attributes(mock_client):
    mock_client.api_key = "sk-test-123"
    assert instrument(mock_client, start_server=False).api_key == "sk-test-123"


def test_proxies_other_methods(mock_client):
    mock_client.count_tokens.return_value = 42
    assert instrument(mock_client, start_server=False).count_tokens() == 42

# HTTP server lifecycle

@patch("agentgauge.start_http_server")
def test_starts_server_on_first_call(mock_server, mock_client):
    instrument(mock_client, port=9999)
    mock_server.assert_called_once_with(9999)


@patch("agentgauge.start_http_server")
def test_starts_server_only_once(mock_server, mock_client):
    instrument(mock_client, port=9999)
    instrument(mock_client, port=9999)
    mock_server.assert_called_once_with(9999)


@patch("agentgauge.start_http_server")
def test_start_server_false_skips(mock_server, mock_client):
    instrument(mock_client, start_server=False)
    mock_server.assert_not_called()
