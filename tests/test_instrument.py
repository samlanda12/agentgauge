"""Tests for the instrument() public API."""

from unittest.mock import MagicMock, patch

import pytest

import agentgauge
from agentgauge import instrument
from agentgauge.anthropic_wrapper import InstrumentedMessages
from agentgauge.openai_wrapper import InstrumentedChatCompletion

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
OPENAI_MODEL = "gpt-4-turbo"


@pytest.fixture
def mock_anthropic_client():
    client = MagicMock()
    client.__class__.__name__ = "Anthropic"
    client.messages.create.return_value = MagicMock(
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    # Remove chat attribute to make it Anthropic-only
    del client.chat
    return client


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.__class__.__name__ = "OpenAI"
    client.chat.completions.create.return_value = MagicMock(
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        choices=[MagicMock(message=MagicMock(tool_calls=None))],
    )
    # Remove messages attribute to make it OpenAI-only
    del client.messages
    return client


@pytest.fixture(autouse=True)
def _reset_server_flag():
    """Ensure the server-started flag is always clean before and after each test."""
    agentgauge._server_started = False
    yield
    agentgauge._server_started = False

# Anthropic wrapping behaviour

def test_returns_instrumented_anthropic_messages(mock_anthropic_client):
    wrapped = instrument(mock_anthropic_client, start_server=False)
    assert isinstance(wrapped.messages, InstrumentedMessages)


def test_anthropic_messages_create_delegates(mock_anthropic_client):
    wrapped = instrument(mock_anthropic_client, start_server=False)
    wrapped.messages.create(model=ANTHROPIC_MODEL, max_tokens=1024, messages=[])
    mock_anthropic_client.messages.create.assert_called_once()


def test_anthropic_proxies_non_messages_attributes(mock_anthropic_client):
    mock_anthropic_client.api_key = "sk-test-123"
    assert instrument(mock_anthropic_client, start_server=False).api_key == "sk-test-123"


def test_anthropic_proxies_other_methods(mock_anthropic_client):
    mock_anthropic_client.count_tokens.return_value = 42
    assert instrument(mock_anthropic_client, start_server=False).count_tokens() == 42

# OpenAI wrapping behaviour

def test_returns_instrumented_openai_chat_completions(mock_openai_client):
    wrapped = instrument(mock_openai_client, start_server=False)
    assert isinstance(wrapped.chat.completions, InstrumentedChatCompletion)


def test_openai_chat_completions_create_delegates(mock_openai_client):
    wrapped = instrument(mock_openai_client, start_server=False)
    wrapped.chat.completions.create(model=OPENAI_MODEL, messages=[])
    mock_openai_client.chat.completions.create.assert_called_once()


def test_openai_proxies_non_chat_attributes(mock_openai_client):
    mock_openai_client.api_key = "sk-test-123"
    assert instrument(mock_openai_client, start_server=False).api_key == "sk-test-123"


def test_openai_proxies_other_methods(mock_openai_client):
    mock_openai_client.embeddings.create.return_value = "embeddings"
    assert instrument(mock_openai_client, start_server=False).embeddings.create() == "embeddings"


def test_invalid_client_raises_error():
    invalid_client = MagicMock(spec=[])
    with pytest.raises(ValueError, match="Client must be an Anthropic, OpenAI, or OpenAI-compatible client"):
        instrument(invalid_client, start_server=False)


# HTTP server lifecycle

@patch("agentgauge.start_http_server")
def test_starts_server_on_first_call(mock_server, mock_anthropic_client):
    instrument(mock_anthropic_client, port=9999)
    mock_server.assert_called_once_with(9999)


@patch("agentgauge.start_http_server")
def test_starts_server_only_once(mock_server, mock_anthropic_client):
    instrument(mock_anthropic_client, port=9999)
    instrument(mock_anthropic_client, port=9999)
    mock_server.assert_called_once_with(9999)


@patch("agentgauge.start_http_server")
def test_start_server_false_skips(mock_server, mock_anthropic_client):
    instrument(mock_anthropic_client, start_server=False)
    mock_server.assert_not_called()
