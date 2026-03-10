from __future__ import annotations

import threading
from typing import Any

from prometheus_client import start_http_server

from .anthropic_wrapper import InstrumentedMessages
from .openai_wrapper import InstrumentedChatCompletion

__version__ = "0.1.0"

_server_started = False
_server_lock = threading.Lock()

DEFAULT_PORT = 9464

class InstrumentedAnthropicClient:
    """Proxy around an Anthropic client that swaps in instrumented messages."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._messages = InstrumentedMessages(client.messages)

    @property
    def messages(self) -> InstrumentedMessages:
        return self._messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class InstrumentedOpenAIClient:
    """Proxy around an OpenAI or OpenAI-compatible client that swaps in instrumented chat completions."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._chat = InstrumentedChatCompletionProxy(client.chat)

    @property
    def chat(self) -> InstrumentedChatCompletionProxy:
        return self._chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class InstrumentedChatCompletionProxy:
    """Proxy for the chat.completions resource."""

    def __init__(self, chat: Any) -> None:
        self._chat = chat
        self._completions = InstrumentedChatCompletion(chat.completions)

    @property
    def completions(self) -> InstrumentedChatCompletion:
        return self._completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


def _is_anthropic_client(client: Any) -> bool:
    """Check if client is an Anthropic client."""
    # Check class name and module first for clarity
    class_name = type(client).__name__
    module = type(client).__module__
    if "Anthropic" in class_name or "anthropic" in module:
        return True
    # Fallback: check for messages.create without OpenAI in class name
    return (
        hasattr(client, "messages")
        and hasattr(client.messages, "create")
        and "OpenAI" not in class_name
    )


def _is_openai_client(client: Any) -> bool:
    """Check if client is an OpenAI or OpenAI-compatible client."""
    # Check class name and module first for clarity
    class_name = type(client).__name__
    module = type(client).__module__
    if "OpenAI" in class_name or "openai" in module:
        return True
    # Fallback: check for chat.completions.create without Anthropic in class name
    return (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "create")
        and "Anthropic" not in class_name
    )


def instrument(
    client: Any, *, port: int = DEFAULT_PORT, start_server: bool = True
) -> InstrumentedAnthropicClient | InstrumentedOpenAIClient:
    """Wrap an LLM client with Prometheus instrumentation.

    Supports Anthropic, OpenAI, and OpenAI-compatible clients.

    Args:
        client: An "anthropic.Anthropic", "openai.OpenAI", or OpenAI-compatible client instance.
        port: Port for the "/metrics" HTTP endpoint. Defaults to 9464.
        start_server: Whether to auto-start the metrics HTTP server.
            Set to "False" in tests or when exposing metrics through
            an existing WSGI/ASGI app.

    Returns:
        An instrumented client that records metrics on every API call.

    Raises:
        ValueError: If client type cannot be determined.
    """
    global _server_started

    if start_server:
        with _server_lock:
            if not _server_started:
                start_http_server(port)
                _server_started = True

    # Check OpenAI first since it's more specific (chat.completions vs messages)
    if _is_openai_client(client):
        return InstrumentedOpenAIClient(client)
    elif _is_anthropic_client(client):
        return InstrumentedAnthropicClient(client)
    else:
        raise ValueError(
            "Client must be an Anthropic, OpenAI, or OpenAI-compatible client. "
            f"Got {type(client).__name__}"
        )
