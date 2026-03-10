from __future__ import annotations

import threading
from typing import Any

from prometheus_client import start_http_server

from .wrapper import InstrumentedMessages

__version__ = "0.1.0"

_server_started = False
_server_lock = threading.Lock()

DEFAULT_PORT = 9464

class InstrumentedClient:
    """Proxy around an Anthropic client that swaps in instrumented messages."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._messages = InstrumentedMessages(client.messages)

    @property
    def messages(self) -> InstrumentedMessages:
        return self._messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def instrument(client: Any, *, port: int = DEFAULT_PORT, start_server: bool = True) -> InstrumentedClient:
    """Wrap an Anthropic client with Prometheus instrumentation.

    Args:
        client: An "anthropic.Anthropic" instance.
        port: Port for the "/metrics" HTTP endpoint. Defaults to 9464.
        start_server: Whether to auto-start the metrics HTTP server.
            Set to "False" in tests or when exposing metrics through
            an existing WSGI/ASGI app.

    Returns:
        An instrumented client that records metrics on every API call.
    """
    global _server_started

    if start_server:
        with _server_lock:
            if not _server_started:
                start_http_server(port)
                _server_started = True

    return InstrumentedClient(client)
