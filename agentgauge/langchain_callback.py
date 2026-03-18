from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as e:
    raise ImportError(
        "langchain-core is required to use AgentGaugeCallbackHandler. "
        "Install it with: pip install agentgauge[langchain]"
    ) from e

from .metrics import (
    LLM_ACTIVE_REQUESTS,
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    LLM_TOOL_CALLS_TOTAL,
)


def _extract_model(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
    """Extract model name from LangChain serialized info and invocation kwargs.

    LangChain provides model info in two places:
    - 'kwargs["invocation_params"]': populated at call time (most reliable).
    - 'serialized["kwargs"]': populated from the LLM constructor arguments.
    """
    invocation_params = kwargs.get("invocation_params") or {}
    for key in ("model", "model_name"):
        val = invocation_params.get(key)
        if val and isinstance(val, str):
            return val

    serialized_kwargs = (serialized or {}).get("kwargs") or {}
    for key in ("model", "model_name"):
        val = serialized_kwargs.get(key)
        if val and isinstance(val, str):
            return val

    return "unknown"


def _record_token_usage(result: LLMResult, model: str) -> None:
    """Record token usage from an LLMResult.

    Handles both OpenAI-style ('token_usage') and Anthropic-style ('usage')
    output formats found in 'LLMResult.llm_output'.
    """
    llm_output = result.llm_output or {}

    # OpenAI-style: {"token_usage": {"prompt_tokens": N, "completion_tokens": N}}
    token_usage = llm_output.get("token_usage") or {}
    if token_usage:
        prompt = token_usage.get("prompt_tokens")
        completion = token_usage.get("completion_tokens")
        if isinstance(prompt, int):
            LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(prompt)
        if isinstance(completion, int):
            LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(completion)
        return

    # Anthropic-style: {"usage": {"input_tokens": N, "output_tokens": N}}
    usage = llm_output.get("usage") or {}
    if usage:
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int):
            LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(input_tokens)
        if isinstance(output_tokens, int):
            LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(output_tokens)


class AgentGaugeCallbackHandler(BaseCallbackHandler):
    """LangChain/LangGraph callback handler that records Prometheus metrics.

    Supports both LangChain chains and LangGraph workflows. Attach to any
    LangChain-compatible LLM to capture request count, latency, token usage,
    and tool calls as native Prometheus metrics.
    """

    def __init__(self) -> None:
        super().__init__()
        self._request_starts: Dict[str, float] = {}
        self._model_names: Dict[str, str] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a text LLM starts a request."""
        model = _extract_model(serialized, kwargs)
        key = str(run_id)
        self._request_starts[key] = time.monotonic()
        self._model_names[key] = model
        LLM_ACTIVE_REQUESTS.labels(model=model).inc()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts a request (ChatOpenAI, ChatAnthropic, etc.)."""
        model = _extract_model(serialized, kwargs)
        key = str(run_id)
        self._request_starts[key] = time.monotonic()
        self._model_names[key] = model
        LLM_ACTIVE_REQUESTS.labels(model=model).inc()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when any LLM or chat model finishes successfully."""
        key = str(run_id)
        model = self._model_names.pop(key, None)
        if model is None:
            return

        if key in self._request_starts:
            duration = time.monotonic() - self._request_starts.pop(key)
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method="invoke").observe(duration)

        LLM_ACTIVE_REQUESTS.labels(model=model).dec()
        LLM_REQUESTS_TOTAL.labels(model=model, method="invoke", status="ok").inc()
        _record_token_usage(response, model)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when any LLM or chat model raises an error."""
        key = str(run_id)
        model = self._model_names.pop(key, None)
        if model is None:
            return

        if key in self._request_starts:
            duration = time.monotonic() - self._request_starts.pop(key)
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method="invoke").observe(duration)

        LLM_ACTIVE_REQUESTS.labels(model=model).dec()
        LLM_REQUESTS_TOTAL.labels(model=model, method="invoke", status="error").inc()

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent tool is invoked.

        Tool calls are labeled with model="unknown" because the LangChain
        callback system does not associate tool invocations with a specific
        model at this hook level.
        """
        tool_name = (serialized or {}).get("name", "unknown")
        LLM_TOOL_CALLS_TOTAL.labels(model="unknown", tool_name=tool_name).inc()
