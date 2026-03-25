from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict
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
    LLM_TOOL_DURATION_SECONDS,
)


class InvocationParams(TypedDict, total=False):
    """Model invocation parameters from LangChain callbacks.

    Passed in kwargs['invocation_params'] at call time.
    """

    model: str
    model_name: str
    _type: str


class SerializedLLM(TypedDict, total=False):
    """Serialized LangChain LLM configuration.

    Contains info from the LLM constructor, including model name.
    """

    name: str
    kwargs: Dict[str, Any]
    invocation_params: InvocationParams


def _extract_model(serialized: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> str:
    """Extract model name from LangChain serialized info and invocation kwargs.

    LangChain provides model info in two places:
    - 'kwargs["invocation_params"]': populated at call time (most reliable).
    - 'serialized["kwargs"]': populated from the LLM constructor arguments.
    """
    invocation_params: InvocationParams = kwargs.get("invocation_params") or {}
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

    All callback methods have async variants for use with async LangGraph workflows.
    The async methods delegate to the sync implementations to avoid code duplication.
    """

    def __init__(self) -> None:
        super().__init__()
        self._request_starts: Dict[str, float] = {}
        self._model_names: Dict[str, str] = {}
        self._tool_starts: Dict[str, float] = {}
        self._tool_names: Dict[str, str] = {}
        self._is_streaming: Dict[str, bool] = {}

    # Sync LLM callbacks

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
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
        self._is_streaming[key] = False
        LLM_ACTIVE_REQUESTS.labels(model=model).inc()

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
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
        self._is_streaming[key] = False
        LLM_ACTIVE_REQUESTS.labels(model=model).inc()

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a new token is streamed from an LLM.

        Marks the request as streaming so the method label can be set correctly
        in on_llm_end.
        """
        key = str(run_id)
        if key in self._is_streaming:
            self._is_streaming[key] = True

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

        is_streaming = self._is_streaming.pop(key, False)

        if key in self._request_starts:
            duration = time.monotonic() - self._request_starts.pop(key)
            method = "stream" if is_streaming else "invoke"
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method=method).observe(duration)

        LLM_ACTIVE_REQUESTS.labels(model=model).dec()
        method = "stream" if is_streaming else "invoke"
        LLM_REQUESTS_TOTAL.labels(model=model, method=method, status="ok").inc()
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

        is_streaming = self._is_streaming.pop(key, False)

        if key in self._request_starts:
            duration = time.monotonic() - self._request_starts.pop(key)
            method = "stream" if is_streaming else "invoke"
            LLM_REQUEST_DURATION_SECONDS.labels(model=model, method=method).observe(duration)

        LLM_ACTIVE_REQUESTS.labels(model=model).dec()
        method = "stream" if is_streaming else "invoke"
        LLM_REQUESTS_TOTAL.labels(model=model, method=method, status="error").inc()

    # Async LLM callbacks

    async def on_llm_start_async(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_llm_start."""
        self.on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_chat_model_start_async(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_chat_model_start."""
        self.on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_llm_new_token_async(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_llm_new_token."""
        self.on_llm_new_token(
            token,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )

    async def on_llm_end_async(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_llm_end."""
        self.on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )

    async def on_llm_error_async(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_llm_error."""
        self.on_llm_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )

    # Sync tool callbacks

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent tool is invoked.

        Tool calls inherit the model label from their parent LLM run when available.
        If the parent LLM run is not tracked or has already completed, model defaults
        to "unknown".
        """
        tool_name = (serialized or {}).get("name", "unknown")
        key = str(run_id)
        self._tool_starts[key] = time.monotonic()
        self._tool_names[key] = tool_name
        # Look up model from parent LLM run if available
        model = self._model_names.get(str(parent_run_id), "unknown") if parent_run_id else "unknown"
        LLM_TOOL_CALLS_TOTAL.labels(model=model, tool_name=tool_name).inc()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent tool finishes successfully."""
        key = str(run_id)
        tool_name = self._tool_names.pop(key, None)
        if tool_name is None:
            return

        if key in self._tool_starts:
            duration = time.monotonic() - self._tool_starts.pop(key)
            LLM_TOOL_DURATION_SECONDS.labels(tool_name=tool_name).observe(duration)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent tool raises an error."""
        key = str(run_id)
        tool_name = self._tool_names.pop(key, None)
        if tool_name is None:
            return

        if key in self._tool_starts:
            duration = time.monotonic() - self._tool_starts.pop(key)
            LLM_TOOL_DURATION_SECONDS.labels(tool_name=tool_name).observe(duration)

    # Async tool callbacks

    async def on_tool_start_async(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_tool_start."""
        self.on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_tool_end_async(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_tool_end."""
        self.on_tool_end(
            output,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )

    async def on_tool_error_async(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_tool_error."""
        self.on_tool_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )
