# agentgauge

Lightweight Prometheus metrics exporter for AI agent pipelines. Wraps LLM client SDKs to expose token usage, latency, and tool calls as native Prometheus metrics. No OpenTelemetry or other external infrastructure required.

Supports:
- **Anthropic** (`anthropic.Anthropic`, `anthropic.AsyncAnthropic`)
- **OpenAI** and **OpenAI-compatible providers** (`openai.OpenAI`, `openai.AsyncOpenAI`, OpenRouter, Together, Groq, etc.)
- **LangChain** and **LangGraph** (via callback handler; no client wrapping)

## Install

```bash
pip install agentgauge
```

## Usage

### Anthropic

```python
import anthropic
from agentgauge import instrument

client = instrument(anthropic.Anthropic())

# Use client exactly as you would normally
response = client.messages.create(...)

# Metrics available at http://localhost:9464/metrics
```

Async clients are supported with the same API:

```python
import anthropic
from agentgauge import instrument

client = instrument(anthropic.AsyncAnthropic())

response = await client.messages.create(...)

async with client.messages.stream(...) as stream:
    async for event in stream:
        ...
```

### OpenAI and OpenAI-compatible providers

Works with OpenAI and any provider using the OpenAI SDK:

```python
from openai import OpenAI
from agentgauge import instrument

# Standard OpenAI
client = instrument(OpenAI())

# Or with an OpenAI-compatible provider (OpenRouter, Together, Groq, etc.)
client = instrument(
    OpenAI(
        api_key="your-api-key",
        base_url="https://your-provider.com/v1",
    )
)

response = client.chat.completions.create(...)

# Metrics available at http://localhost:9464/metrics
```

Async clients are supported with the same API:

```python
from openai import AsyncOpenAI
from agentgauge import instrument

client = instrument(AsyncOpenAI())

response = await client.chat.completions.create(...)

stream = await client.chat.completions.stream(...)
async with stream as s:
    async for chunk in s:
        ...
```

### LangChain

```bash
pip install agentgauge[langchain] langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from agentgauge import agentgaugeCallbackHandler

handler = agentgaugeCallbackHandler()

llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

response = llm.invoke("Hello!")
```

### LangGraph

Uses the same callback handler as LangChain. Just pass it via `RunnableConfig` when invoking the agent. This ensures it propagates to all graph nodes; LLM calls and tool calls. Attaching only to the LLM constructor will miss tool-node callbacks.

```bash
pip install agentgauge[langchain] langchain-openai langgraph
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from agentgauge import agentgaugeCallbackHandler

handler = agentgaugeCallbackHandler()

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, tools=[...])

config = RunnableConfig(callbacks=[handler])
result = agent.invoke({"messages": [...]}, config=config)
```

## Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `llm_requests_total` | Counter | `model`, `method`, `status` |
| `llm_request_duration_seconds` | Histogram | `model`, `method` |
| `llm_tokens_total` | Counter | `model`, `token_type` |
| `llm_active_requests` | Gauge | `model` |
| `llm_tool_calls_total` | Counter | `model`, `tool_name` |
| `llm_tool_duration_seconds` | Histogram | `tool_name` |
| `llm_cache_tokens_total` | Counter | `model`, `cache_type` |

> **Note:** `llm_tool_duration_seconds` is only available for LangChain/LangGraph workflows where tool execution is tracked via callbacks. Direct SDK wrappers (Anthropic, OpenAI) only see tool calls in the LLM response, not the actual tool execution.

## Common Queries

| Query | Description |
|-------|-------------|
| `rate(llm_tokens_total[1h])` | Token usage over time |
| `llm_active_requests` | Current active requests |
| `sum(rate(llm_requests_total{status="error"}[5m])) / sum(rate(llm_requests_total[5m]))` | Error rate |
| `histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))` | p95 latency |

## Prometheus config

See `prometheus.yml` for example configuration.
