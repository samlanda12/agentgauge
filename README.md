# agentgauge

Lightweight Prometheus metrics exporter for AI agent pipelines. Wraps LLM client SDKs to expose token usage, latency, and tool calls as native Prometheus metrics. No OpenTelemetry or other external infrastructure required.

Supports:
- **Anthropic** (`anthropic.Anthropic`)
- **OpenAI** and **OpenAI-compatible providers** (OpenRouter, Together, Groq, etc.)

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

## Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `llm_requests_total` | Counter | `model`, `method`, `status` |
| `llm_request_duration_seconds` | Histogram | `model`, `method` |
| `llm_tokens_total` | Counter | `model`, `token_type` |
| `llm_active_requests` | Gauge | `model` |
| `llm_tool_calls_total` | Counter | `model`, `tool_name` |

## Common Queries

| Query | Description |
|-------|-------------|
| `rate(llm_tokens_total[1h])` | Token usage over time |
| `llm_active_requests` | Current active requests |
| `sum(rate(llm_requests_total{status="error"}[5m])) / sum(rate(llm_requests_total[5m]))` | Error rate |
| `histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))` | p95 latency |

## Prometheus config

See `prometheus.yml` for example configuration.
