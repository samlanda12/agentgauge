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

## Cost Estimation

agentgauge exports token counts by model, so you can compute costs in Grafana or
PromQL using your own per-token rates. This keeps pricing accurate to your actual
contracts and avoids bundling data that goes stale.

Example recording rule (`prometheus.rules.yml`):

```yaml
groups:
  - name: llm_cost
    rules:
      - record: llm_cost_per_second
        # Map each model to its $/token rate directly in PromQL
        expr: |
          (
              rate(llm_tokens_total{token_type="input"}[5m])
            * on(model) group_left
              (
                  (label_replace(vector(0.000003), "model", "claude-sonnet-4-20250514", "", ""))
                or
                  (label_replace(vector(0.0000025), "model", "gpt-4o", "", ""))
              )
          )
          +
          (
              rate(llm_tokens_total{token_type="output"}[5m])
            * on(model) group_left
              (
                  (label_replace(vector(0.000015), "model", "claude-sonnet-4-20250514", "", ""))
                or
                  (label_replace(vector(0.00001), "model", "gpt-4o", "", ""))
              )
          )
```

Or for simpler setups, compute cost in a Grafana panel transformation.

## Prometheus config

See `prometheus.yml` for example configuration.
