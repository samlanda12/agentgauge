from prometheus_client import Counter, Gauge, Histogram

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total number of LLM API requests",
    ["model", "method", "status"],
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "LLM API request duration in seconds",
    ["model", "method"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total number of LLM tokens consumed",
    ["model", "token_type"],
)

LLM_ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of in-flight LLM API requests",
    ["model"],
)

LLM_TOOL_CALLS_TOTAL = Counter(
    "llm_tool_calls_total",
    "Total number of tool calls made by the LLM",
    ["model", "tool_name"],
)
