import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from agentgauge import AgentGaugeCallbackHandler
from prometheus_client import start_http_server

load_dotenv()

# Start metrics server
start_http_server(9464)

# Create the callback handler
handler = AgentGaugeCallbackHandler()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# Define a simple tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulated weather data
    return f"The weather in {location} is sunny and 72°F"


# Create a LangGraph agent
agent = create_react_agent(llm, tools=[get_weather])

# Pass the callback via RunnableConfig so it propagates to all nodes in the
# graph (LLM calls AND tool calls). Attaching only to the LLM constructor
# will miss tool-node callbacks.
config = RunnableConfig(callbacks=[handler])

print("Making LangGraph agent call...")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)
print(f"Response: {result['messages'][-1].content}")
print()
print("Metrics available at http://localhost:9464/metrics")
print("Press Ctrl+C to stop.")

while True:
    time.sleep(1)