import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from prometheus_client import start_http_server

from agentgauge import AgentGaugeCallbackHandler

load_dotenv()

# Start metrics server
start_http_server(9464)

# Create the callback handler
handler = AgentGaugeCallbackHandler()

# Attach to any LangChain chat model
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    callbacks=[handler],
)

print("Making LangChain API call...")
response = llm.invoke("Reply with exactly: hello world")
print(f"Response: {response.content}")
print()
print("Metrics available at http://localhost:9464/metrics")
print("Press Ctrl+C to stop.")

while True:
    time.sleep(1)