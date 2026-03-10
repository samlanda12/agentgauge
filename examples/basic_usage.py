import os
import time

import anthropic
from dotenv import load_dotenv

from agentgauge import instrument

load_dotenv()

client = instrument(anthropic.Anthropic(), port=9464)

print("Making API call...")
response = client.messages.create(
    model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
    max_tokens=64,
    messages=[{"role": "user", "content": "Reply with exactly: hello world"}],
)
print(f"Response: {response.content[0].text}")
print(f"Tokens — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")
print()
print("Metrics available at http://localhost:9464/metrics")
print("Press Ctrl+C to stop.")

while True:
    time.sleep(1)
