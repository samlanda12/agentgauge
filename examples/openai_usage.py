import os
import time

from openai import OpenAI
from dotenv import load_dotenv

from agentgauge import instrument

load_dotenv()

# Standard OpenAI
client = instrument(OpenAI(api_key=os.getenv("OPENAI_API_KEY")), port=9464)

print("Making OpenAI API call...")
response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
    max_tokens=64,
    messages=[{"role": "user", "content": "Reply with exactly: hello world"}],
)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens — prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}")
print()
print("Metrics available at http://localhost:9464/metrics")
print("Press Ctrl+C to stop.")

while True:
    time.sleep(1)
