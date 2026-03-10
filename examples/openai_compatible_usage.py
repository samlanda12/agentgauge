import os
import time

from openai import OpenAI
from dotenv import load_dotenv

from agentgauge import instrument

load_dotenv()

# Example: OpenAI-compatible provider (OpenRouter, Together, Groq, etc.)
# Set PROVIDER_BASE_URL and PROVIDER_API_KEY in your .env file
# Examples:
#   OpenRouter: https://openrouter.ai/api/v1
#   Together: https://api.together.xyz/v1
#   Groq: https://api.groq.com/openai/v1

client = instrument(
    OpenAI(
        api_key=os.getenv("PROVIDER_API_KEY"),
        base_url=os.getenv("PROVIDER_BASE_URL"),
    ),
    port=9464
)

print(f"Making API call to {os.getenv('PROVIDER_BASE_URL', 'OpenAI')}...")
response = client.chat.completions.create(
    model=os.getenv("PROVIDER_MODEL", "claude-haiku-4.5"),
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
