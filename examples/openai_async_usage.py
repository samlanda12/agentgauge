import asyncio
import os

from openai import AsyncOpenAI
from dotenv import load_dotenv

from agentgauge import instrument

load_dotenv()


async def main() -> None:
    client = instrument(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")), port=9464)

    # create() call
    print("Making async create() call...")
    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly: hello world"}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens — prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}")
    print()

    # stream() call
    print("Making async stream() call...")
    stream = await client.chat.completions.stream(
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        max_tokens=64,
        messages=[{"role": "user", "content": "Count to five, one word per line."}],
    )
    async with stream as s:
        async for chunk in s:
            pass  # consume the stream
    print("Stream completed (tokens tracked via stream_options)")
    print()

    print("Metrics available at http://localhost:9464/metrics")
    print("Press Ctrl+C to stop.")
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
