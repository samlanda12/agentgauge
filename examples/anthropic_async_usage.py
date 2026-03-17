import asyncio
import os
import time

import anthropic
from dotenv import load_dotenv

from agentgauge import instrument

load_dotenv()


async def main() -> None:
    client = instrument(anthropic.AsyncAnthropic(), port=9464)

    # create() call
    print("Making async create() call...")
    response = await client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly: hello world"}],
    )
    print(f"Response: {response.content[0].text}")
    print(f"Tokens — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")
    print()

    # stream() call
    print("Making async stream() call...")
    async with client.messages.stream(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        max_tokens=64,
        messages=[{"role": "user", "content": "Count to five, one word per line."}],
    ) as stream:
        async for event in stream:
            pass  # consume the stream
        final = await stream.get_final_message()
    print(f"Stream final message stop reason: {final.stop_reason}")
    print(f"Tokens — input: {final.usage.input_tokens}, output: {final.usage.output_tokens}")
    print()

    print("Metrics available at http://localhost:9464/metrics")
    print("Press Ctrl+C to stop.")
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
