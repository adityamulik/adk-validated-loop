"""Entry point for running the validated-loop pipeline from the CLI.

Usage
-----
Interactive session::

    python main.py

Non-interactive (pipe a single prompt)::

    echo "Explain transformer attention" | python main.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


async def _run(prompt: str) -> None:
    """Execute one turn of the pipeline and print the final output."""
    import google.genai as genai
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService

    from loop_agent.agent import root_agent

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "Error: GOOGLE_API_KEY is not set.\n"
            "Copy .env.example → .env and add your key."
        )

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="adk-validated-loop",
        user_id="cli-user",
    )

    runner = Runner(
        agent=root_agent,
        app_name="adk-validated-loop",
        session_service=session_service,
    )

    user_content = genai.types.Content(
        role="user",
        parts=[genai.types.Part(text=prompt)],
    )

    print(f"\n>>> {prompt}\n")
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=user_content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


def main() -> None:
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        prompt = input("Enter your prompt: ").strip()

    if not prompt:
        sys.exit("No prompt provided.")

    asyncio.run(_run(prompt))


if __name__ == "__main__":
    main()
