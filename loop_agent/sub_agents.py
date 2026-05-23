"""Five LlmAgent sub-agents that make up the validated-loop pipeline.

Each agent is constructed via :func:`_make_agent`, which wires in:

- ``on_model_error_callback`` → converts HTTP 429 responses to
  :class:`.RateLimitError` so that the dedicated rate-limit retry policy
  fires instead of the generic one.
- ``retry_config`` → :data:`.RATE_LIMIT_RETRY_CONFIG` (targets 429 errors).

Pipeline order
--------------
1. ``input_formatter_agent``   – normalises and re-states the user request.
2. ``research_agent``          – gathers relevant context / facts.
3. ``drafting_agent``          – produces an initial response draft.
4. ``validation_agent``        – checks the draft; *escalates* when satisfied.
5. ``revision_agent``          – rewrites the draft if validation rejected it.

The loop continues until ``validation_agent`` escalates or
``max_iterations`` is reached.
"""

from __future__ import annotations

import os

from google.adk import Agent

from .retry import RATE_LIMIT_RETRY_CONFIG, on_rate_limit_error

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gemini-2.0-flash")


def _make_agent(name: str, description: str, instruction: str) -> Agent:
    """Return an ``Agent`` pre-configured with rate-limit retry handling.

    Args:
        name: Unique agent name (used internally by ADK).
        description: Short description of the agent's purpose.
        instruction: System prompt / instruction for the LLM.

    Returns:
        A :class:`~google.adk.Agent` instance ready for use as a sub-agent.
    """
    return Agent(
        name=name,
        model=_DEFAULT_MODEL,
        description=description,
        instruction=instruction,
        on_model_error_callback=on_rate_limit_error,
        retry_config=RATE_LIMIT_RETRY_CONFIG,
    )


# ---------------------------------------------------------------------------
# The five sub-agents
# ---------------------------------------------------------------------------

input_formatter_agent: Agent = _make_agent(
    name="input_formatter_agent",
    description="Normalises and re-states the user request for downstream agents.",
    instruction=(
        "You are the first stage of a multi-step pipeline.\n"
        "Your job is to read the user's message and re-state it clearly and "
        "concisely in a single paragraph, removing any ambiguity. "
        "Do not answer the question yet — only reformulate it.\n"
        "Output your reformulation under the heading 'TASK:'."
    ),
)

research_agent: Agent = _make_agent(
    name="research_agent",
    description="Gathers relevant context and key facts needed to answer the task.",
    instruction=(
        "You are the research stage of a multi-step pipeline.\n"
        "Read the TASK from the conversation history and list the key facts, "
        "concepts, or context points needed to answer it well. "
        "Be factual and concise; do not write the final answer yet.\n"
        "Output your findings under the heading 'RESEARCH:'."
    ),
)

drafting_agent: Agent = _make_agent(
    name="drafting_agent",
    description="Produces an initial response draft based on the research.",
    instruction=(
        "You are the drafting stage of a multi-step pipeline.\n"
        "Using the TASK and RESEARCH from the conversation history, write a "
        "complete, well-structured answer. Aim for clarity and accuracy.\n"
        "Output your draft under the heading 'DRAFT:'."
    ),
)

validation_agent: Agent = _make_agent(
    name="validation_agent",
    description=(
        "Reviews the draft and escalates when it meets quality criteria; "
        "otherwise records specific issues for revision."
    ),
    instruction=(
        "You are the quality-validation stage of a multi-step pipeline.\n"
        "Review the DRAFT from the conversation history against these criteria:\n"
        "  1. Directly answers the TASK.\n"
        "  2. Is factually consistent with the RESEARCH.\n"
        "  3. Is clear, complete, and free of obvious errors.\n\n"
        "If ALL criteria are met:\n"
        "  - Output 'VALIDATION: PASS'\n"
        "  - Then use the ``escalate`` transfer to signal the loop is done.\n\n"
        "If any criterion fails:\n"
        "  - Output 'VALIDATION: FAIL'\n"
        "  - List each issue under 'ISSUES:' so the revision agent can fix them.\n"
        "  - Do NOT escalate."
    ),
)

revision_agent: Agent = _make_agent(
    name="revision_agent",
    description="Rewrites the draft to address issues identified by the validation agent.",
    instruction=(
        "You are the revision stage of a multi-step pipeline.\n"
        "The validation agent found issues with the current DRAFT. "
        "Read the ISSUES listed in the conversation history and produce an "
        "improved draft that resolves every issue.\n"
        "Output the improved version under the heading 'DRAFT:' so it "
        "replaces the previous draft in the next validation pass."
    ),
)
