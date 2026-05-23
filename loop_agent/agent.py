"""Root agent definition for the adk-validated-loop pipeline.

This module exposes ``root_agent``, an ADK :class:`~google.adk.agents.LoopAgent`
that orchestrates five sub-agents in a validate-and-revise loop:

    input_formatter → research → drafting → validation → revision
                         ↑                        |
                         └─── (loop back) ────────┘

The loop terminates when:

- ``validation_agent`` escalates (quality criteria are met), **or**
- ``max_iterations`` is reached (safety cap).

.. deprecated::
    :class:`~google.adk.agents.LoopAgent` is deprecated in ADK 2.x in favour
    of :class:`~google.adk.Workflow` with cyclic edges.  It is used here
    because it is the API explicitly referenced in the project description.
    A future migration path is shown in the README.

Usage::

    from loop_agent.agent import root_agent
"""

from __future__ import annotations

import warnings

from .sub_agents import (
    drafting_agent,
    input_formatter_agent,
    research_agent,
    revision_agent,
    validation_agent,
)

# LoopAgent is deprecated; suppress the warning so downstream code that imports
# this module does not receive spurious deprecation noise.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.agents import LoopAgent

    #: The root agent for the pipeline.
    #:
    #: Pass this to :class:`~google.adk.Runner` or use ``adk run`` / ``adk web``
    #: to interact with it:
    #:
    #: .. code-block:: bash
    #:
    #:     adk run loop_agent
    root_agent: LoopAgent = LoopAgent(
        name="validated_loop_agent",
        description=(
            "A five-stage validate-and-revise pipeline that iterates until "
            "the drafted answer passes all quality checks or the iteration "
            "cap is reached."
        ),
        sub_agents=[
            input_formatter_agent,
            research_agent,
            drafting_agent,
            validation_agent,
            revision_agent,
        ],
        max_iterations=5,
    )
