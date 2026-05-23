"""Retry configuration and helpers for the adk-validated-loop pipeline.

Two retry profiles are provided:

- :data:`RATE_LIMIT_RETRY_CONFIG` — targets ``RateLimitError`` (HTTP 429).
  Uses a longer initial back-off to respect upstream rate-limit windows.

- :data:`DEFAULT_RETRY_CONFIG` — catches any exception.  Applied as a
  fallback on every sub-agent so that transient network / server errors
  are automatically retried.

Both configs are passed directly to each ``LlmAgent`` via the
``retry_config`` kwarg; the 429-specific one also requires the
:func:`on_rate_limit_error` callback to be wired in as
``on_model_error_callback`` on each agent.
"""

from __future__ import annotations

from typing import Optional

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.workflow import RetryConfig

from .exceptions import RateLimitError

# ---------------------------------------------------------------------------
# Retry profiles
# ---------------------------------------------------------------------------

#: Retry policy applied specifically when the API signals rate-limiting
#: (HTTP 429).  Five attempts with exponential back-off capped at 2 minutes.
RATE_LIMIT_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_delay=5.0,
    max_delay=120.0,
    backoff_factor=2.0,
    jitter=0.5,
    exceptions=["RateLimitError"],
)

#: General retry policy for transient failures (network blips, 5xx errors,
#: etc.).  Three attempts with a shorter back-off window.
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=1.0,
    exceptions=None,  # retry on *any* exception
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def on_rate_limit_error(
    ctx,  # google.adk.agents.context.Context
    request: LlmRequest,
    error: Exception,
) -> Optional[LlmResponse]:
    """Intercept HTTP 429 errors and re-raise as :class:`.RateLimitError`.

    This callback is attached to each sub-agent as ``on_model_error_callback``.
    When the underlying model call fails with a ``ClientError`` whose status
    code is 429, the callback raises :class:`.RateLimitError` instead.
    The node retry mechanism then sees ``RateLimitError`` in
    :data:`RATE_LIMIT_RETRY_CONFIG` and applies the patient back-off policy.

    For all other errors the callback returns ``None``, which causes the
    original exception to propagate normally (and may trigger
    :data:`DEFAULT_RETRY_CONFIG` at the node level).

    Args:
        ctx: The current agent :class:`~google.adk.agents.context.Context`.
        request: The :class:`~google.adk.models.llm_request.LlmRequest` that
            triggered the error.
        error: The exception raised by the model call.

    Returns:
        ``None`` to propagate non-429 errors unchanged.

    Raises:
        RateLimitError: When *error* is an HTTP 429 ``ClientError``.
    """
    try:
        from google.genai.errors import ClientError

        if isinstance(error, ClientError) and getattr(error, "code", None) == 429:
            raise RateLimitError(
                f"API rate limit exceeded (HTTP 429): {error}"
            ) from error
    except ImportError:
        pass  # google-genai not available; let the original error propagate

    return None
