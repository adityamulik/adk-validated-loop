"""Custom exceptions for the adk-validated-loop pipeline."""

from __future__ import annotations


class RateLimitError(Exception):
    """Raised when the upstream API returns HTTP 429 Too Many Requests.

    By subclassing :class:`Exception` (not ``ClientError``), this exception
    can be used as a precise target in :class:`~google.adk.workflow.RetryConfig`
    without accidentally silencing unrelated client-side errors.

    The ``on_model_error_callback`` attached to each sub-agent detects a 429
    ``ClientError`` and re-raises it as ``RateLimitError`` so that the node
    retry mechanism can apply a dedicated, more-patient backoff policy.
    """
