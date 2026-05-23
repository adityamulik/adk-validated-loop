"""Structural and unit tests for the adk-validated-loop pipeline.

These tests validate the *configuration* of the pipeline without making
live LLM API calls.  They verify that:

- All five sub-agents are created and have the expected names.
- Every sub-agent has the rate-limit retry policy attached.
- Every sub-agent has the ``on_model_error_callback`` wired in.
- The ``LoopAgent`` is configured with the correct sub-agent order and cap.
- The ``RateLimitError`` is raised correctly by :func:`on_rate_limit_error`.
- The :data:`RATE_LIMIT_RETRY_CONFIG` and :data:`DEFAULT_RETRY_CONFIG` have
  the expected parameters.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from loop_agent.exceptions import RateLimitError
from loop_agent.retry import (
    DEFAULT_RETRY_CONFIG,
    RATE_LIMIT_RETRY_CONFIG,
    on_rate_limit_error,
)
from loop_agent.sub_agents import (
    drafting_agent,
    input_formatter_agent,
    research_agent,
    revision_agent,
    validation_agent,
)
from loop_agent.agent import root_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_AGENT_NAMES = [
    "input_formatter_agent",
    "research_agent",
    "drafting_agent",
    "validation_agent",
    "revision_agent",
]


# ---------------------------------------------------------------------------
# RateLimitError
# ---------------------------------------------------------------------------


class TestRateLimitError:
    def test_is_exception_subclass(self):
        assert issubclass(RateLimitError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(RateLimitError, match="boom"):
            raise RateLimitError("boom")


# ---------------------------------------------------------------------------
# on_rate_limit_error callback
# ---------------------------------------------------------------------------


class TestOnRateLimitErrorCallback:
    """Tests for the model error callback that converts 429 → RateLimitError."""

    def _make_client_error(self, code: int):
        """Create a minimal fake ClientError with the given HTTP status code."""

        class _FakeClientError(Exception):
            pass

        err = _FakeClientError(f"HTTP {code}")
        err.code = code  # type: ignore[attr-defined]

        # Patch isinstance check to work with google.genai.errors.ClientError
        try:
            from google.genai.errors import ClientError

            class _Err(ClientError):
                def __init__(self, code_: int):
                    Exception.__init__(self, f"HTTP {code_}")
                    self.code = code_
                    self.response = None

            return _Err(code)
        except Exception:
            return err

    def test_429_raises_rate_limit_error(self):
        err = self._make_client_error(429)
        with pytest.raises(RateLimitError):
            on_rate_limit_error(None, None, err)

    def test_non_429_returns_none(self):
        err = self._make_client_error(500)
        result = on_rate_limit_error(None, None, err)
        assert result is None

    def test_non_client_error_returns_none(self):
        result = on_rate_limit_error(None, None, ValueError("something else"))
        assert result is None


# ---------------------------------------------------------------------------
# RetryConfig instances
# ---------------------------------------------------------------------------


class TestRetryConfigs:
    def test_rate_limit_config_targets_rate_limit_error(self):
        assert RATE_LIMIT_RETRY_CONFIG.exceptions is not None
        assert "RateLimitError" in RATE_LIMIT_RETRY_CONFIG.exceptions

    def test_rate_limit_config_max_attempts(self):
        assert RATE_LIMIT_RETRY_CONFIG.max_attempts == 5

    def test_rate_limit_config_initial_delay(self):
        assert RATE_LIMIT_RETRY_CONFIG.initial_delay == 5.0

    def test_default_config_retries_all_exceptions(self):
        assert DEFAULT_RETRY_CONFIG.exceptions is None

    def test_default_config_max_attempts(self):
        assert DEFAULT_RETRY_CONFIG.max_attempts == 3


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------


class TestSubAgents:
    @pytest.mark.parametrize(
        "agent, expected_name",
        [
            (input_formatter_agent, "input_formatter_agent"),
            (research_agent, "research_agent"),
            (drafting_agent, "drafting_agent"),
            (validation_agent, "validation_agent"),
            (revision_agent, "revision_agent"),
        ],
    )
    def test_agent_name(self, agent, expected_name):
        assert agent.name == expected_name

    @pytest.mark.parametrize(
        "agent",
        [
            input_formatter_agent,
            research_agent,
            drafting_agent,
            validation_agent,
            revision_agent,
        ],
    )
    def test_agent_has_rate_limit_retry_config(self, agent):
        assert agent.retry_config is not None
        assert agent.retry_config.max_attempts == RATE_LIMIT_RETRY_CONFIG.max_attempts
        assert (
            agent.retry_config.exceptions == RATE_LIMIT_RETRY_CONFIG.exceptions
        )

    @pytest.mark.parametrize(
        "agent",
        [
            input_formatter_agent,
            research_agent,
            drafting_agent,
            validation_agent,
            revision_agent,
        ],
    )
    def test_agent_has_on_model_error_callback(self, agent):
        assert agent.on_model_error_callback is not None

    @pytest.mark.parametrize(
        "agent",
        [
            input_formatter_agent,
            research_agent,
            drafting_agent,
            validation_agent,
            revision_agent,
        ],
    )
    def test_agent_instruction_is_non_empty(self, agent):
        assert isinstance(agent.instruction, str) and len(agent.instruction) > 0


# ---------------------------------------------------------------------------
# root_agent (LoopAgent)
# ---------------------------------------------------------------------------


class TestRootAgent:
    def test_root_agent_name(self):
        assert root_agent.name == "validated_loop_agent"

    def test_root_agent_has_five_sub_agents(self):
        assert len(root_agent.sub_agents) == 5

    def test_root_agent_sub_agent_names_in_order(self):
        names = [a.name for a in root_agent.sub_agents]
        assert names == _EXPECTED_AGENT_NAMES

    def test_root_agent_max_iterations(self):
        assert root_agent.max_iterations == 5

    def test_root_agent_is_loop_agent(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from google.adk.agents import LoopAgent

        assert isinstance(root_agent, LoopAgent)
