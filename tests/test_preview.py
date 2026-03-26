"""Unit tests for tone preview generation.

Minimum 5 test cases covering preview generation with mocked LLM,
error handling, and output format validation.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from opinionforge.core.preview import generate_preview, LLMClient
from opinionforge.models.config import StanceConfig
from opinionforge.models.topic import TopicContext


def _make_topic() -> TopicContext:
    """Create a sample TopicContext for testing."""
    return TopicContext(
        raw_input="The decline of local journalism",
        input_type="text",
        title="The Decline of Local Journalism in Rural America",
        summary="Local newspapers are closing at an alarming rate, leaving communities without essential coverage.",
        key_claims=["60% of rural counties have lost their local newspaper"],
        key_entities=["Rural America"],
        subject_domain="politics",
    )


class MockLLMClient:
    """Mock LLM client that returns predictable responses."""

    def __init__(self, response: str = "This is a preview sentence. It captures the tone.") -> None:
        self._response = response

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Return the pre-configured response."""
        return self._response


class FailingLLMClient:
    """Mock LLM client that raises an exception."""

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Always raise an API error."""
        raise Exception("API rate limit exceeded")


class TestGeneratePreview:
    """Tests for generate_preview function."""

    def test_preview_returns_string(self) -> None:
        """Preview generation with mocked LLM returns a string."""
        topic = _make_topic()
        client = MockLLMClient()
        result = generate_preview(
            topic=topic,
            voice_prompt="Write with sharp analytical prose.",
            spectrum=StanceConfig(position=0),
            client=client,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_preview_uses_injected_client(self) -> None:
        """Preview uses the injected LLM client, not a real API."""
        topic = _make_topic()
        expected_text = "Custom preview response for testing."
        client = MockLLMClient(response=expected_text)
        result = generate_preview(
            topic=topic,
            voice_prompt="Write with vigor.",
            spectrum=StanceConfig(position=-50),
            client=client,
        )
        assert result == expected_text

    def test_preview_handles_api_error_gracefully(self) -> None:
        """Preview raises RuntimeError with clear message on API failure."""
        topic = _make_topic()
        client = FailingLLMClient()
        with pytest.raises(RuntimeError, match="Failed to generate preview"):
            generate_preview(
                topic=topic,
                voice_prompt="Write analytically.",
                spectrum=StanceConfig(position=0),
                client=client,
            )

    def test_preview_passes_voice_prompt_as_system(self) -> None:
        """The voice_prompt is passed as the system prompt to the LLM."""
        topic = _make_topic()
        spy_client = MagicMock(spec=LLMClient)
        spy_client.generate.return_value = "Preview text."

        generate_preview(
            topic=topic,
            voice_prompt="UNIQUE_SYSTEM_PROMPT_MARKER",
            spectrum=StanceConfig(position=0),
            client=spy_client,
        )

        call_args = spy_client.generate.call_args
        assert call_args.kwargs.get("system_prompt") == "UNIQUE_SYSTEM_PROMPT_MARKER" or \
               call_args[0][0] == "UNIQUE_SYSTEM_PROMPT_MARKER"  # positional fallback

    def test_preview_includes_topic_in_user_prompt(self) -> None:
        """The user prompt includes the topic title and summary."""
        topic = _make_topic()
        spy_client = MagicMock(spec=LLMClient)
        spy_client.generate.return_value = "Preview text."

        generate_preview(
            topic=topic,
            voice_prompt="voice",
            spectrum=StanceConfig(position=0),
            client=spy_client,
        )

        call_args = spy_client.generate.call_args
        # Check user_prompt contains topic details
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[0][1]
        assert "Decline of Local Journalism" in user_prompt
        assert "closing" in user_prompt.lower() or "newspaper" in user_prompt.lower()

    def test_preview_respects_max_tokens(self) -> None:
        """The preview request uses a small max_tokens value."""
        topic = _make_topic()
        spy_client = MagicMock(spec=LLMClient)
        spy_client.generate.return_value = "Short preview."

        generate_preview(
            topic=topic,
            voice_prompt="voice",
            spectrum=StanceConfig(position=0),
            client=spy_client,
        )

        call_args = spy_client.generate.call_args
        max_tokens = call_args.kwargs.get("max_tokens") or call_args[0][2]
        assert max_tokens <= 500  # should be a short call
