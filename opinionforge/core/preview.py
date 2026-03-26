"""Tone preview generator that produces a 2-3 sentence preview using the composed voice prompt.

The preview captures the opening hook or thesis statement style without
performing a full research cycle.
"""

from __future__ import annotations

from typing import Protocol

from opinionforge.config import Settings, get_settings
from opinionforge.models.config import StanceConfig
from opinionforge.models.topic import TopicContext


class LLMClient(Protocol):
    """Protocol for LLM client abstraction, enabling dependency injection for tests."""

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text from the LLM.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text string.
        """
        ...


class AnthropicLLMClient:
    """LLM client backed by the Anthropic Claude API.

    Args:
        api_key: The Anthropic API key.
        model: The model to use. Defaults to 'claude-sonnet-4-20250514'.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text using the Anthropic Claude API.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text string.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class OpenAILLMClient:
    """LLM client backed by the OpenAI API.

    Args:
        api_key: The OpenAI API key.
        model: The model to use. Defaults to 'gpt-4o'.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate text using the OpenAI API.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text string.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Create an LLM client based on configuration.

    Args:
        settings: Application settings. Uses default settings if None.

    Returns:
        An LLMClient implementation for the configured provider.
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.require_llm_api_key()

    if settings.opinionforge_llm_provider == "anthropic":
        return AnthropicLLMClient(api_key=api_key)
    else:
        return OpenAILLMClient(api_key=api_key)


def generate_preview(
    topic: TopicContext,
    voice_prompt: str,
    spectrum: StanceConfig,
    *,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> str:
    """Generate a 2-3 sentence tone preview in the selected voice.

    Captures the opening hook or thesis statement style without performing
    a full research cycle. Uses a single short LLM call.

    Args:
        topic: The normalized topic context.
        voice_prompt: The composed voice prompt (after blending and spectrum).
        spectrum: The stance configuration for context.
        client: Optional LLM client for dependency injection (used in tests).
        settings: Optional settings override.

    Returns:
        A 2-3 sentence preview string.

    Raises:
        RuntimeError: If the LLM API call fails.
    """
    if client is None:
        client = create_llm_client(settings)

    user_prompt = (
        f"Write a 2-3 sentence preview (opening hook or thesis statement) for an opinion piece "
        f"about the following topic. This is a tone preview only -- do not write the full piece.\n\n"
        f"Topic: {topic.title}\n"
        f"Summary: {topic.summary}\n"
    )
    if topic.key_claims:
        user_prompt += f"Key claims: {'; '.join(topic.key_claims[:3])}\n"

    try:
        return client.generate(
            system_prompt=voice_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate preview: {exc}. "
            "Check your API key configuration and network connection."
        ) from exc
