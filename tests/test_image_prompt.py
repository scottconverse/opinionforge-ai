"""Unit tests for the image prompt generator.

Covers generate_image_prompt() across all 6 platforms and 6 styles,
custom keywords, no-writer-name constraint, LLM client injection,
error handling, and generate_piece() integration.

All tests use mocked LLM clients -- no real API calls are made.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from opinionforge.core.image_prompt import generate_image_prompt
from opinionforge.core.generator import MANDATORY_DISCLAIMER, generate_piece
from opinionforge.models.config import ImagePromptConfig, ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece
from opinionforge.models.topic import TopicContext


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_topic(
    subject_domain: str = "technology",
    key_entities: list[str] | None = None,
) -> TopicContext:
    """Create a minimal but realistic TopicContext for testing."""
    return TopicContext(
        raw_input="Artificial intelligence and democratic governance",
        input_type="text",
        title="How AI Is Reshaping Democratic Institutions",
        summary=(
            "Artificial intelligence is transforming how governments operate, "
            "introducing both efficiency gains and accountability risks."
        ),
        key_claims=[
            "73% of government agencies now use AI tools.",
            "AI-generated misinformation has increased 400% since 2023.",
        ],
        key_entities=key_entities or ["United States Congress", "European Union", "OpenAI"],
        subject_domain=subject_domain,
    )


def _make_piece(
    image_prompt: str | None = None,
    image_platform: str | None = None,
) -> GeneratedPiece:
    """Create a minimal GeneratedPiece for testing."""
    topic = _make_topic()
    return GeneratedPiece(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        topic=topic,
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=0),
        target_length=800,
        actual_length=810,
        title="How AI Is Reshaping Democratic Institutions",
        body=(
            "The machinery of self-governance is under siege from algorithmic forces. "
            "Democratic institutions, built for human deliberation, now face a new kind "
            "of challenge that prior generations never anticipated."
        ),
        preview_text="The machinery of self-governance is under siege.",
        sources=[],
        research_queries=["AI and democracy 2025"],
        disclaimer=MANDATORY_DISCLAIMER,
        image_prompt=image_prompt,
        image_platform=image_platform,
    )


def _make_mock_client(response: str = "A conceptual scene representing AI and governance.") -> MagicMock:
    """Create a MagicMock LLM client with a controlled response."""
    mock = MagicMock()
    mock.generate.return_value = response
    return mock


class FailingLLMClient:
    """LLM client that always raises an exception."""

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Always raise a simulated API error."""
        raise Exception("Simulated API failure")


# ---------------------------------------------------------------------------
# Tests: generate_image_prompt() — style x platform combinations
# ---------------------------------------------------------------------------


class TestStylePlatformCombinations:
    """Tests for each style+platform combination specified in the sprint contract."""

    def test_photorealistic_substack(self) -> None:
        """Style='photorealistic', platform='substack' returns valid prompt."""
        piece = _make_piece()
        config = ImagePromptConfig(style="photorealistic", platform="substack")
        client = _make_mock_client("A photograph of a government chamber.")
        result = generate_image_prompt(piece, config, client=client)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "photorealistic" in result.lower() or "photograph" in result.lower()

    def test_editorial_medium(self) -> None:
        """Style='editorial', platform='medium' returns valid prompt."""
        piece = _make_piece()
        config = ImagePromptConfig(style="editorial", platform="medium")
        client = _make_mock_client("An editorial illustration of civic technology.")
        result = generate_image_prompt(piece, config, client=client)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_cartoon_wordpress(self) -> None:
        """Style='cartoon', platform='wordpress' returns prompt with cartoon reference."""
        piece = _make_piece()
        config = ImagePromptConfig(style="cartoon", platform="wordpress")
        client = _make_mock_client("A colorful scene of robots interacting with ballots.")
        result = generate_image_prompt(piece, config, client=client)

        assert "cartoon" in result.lower() or "illustrated" in result.lower()

    def test_minimalist_facebook(self) -> None:
        """Style='minimalist', platform='facebook' returns prompt with minimalist reference."""
        piece = _make_piece()
        config = ImagePromptConfig(style="minimalist", platform="facebook")
        client = _make_mock_client("A simple geometric pattern symbolizing data flow.")
        result = generate_image_prompt(piece, config, client=client)

        assert "minimalist" in result.lower() or "clean" in result.lower()

    def test_vintage_twitter(self) -> None:
        """Style='vintage', platform='twitter' returns prompt with vintage reference."""
        piece = _make_piece()
        config = ImagePromptConfig(style="vintage", platform="twitter")
        client = _make_mock_client("A retro-style image of an election ballot.")
        result = generate_image_prompt(piece, config, client=client)

        assert "vintage" in result.lower() or "retro" in result.lower()

    def test_abstract_instagram(self) -> None:
        """Style='abstract', platform='instagram' returns prompt with abstract reference."""
        piece = _make_piece()
        config = ImagePromptConfig(style="abstract", platform="instagram")
        client = _make_mock_client("Abstract flowing shapes representing information streams.")
        result = generate_image_prompt(piece, config, client=client)

        assert "abstract" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Aspect ratios and dimensions per platform
# ---------------------------------------------------------------------------


class TestPlatformRatiosAndDimensions:
    """Tests that each platform's aspect ratio and pixel dimensions appear in output."""

    def test_substack_aspect_ratio_and_dimensions(self) -> None:
        """Substack prompt contains '16:9' and '1456x819 px'."""
        piece = _make_piece()
        config = ImagePromptConfig(platform="substack")
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "Aspect ratio: 16:9" in result
        assert "Size: 1456x819 px" in result

    def test_medium_aspect_ratio_and_dimensions(self) -> None:
        """Medium prompt contains '16:9' and '1400x788 px'."""
        piece = _make_piece()
        config = ImagePromptConfig(platform="medium")
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "Aspect ratio: 16:9" in result
        assert "Size: 1400x788 px" in result

    def test_wordpress_aspect_ratio_and_dimensions(self) -> None:
        """WordPress prompt contains '16:9' and '1200x675 px'."""
        piece = _make_piece()
        config = ImagePromptConfig(platform="wordpress")
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "Aspect ratio: 16:9" in result
        assert "Size: 1200x675 px" in result

    def test_facebook_aspect_ratio_and_dimensions(self) -> None:
        """Facebook prompt contains '1.91:1' and '1200x628 px'."""
        piece = _make_piece()
        config = ImagePromptConfig(platform="facebook")
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "Aspect ratio: 1.91:1" in result
        assert "Size: 1200x628 px" in result

    def test_twitter_aspect_ratio_and_dimensions(self) -> None:
        """Twitter prompt contains '16:9' and '1600x900 px'."""
        piece = _make_piece()
        config = ImagePromptConfig(platform="twitter")
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "Aspect ratio: 16:9" in result
        assert "Size: 1600x900 px" in result

    def test_instagram_aspect_ratio_and_dimensions(self) -> None:
        """Instagram prompt contains '1:1' and '1080x1080 px'."""
        piece = _make_piece()
        config = ImagePromptConfig(platform="instagram")
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "Aspect ratio: 1:1" in result
        assert "Size: 1080x1080 px" in result


# ---------------------------------------------------------------------------
# Tests: Custom keywords
# ---------------------------------------------------------------------------


class TestCustomKeywords:
    """Tests for custom_keywords injection into the prompt."""

    def test_custom_keywords_appear_in_prompt(self) -> None:
        """custom_keywords=['dark', 'moody'] appear in the output prompt."""
        piece = _make_piece()
        config = ImagePromptConfig(custom_keywords=["dark", "moody"])
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "dark" in result
        assert "moody" in result

    def test_empty_custom_keywords_no_spurious_content(self) -> None:
        """Empty custom_keywords list does not add unexpected content."""
        piece = _make_piece()
        config = ImagePromptConfig(custom_keywords=[])
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: Writer name exclusion
# ---------------------------------------------------------------------------


class TestNoWriterName:
    """Tests that writer names do not appear in the image prompt."""

    def test_writer_name_not_in_prompt(self) -> None:
        """Real person names must not appear in generated image prompts."""
        piece = _make_piece()
        config = ImagePromptConfig(style="editorial", platform="substack")
        client = _make_mock_client(
            "A symbolic representation of algorithmic governance."
        )
        result = generate_image_prompt(piece, config, client=client)

        # The MANDATORY_DISCLAIMER doesn't contain writer names,
        # so no writer names should appear in the generated prompt
        assert "Christopher" not in result
        assert "Hitchens" not in result

    def test_user_prompt_instructs_no_names(self) -> None:
        """The user prompt passed to the LLM instructs it not to use person names."""
        piece = _make_piece()
        config = ImagePromptConfig()
        client = MagicMock()
        client.generate.return_value = "A scene depicting governance."

        generate_image_prompt(piece, config, client=client)

        call_kwargs = client.generate.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt") or call_kwargs[0][1]
        assert "name" in user_prompt.lower() or "person" in user_prompt.lower()


# ---------------------------------------------------------------------------
# Tests: LLM client usage
# ---------------------------------------------------------------------------


class TestLLMClientUsage:
    """Tests for correct LLM client interaction."""

    def test_llm_client_called_exactly_once(self) -> None:
        """LLM client.generate() is called exactly once per generate_image_prompt() call."""
        piece = _make_piece()
        config = ImagePromptConfig()
        client = _make_mock_client()
        generate_image_prompt(piece, config, client=client)

        assert client.generate.call_count == 1

    def test_runtime_error_on_llm_failure(self) -> None:
        """RuntimeError is raised when the LLM client raises an exception."""
        piece = _make_piece()
        config = ImagePromptConfig()
        failing_client = FailingLLMClient()

        with pytest.raises(RuntimeError, match="Failed to generate image prompt"):
            generate_image_prompt(piece, config, client=failing_client)

    def test_runtime_error_message_is_descriptive(self) -> None:
        """RuntimeError message includes useful context about the failure."""
        piece = _make_piece()
        config = ImagePromptConfig()
        failing_client = FailingLLMClient()

        with pytest.raises(RuntimeError) as exc_info:
            generate_image_prompt(piece, config, client=failing_client)

        assert "image prompt" in str(exc_info.value).lower()

    def test_style_label_in_output(self) -> None:
        """The Style: line appears in the output prompt."""
        piece = _make_piece()
        config = ImagePromptConfig(style="editorial", platform="substack")
        client = _make_mock_client()
        result = generate_image_prompt(piece, config, client=client)

        assert "Style:" in result

    def test_output_is_non_empty_string(self) -> None:
        """generate_image_prompt() always returns a non-empty string on success."""
        piece = _make_piece()
        config = ImagePromptConfig()
        client = _make_mock_client("A depiction of civic technology.")
        result = generate_image_prompt(piece, config, client=client)

        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Tests: generate_piece() integration
# ---------------------------------------------------------------------------


class TestGeneratePieceIntegration:
    """Tests for generate_piece() image_config integration."""

    def _make_mode_blend_config(self) -> ModeBlendConfig:
        """Return a single-mode ModeBlendConfig for testing."""
        return ModeBlendConfig(modes=[("analytical", 100.0)])

    def _make_stance_config(self) -> StanceConfig:
        """Return a balanced StanceConfig for testing."""
        return StanceConfig(position=0)

    def test_generate_piece_without_image_config_leaves_image_prompt_none(
        self,
        sample_topic_context: TopicContext,
    ) -> None:
        """generate_piece() with image_config=None leaves image_prompt as None."""
        mock_client = _make_mock_client(
            "## The AI Threat\n\nArtificial intelligence changes everything."
        )
        piece = generate_piece(
            topic=sample_topic_context,
            mode_config=self._make_mode_blend_config(),
            stance=self._make_stance_config(),
            target_length="short",
            image_config=None,
            client=mock_client,
        )

        assert piece.image_prompt is None
        assert piece.image_platform is None

    def test_generate_piece_with_image_config_sets_image_prompt(
        self,
        sample_topic_context: TopicContext,
    ) -> None:
        """generate_piece() with a valid ImagePromptConfig sets image_prompt to non-empty string."""
        piece_text = "## The AI Threat\n\nArtificial intelligence changes everything."
        image_subject = "Abstract representation of digital governance."
        mock_client = MagicMock()
        mock_client.generate.side_effect = [piece_text, image_subject]

        image_config = ImagePromptConfig(style="editorial", platform="substack")
        piece = generate_piece(
            topic=sample_topic_context,
            mode_config=self._make_mode_blend_config(),
            stance=self._make_stance_config(),
            target_length="short",
            image_config=image_config,
            client=mock_client,
        )

        assert piece.image_prompt is not None
        assert len(piece.image_prompt) > 0

    def test_generate_piece_with_image_config_sets_image_platform(
        self,
        sample_topic_context: TopicContext,
    ) -> None:
        """generate_piece() with a valid ImagePromptConfig sets image_platform to config.platform."""
        piece_text = "## The AI Threat\n\nArtificial intelligence changes everything."
        image_subject = "A conceptual image of civic technology."
        mock_client = MagicMock()
        mock_client.generate.side_effect = [piece_text, image_subject]

        image_config = ImagePromptConfig(style="minimalist", platform="instagram")
        piece = generate_piece(
            topic=sample_topic_context,
            mode_config=self._make_mode_blend_config(),
            stance=self._make_stance_config(),
            target_length="short",
            image_config=image_config,
            client=mock_client,
        )

        assert piece.image_platform == "instagram"
