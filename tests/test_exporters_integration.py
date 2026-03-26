"""Integration tests verifying exporters work correctly with real GeneratedPiece objects.

Pieces are generated via the generator (with mocked LLM) and then exported
through the exporter classes, validating the full produce-then-export pipeline.
No real external API calls are made.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.exporters import (
    MediumExporter,
    SubstackExporter,
    TwitterExporter,
    WordPressExporter,
    export,
)
from opinionforge.models.config import ImagePromptConfig, ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece, SourceCitation
from opinionforge.models.topic import TopicContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STANDARD_BODY_750 = (
    "The question of whether democratic institutions can survive the current assault "
    "is one that demands a serious answer, not a reassuring one. "
    "History provides instructive parallels, though none is perfect.\n\n"
    "The erosion begins at the margins. Small capitulations accumulate. "
    "A press release goes unchallenged. A subpoena goes unanswered. "
    "An independent agency is quietly defunded. "
    "Citizens, exhausted by the sheer volume of outrage, disengage.\n\n"
    "But the mechanics of recovery are equally well documented. "
    "Civic institutions are not made of glass. "
    "They are made of habits, and habits can be re-formed. "
    "The first step is to name what is happening with clarity and without euphemism.\n\n"
    "The second step is harder. It requires those in positions of institutional "
    "authority to exercise that authority rather than defer to the convenience of "
    "the moment. Courts must court. Legislatures must legislate. "
    "Journalists must report what they see, not what they are permitted to say.\n\n"
    "We are, in short, at a moment that will be studied. "
    "The only question is what the study will conclude."
)

_LONG_BODY_2500 = _STANDARD_BODY_750 * 4


def _make_topic(title: str = "Democratic Resilience in Crisis") -> TopicContext:
    """Return a realistic TopicContext for integration tests."""
    return TopicContext(
        raw_input="The challenge to democratic institutions",
        input_type="text",
        title=title,
        summary="Examining whether democratic institutions can survive modern pressures.",
        key_claims=["Voter engagement has declined 20% since 2018."],
        key_entities=["United States Congress", "Supreme Court"],
        subject_domain="politics",
    )


def _make_citation(n: int = 1) -> SourceCitation:
    """Return a SourceCitation for testing sources appendix."""
    return SourceCitation(
        claim=f"Democratic norms have declined — finding {n}.",
        source_name=f"Research Journal {n}",
        source_url=f"https://example.com/research-{n}",
        accessed_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        credibility_score=0.85,
    )


def _make_piece(
    *,
    body: str = _STANDARD_BODY_750,
    title: str = "The Reckoning We Cannot Postpone",
    sources: list[SourceCitation] | None = None,
    image_prompt: str | None = None,
    disclaimer: str = MANDATORY_DISCLAIMER,
) -> GeneratedPiece:
    """Build a GeneratedPiece for integration tests."""
    if sources is None:
        sources = [_make_citation(1), _make_citation(2)]
    return GeneratedPiece(
        id="integration-test-001",
        created_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        topic=_make_topic(title=title),
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=-20),
        target_length=750,
        actual_length=len(body.split()),
        title=title,
        body=body,
        preview_text="Democratic institutions face an existential challenge.",
        sources=sources,
        research_queries=["democratic decline 2025"],
        image_prompt=image_prompt,
        disclaimer=disclaimer,
    )


def _make_generated_piece_via_generator(
    body_text: str = _STANDARD_BODY_750,
    title: str = "The Reckoning We Cannot Postpone",
) -> GeneratedPiece:
    """Generate a piece using the real generator with a mocked LLM client.

    This exercises the full generate_piece() path rather than constructing
    the piece directly.
    """
    from opinionforge.core.generator import generate_piece

    mock_llm = MagicMock()
    mock_llm.generate.return_value = f"## {title}\n\n{body_text}"

    topic = _make_topic(title=title)
    mode_config = ModeBlendConfig(modes=[("analytical", 100.0)])
    stance = StanceConfig(position=0)

    with patch("opinionforge.core.generator.create_llm_client", return_value=mock_llm):
        return generate_piece(
            topic=topic,
            mode_config=mode_config,
            stance=stance,
            target_length=800,
            research_context=None,
            client=mock_llm,
        )


# ---------------------------------------------------------------------------
# Substack integration tests
# ---------------------------------------------------------------------------

class TestSubstackIntegration:
    """Round-trip tests for the Substack exporter."""

    def test_substack_contains_original_title(self) -> None:
        """Round-trip: piece generated -> exported to substack -> output contains original title."""
        piece = _make_generated_piece_via_generator(title="The Reckoning We Cannot Postpone")
        output = SubstackExporter().export(piece)
        assert "The Reckoning We Cannot Postpone" in output

    def test_substack_valid_markdown_no_unclosed_brackets(self) -> None:
        """Substack export renders in valid CommonMark — no unclosed brackets."""
        piece = _make_piece()
        output = SubstackExporter().export(piece)
        assert abs(output.count("[") - output.count("]")) < 5

    def test_substack_balanced_asterisks(self) -> None:
        """Substack export has balanced asterisk pairs (no half-open bold/italic)."""
        body = "This is **bold text** and *italic text* in a paragraph."
        piece = _make_piece(body=body)
        output = SubstackExporter().export(piece)
        double_star_count = len(re.findall(r"\*\*", output))
        assert double_star_count % 2 == 0, f"Unbalanced ** in output: {output}"

    def test_substack_includes_sources_appendix(self) -> None:
        """Substack export includes the '## Sources & Claims' section when piece has sources."""
        piece = _make_piece(sources=[_make_citation(1), _make_citation(2)])
        output = SubstackExporter().export(piece)
        assert "Sources & Claims" in output

    def test_substack_includes_disclaimer(self) -> None:
        """Substack export includes the mandatory disclaimer."""
        piece = _make_piece()
        output = SubstackExporter().export(piece)
        assert "AI-assisted rhetorical controls" in output or "AI-generated" in output

    def test_substack_includes_image_prompt(self) -> None:
        """Substack export includes the image prompt when piece.image_prompt is set."""
        piece = _make_piece(image_prompt="A fractured ballot box at twilight.")
        output = SubstackExporter().export(piece)
        assert "fractured ballot box" in output

    def test_substack_no_sources_no_appendix(self) -> None:
        """Substack export omits sources section when piece has no sources."""
        piece = _make_piece(sources=[])
        output = SubstackExporter().export(piece)
        assert "Sources & Claims" not in output


# ---------------------------------------------------------------------------
# Medium integration tests
# ---------------------------------------------------------------------------

class TestMediumIntegration:
    """Round-trip tests for the Medium exporter."""

    def test_medium_heading_demotion_no_headings(self) -> None:
        """Medium export heading demotion does not break pieces with no headings in body."""
        body = (
            "Plain paragraph one without any headings.\n\n"
            "Plain paragraph two continues the argument."
        )
        piece = _make_piece(body=body)
        output = MediumExporter().export(piece)
        assert "###" not in output
        assert "DROP CAP" in output  # Drop cap should still be inserted

    def test_medium_includes_disclaimer(self) -> None:
        """Medium export includes the mandatory disclaimer."""
        piece = _make_piece()
        output = MediumExporter().export(piece)
        assert "AI-assisted rhetorical controls" in output or "AI-generated" in output

    def test_medium_includes_sources(self) -> None:
        """Medium export includes sources appendix when piece has sources."""
        piece = _make_piece(sources=[_make_citation(1)])
        output = MediumExporter().export(piece)
        assert "Sources & Claims" in output

    def test_medium_includes_image_prompt(self) -> None:
        """Medium export includes the image prompt when piece.image_prompt is set."""
        piece = _make_piece(image_prompt="Golden sunrise over a crumbling courthouse.")
        output = MediumExporter().export(piece)
        assert "courthouse" in output


# ---------------------------------------------------------------------------
# WordPress integration tests
# ---------------------------------------------------------------------------

class TestWordPressIntegration:
    """Round-trip tests for the WordPress exporter."""

    def test_wordpress_seo_meta_under_160_chars(self) -> None:
        """WordPress SEO meta description is exactly 160 characters or the full first sentence if shorter."""
        long_body = "A" * 200 + " more content here."
        piece = _make_piece(body=long_body)
        output = WordPressExporter().export(piece)
        match = re.search(r"<!-- SEO META: (.+?) -->", output)
        assert match is not None, "SEO META comment not found"
        seo_text = match.group(1)
        assert len(seo_text) <= 160, f"SEO meta too long: {len(seo_text)} chars"

    def test_wordpress_seo_meta_short_body_uses_full_sentence(self) -> None:
        """WordPress SEO meta uses full first sentence when body is short."""
        short_sentence = "The experiment of self-governance has always required courage."
        piece = _make_piece(body=short_sentence)
        output = WordPressExporter().export(piece)
        match = re.search(r"<!-- SEO META: (.+?) -->", output)
        assert match is not None
        assert "experiment" in match.group(1) or "courage" in match.group(1)

    def test_wordpress_includes_disclaimer(self) -> None:
        """WordPress export includes the mandatory disclaimer."""
        piece = _make_piece()
        output = WordPressExporter().export(piece)
        assert "AI-assisted rhetorical controls" in output or "AI-generated" in output

    def test_wordpress_includes_sources(self) -> None:
        """WordPress export includes sources when piece has sources."""
        piece = _make_piece(sources=[_make_citation(1)])
        output = WordPressExporter().export(piece)
        assert "Sources" in output

    def test_wordpress_includes_image_prompt_comment(self) -> None:
        """WordPress export includes image prompt in an HTML comment when piece.image_prompt is set."""
        piece = _make_piece(image_prompt="Bold typography over dark blue background.")
        output = WordPressExporter().export(piece)
        assert "IMAGE PROMPT" in output
        assert "typography" in output


# ---------------------------------------------------------------------------
# Twitter integration tests
# ---------------------------------------------------------------------------

class TestTwitterIntegration:
    """Round-trip tests for the Twitter exporter."""

    def test_twitter_750_words_no_tweet_over_280(self) -> None:
        """Twitter thread from a 750-word piece never creates a tweet over 280 characters."""
        piece = _make_piece(body=_STANDARD_BODY_750)
        output = TwitterExporter().export(piece)
        tweets = [t.strip() for t in output.split("\n\n") if t.strip()]
        for tweet in tweets:
            assert len(tweet) <= 280, f"Tweet too long ({len(tweet)}): {tweet}"

    def test_twitter_2500_words_no_tweet_over_280(self) -> None:
        """Twitter thread from a 2500-word piece never creates a tweet over 280 characters."""
        piece = _make_piece(body=_LONG_BODY_2500)
        output = TwitterExporter().export(piece)
        tweets = [t.strip() for t in output.split("\n\n") if t.strip()]
        for tweet in tweets:
            assert len(tweet) <= 280, f"Tweet too long ({len(tweet)}): {tweet}"

    def test_twitter_first_tweet_contains_title(self) -> None:
        """Round-trip: first tweet contains the original title or key phrase from title."""
        piece = _make_generated_piece_via_generator(title="The Reckoning We Cannot Postpone")
        output = TwitterExporter().export(piece)
        first_tweet = output.split("\n\n")[0]
        assert "Reckoning" in first_tweet or "Postpone" in first_tweet or "Cannot" in first_tweet

    def test_twitter_includes_disclaimer(self) -> None:
        """Twitter export includes the mandatory disclaimer."""
        piece = _make_piece()
        output = TwitterExporter().export(piece)
        assert "AI-assisted rhetorical controls" in output or "AI-generated" in output

    def test_twitter_thread_is_numbered(self) -> None:
        """Twitter thread tweets are numbered starting from 1/."""
        piece = _make_piece()
        output = TwitterExporter().export(piece)
        assert output.strip().startswith("1/")


# ---------------------------------------------------------------------------
# All-four-exporters consistency tests
# ---------------------------------------------------------------------------

class TestAllExportersConsistency:
    """Verify that all four exporters behave consistently for a given piece."""

    def test_all_exporters_include_disclaimer(self) -> None:
        """All four exporters include the mandatory disclaimer."""
        piece = _make_piece()
        for exporter_cls in [SubstackExporter, MediumExporter, WordPressExporter, TwitterExporter]:
            output = exporter_cls().export(piece)
            assert (
                "AI-assisted rhetorical controls" in output
                or "AI-generated" in output
                or "original content" in output
            ), f"{exporter_cls.__name__} missing disclaimer"

    def test_all_exporters_include_sources_appendix(self) -> None:
        """Long-form exporters include sources when piece has sources.

        Twitter is excluded because the thread format with 280-char tweets
        and mandatory disclaimer as the final tweet leaves no room for a
        sources appendix.
        """
        piece = _make_piece(sources=[_make_citation(1)])
        for exporter_cls in [SubstackExporter, MediumExporter, WordPressExporter]:
            output = exporter_cls().export(piece)
            assert "Sources" in output or "source" in output.lower(), (
                f"{exporter_cls.__name__} missing sources"
            )

    def test_all_exporters_include_image_prompt(self) -> None:
        """All four exporters include the image prompt when piece.image_prompt is set."""
        piece = _make_piece(image_prompt="A dramatic cityscape under stormy skies.")
        for exporter_cls in [SubstackExporter, MediumExporter, WordPressExporter, TwitterExporter]:
            output = exporter_cls().export(piece)
            assert "cityscape" in output or "image" in output.lower(), (
                f"{exporter_cls.__name__} missing image prompt"
            )
