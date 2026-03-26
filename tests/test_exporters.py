"""Unit tests for all four exporters, the export dispatcher, and ImagePromptConfig.

All four exporters are tested for mandatory disclaimer inclusion.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_topic() -> TopicContext:
    """Construct a minimal TopicContext for testing."""
    return TopicContext(
        raw_input="The future of democracy",
        input_type="text",
        title="Democracy Under Pressure",
        summary="Democratic institutions face mounting challenges.",
        key_claims=["Voter turnout fell 10% in the last decade."],
        key_entities=["United States", "European Union"],
        subject_domain="politics",
    )


def _make_citation(n: int = 1) -> SourceCitation:
    """Construct a minimal SourceCitation for testing."""
    return SourceCitation(
        claim=f"Claim number {n} as cited in the piece.",
        source_name=f"Source {n}",
        source_url=f"https://example.com/source-{n}",
        accessed_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        credibility_score=0.9,
    )


def _make_piece(
    *,
    body: str | None = None,
    subtitle: str | None = "Why it matters now",
    image_prompt: str | None = "A crumbling ballot box in neon light.",
    sources: list[SourceCitation] | None = None,
    disclaimer: str = MANDATORY_DISCLAIMER,
) -> GeneratedPiece:
    """Construct a minimal GeneratedPiece for testing."""
    if body is None:
        body = (
            "## The Stakes Are High\n\n"
            "It would be convenient to pretend that nothing fundamental has changed. "
            "It has. The machinery of democratic governance is being corroded from within "
            "by forces that benefit from its collapse.\n\n"
            "## A Second Section\n\n"
            "The evidence is everywhere if you choose to look. Voter suppression "
            "laws are proliferating. Independent journalism is dying. And the courts "
            "have been packed with ideologues masquerading as jurists.\n\n"
            "We must be honest about what is at stake here. The experiment of "
            "self-governance is not self-sustaining. It requires active, informed, "
            "and courageous citizens to preserve it against those who would hollow it out "
            "for their own enrichment and the perpetuation of their own power."
        )
    if sources is None:
        sources = [_make_citation(1), _make_citation(2)]
    return GeneratedPiece(
        id="test-001",
        created_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        topic=_make_topic(),
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=-30, intensity=0.5),
        target_length=600,
        actual_length=len(body.split()),
        title="Democracy Is Not Self-Sustaining",
        subtitle=subtitle,
        body=body,
        preview_text="A warning about the fragility of democratic institutions.",
        sources=sources,
        research_queries=["democracy decline 2025"],
        image_prompt=image_prompt,
        disclaimer=disclaimer,
    )


def _make_long_piece() -> GeneratedPiece:
    """Return a piece whose body is >= 300 words for pull-quote tests."""
    sentences = (
        "The erosion of democratic norms is a slow process that rarely announces itself. "
        "Each small capitulation seems trivial in isolation. "
        "The press loses its independence one editorial decision at a time. "
        "Courts drift toward ideological capture over the course of decades. "
        "Citizens disengage from civic life gradually, imperceptibly. "
        "And yet the cumulative effect is catastrophic. "
        "What was once unthinkable becomes merely controversial. "
        "What is merely controversial becomes normal. "
        "What is normal becomes mandatory. "
        "This is the arc of democratic decay. "
    )
    # Repeat to exceed 300 words
    body = (sentences * 8).strip()
    return _make_piece(body=body)


# ---------------------------------------------------------------------------
# SubstackExporter tests
# ---------------------------------------------------------------------------

class TestSubstackExporter:
    def test_starts_with_h1_title(self) -> None:
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        assert result.startswith("# Democracy Is Not Self-Sustaining")

    def test_subtitle_appears_after_title(self) -> None:
        piece = _make_piece(subtitle="Why it matters now")
        result = SubstackExporter().export(piece)
        assert "Why it matters now" in result
        title_pos = result.index("# Democracy Is Not Self-Sustaining")
        subtitle_pos = result.index("Why it matters now")
        assert subtitle_pos > title_pos

    def test_no_subtitle_omitted(self) -> None:
        piece = _make_piece(subtitle=None)
        result = SubstackExporter().export(piece)
        lines = result.splitlines()
        non_empty = [ln for ln in lines if ln.strip()]
        assert non_empty[0] == "# Democracy Is Not Self-Sustaining"

    def test_no_html_tags(self) -> None:
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        assert "<div" not in result
        assert "<p>" not in result
        assert "<span" not in result
        assert "<br" not in result

    def test_no_html_tags_in_body_with_html(self) -> None:
        piece = _make_piece(body="<p>Hello <b>world</b></p>\n\nSecond paragraph.")
        result = SubstackExporter().export(piece)
        assert "<p>" not in result
        assert "<b>" not in result

    def test_contains_sources_appendix(self) -> None:
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        assert "## Sources & Claims" in result

    def test_sources_appendix_numbered_format(self) -> None:
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        assert '1. "Claim number 1 as cited in the piece."' in result
        assert "accessed 2026-03-25" in result

    def test_contains_mandatory_disclaimer(self) -> None:
        """SubstackExporter always includes the fixed mandatory disclaimer."""
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_disclaimer_text_equals_mandatory_constant(self) -> None:
        """The disclaimer in Substack output equals exactly MANDATORY_DISCLAIMER."""
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_disclaimer_present_regardless_of_piece_disclaimer_field(self) -> None:
        """Disclaimer appears even if piece.disclaimer is empty — it is mandatory."""
        piece = _make_piece(disclaimer="")
        result = SubstackExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_contains_image_prompt(self) -> None:
        piece = _make_piece(image_prompt="A crumbling ballot box in neon light.")
        result = SubstackExporter().export(piece)
        assert "**Header image prompt:**" in result
        assert "A crumbling ballot box in neon light." in result

    def test_no_image_prompt_when_none(self) -> None:
        piece = _make_piece(image_prompt=None)
        result = SubstackExporter().export(piece)
        assert "Header image prompt" not in result

    def test_no_trailing_whitespace(self) -> None:
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        for line in result.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace on line: {repr(line)}"

    def test_headings_are_atx_style(self) -> None:
        piece = _make_piece()
        result = SubstackExporter().export(piece)
        assert re.search(r"^#+ ", result, re.MULTILINE)

    def test_empty_sources_no_appendix_header(self) -> None:
        piece = _make_piece(sources=[])
        result = SubstackExporter().export(piece)
        assert "## Sources & Claims" not in result


# ---------------------------------------------------------------------------
# MediumExporter tests
# ---------------------------------------------------------------------------

class TestMediumExporter:
    def test_starts_with_h1_title(self) -> None:
        piece = _make_piece()
        result = MediumExporter().export(piece)
        assert result.startswith("# Democracy Is Not Self-Sustaining")

    def test_no_html_tags(self) -> None:
        piece = _make_piece()
        result = MediumExporter().export(piece)
        assert "<p>" not in result
        assert "<div" not in result
        assert "<span" not in result

    def test_drop_cap_marker_present(self) -> None:
        piece = _make_piece()
        result = MediumExporter().export(piece)
        assert "> DROP CAP" in result

    def test_pull_quote_for_long_piece(self) -> None:
        piece = _make_long_piece()
        result = MediumExporter().export(piece)
        blockquote_lines = [
            ln for ln in result.splitlines()
            if ln.startswith(">") and "DROP CAP" not in ln
        ]
        assert len(blockquote_lines) >= 1

    def test_no_pull_quote_for_short_piece(self) -> None:
        piece = _make_piece(body="Short body. Only a few words here.")
        result = MediumExporter().export(piece)
        blockquote_lines = [
            ln for ln in result.splitlines()
            if ln.startswith(">") and "DROP CAP" not in ln
        ]
        assert len(blockquote_lines) == 0

    def test_heading_demotion(self) -> None:
        piece = _make_piece(body="## Section Title\n\nParagraph text here.")
        result = MediumExporter().export(piece)
        assert "### Section Title" in result
        assert not re.search(r"^## Section Title", result, re.MULTILINE)

    def test_h1_heading_demotion(self) -> None:
        piece = _make_piece(body="# Big Section\n\nParagraph text here.")
        result = MediumExporter().export(piece)
        assert "## Big Section" in result

    def test_contains_sources_appendix(self) -> None:
        piece = _make_piece()
        result = MediumExporter().export(piece)
        assert "## Sources & Claims" in result

    def test_contains_mandatory_disclaimer(self) -> None:
        """MediumExporter always includes the fixed mandatory disclaimer."""
        piece = _make_piece()
        result = MediumExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_disclaimer_text_equals_mandatory_constant(self) -> None:
        """The disclaimer in Medium output equals exactly MANDATORY_DISCLAIMER."""
        piece = _make_piece()
        result = MediumExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_disclaimer_present_regardless_of_piece_disclaimer_field(self) -> None:
        """Disclaimer appears even if piece.disclaimer is empty — it is mandatory."""
        piece = _make_piece(disclaimer="")
        result = MediumExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_no_conditional_disclaimer_omission(self) -> None:
        """No conditional logic omits the disclaimer for Medium."""
        piece = _make_piece(disclaimer="something else")
        result = MediumExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_contains_image_prompt(self) -> None:
        piece = _make_piece(image_prompt="Sunset over Capitol Hill.")
        result = MediumExporter().export(piece)
        assert "**Header image prompt:**" in result
        assert "Sunset over Capitol Hill." in result


# ---------------------------------------------------------------------------
# WordPressExporter tests
# ---------------------------------------------------------------------------

class TestWordPressExporter:
    def test_contains_wp_paragraph_blocks(self) -> None:
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert "<!-- wp:paragraph -->" in result
        assert "<!-- /wp:paragraph -->" in result

    def test_title_in_wp_heading_block(self) -> None:
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert '<!-- wp:heading {"level":1} -->' in result
        assert "<h1>Democracy Is Not Self-Sustaining</h1>" in result
        assert "<!-- /wp:heading -->" in result

    def test_featured_image_placeholder(self) -> None:
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert "<!-- FEATURED IMAGE PLACEHOLDER -->" in result

    def test_seo_meta_comment(self) -> None:
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert "<!-- SEO META:" in result
        match = re.search(r"<!-- SEO META: (.+?) -->", result)
        assert match is not None
        assert len(match.group(1)) <= 160

    def test_image_prompt_comment(self) -> None:
        piece = _make_piece(image_prompt="A crumbling ballot box.")
        result = WordPressExporter().export(piece)
        assert "<!-- IMAGE PROMPT: A crumbling ballot box. -->" in result

    def test_no_image_prompt_comment_when_none(self) -> None:
        piece = _make_piece(image_prompt=None)
        result = WordPressExporter().export(piece)
        assert "<!-- IMAGE PROMPT:" not in result

    def test_disclaimer_in_disclaimer_paragraph(self) -> None:
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert "opinionforge-disclaimer" in result

    def test_contains_mandatory_disclaimer(self) -> None:
        """WordPressExporter always includes the fixed mandatory disclaimer."""
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_disclaimer_text_equals_mandatory_constant(self) -> None:
        """The disclaimer in WordPress output equals exactly MANDATORY_DISCLAIMER."""
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_disclaimer_present_regardless_of_piece_disclaimer_field(self) -> None:
        """Disclaimer appears even if piece.disclaimer is empty — it is mandatory."""
        piece = _make_piece(disclaimer="")
        result = WordPressExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_no_conditional_disclaimer_omission(self) -> None:
        """No conditional logic omits the disclaimer for WordPress."""
        piece = _make_piece(disclaimer="something else")
        result = WordPressExporter().export(piece)
        assert MANDATORY_DISCLAIMER in result

    def test_sources_in_wp_paragraph_block(self) -> None:
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert "Sources" in result
        assert "<!-- wp:paragraph -->" in result

    def test_output_contains_h1(self) -> None:
        piece = _make_piece()
        result = WordPressExporter().export(piece)
        assert "<h1>" in result


# ---------------------------------------------------------------------------
# TwitterExporter tests
# ---------------------------------------------------------------------------

class TestTwitterExporter:
    def _get_tweets(self, piece: GeneratedPiece) -> list[str]:
        result = TwitterExporter().export(piece)
        return [t.strip() for t in result.split("\n\n") if t.strip()]

    def test_returns_multiple_tweets_separated_by_blank_lines(self) -> None:
        piece = _make_piece()
        result = TwitterExporter().export(piece)
        assert "\n\n" in result

    def test_no_tweet_exceeds_280_chars(self) -> None:
        piece = _make_long_piece()
        tweets = self._get_tweets(piece)
        for tweet in tweets:
            assert len(tweet) <= 280, f"Tweet too long ({len(tweet)} chars): {tweet!r}"

    def test_thread_has_between_5_and_15_tweets(self) -> None:
        piece = _make_piece()
        tweets = self._get_tweets(piece)
        assert _MIN_TWEETS <= len(tweets) <= _MAX_TWEETS

    def test_long_piece_thread_within_limits(self) -> None:
        piece = _make_long_piece()
        tweets = self._get_tweets(piece)
        assert _MIN_TWEETS <= len(tweets) <= _MAX_TWEETS

    def test_tweets_numbered_with_n_slash_format(self) -> None:
        piece = _make_piece()
        tweets = self._get_tweets(piece)
        for i, tweet in enumerate(tweets, start=1):
            assert tweet.startswith(f"{i}/"), f"Tweet {i} missing prefix: {tweet!r}"

    def test_first_tweet_contains_title(self) -> None:
        piece = _make_piece()
        tweets = self._get_tweets(piece)
        assert piece.title in tweets[0]

    def test_contains_mandatory_disclaimer(self) -> None:
        """TwitterExporter always includes the mandatory disclaimer text."""
        piece = _make_piece()
        result = TwitterExporter().export(piece)
        assert "AI-assisted rhetorical controls" in result

    def test_disclaimer_is_final_tweet(self) -> None:
        """The disclaimer appears as the final tweet in the thread."""
        piece = _make_piece()
        tweets = self._get_tweets(piece)
        last_tweet = tweets[-1]
        assert "AI-assisted rhetorical controls" in last_tweet

    def test_disclaimer_present_regardless_of_piece_disclaimer_field(self) -> None:
        """Twitter disclaimer uses the fixed constant, not piece.disclaimer."""
        piece = _make_piece(disclaimer="custom disclaimer text that is different")
        result = TwitterExporter().export(piece)
        assert "AI-assisted rhetorical controls" in result

    def test_no_conditional_disclaimer_omission(self) -> None:
        """No conditional logic omits the disclaimer for Twitter."""
        piece = _make_piece(disclaimer="")
        result = TwitterExporter().export(piece)
        assert "AI-assisted rhetorical controls" in result

    def test_final_tweet_is_disclaimer_not_cta(self) -> None:
        """The final tweet is the disclaimer, not a CTA."""
        piece = _make_piece()
        tweets = self._get_tweets(piece)
        last_tweet = tweets[-1]
        assert "AI-assisted rhetorical controls" in last_tweet

    def test_sources_appendix_not_in_body_tweets(self) -> None:
        piece = _make_piece()
        tweets = self._get_tweets(piece)
        for tweet in tweets[:-1]:
            assert "## Sources & Claims" not in tweet


# ---------------------------------------------------------------------------
# export() dispatcher tests
# ---------------------------------------------------------------------------

_MIN_TWEETS = 5
_MAX_TWEETS = 15


class TestExportDispatcher:
    def test_dispatches_substack(self) -> None:
        piece = _make_piece()
        result = export(piece, "substack")
        assert result.startswith("# Democracy Is Not Self-Sustaining")

    def test_dispatches_medium(self) -> None:
        piece = _make_piece()
        result = export(piece, "medium")
        assert result.startswith("# Democracy Is Not Self-Sustaining")

    def test_dispatches_wordpress(self) -> None:
        piece = _make_piece()
        result = export(piece, "wordpress")
        assert "<!-- wp:heading" in result

    def test_dispatches_twitter(self) -> None:
        piece = _make_piece()
        result = export(piece, "twitter")
        assert "1/" in result

    def test_raises_value_error_for_unknown_format(self) -> None:
        piece = _make_piece()
        with pytest.raises(ValueError, match="Unknown export format"):
            export(piece, "myspace")

    def test_returns_non_empty_string_for_all_formats(self) -> None:
        piece = _make_piece()
        for fmt in ("substack", "medium", "wordpress", "twitter"):
            result = export(piece, fmt)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_substack_contains_mandatory_disclaimer(self) -> None:
        """export() for substack returns output containing mandatory disclaimer."""
        piece = _make_piece()
        result = export(piece, "substack")
        assert MANDATORY_DISCLAIMER in result

    def test_medium_contains_mandatory_disclaimer(self) -> None:
        """export() for medium returns output containing mandatory disclaimer."""
        piece = _make_piece()
        result = export(piece, "medium")
        assert MANDATORY_DISCLAIMER in result

    def test_wordpress_contains_mandatory_disclaimer(self) -> None:
        """export() for wordpress returns output containing mandatory disclaimer."""
        piece = _make_piece()
        result = export(piece, "wordpress")
        assert MANDATORY_DISCLAIMER in result

    def test_twitter_contains_mandatory_disclaimer(self) -> None:
        """export() for twitter returns output containing mandatory disclaimer."""
        piece = _make_piece()
        result = export(piece, "twitter")
        assert "AI-assisted rhetorical controls" in result


# ---------------------------------------------------------------------------
# BaseExporter._format_sources() tests
# ---------------------------------------------------------------------------

class TestFormatSources:
    def test_empty_sources_returns_empty_string(self) -> None:
        piece = _make_piece(sources=[])
        exporter = SubstackExporter()
        result = exporter._format_sources(piece)
        assert result == ""

    def test_two_citations_numbered_format(self) -> None:
        piece = _make_piece(sources=[_make_citation(1), _make_citation(2)])
        exporter = SubstackExporter()
        result = exporter._format_sources(piece)
        assert "## Sources & Claims" in result
        assert '1. "Claim number 1 as cited in the piece."' in result
        assert '2. "Claim number 2 as cited in the piece."' in result
        assert "\u2014" in result  # em-dash
        assert "accessed 2026-03-25" in result

    def test_citation_contains_markdown_link(self) -> None:
        piece = _make_piece(sources=[_make_citation(1)])
        exporter = SubstackExporter()
        result = exporter._format_sources(piece)
        assert "[Source 1](https://example.com/source-1)" in result


# ---------------------------------------------------------------------------
# BaseExporter._format_disclaimer() tests
# ---------------------------------------------------------------------------

class TestFormatDisclaimer:
    def test_format_disclaimer_returns_mandatory_constant(self) -> None:
        """_format_disclaimer() returns the fixed mandatory disclaimer constant."""
        piece = _make_piece()
        exporter = SubstackExporter()
        result = exporter._format_disclaimer(piece)
        assert result == MANDATORY_DISCLAIMER

    def test_format_disclaimer_not_dynamic(self) -> None:
        """_format_disclaimer() does not construct any dynamic disclaimer text."""
        piece = _make_piece(disclaimer="Some custom disclaimer")
        exporter = SubstackExporter()
        result = exporter._format_disclaimer(piece)
        # Must return the fixed constant, not piece.disclaimer
        assert result == MANDATORY_DISCLAIMER
        assert result != "Some custom disclaimer"

    def test_mandatory_disclaimer_text_is_correct(self) -> None:
        """The mandatory disclaimer constant matches the exact required text."""
        assert MANDATORY_DISCLAIMER == (
            "This piece was generated with AI-assisted rhetorical controls. "
            "It is original content and is not written by, endorsed by, or affiliated with any real person."
        )


# ---------------------------------------------------------------------------
# ImagePromptConfig tests
# ---------------------------------------------------------------------------

class TestImagePromptConfig:
    def test_defaults(self) -> None:
        cfg = ImagePromptConfig()
        assert cfg.style == "editorial"
        assert cfg.platform == "substack"
        assert cfg.custom_keywords == []

    def test_aspect_ratio_substack(self) -> None:
        assert ImagePromptConfig(platform="substack").aspect_ratio == "16:9"

    def test_aspect_ratio_medium(self) -> None:
        assert ImagePromptConfig(platform="medium").aspect_ratio == "16:9"

    def test_aspect_ratio_wordpress(self) -> None:
        assert ImagePromptConfig(platform="wordpress").aspect_ratio == "16:9"

    def test_aspect_ratio_twitter(self) -> None:
        assert ImagePromptConfig(platform="twitter").aspect_ratio == "16:9"

    def test_aspect_ratio_facebook(self) -> None:
        assert ImagePromptConfig(platform="facebook").aspect_ratio == "1.91:1"

    def test_aspect_ratio_instagram(self) -> None:
        assert ImagePromptConfig(platform="instagram").aspect_ratio == "1:1"

    def test_dimensions_substack(self) -> None:
        assert ImagePromptConfig(platform="substack").dimensions == (1456, 819)

    def test_dimensions_medium(self) -> None:
        assert ImagePromptConfig(platform="medium").dimensions == (1400, 788)

    def test_dimensions_wordpress(self) -> None:
        assert ImagePromptConfig(platform="wordpress").dimensions == (1200, 675)

    def test_dimensions_facebook(self) -> None:
        assert ImagePromptConfig(platform="facebook").dimensions == (1200, 628)

    def test_dimensions_twitter(self) -> None:
        assert ImagePromptConfig(platform="twitter").dimensions == (1600, 900)

    def test_dimensions_instagram(self) -> None:
        assert ImagePromptConfig(platform="instagram").dimensions == (1080, 1080)

    def test_invalid_style_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ImagePromptConfig(style="oil_painting")  # type: ignore[arg-type]

    def test_invalid_platform_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ImagePromptConfig(platform="myspace")  # type: ignore[arg-type]

    def test_all_styles_accepted(self) -> None:
        for style in ("photorealistic", "editorial", "cartoon", "minimalist", "vintage", "abstract"):
            cfg = ImagePromptConfig(style=style)  # type: ignore[arg-type]
            assert cfg.style == style

    def test_custom_keywords(self) -> None:
        cfg = ImagePromptConfig(custom_keywords=["gritty", "urban"])
        assert cfg.custom_keywords == ["gritty", "urban"]
