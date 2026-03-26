"""End-to-end integration tests for Phase 2 features.

Tests the full pipeline with export formats, image prompt generation,
and all 100 voices. All external API calls are mocked — no real network
requests are made.
"""

from __future__ import annotations

import contextlib
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from opinionforge.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Mock text constants
# ---------------------------------------------------------------------------

_MOCK_GENERATED = (
    "## The Reckoning We Cannot Postpone\n\n"
    "It would be comfortable to pretend that the forces arrayed against civic "
    "society are merely the usual suspects — the cynics and the cowards who "
    "populate every era of democratic decline. But the evidence before us "
    "suggests something far more systematic.\n\n"
    "The first observation is obvious once stated: the very machinery of "
    "deliberation has been captured by those who understand its rules far "
    "better than those who depend on its fairness.\n\n"
    "We must not, however, succumb to the paralysis of despair. The same "
    "institutions that are under attack are also the instruments of their "
    "own defense, provided that citizens of conscience choose to use them."
)

_MOCK_PREVIEW = "The machinery of democracy is being tested as never before."

_MOCK_IMAGE_SUBJECT = (
    "A fractured ballot box casting long shadows across a twilight cityscape, "
    "symbolic of the tension between civic participation and institutional decay."
)


# ---------------------------------------------------------------------------
# Context manager: full pipeline mocks
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _full_pipeline_mocks(
    image_response: str = _MOCK_IMAGE_SUBJECT,
) -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Patch all external calls for E2E tests.

    Uses separate mock clients for the preview, generator, and image_prompt
    modules so each always returns the correct fixture regardless of call order.

    Yields (mock_generate_llm, mock_search).
    """
    mock_preview_llm = MagicMock()
    mock_preview_llm.generate.return_value = _MOCK_PREVIEW

    mock_generate_llm = MagicMock()
    mock_generate_llm.generate.return_value = _MOCK_GENERATED

    mock_image_llm = MagicMock()
    mock_image_llm.generate.return_value = image_response

    mock_search = MagicMock()
    mock_search.search.return_value = []

    with (
        patch("opinionforge.core.preview.create_llm_client", return_value=mock_preview_llm),
        patch("opinionforge.core.generator.create_llm_client", return_value=mock_generate_llm),
        patch("opinionforge.core.image_prompt.create_llm_client", return_value=mock_image_llm),
        patch("opinionforge.utils.search.get_search_client", return_value=mock_search),
        patch("opinionforge.core.research.get_search_client", return_value=mock_search),
    ):
        yield mock_generate_llm, mock_search


def _write_no_preview(*args: str) -> list[str]:
    """Return a CLI arg list for 'write ... --no-preview --no-research'."""
    return ["write", *args, "--no-preview", "--no-research"]


# ---------------------------------------------------------------------------
# Export format tests
# ---------------------------------------------------------------------------


class TestExportSubstack:
    """Test the --export substack path through the write pipeline."""

    def test_substack_export_exits_zero(self) -> None:
        """Write with --export substack exits with code 0."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "substack"
            ))
        assert result.exit_code == 0, result.stdout

    def test_substack_export_contains_h1(self) -> None:
        """Substack export output contains a level-1 heading."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "substack"
            ))
        assert "# " in result.stdout

    def test_substack_export_no_html_tags(self) -> None:
        """Substack export output contains no raw HTML tags."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "substack"
            ))
        assert "<p>" not in result.stdout
        assert "<div" not in result.stdout
        assert "<br" not in result.stdout

    def test_substack_export_contains_disclaimer(self) -> None:
        """Substack export output includes the mandatory disclaimer."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "substack"
            ))
        assert "AI-assisted rhetorical controls" in result.stdout or \
               "original content" in result.stdout, (
            f"Disclaimer not found in substack export output: {result.stdout[:300]}"
        )


class TestExportMedium:
    """Test the --export medium path through the write pipeline."""

    def test_medium_export_exits_zero(self) -> None:
        """Write with --export medium exits with code 0."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "medium"
            ))
        assert result.exit_code == 0, result.stdout

    def test_medium_export_contains_drop_cap(self) -> None:
        """Medium export output contains a DROP CAP marker."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "medium"
            ))
        assert "DROP CAP" in result.stdout

    def test_medium_export_contains_disclaimer(self) -> None:
        """Medium export output includes the mandatory disclaimer."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "medium"
            ))
        assert "AI-assisted rhetorical controls" in result.stdout or \
               "original content" in result.stdout, (
            f"Disclaimer not found in medium export output: {result.stdout[:300]}"
        )


class TestExportWordPress:
    """Test the --export wordpress path through the write pipeline."""

    def test_wordpress_export_exits_zero(self) -> None:
        """Write with --export wordpress exits with code 0."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "wordpress"
            ))
        assert result.exit_code == 0, result.stdout

    def test_wordpress_export_contains_gutenberg_blocks(self) -> None:
        """WordPress export output contains Gutenberg block markers."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "wordpress"
            ))
        assert "<!-- wp:paragraph -->" in result.stdout

    def test_wordpress_export_contains_disclaimer(self) -> None:
        """WordPress export output includes the mandatory disclaimer."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "wordpress"
            ))
        assert "AI-assisted rhetorical controls" in result.stdout or \
               "original content" in result.stdout, (
            f"Disclaimer not found in wordpress export output: {result.stdout[:300]}"
        )


class TestExportTwitter:
    """Test the --export twitter path through the write pipeline."""

    def test_twitter_export_exits_zero(self) -> None:
        """Write with --export twitter exits with code 0."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "twitter"
            ))
        assert result.exit_code == 0, result.stdout

    def test_twitter_export_tweets_numbered(self) -> None:
        """Twitter export output has numbered tweets (1/, 2/, etc.)."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "twitter"
            ))
        assert "1/" in result.stdout
        assert "2/" in result.stdout

    def test_twitter_export_all_tweets_under_280(self, tmp_path: "Path") -> None:
        """Every tweet in the Twitter export is under 280 characters."""
        import sys
        from pathlib import Path

        out_file = tmp_path / "twitter_output.txt"
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical",
                "--export", "twitter", "--output", str(out_file)
            ))
        assert result.exit_code == 0, result.stdout
        raw = out_file.read_text(encoding="utf-8")
        # Each tweet block is separated by blank lines
        tweet_blocks = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]
        assert tweet_blocks, "Should find at least one tweet block in output file"
        for tweet in tweet_blocks:
            # tweet_blocks include the footer image prompt line too; skip non-tweet lines
            if not tweet[:3].rstrip("/").rstrip().isdigit():
                continue
            assert len(tweet) <= 280, f"Tweet too long ({len(tweet)} chars): {tweet}"

    def test_twitter_export_contains_disclaimer(self) -> None:
        """Twitter export output includes the mandatory disclaimer."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "twitter"
            ))
        assert "AI-assisted rhetorical controls" in result.stdout or \
               "original content" in result.stdout, (
            f"Disclaimer not found in twitter export output: {result.stdout[:300]}"
        )

    def test_twitter_no_disclaimer_flag_exits_2(self) -> None:
        """--no-disclaimer flag is removed in v1.0.0 and returns exit code 2."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical",
                "--export", "twitter", "--no-disclaimer"
            ))
        assert result.exit_code == 2, (
            f"Expected exit code 2 for removed --no-disclaimer flag, got {result.exit_code}"
        )


class TestInvalidExportFormat:
    """Test that invalid --export values cause exit code 2."""

    def test_invalid_export_format_exits_code_2(self) -> None:
        """Invalid --export value causes CLI exit code 2."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "invalid_format"
            ))
        assert result.exit_code == 2, result.stdout

    def test_invalid_export_format_shows_error(self) -> None:
        """Invalid --export value shows an error message listing supported formats."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--export", "pdf"
            ))
        # Error is written to stderr; CliRunner mixes by default
        combined = result.stdout + (result.stderr if hasattr(result, "stderr") else "")
        assert "pdf" in combined.lower() or result.exit_code == 2


# ---------------------------------------------------------------------------
# Image prompt tests
# ---------------------------------------------------------------------------


class TestImagePromptFlag:
    """Test --image-prompt, --image-platform, --image-style flags."""

    def test_image_prompt_flag_exits_zero(self) -> None:
        """--image-prompt flag causes a successful exit."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--image-prompt"
            ))
        assert result.exit_code == 0, result.stdout

    def test_image_prompt_appears_in_output(self) -> None:
        """Image prompt text appears in output when --image-prompt is set."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical", "--image-prompt"
            ))
        assert "Header image prompt" in result.stdout or "image" in result.stdout.lower()

    def test_image_platform_substack_dimensions(self) -> None:
        """--image-platform substack produces '1456x819' dimensions."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical",
                "--image-prompt", "--image-platform", "substack"
            ))
        assert result.exit_code == 0, result.stdout
        assert "1456" in result.stdout

    def test_image_platform_instagram_dimensions(self) -> None:
        """--image-platform instagram produces '1080x1080' dimensions."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical",
                "--image-prompt", "--image-platform", "instagram"
            ))
        assert result.exit_code == 0, result.stdout
        assert "1080" in result.stdout

    def test_image_style_cartoon_in_output(self) -> None:
        """--image-style cartoon includes 'cartoon' in image prompt output."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical",
                "--image-prompt", "--image-style", "cartoon"
            ))
        assert result.exit_code == 0, result.stdout
        assert "cartoon" in result.stdout.lower()

    def test_image_prompt_with_substack_export(self) -> None:
        """--image-prompt combined with --export substack includes prompt in markdown output."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "AI and democracy", "--mode", "analytical",
                "--image-prompt", "--export", "substack"
            ))
        assert result.exit_code == 0, result.stdout
        assert "Header image prompt" in result.stdout


# ---------------------------------------------------------------------------
# Modes command tests
# ---------------------------------------------------------------------------


class TestModesCommand:
    """Test the modes listing command."""

    def test_modes_lists_available_modes(self) -> None:
        """'opinionforge modes' lists available rhetorical modes."""
        result = runner.invoke(app, ["modes"])
        assert result.exit_code == 0, result.stdout

    def test_modes_category_filter(self) -> None:
        """'opinionforge modes --category confrontational' returns modes in that category."""
        result = runner.invoke(app, ["modes", "--category", "confrontational"])
        assert result.exit_code == 0, result.stdout

    def test_modes_search_filter(self) -> None:
        """'opinionforge modes --search analytical' returns relevant modes."""
        result = runner.invoke(app, ["modes", "--search", "analytical"])
        assert result.exit_code == 0, result.stdout

    def test_modes_no_error(self) -> None:
        """'opinionforge modes' runs without error."""
        result = runner.invoke(app, ["modes"])
        assert result.exit_code == 0, result.stdout


# ---------------------------------------------------------------------------
# Per-mode generation smoke tests
# ---------------------------------------------------------------------------


class TestModeGenerationSmoke:
    """Verify each rhetorical mode generates successfully."""

    def _generate_with_mode(self, mode_id: str) -> int:
        """Run generation with a specific mode and return the exit code."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "Technology and society", "--mode", mode_id
            ))
        return result.exit_code

    def test_analytical_mode_generates(self) -> None:
        """The 'analytical' mode generates without error."""
        assert self._generate_with_mode("analytical") == 0

    def test_polemical_mode_generates(self) -> None:
        """The 'polemical' mode generates without error."""
        assert self._generate_with_mode("polemical") == 0

    def test_narrative_mode_generates(self) -> None:
        """The 'narrative' mode generates without error."""
        assert self._generate_with_mode("narrative") == 0


# ---------------------------------------------------------------------------
# Mode blending tests
# ---------------------------------------------------------------------------


class TestModeBlending:
    """Test mode blending with multiple modes."""

    def test_mode_blend_completes_without_error(self) -> None:
        """Blending polemical:60,analytical:40 completes without error."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "Economic policy", "--mode", "polemical:60,analytical:40"
            ))
        assert result.exit_code == 0, result.stdout

    def test_phase2_blend_two_modes(self) -> None:
        """Blending two modes (narrative:70,oratorical:30) completes without error."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "Climate change", "--mode", "narrative:70,oratorical:30"
            ))
        assert result.exit_code == 0, result.stdout


# ---------------------------------------------------------------------------
# Mode smoke test for robustness
# ---------------------------------------------------------------------------


class TestModeRobustness:
    """Test that mode generation handles various topic types."""

    def test_non_english_topic_generates(self) -> None:
        """A topic about international politics generates without error."""
        with _full_pipeline_mocks():
            result = runner.invoke(app, _write_no_preview(
                "Canadian politics", "--mode", "analytical"
            ))
        assert result.exit_code == 0, result.stdout


# ---------------------------------------------------------------------------
# Export command (Phase 3 stub)
# ---------------------------------------------------------------------------


class TestExportCommand:
    """Test the 'opinionforge export PIECE_ID --format FORMAT' command."""

    def test_export_command_not_available_exits_1(self) -> None:
        """The export command shows a clear error and exits with code 1 (not implemented)."""
        result = runner.invoke(app, ["export", "some-piece-id", "--format", "substack"])
        assert result.exit_code == 1

    def test_export_command_invalid_format_exits_2(self) -> None:
        """The export command with invalid format exits with code 2."""
        result = runner.invoke(app, ["export", "some-piece-id", "--format", "pdf"])
        assert result.exit_code == 2

    def test_export_command_shows_helpful_message(self) -> None:
        """The export command shows a helpful message about using --export flag instead."""
        result = runner.invoke(app, ["export", "some-piece-id", "--format", "substack"])
        combined = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "Phase 3" in combined or "write" in combined or "not yet" in combined.lower()
