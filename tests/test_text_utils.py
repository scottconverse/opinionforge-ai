"""Unit tests for text processing utilities.

Minimum 8 test cases covering word_count, truncate_text, and format_citations.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from opinionforge.models.piece import SourceCitation
from opinionforge.utils.text import format_citations, truncate_text, word_count


class TestWordCount:
    """Tests for the word_count function."""

    def test_word_count_normal_text(self) -> None:
        """word_count returns correct count for normal text."""
        assert word_count("The quick brown fox jumps over the lazy dog") == 9

    def test_word_count_single_word(self) -> None:
        """word_count returns 1 for a single word."""
        assert word_count("hello") == 1

    def test_word_count_empty_string(self) -> None:
        """word_count returns 0 for an empty string."""
        assert word_count("") == 0

    def test_word_count_whitespace_only(self) -> None:
        """word_count returns 0 for whitespace-only strings."""
        assert word_count("   \t\n  ") == 0

    def test_word_count_multiple_spaces(self) -> None:
        """word_count handles multiple spaces between words correctly."""
        assert word_count("hello    world   foo") == 3

    def test_word_count_newlines(self) -> None:
        """word_count counts words across newlines."""
        assert word_count("line one\nline two\nline three") == 6


class TestTruncateText:
    """Tests for the truncate_text function."""

    def test_truncate_at_word_boundary(self) -> None:
        """truncate_text truncates at a word boundary, not mid-word."""
        result = truncate_text("one two three four five", 3)
        assert result == "one two three"

    def test_truncate_returns_full_text_when_under_limit(self) -> None:
        """truncate_text returns the full text when word count is under the limit."""
        text = "short text"
        result = truncate_text(text, 100)
        assert result == text

    def test_truncate_exact_limit(self) -> None:
        """truncate_text returns full text when word count exactly equals the limit."""
        text = "one two three"
        result = truncate_text(text, 3)
        assert result == text

    def test_truncate_empty_string(self) -> None:
        """truncate_text handles empty string."""
        assert truncate_text("", 10) == ""

    def test_truncate_zero_max_words_raises(self) -> None:
        """truncate_text raises ValueError when max_words is 0."""
        with pytest.raises(ValueError, match="max_words must be a positive integer"):
            truncate_text("hello world", 0)

    def test_truncate_negative_max_words_raises(self) -> None:
        """truncate_text raises ValueError when max_words is negative."""
        with pytest.raises(ValueError, match="max_words must be a positive integer"):
            truncate_text("one two three", -1)


class TestFormatCitations:
    """Tests for the format_citations function."""

    def test_format_citations_prd_format(self) -> None:
        """format_citations produces the correct PRD format string."""
        sources = [
            SourceCitation(
                claim="60% of rural counties have lost their local newspaper",
                source_name="The New York Times",
                source_url="https://www.nytimes.com/article",
                accessed_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
                credibility_score=0.9,
            ),
        ]
        result = format_citations(sources)
        expected = (
            '"60% of rural counties have lost their local newspaper" '
            "-- [The New York Times](https://www.nytimes.com/article), "
            "accessed 2026-03-25"
        )
        assert result == expected

    def test_format_citations_multiple(self) -> None:
        """format_citations formats multiple citations separated by newlines."""
        sources = [
            SourceCitation(
                claim="Claim one",
                source_name="Source A",
                source_url="https://example.com/a",
                accessed_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
                credibility_score=0.8,
            ),
            SourceCitation(
                claim="Claim two",
                source_name="Source B",
                source_url="https://example.com/b",
                accessed_at=datetime(2026, 2, 20, tzinfo=timezone.utc),
                credibility_score=0.7,
            ),
        ]
        result = format_citations(sources)
        lines = result.split("\n")
        assert len(lines) == 2
        assert '"Claim one"' in lines[0]
        assert '"Claim two"' in lines[1]
        assert "accessed 2026-01-15" in lines[0]
        assert "accessed 2026-02-20" in lines[1]

    def test_format_citations_empty_list(self) -> None:
        """format_citations returns empty string for empty list."""
        assert format_citations([]) == ""

    def test_format_citations_includes_political_lean_source(self) -> None:
        """format_citations works with sources that have political_lean set."""
        sources = [
            SourceCitation(
                claim="A politically leaning claim",
                source_name="National Review",
                source_url="https://www.nationalreview.com/article",
                accessed_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
                political_lean="right",
                credibility_score=0.7,
            ),
        ]
        result = format_citations(sources)
        assert "[National Review]" in result
        assert "accessed 2026-03-01" in result
