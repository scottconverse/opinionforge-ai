"""Unit tests for topic ingestion (text, URL, file).

Minimum 12 test cases covering text ingestion, URL ingestion with mocked
httpx, URL fallback on errors, file ingestion, missing files, empty input,
and text truncation.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from opinionforge.core.topic import (
    ingest_text,
    ingest_url,
    ingest_file,
    _validate_url,
    _truncate_text,
)
from opinionforge.models.topic import TopicContext


# ---------------------------------------------------------------------------
# Text ingestion tests
# ---------------------------------------------------------------------------


class TestIngestText:
    """Tests for ingest_text function."""

    def test_valid_text_produces_topic_context(self) -> None:
        """Plain text input produces a valid TopicContext."""
        result = ingest_text("The collapse of local journalism in rural America")
        assert isinstance(result, TopicContext)
        assert result.input_type == "text"
        assert result.title
        assert result.summary

    def test_text_preserves_raw_input(self) -> None:
        """TopicContext stores the raw input text."""
        text = "Climate change and its impact on coastal cities"
        result = ingest_text(text)
        assert text in result.raw_input

    def test_empty_text_raises_value_error(self) -> None:
        """Empty text raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ingest_text("")

    def test_whitespace_only_raises_value_error(self) -> None:
        """Whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ingest_text("   \n\t  ")

    def test_text_with_claims_extracts_metadata(self) -> None:
        """Text with claim-like sentences extracts key claims."""
        text = (
            "According to a 2024 study, 60 percent of rural counties have lost "
            "their local newspaper. The report shows that communities without "
            "local journalism see higher corruption rates."
        )
        result = ingest_text(text)
        assert result.subject_domain  # should classify into a domain

    def test_text_truncation_at_10000_words(self) -> None:
        """Text longer than 10,000 words is truncated with a summarization note."""
        long_text = "word " * 15_000
        result = ingest_text(long_text)
        source_words = result.source_text.split() if result.source_text else []
        assert len(source_words) <= 10_000
        assert "Truncated" in result.summary


# ---------------------------------------------------------------------------
# URL validation tests
# ---------------------------------------------------------------------------


class TestURLValidation:
    """Tests for URL validation."""

    def test_valid_http_url(self) -> None:
        """Valid HTTP URL passes validation."""
        assert _validate_url("http://example.com/article") is True

    def test_valid_https_url(self) -> None:
        """Valid HTTPS URL passes validation."""
        assert _validate_url("https://www.nytimes.com/2024/01/01/opinion/test.html") is True

    def test_malformed_url_rejected(self) -> None:
        """Malformed URL is rejected."""
        assert _validate_url("not-a-url") is False

    def test_missing_scheme_rejected(self) -> None:
        """URL without scheme is rejected."""
        assert _validate_url("example.com/article") is False


# ---------------------------------------------------------------------------
# URL ingestion tests
# ---------------------------------------------------------------------------


class TestIngestUrl:
    """Tests for ingest_url function."""

    def test_url_ingestion_with_mocked_response(self) -> None:
        """URL ingestion with mocked httpx response produces valid TopicContext."""
        mock_html = "<html><body><h1>Test Article</h1><p>This is a test article about politics and government policy reform. According to a new study, policy changes affect millions of citizens.</p></body></html>"

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("opinionforge.core.topic.httpx.get", return_value=mock_response):
            with patch("trafilatura.extract", return_value="This is a test article about politics and government policy reform. According to a new study, policy changes affect millions of citizens."):
                result = ingest_url("https://example.com/test-article")

        assert isinstance(result, TopicContext)
        assert result.input_type == "url"
        assert result.source_url == "https://example.com/test-article"

    def test_url_ingestion_fallback_on_404(self) -> None:
        """URL ingestion falls back to treating URL as topic text on 404."""
        import httpx as httpx_mod

        with patch("opinionforge.core.topic.httpx.get", side_effect=httpx_mod.HTTPStatusError("404", request=MagicMock(), response=MagicMock())):
            result = ingest_url("https://example.com/missing-page")

        assert isinstance(result, TopicContext)
        assert result.input_type == "url"
        assert "Unable to fetch" in result.summary or "topic reference" in result.summary.lower()

    def test_url_ingestion_fallback_on_timeout(self) -> None:
        """URL ingestion falls back on timeout."""
        import httpx as httpx_mod

        with patch("opinionforge.core.topic.httpx.get", side_effect=httpx_mod.TimeoutException("timeout")):
            result = ingest_url("https://example.com/slow-page")

        assert isinstance(result, TopicContext)
        assert result.source_url == "https://example.com/slow-page"

    def test_empty_url_raises_error(self) -> None:
        """Empty URL raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ingest_url("")

    def test_malformed_url_raises_error(self) -> None:
        """Malformed URL raises ValueError before attempting fetch."""
        with pytest.raises(ValueError, match="Invalid URL"):
            ingest_url("not-a-valid-url")


# ---------------------------------------------------------------------------
# File ingestion tests
# ---------------------------------------------------------------------------


class TestIngestFile:
    """Tests for ingest_file function."""

    def test_file_ingestion_with_existing_file(self, tmp_path) -> None:
        """File ingestion with an existing file produces valid TopicContext."""
        test_file = tmp_path / "topic.txt"
        test_file.write_text(
            "The future of artificial intelligence in healthcare. "
            "According to recent data, AI diagnostics improve accuracy by 30 percent.",
            encoding="utf-8",
        )

        result = ingest_file(str(test_file))
        assert isinstance(result, TopicContext)
        assert result.input_type == "file"
        assert result.source_text

    def test_file_ingestion_missing_file_raises_error(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            ingest_file("/nonexistent/path/topic.txt")

    def test_file_ingestion_empty_file_raises_error(self, tmp_path) -> None:
        """Empty file raises ValueError."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            ingest_file(str(empty_file))


# ---------------------------------------------------------------------------
# Truncation tests
# ---------------------------------------------------------------------------


class TestTruncation:
    """Tests for text truncation logic."""

    def test_short_text_not_truncated(self) -> None:
        """Text under 10,000 words is not truncated."""
        text = "word " * 100
        result, was_truncated = _truncate_text(text)
        assert not was_truncated
        assert len(result.split()) == 100

    def test_long_text_truncated(self) -> None:
        """Text over 10,000 words is truncated to exactly 10,000."""
        text = "word " * 15_000
        result, was_truncated = _truncate_text(text)
        assert was_truncated
        assert len(result.split()) == 10_000
