"""Unit tests for URL fetching utility.

Minimum 6 test cases covering successful fetches, error handling,
and content extraction with mocked HTTP responses.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx
import pytest

from opinionforge.utils.fetcher import FetchResult, fetch_url


class TestFetchUrl:
    """Tests for the fetch_url function."""

    @patch("opinionforge.utils.fetcher.trafilatura")
    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_successful_fetch(self, mock_get: MagicMock, mock_traf: MagicMock) -> None:
        """Successful URL fetch with mocked httpx response returns FetchResult with success=True."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "<html><body>Article</body></html>"
        mock_get.return_value = mock_response

        mock_traf.extract.return_value = "Extracted article text content."
        mock_meta_obj = MagicMock()
        mock_meta_obj.title = "Test Article Title"
        mock_traf.extract_metadata.return_value = mock_meta_obj

        result = fetch_url("https://example.com/article")

        assert result.success is True
        assert result.url == "https://example.com/article"
        assert result.text == "Extracted article text content."
        assert result.title == "Test Article Title"
        assert result.error is None

    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_404_response(self, mock_get: MagicMock) -> None:
        """404 response returns FetchResult with success=False and error message."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_url("https://example.com/missing")

        assert result.success is False
        assert "404" in result.error
        assert result.text is None

    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_timeout(self, mock_get: MagicMock) -> None:
        """Timeout returns FetchResult with success=False."""
        mock_get.side_effect = httpx.TimeoutException("Connection timed out")

        result = fetch_url("https://example.com/slow")

        assert result.success is False
        assert "timed out" in result.error.lower()
        assert result.text is None

    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_connection_error(self, mock_get: MagicMock) -> None:
        """Connection failure returns FetchResult with success=False."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        result = fetch_url("https://example.com/down")

        assert result.success is False
        assert result.error is not None
        assert "Connection" in result.error

    @patch("opinionforge.utils.fetcher.trafilatura")
    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_extraction_failure(self, mock_get: MagicMock, mock_traf: MagicMock) -> None:
        """When trafilatura cannot extract content, returns success=False."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "<html><body></body></html>"
        mock_get.return_value = mock_response

        mock_traf.extract.return_value = None

        result = fetch_url("https://example.com/noarticle")

        assert result.success is False
        assert "extract" in result.error.lower()

    @patch("opinionforge.utils.fetcher.trafilatura")
    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_extracted_text_is_nonempty(self, mock_get: MagicMock, mock_traf: MagicMock) -> None:
        """Extracted text is non-empty for valid HTML content."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "<html><body>Content</body></html>"
        mock_get.return_value = mock_response

        mock_traf.extract.return_value = "This is meaningful extracted content from the article."
        mock_traf.extract_metadata.return_value = None

        result = fetch_url("https://example.com/article")

        assert result.success is True
        assert result.text is not None
        assert len(result.text) > 0

    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_ssl_error(self, mock_get: MagicMock) -> None:
        """SSL errors are handled gracefully."""
        mock_get.side_effect = httpx.HTTPError("SSL certificate verify failed")

        result = fetch_url("https://badssl.com/article")

        assert result.success is False
        assert result.error is not None

    @patch("opinionforge.utils.fetcher.httpx.get")
    def test_500_error(self, mock_get: MagicMock) -> None:
        """Server errors (500) return FetchResult with success=False."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = fetch_url("https://example.com/servererror")

        assert result.success is False
        assert "500" in result.error

    def test_fetch_result_has_fetched_at(self) -> None:
        """FetchResult always has a fetched_at timestamp."""
        with patch("opinionforge.utils.fetcher.httpx.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("timeout")
            result = fetch_url("https://example.com/timeout")
            assert result.fetched_at is not None
