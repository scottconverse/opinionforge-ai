"""Unit tests for search API client adapters.

Minimum 8 test cases covering Tavily, Brave, SerpAPI clients and the factory function.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import sys

import httpx
import pytest

from opinionforge.config import Settings
from opinionforge.utils.search import (
    BraveSearchClient,
    SearchResult,
    SerpAPISearchClient,
    TavilySearchClient,
    get_search_client,
)


class TestTavilySearchClient:
    """Tests for TavilySearchClient."""

    def test_returns_search_results(self) -> None:
        """TavilySearchClient returns list of SearchResult from mocked response."""
        mock_tavily_client = MagicMock()
        mock_tavily_client.search.return_value = {
            "results": [
                {
                    "url": "https://example.com/1",
                    "title": "Result 1",
                    "content": "Snippet one",
                    "raw_content": "Full content one",
                },
                {
                    "url": "https://example.com/2",
                    "title": "Result 2",
                    "content": "Snippet two",
                    "raw_content": None,
                },
            ]
        }

        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_tavily_client

        client = TavilySearchClient(api_key="test-key-placeholder")
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            results = client.search("test query", max_results=5)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].url == "https://example.com/1"
        assert results[0].title == "Result 1"
        assert results[0].snippet == "Snippet one"
        assert results[0].raw_content == "Full content one"

    def test_auth_error_exits_5(self) -> None:
        """TavilySearchClient exits with code 5 on unauthorized error."""
        mock_tavily_client = MagicMock()
        mock_tavily_client.search.side_effect = Exception("Unauthorized: invalid API key")

        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_tavily_client

        client = TavilySearchClient(api_key="test-key-placeholder")
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            with pytest.raises(SystemExit) as exc_info:
                client.search("test")
            assert exc_info.value.code == 5

    def test_rate_limit_exits_6(self) -> None:
        """TavilySearchClient exits with code 6 on rate limit error."""
        mock_tavily_client = MagicMock()
        mock_tavily_client.search.side_effect = Exception("Rate limit exceeded")

        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_tavily_client

        client = TavilySearchClient(api_key="test-key-placeholder")
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            with pytest.raises(SystemExit) as exc_info:
                client.search("test")
            assert exc_info.value.code == 6


class TestBraveSearchClient:
    """Tests for BraveSearchClient."""

    @patch("opinionforge.utils.search.httpx.get")
    def test_returns_search_results(self, mock_get: MagicMock) -> None:
        """BraveSearchClient returns list of SearchResult from mocked response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "url": "https://example.com/brave1",
                        "title": "Brave Result",
                        "description": "Brave snippet",
                    },
                ]
            }
        }
        mock_get.return_value = mock_response

        client = BraveSearchClient(api_key="test-key-placeholder")
        results = client.search("test query")

        assert len(results) == 1
        assert results[0].url == "https://example.com/brave1"
        assert results[0].title == "Brave Result"
        assert results[0].snippet == "Brave snippet"
        assert results[0].raw_content is None

    @patch("opinionforge.utils.search.httpx.get")
    def test_auth_error_exits_5(self, mock_get: MagicMock) -> None:
        """BraveSearchClient exits with code 5 on 401 response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        client = BraveSearchClient(api_key="test-key-placeholder")
        with pytest.raises(SystemExit) as exc_info:
            client.search("test")
        assert exc_info.value.code == 5

    @patch("opinionforge.utils.search.httpx.get")
    def test_rate_limit_exits_6(self, mock_get: MagicMock) -> None:
        """BraveSearchClient exits with code 6 on 429 response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        client = BraveSearchClient(api_key="test-key-placeholder")
        with pytest.raises(SystemExit) as exc_info:
            client.search("test")
        assert exc_info.value.code == 6


class TestSerpAPISearchClient:
    """Tests for SerpAPISearchClient."""

    @patch("opinionforge.utils.search.httpx.get")
    def test_returns_search_results(self, mock_get: MagicMock) -> None:
        """SerpAPISearchClient returns list of SearchResult from mocked response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "organic_results": [
                {
                    "link": "https://example.com/serp1",
                    "title": "SerpAPI Result",
                    "snippet": "SerpAPI snippet",
                },
            ]
        }
        mock_get.return_value = mock_response

        client = SerpAPISearchClient(api_key="test-key-placeholder")
        results = client.search("test query")

        assert len(results) == 1
        assert results[0].url == "https://example.com/serp1"
        assert results[0].title == "SerpAPI Result"
        assert results[0].snippet == "SerpAPI snippet"

    @patch("opinionforge.utils.search.httpx.get")
    def test_auth_error_exits_5(self, mock_get: MagicMock) -> None:
        """SerpAPISearchClient exits with code 5 on 403 response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        client = SerpAPISearchClient(api_key="test-key-placeholder")
        with pytest.raises(SystemExit) as exc_info:
            client.search("test")
        assert exc_info.value.code == 5

    @patch("opinionforge.utils.search.httpx.get")
    def test_rate_limit_exits_6(self, mock_get: MagicMock) -> None:
        """SerpAPISearchClient exits with code 6 on 429 response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        client = SerpAPISearchClient(api_key="test-key-placeholder")
        with pytest.raises(SystemExit) as exc_info:
            client.search("test")
        assert exc_info.value.code == 6


class TestGetSearchClient:
    """Tests for the get_search_client factory function."""

    def test_tavily_provider(self) -> None:
        """get_search_client returns TavilySearchClient for 'tavily' provider."""
        settings = Settings(
            opinionforge_search_api_key="test-key-placeholder",
            opinionforge_search_provider="tavily",
        )
        client = get_search_client(settings)
        assert isinstance(client, TavilySearchClient)

    def test_brave_provider(self) -> None:
        """get_search_client returns BraveSearchClient for 'brave' provider."""
        settings = Settings(
            opinionforge_search_api_key="test-key-placeholder",
            opinionforge_search_provider="brave",
        )
        client = get_search_client(settings)
        assert isinstance(client, BraveSearchClient)

    def test_serpapi_provider(self) -> None:
        """get_search_client returns SerpAPISearchClient for 'serpapi' provider."""
        settings = Settings(
            opinionforge_search_api_key="test-key-placeholder",
            opinionforge_search_provider="serpapi",
        )
        client = get_search_client(settings)
        assert isinstance(client, SerpAPISearchClient)

    def test_missing_api_key_exits_5(self) -> None:
        """get_search_client exits with code 5 when API key is missing."""
        settings = Settings(
            opinionforge_search_api_key=None,
            opinionforge_search_provider="tavily",
        )
        with pytest.raises(SystemExit) as exc_info:
            get_search_client(settings)
        assert exc_info.value.code == 5

    def test_invalid_provider(self) -> None:
        """get_search_client raises ValueError for an unknown provider string.

        Pydantic's Literal validator blocks invalid provider values through the
        normal Settings constructor, so we patch the provider attribute directly
        on a valid Settings instance to exercise the ValueError branch inside
        get_search_client().
        """
        settings = Settings(
            opinionforge_search_api_key="test-key-placeholder",
            opinionforge_search_provider="tavily",
        )
        # Override the validated field at the instance level to simulate
        # an unknown provider reaching get_search_client().
        object.__setattr__(settings, "opinionforge_search_provider", "unknown_provider")
        with pytest.raises(ValueError, match="Unsupported search provider"):
            get_search_client(settings)
