"""Web search API integration with adapters for Tavily, Brave Search, and SerpAPI.

Provides a common abstract interface for web search across multiple providers,
with a factory function that selects the correct client based on configuration.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from opinionforge.config import Settings, get_settings


class SearchResult(BaseModel):
    """A single web search result.

    Attributes:
        url: The URL of the search result.
        title: The title of the search result page.
        snippet: A short text snippet or description.
        raw_content: Full page content if available from the search API.
    """

    url: str
    title: str
    snippet: str
    raw_content: str | None = None


class SearchClient(ABC):
    """Abstract base class defining the search client interface."""

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Execute a web search and return results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            A list of SearchResult objects.

        Raises:
            SystemExit: With exit code 5 for missing/invalid API keys,
                or exit code 6 for rate limiting.
        """
        ...


class TavilySearchClient(SearchClient):
    """Search client using the Tavily search API via tavily-python.

    Args:
        api_key: The Tavily API key.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Execute a search using Tavily.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            A list of SearchResult objects.

        Raises:
            SystemExit: With exit code 5 for auth errors, exit code 6 for rate limits.
        """
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self._api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                include_raw_content=True,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if "unauthorized" in error_msg or "invalid" in error_msg or "api key" in error_msg:
                print(
                    f"Error: Tavily API key is invalid or unauthorized. "
                    f"Please check your OPINIONFORGE_SEARCH_API_KEY.",
                    file=sys.stderr,
                )
                raise SystemExit(5) from exc
            if "rate" in error_msg and "limit" in error_msg:
                print(
                    f"Error: Tavily API rate limit exceeded. Please wait and try again.",
                    file=sys.stderr,
                )
                raise SystemExit(6) from exc
            raise

        results: list[SearchResult] = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    raw_content=item.get("raw_content"),
                )
            )
        return results


class BraveSearchClient(SearchClient):
    """Search client using the Brave Search API via httpx.

    Args:
        api_key: The Brave Search API key.
    """

    _BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Execute a search using Brave Search API.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            A list of SearchResult objects.

        Raises:
            SystemExit: With exit code 5 for auth errors, exit code 6 for rate limits.
        """
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": max_results}

        try:
            response = httpx.get(
                self._BASE_URL,
                headers=headers,
                params=params,
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Brave Search API request failed: {exc}") from exc

        if response.status_code == 401 or response.status_code == 403:
            print(
                "Error: Brave Search API key is invalid or unauthorized. "
                "Please check your OPINIONFORGE_SEARCH_API_KEY.",
                file=sys.stderr,
            )
            raise SystemExit(5)

        if response.status_code == 429:
            print(
                "Error: Brave Search API rate limit exceeded. Please wait and try again.",
                file=sys.stderr,
            )
            raise SystemExit(6)

        response.raise_for_status()
        data = response.json()

        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                    raw_content=None,
                )
            )
        return results


class SerpAPISearchClient(SearchClient):
    """Search client using the SerpAPI via httpx.

    Args:
        api_key: The SerpAPI key.
    """

    _BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Execute a search using SerpAPI.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            A list of SearchResult objects.

        Raises:
            SystemExit: With exit code 5 for auth errors, exit code 6 for rate limits.
        """
        params: dict[str, Any] = {
            "q": query,
            "api_key": self._api_key,
            "engine": "google",
            "num": max_results,
        }

        try:
            response = httpx.get(
                self._BASE_URL,
                params=params,
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"SerpAPI request failed: {exc}") from exc

        if response.status_code == 401 or response.status_code == 403:
            print(
                "Error: SerpAPI key is invalid or unauthorized. "
                "Please check your OPINIONFORGE_SEARCH_API_KEY.",
                file=sys.stderr,
            )
            raise SystemExit(5)

        if response.status_code == 429:
            print(
                "Error: SerpAPI rate limit exceeded. Please wait and try again.",
                file=sys.stderr,
            )
            raise SystemExit(6)

        response.raise_for_status()
        data = response.json()

        results: list[SearchResult] = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append(
                SearchResult(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    raw_content=None,
                )
            )
        return results


def get_search_client(settings: Settings | None = None) -> SearchClient:
    """Factory function that returns the correct search client based on config.

    Args:
        settings: Application settings. Uses default settings if None.

    Returns:
        A SearchClient implementation for the configured provider.

    Raises:
        ValueError: If the configured search provider is not supported.
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.require_search_api_key()
    provider = settings.opinionforge_search_provider

    if provider == "tavily":
        return TavilySearchClient(api_key=api_key)
    elif provider == "brave":
        return BraveSearchClient(api_key=api_key)
    elif provider == "serpapi":
        return SerpAPISearchClient(api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported search provider: '{provider}'. "
            f"Valid providers: tavily, brave, serpapi."
        )
