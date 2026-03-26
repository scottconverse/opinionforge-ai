"""URL fetching and HTML content extraction utility.

Uses httpx for HTTP requests and trafilatura for article text extraction
from HTML pages.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import trafilatura
from pydantic import BaseModel


_USER_AGENT = "OpinionForge/0.1 (https://github.com/opinionforge; research bot)"
_TIMEOUT_SECONDS = 30.0


class FetchResult(BaseModel):
    """Result of fetching and extracting content from a URL.

    Attributes:
        url: The URL that was fetched.
        title: Extracted article title, if available.
        text: Extracted article text content.
        fetched_at: Timestamp when the URL was fetched.
        success: Whether the fetch and extraction succeeded.
        error: Error message if the fetch failed.
    """

    url: str
    title: str | None = None
    text: str | None = None
    fetched_at: datetime
    success: bool
    error: str | None = None


def fetch_url(url: str) -> FetchResult:
    """Fetch a URL and extract article content using trafilatura.

    Args:
        url: The URL to fetch.

    Returns:
        A FetchResult with extracted content or error information.
    """
    now = datetime.now(timezone.utc)

    try:
        response = httpx.get(
            url,
            timeout=_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
    except httpx.TimeoutException:
        return FetchResult(
            url=url,
            fetched_at=now,
            success=False,
            error=f"Request timed out after {_TIMEOUT_SECONDS} seconds.",
        )
    except httpx.ConnectError as exc:
        return FetchResult(
            url=url,
            fetched_at=now,
            success=False,
            error=f"Connection failed: {exc}",
        )
    except httpx.HTTPError as exc:
        return FetchResult(
            url=url,
            fetched_at=now,
            success=False,
            error=f"HTTP error: {exc}",
        )

    if response.status_code == 404:
        return FetchResult(
            url=url,
            fetched_at=now,
            success=False,
            error="HTTP 404: content not found at this URL.",
        )

    if response.status_code >= 400:
        return FetchResult(
            url=url,
            fetched_at=now,
            success=False,
            error=f"HTTP error {response.status_code}.",
        )

    # Extract article content with trafilatura
    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=False,
    )

    if not extracted:
        return FetchResult(
            url=url,
            fetched_at=now,
            success=False,
            error="Could not extract article content from the page.",
        )

    # Attempt to extract the title
    metadata = trafilatura.extract_metadata(response.text)
    title = metadata.title if metadata and metadata.title else None

    return FetchResult(
        url=url,
        title=title,
        text=extracted,
        fetched_at=now,
        success=True,
    )
