"""Source research engine: web search, claim extraction, and citation formatting.

Orchestrates multi-query web searches, evaluates source credibility,
extracts claims, and produces structured research results with citations.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from opinionforge.models.config import StanceConfig
from opinionforge.models.piece import SourceCitation
from opinionforge.models.topic import TopicContext
from opinionforge.utils.fetcher import FetchResult, fetch_url
from opinionforge.utils.search import SearchClient, SearchResult, get_search_client
from opinionforge.utils.text import word_count


class ResearchResult(BaseModel):
    """Structured output of the research engine.

    Attributes:
        sources: List of source citations with claims.
        queries_used: Search queries that were executed.
        warning: Warning message if research is thin.
    """

    sources: list[SourceCitation]
    queries_used: list[str]
    warning: str | None = None


# --- Domain reputation and political lean data ---

# Credibility tiers: maps domain -> credibility score (0.0-1.0)
_CREDIBILITY_MAP: dict[str, float] = {
    # Tier 1: Major broadsheets and wire services
    "nytimes.com": 0.9,
    "washingtonpost.com": 0.9,
    "wsj.com": 0.9,
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "bbc.com": 0.9,
    "bbc.co.uk": 0.9,
    "economist.com": 0.9,
    "theguardian.com": 0.85,
    "ft.com": 0.9,
    "bloomberg.com": 0.9,
    "npr.org": 0.85,
    "pbs.org": 0.85,
    # Tier 2: Respected outlets
    "politico.com": 0.8,
    "theatlantic.com": 0.8,
    "newyorker.com": 0.8,
    "foreignaffairs.com": 0.85,
    "nature.com": 0.95,
    "science.org": 0.95,
    "usatoday.com": 0.75,
    "latimes.com": 0.8,
    "chicagotribune.com": 0.75,
    "time.com": 0.8,
    # Tier 3: Opinion-forward but factual
    "nationalreview.com": 0.7,
    "thenation.com": 0.7,
    "reason.com": 0.7,
    "jacobin.com": 0.65,
    "spectator.co.uk": 0.7,
    "foxnews.com": 0.6,
    "msnbc.com": 0.6,
    "cnn.com": 0.7,
    "vox.com": 0.7,
    # Tier 4: Think tanks and research
    "brookings.edu": 0.85,
    "heritage.org": 0.7,
    "cato.org": 0.7,
    "rand.org": 0.9,
    "pewresearch.org": 0.9,
    "cfr.org": 0.85,
}

# Political lean map: domain -> lean label
_POLITICAL_LEAN_MAP: dict[str, str] = {
    "nytimes.com": "center-left",
    "washingtonpost.com": "center-left",
    "wsj.com": "center-right",
    "reuters.com": "center",
    "apnews.com": "center",
    "bbc.com": "center",
    "bbc.co.uk": "center",
    "economist.com": "center",
    "theguardian.com": "left",
    "ft.com": "center",
    "bloomberg.com": "center",
    "npr.org": "center-left",
    "pbs.org": "center",
    "politico.com": "center",
    "theatlantic.com": "center-left",
    "newyorker.com": "center-left",
    "foreignaffairs.com": "center",
    "usatoday.com": "center",
    "latimes.com": "center-left",
    "time.com": "center-left",
    "nationalreview.com": "right",
    "thenation.com": "left",
    "reason.com": "right",
    "jacobin.com": "left",
    "foxnews.com": "right",
    "msnbc.com": "left",
    "cnn.com": "center-left",
    "vox.com": "left",
    "brookings.edu": "center-left",
    "heritage.org": "right",
    "cato.org": "center-right",
    "rand.org": "center",
    "pewresearch.org": "center",
    "cfr.org": "center",
}


def _extract_domain(url: str) -> str:
    """Extract the registrable domain from a URL.

    Args:
        url: A full URL.

    Returns:
        The domain (e.g., 'nytimes.com').
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Strip 'www.' prefix
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def score_credibility(url: str) -> float:
    """Score a source URL's credibility based on domain reputation.

    Uses a static mapping of known domains. Unknown domains receive a
    default score of 0.5.

    Args:
        url: The source URL.

    Returns:
        A credibility score between 0.0 and 1.0.
    """
    domain = _extract_domain(url)
    return _CREDIBILITY_MAP.get(domain, 0.5)


def tag_political_lean(url: str) -> str | None:
    """Tag a source URL with its political lean based on known publications.

    Uses a static mapping. Unknown sources return None.

    Args:
        url: The source URL.

    Returns:
        A political lean label or None if unknown.
    """
    domain = _extract_domain(url)
    return _POLITICAL_LEAN_MAP.get(domain)


def score_relevance(snippet: str, topic_title: str, topic_summary: str) -> float:
    """Score how relevant a search result snippet is to the topic.

    Uses simple keyword overlap heuristic.

    Args:
        snippet: The search result snippet text.
        topic_title: The topic title.
        topic_summary: The topic summary.

    Returns:
        A relevance score between 0.0 and 1.0.
    """
    if not snippet:
        return 0.0

    # Combine topic keywords
    topic_words = set(
        (topic_title + " " + topic_summary).lower().split()
    )
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "of", "in", "to", "for", "with", "on", "at", "from", "by",
        "about", "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "this", "that", "these", "those", "it", "its",
    }
    topic_words -= stop_words

    snippet_words = set(snippet.lower().split()) - stop_words

    if not topic_words:
        return 0.5

    overlap = len(topic_words & snippet_words)
    score = min(overlap / max(len(topic_words) * 0.3, 1), 1.0)
    return round(score, 2)


def score_recency(published_date: datetime | None, reference_date: datetime | None = None) -> float:
    """Score how recent a source is on a 0.0-1.0 scale.

    Uses the published date of an article (or fetched_at as a proxy when
    no publication date is available). Articles published within the last
    30 days score 1.0; scores decay linearly to 0.0 at 2 years old.

    Args:
        published_date: The publication or access date of the source.
            If None, a neutral score of 0.5 is returned.
        reference_date: The date to measure recency from (defaults to now UTC).

    Returns:
        A recency score between 0.0 and 1.0.
    """
    if published_date is None:
        return 0.5

    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    # Ensure both dates are timezone-aware
    if published_date.tzinfo is None:
        published_date = published_date.replace(tzinfo=timezone.utc)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    age_days = (reference_date - published_date).total_seconds() / 86400.0
    if age_days < 0:
        # Future-dated source — treat as very recent
        return 1.0

    # Full score for <= 30 days; linear decay to 0.0 at 730 days (2 years)
    max_age_days = 730.0
    if age_days <= 30:
        return 1.0
    if age_days >= max_age_days:
        return 0.0
    return round(1.0 - (age_days - 30) / (max_age_days - 30), 2)


def generate_search_queries(topic: TopicContext) -> list[str]:
    """Generate multiple search queries from a topic for comprehensive research.

    Produces at least 3 distinct queries covering factual background,
    recent developments, expert opinions, statistics, and counterarguments.

    Args:
        topic: The normalized topic context.

    Returns:
        A list of at least 3 search query strings.

    Raises:
        ValueError: If topic.title is empty or whitespace-only.
    """
    title = topic.title
    if not title or not title.strip():
        raise ValueError("Topic title must not be empty.")
    domain = topic.subject_domain

    queries = [
        # Factual background
        f"{title} facts background overview",
        # Recent developments
        f"{title} latest news developments 2024 2025 2026",
        # Expert opinions
        f"{title} expert analysis opinion",
        # Statistics and data
        f"{title} statistics data research study",
        # Counterarguments
        f"{title} criticism counterargument opposing view debate",
    ]

    # Add domain-specific query
    if domain and domain != "general":
        queries.append(f"{title} {domain} policy implications")

    # Add queries based on key claims if available
    if topic.key_claims:
        for claim in topic.key_claims[:2]:
            # Shorten claim for search
            claim_words = claim.split()[:8]
            queries.append(" ".join(claim_words))

    return queries


def _min_sources_for_length(target_words: int) -> int:
    """Calculate minimum required sources based on target piece length.

    3 for short op-eds (<=800 words), 8 for features (>=2500 words),
    scaled linearly between.

    Args:
        target_words: Target word count for the piece.

    Returns:
        Minimum number of sources required.
    """
    if target_words <= 800:
        return 3
    if target_words >= 2500:
        return 8
    # Linear interpolation between 3 and 8
    ratio = (target_words - 800) / (2500 - 800)
    return int(3 + ratio * 5)


def _extract_claims_simple(text: str, topic_title: str) -> list[str]:
    """Extract key claims from text using simple heuristics.

    Looks for sentences containing statistics, quotes, data points,
    and factual assertions.

    Args:
        text: The article text to extract claims from.
        topic_title: The topic title for relevance filtering.

    Returns:
        A list of claim strings.
    """
    import re

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: list[str] = []

    claim_indicators = [
        "percent", "%", "million", "billion", "trillion",
        "according to", "study", "research", "report",
        "data shows", "evidence", "found that", "survey",
        "statistics", "analysis", "estimated",
    ]

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 300:
            continue
        if any(indicator in sent.lower() for indicator in claim_indicators):
            claims.append(sent)
            if len(claims) >= 5:
                break

    return claims


def _source_name_from_url(url: str) -> str:
    """Derive a human-readable source name from a URL domain.

    Args:
        url: The source URL.

    Returns:
        A cleaned-up source name.
    """
    domain = _extract_domain(url)
    if not domain:
        return "Unknown Source"

    # Known display names
    _DISPLAY_NAMES: dict[str, str] = {
        "nytimes.com": "The New York Times",
        "washingtonpost.com": "The Washington Post",
        "wsj.com": "The Wall Street Journal",
        "reuters.com": "Reuters",
        "apnews.com": "Associated Press",
        "bbc.com": "BBC",
        "bbc.co.uk": "BBC",
        "theguardian.com": "The Guardian",
        "economist.com": "The Economist",
        "bloomberg.com": "Bloomberg",
        "ft.com": "Financial Times",
        "npr.org": "NPR",
        "pbs.org": "PBS",
        "politico.com": "Politico",
        "theatlantic.com": "The Atlantic",
        "newyorker.com": "The New Yorker",
        "foxnews.com": "Fox News",
        "cnn.com": "CNN",
        "msnbc.com": "MSNBC",
        "vox.com": "Vox",
        "nationalreview.com": "National Review",
        "thenation.com": "The Nation",
        "reason.com": "Reason",
        "pewresearch.org": "Pew Research Center",
    }

    if domain in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[domain]

    # Fallback: capitalize domain parts
    name = domain.split(".")[0].replace("-", " ").title()
    return name


def research_topic(
    topic: TopicContext,
    spectrum: StanceConfig,
    min_sources: int = 5,
    *,
    search_client: SearchClient | None = None,
    fetcher: callable | None = None,
    target_length: int = 800,
) -> ResearchResult:
    """Conduct multi-query research on a topic and return structured results.

    Generates search queries, executes searches, fetches source content,
    extracts claims, scores sources, and formats citations.

    Args:
        topic: The normalized topic context.
        spectrum: Stance configuration (affects source weighting).
        min_sources: Minimum number of sources to aim for.
        search_client: Optional search client for dependency injection.
        fetcher: Optional URL fetcher function for dependency injection.
        target_length: Target word count for the piece (affects min source count).

    Returns:
        A ResearchResult with sources, queries, and optional warning.
    """
    # Use injected or default search client
    if search_client is None:
        search_client = get_search_client()

    if fetcher is None:
        fetcher = fetch_url

    # Calculate minimum sources for piece length
    required_min = _min_sources_for_length(target_length)
    effective_min = max(min_sources, required_min)

    # Generate search queries
    queries = generate_search_queries(topic)

    # Deduplicate search results by URL
    seen_urls: set[str] = set()
    all_results: list[SearchResult] = []

    for query in queries:
        try:
            results = search_client.search(query, max_results=5)
        except SystemExit:
            # Re-raise SystemExit for API key / rate limit errors
            raise
        except Exception:
            # Gracefully handle search errors -- continue with other queries
            continue

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                all_results.append(result)

    # Handle zero results
    if not all_results:
        return ResearchResult(
            sources=[],
            queries_used=queries,
            warning="No search results found. Source research could not be conducted.",
        )

    # Build spectrum lean preference mapping.
    # spectrum.position ranges from -10 (hard left) to +10 (hard right).
    # At 0 (neutral) all leans are weighted equally.
    # At extremes, sources matching the position get a boost.
    _LEAN_POSITIONS: dict[str, float] = {
        "left": -7.0,
        "center-left": -3.5,
        "center": 0.0,
        "center-right": 3.5,
        "right": 7.0,
    }

    def _spectrum_weight(url: str, position: float) -> float:
        """Return a 0.9-1.1 multiplier based on how well the source lean
        matches the requested spectrum position."""
        lean = tag_political_lean(url)
        if lean is None or position == 0:
            return 1.0
        lean_pos = _LEAN_POSITIONS.get(lean, 0.0)
        # Distance: 0 = perfect match, 14 = maximum possible distance
        distance = abs(position - lean_pos)
        # Normalise to [0, 1] and map to weight [0.9, 1.1]
        normalized = distance / 14.0
        return round(1.1 - normalized * 0.2, 4)

    # Score and sort results
    scored: list[tuple[SearchResult, float, float]] = []
    for result in all_results:
        relevance = score_relevance(
            result.snippet, topic.title, topic.summary
        )
        credibility = score_credibility(result.url)
        spectrum_w = _spectrum_weight(result.url, spectrum.position)
        combined = (relevance * 0.6 + credibility * 0.4) * spectrum_w
        scored.append((result, relevance, combined))

    scored.sort(key=lambda x: x[2], reverse=True)

    # Take top results up to a reasonable limit
    max_to_fetch = min(len(scored), effective_min + 5)
    top_results = scored[:max_to_fetch]

    # Build source citations
    now = datetime.now(timezone.utc)
    sources: list[SourceCitation] = []

    for result, relevance, _ in top_results:
        # Fetch content for claim extraction
        fetch_result: FetchResult | None = None
        claims: list[str] = []
        published_date: datetime | None = None

        try:
            fetch_result = fetcher(result.url)
            if fetch_result.success and fetch_result.text:
                claims = _extract_claims_simple(fetch_result.text, topic.title)
            # Use fetched_at as the best available date proxy (guard against mocks)
            if fetch_result is not None and isinstance(fetch_result.fetched_at, datetime):
                published_date = fetch_result.fetched_at
        except Exception:
            pass

        # Use raw_content from search if fetch failed
        if not claims and result.raw_content:
            claims = _extract_claims_simple(result.raw_content, topic.title)

        # Use snippet as fallback claim
        if not claims and result.snippet:
            claims = [result.snippet]

        credibility = score_credibility(result.url)
        recency = score_recency(published_date, now)
        lean = tag_political_lean(result.url)
        source_name = _source_name_from_url(result.url)

        for claim in claims[:2]:  # Max 2 claims per source
            sources.append(
                SourceCitation(
                    claim=claim,
                    source_name=source_name,
                    source_url=result.url,
                    accessed_at=now,
                    political_lean=lean,
                    credibility_score=credibility,
                    recency_score=recency,
                )
            )

    # Deduplicate by claim text (keep first occurrence)
    seen_claims: set[str] = set()
    unique_sources: list[SourceCitation] = []
    for source in sources:
        claim_key = source.claim.lower().strip()
        if claim_key not in seen_claims:
            seen_claims.add(claim_key)
            unique_sources.append(source)

    # Generate warning if thin
    warning: str | None = None
    unique_urls = {s.source_url for s in unique_sources}
    if len(unique_urls) < effective_min:
        warning = (
            f"Thin research: found {len(unique_urls)} unique sources, "
            f"but {effective_min} are recommended for a {target_length}-word piece."
        )

    return ResearchResult(
        sources=unique_sources,
        queries_used=queries,
        warning=warning,
    )
