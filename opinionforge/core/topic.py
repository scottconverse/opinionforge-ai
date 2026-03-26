"""Topic ingestion and analysis: text, URL, and file input normalization to TopicContext.

Supports three input methods:
- Plain text descriptions
- URLs (fetched with httpx, extracted with trafilatura)
- Local text files
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from opinionforge.models.topic import TopicContext


_MAX_WORDS = 10_000


def _validate_url(url: str) -> bool:
    """Check whether a string is a well-formed URL.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def _truncate_text(text: str, max_words: int = _MAX_WORDS) -> tuple[str, bool]:
    """Truncate text to a maximum word count.

    Args:
        text: The text to truncate.
        max_words: Maximum number of words to keep.

    Returns:
        A tuple of (possibly truncated text, whether truncation occurred).
    """
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True


def _extract_basic_metadata(text: str, input_type: str) -> dict[str, str | list[str]]:
    """Extract basic metadata from text without an LLM.

    Provides fallback metadata extraction using simple heuristics.
    In production, this would be enhanced with LLM-based extraction.

    Args:
        text: The source text to analyze.
        input_type: The input type ('text', 'url', 'file').

    Returns:
        A dictionary with title, summary, key_claims, key_entities,
        and subject_domain fields.
    """
    # Title: first sentence or line, truncated
    lines = text.strip().split("\n")
    first_line = lines[0].strip() if lines else text[:100]
    # Remove markdown headers
    title = re.sub(r"^#+\s*", "", first_line)
    if len(title) > 120:
        title = title[:117] + "..."

    # Summary: first 2-3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:3])
    if len(summary) > 500:
        summary = summary[:497] + "..."

    # Key claims: sentences containing claim-like patterns
    key_claims: list[str] = []
    for sent in sentences[:20]:
        if any(kw in sent.lower() for kw in ["according to", "study", "report", "data", "percent", "million", "billion"]):
            if len(sent) < 300:
                key_claims.append(sent.strip())
            if len(key_claims) >= 5:
                break

    # Key entities: capitalized multi-word sequences (basic NER)
    entity_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    entities = list(set(entity_pattern.findall(text)))[:10]

    # Subject domain: keyword-based classification
    domain_keywords = {
        "politics": ["congress", "senate", "election", "democrat", "republican", "president", "vote", "legislation", "policy"],
        "economics": ["economy", "gdp", "inflation", "market", "trade", "fiscal", "monetary", "employment", "recession"],
        "technology": ["ai", "artificial intelligence", "software", "tech", "digital", "algorithm", "data", "internet"],
        "health": ["health", "medical", "vaccine", "disease", "hospital", "patient", "treatment", "pandemic"],
        "environment": ["climate", "environment", "carbon", "emission", "pollution", "renewable", "sustainability"],
        "culture": ["culture", "art", "music", "film", "literature", "media", "society"],
        "education": ["education", "school", "university", "student", "teacher", "curriculum"],
        "foreign_policy": ["foreign", "diplomatic", "nato", "united nations", "treaty", "geopolitical"],
    }

    text_lower = text.lower()
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            domain_scores[domain] = score

    subject_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"

    return {
        "title": title,
        "summary": summary,
        "key_claims": key_claims,
        "key_entities": entities,
        "subject_domain": subject_domain,
    }


def ingest_text(text: str) -> TopicContext:
    """Normalize plain text input into a TopicContext.

    Args:
        text: The plain text topic input.

    Returns:
        A validated TopicContext instance.

    Raises:
        ValueError: If the text is empty or blank.
    """
    if not text or not text.strip():
        raise ValueError(
            "Topic text cannot be empty. Please provide a topic description, "
            "paste an article, or use --url to provide a URL."
        )

    text = text.strip()
    source_text, was_truncated = _truncate_text(text)

    metadata = _extract_basic_metadata(source_text, "text")

    summary = metadata["summary"]
    if was_truncated:
        summary = f"[Truncated to {_MAX_WORDS:,} words] {summary}"

    return TopicContext(
        raw_input=text[:1000],  # Store first 1000 chars of original
        input_type="text",
        title=metadata["title"],
        summary=summary,
        key_claims=metadata["key_claims"],
        key_entities=metadata["key_entities"],
        subject_domain=metadata["subject_domain"],
        source_text=source_text,
    )


def ingest_url(url: str) -> TopicContext:
    """Fetch URL content and produce a TopicContext.

    Uses httpx for fetching and trafilatura for content extraction.
    Falls back to treating the URL as topic text when fetch fails.

    Args:
        url: The URL to fetch and analyze.

    Returns:
        A validated TopicContext instance.

    Raises:
        ValueError: If the URL is malformed.
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")

    url = url.strip()
    if not _validate_url(url):
        raise ValueError(
            f"Invalid URL format: '{url}'. Please provide a valid URL "
            "starting with http:// or https://."
        )

    try:
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException):
        # Fallback: treat URL as topic text
        return TopicContext(
            raw_input=url,
            input_type="url",
            title=f"Topic from URL: {url}",
            summary=f"Unable to fetch content from {url}. Using URL as topic reference.",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
            source_url=url,
        )

    # Extract article content with trafilatura
    import trafilatura

    extracted = trafilatura.extract(response.text)

    if not extracted:
        # Fallback if trafilatura cannot extract
        return TopicContext(
            raw_input=url,
            input_type="url",
            title=f"Topic from URL: {url}",
            summary=f"Could not extract article content from {url}. Using URL as topic reference.",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
            source_url=url,
        )

    source_text, was_truncated = _truncate_text(extracted)
    metadata = _extract_basic_metadata(source_text, "url")

    summary = metadata["summary"]
    if was_truncated:
        summary = f"[Truncated to {_MAX_WORDS:,} words] {summary}"

    return TopicContext(
        raw_input=url,
        input_type="url",
        title=metadata["title"],
        summary=summary,
        key_claims=metadata["key_claims"],
        key_entities=metadata["key_entities"],
        subject_domain=metadata["subject_domain"],
        source_url=url,
        source_text=source_text,
        fetched_at=datetime.now(timezone.utc),
    )


def ingest_file(path: str) -> TopicContext:
    """Read a local text file and produce a TopicContext.

    Args:
        path: Path to a local text file.

    Returns:
        A validated TopicContext instance.

    Raises:
        FileNotFoundError: With exit code 1 context when the file does not exist.
        ValueError: If the file is empty.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: '{path}'. Please check the file path and try again."
        )

    text = file_path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(
            f"File '{path}' is empty. Please provide a file with topic content."
        )

    source_text, was_truncated = _truncate_text(text)
    metadata = _extract_basic_metadata(source_text, "file")

    summary = metadata["summary"]
    if was_truncated:
        summary = f"[Truncated to {_MAX_WORDS:,} words] {summary}"

    return TopicContext(
        raw_input=f"file://{file_path.resolve()}",
        input_type="file",
        title=metadata["title"],
        summary=summary,
        key_claims=metadata["key_claims"],
        key_entities=metadata["key_entities"],
        subject_domain=metadata["subject_domain"],
        source_text=source_text,
    )
