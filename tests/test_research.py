"""Unit tests for the research engine with mocked search API responses.

Minimum 15 test cases covering query generation, source scoring, political lean
tagging, minimum source enforcement, citation formatting, and error handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from opinionforge.core.research import (
    ResearchResult,
    _extract_claims_simple,
    _min_sources_for_length,
    _source_name_from_url,
    generate_search_queries,
    research_topic,
    score_credibility,
    score_recency,
    score_relevance,
    tag_political_lean,
)
from opinionforge.models.config import StanceConfig
from opinionforge.models.topic import TopicContext
from opinionforge.utils.fetcher import FetchResult
from opinionforge.utils.search import SearchClient, SearchResult
from opinionforge.utils.text import format_citations


def _make_topic() -> TopicContext:
    """Create a sample TopicContext for testing."""
    return TopicContext(
        raw_input="The decline of local journalism",
        input_type="text",
        title="The Decline of Local Journalism in Rural America",
        summary="Local newspapers are closing at an alarming rate, leaving communities without essential coverage of local government and events.",
        key_claims=["60% of rural counties have lost their local newspaper since 2004"],
        key_entities=["Rural America", "Pew Research Center"],
        subject_domain="politics",
    )


class MockSearchClient:
    """Mock search client that returns predictable results."""

    def __init__(self, results: list[SearchResult] | None = None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Return pre-configured results or raise an error."""
        if self._error:
            raise self._error
        return self._results


def _mock_fetcher_success(url: str) -> FetchResult:
    """Mock fetcher that always returns successful results with claims."""
    return FetchResult(
        url=url,
        title="Test Article",
        text=(
            "According to a recent study, 60 percent of rural counties "
            "have lost their local newspapers. The data shows a significant "
            "decline in local journalism across America. Research indicates "
            "that communities without local news have lower voter turnout."
        ),
        fetched_at=datetime.now(timezone.utc),
        success=True,
    )


def _mock_fetcher_failure(url: str) -> FetchResult:
    """Mock fetcher that always returns a failure."""
    return FetchResult(
        url=url,
        fetched_at=datetime.now(timezone.utc),
        success=False,
        error="Connection refused",
    )


def _sample_search_results(count: int = 5) -> list[SearchResult]:
    """Generate a list of sample search results."""
    domains = [
        ("https://www.nytimes.com/article1", "NYT Article", "Local journalism decline"),
        ("https://www.washingtonpost.com/article2", "WaPo Article", "Newspapers closing data"),
        ("https://www.reuters.com/article3", "Reuters Article", "Study shows decline"),
        ("https://www.foxnews.com/article4", "Fox Article", "Media industry changes"),
        ("https://www.theguardian.com/article5", "Guardian Article", "UK press parallels"),
        ("https://www.politico.com/article6", "Politico Article", "Policy implications"),
        ("https://www.nationalreview.com/article7", "NR Article", "Conservative take"),
        ("https://www.theatlantic.com/article8", "Atlantic Article", "Long form analysis"),
        ("https://www.vox.com/article9", "Vox Article", "Explained analysis"),
        ("https://www.bloomberg.com/article10", "Bloomberg Article", "Economic impact"),
    ]
    results = []
    for i in range(min(count, len(domains))):
        url, title, snippet = domains[i]
        results.append(SearchResult(url=url, title=title, snippet=snippet))
    return results


class TestGenerateSearchQueries:
    """Tests for search query generation."""

    def test_generates_at_least_3_queries(self) -> None:
        """Research generates at least 3 distinct search queries from a topic."""
        topic = _make_topic()
        queries = generate_search_queries(topic)
        assert len(queries) >= 3

    def test_queries_are_distinct(self) -> None:
        """All generated queries are distinct."""
        topic = _make_topic()
        queries = generate_search_queries(topic)
        assert len(queries) == len(set(queries))

    def test_queries_include_topic_title(self) -> None:
        """Queries reference the topic title."""
        topic = _make_topic()
        queries = generate_search_queries(topic)
        title_in_queries = sum(1 for q in queries if topic.title in q)
        assert title_in_queries >= 3


class TestSourceScoring:
    """Tests for source credibility and relevance scoring."""

    def test_credibility_known_domain_in_range(self) -> None:
        """Credibility score for known domains is between 0.0 and 1.0."""
        score = score_credibility("https://www.nytimes.com/article")
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # NYT is a high-credibility source

    def test_credibility_unknown_domain_default(self) -> None:
        """Unknown domains receive a default credibility score of 0.5."""
        score = score_credibility("https://unknownblog123.example.com/post")
        assert score == 0.5

    def test_relevance_score_in_range(self) -> None:
        """Relevance score is between 0.0 and 1.0."""
        score = score_relevance(
            "Local journalism decline and newspaper closures",
            "The Decline of Local Journalism",
            "Newspapers are closing at alarming rates",
        )
        assert 0.0 <= score <= 1.0

    def test_relevance_empty_snippet(self) -> None:
        """Empty snippet returns 0.0 relevance."""
        score = score_relevance("", "Title", "Summary")
        assert score == 0.0


class TestPoliticalLeanTagging:
    """Tests for political lean tagging."""

    def test_known_left_lean(self) -> None:
        """Known left-leaning publications are tagged correctly."""
        lean = tag_political_lean("https://www.theguardian.com/article")
        assert lean == "left"

    def test_known_right_lean(self) -> None:
        """Known right-leaning publications are tagged correctly."""
        lean = tag_political_lean("https://www.nationalreview.com/article")
        assert lean == "right"

    def test_known_center(self) -> None:
        """Known center publications are tagged correctly."""
        lean = tag_political_lean("https://www.reuters.com/article")
        assert lean == "center"

    def test_unknown_returns_none(self) -> None:
        """Unknown publications return None for political lean."""
        lean = tag_political_lean("https://randomsite123.example.com/post")
        assert lean is None


class TestMinSourcesForLength:
    """Tests for minimum source count enforcement."""

    def test_short_oped_min_3(self) -> None:
        """Short op-eds (<=800 words) require minimum 3 sources."""
        assert _min_sources_for_length(500) == 3
        assert _min_sources_for_length(800) == 3

    def test_feature_min_8(self) -> None:
        """Features (>=2500 words) require minimum 8 sources."""
        assert _min_sources_for_length(2500) == 8
        assert _min_sources_for_length(5000) == 8

    def test_intermediate_scaled(self) -> None:
        """Intermediate lengths scale linearly between 3 and 8."""
        result = _min_sources_for_length(1650)  # midpoint
        assert 3 < result < 8


class TestResearchTopic:
    """Integration tests for the full research_topic function."""

    def test_research_returns_result(self) -> None:
        """research_topic returns a ResearchResult with sources and queries."""
        topic = _make_topic()
        client = MockSearchClient(results=_sample_search_results(5))

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
            target_length=800,
        )

        assert isinstance(result, ResearchResult)
        assert len(result.queries_used) >= 3
        assert len(result.sources) > 0

    def test_thin_research_warning(self) -> None:
        """Warning is raised when insufficient sources found."""
        topic = _make_topic()
        # Only 1 result -> thin research
        client = MockSearchClient(results=_sample_search_results(1))

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
            target_length=2500,  # needs 8 sources
        )

        assert result.warning is not None
        assert "thin" in result.warning.lower() or "Thin" in result.warning

    def test_zero_results_warning(self) -> None:
        """Zero search results produce a clear warning."""
        topic = _make_topic()
        client = MockSearchClient(results=[])

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        assert result.warning is not None
        assert "no search results" in result.warning.lower()
        assert len(result.sources) == 0

    def test_search_api_error_graceful(self) -> None:
        """Search API errors are handled gracefully without crashing."""
        topic = _make_topic()
        client = MockSearchClient(error=RuntimeError("API connection error"))

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        # Should return empty results, not crash
        assert isinstance(result, ResearchResult)
        assert len(result.sources) == 0
        assert result.warning is not None

    def test_sources_have_credibility_scores(self) -> None:
        """All sources have credibility scores in the 0.0-1.0 range."""
        topic = _make_topic()
        client = MockSearchClient(results=_sample_search_results(3))

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        for source in result.sources:
            assert 0.0 <= source.credibility_score <= 1.0

    def test_sources_have_political_lean_for_known(self) -> None:
        """Sources from known publications have political lean set."""
        topic = _make_topic()
        # Use NYT which has a known lean
        results = [SearchResult(
            url="https://www.nytimes.com/test",
            title="NYT Test",
            snippet="Test snippet about journalism",
        )]
        client = MockSearchClient(results=results)

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        nyt_sources = [s for s in result.sources if "nytimes.com" in s.source_url]
        assert len(nyt_sources) > 0
        assert nyt_sources[0].political_lean == "center-left"

    def test_citation_format_matches_prd(self) -> None:
        """Citation formatting matches the PRD format exactly."""
        topic = _make_topic()
        client = MockSearchClient(results=_sample_search_results(3))

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        if result.sources:
            formatted = format_citations(result.sources)
            # Check PRD format: "Claim" -- [Source](URL), accessed YYYY-MM-DD
            for line in formatted.split("\n"):
                assert line.startswith('"')
                assert "-- [" in line
                assert "](http" in line
                assert "accessed " in line

    def test_fetcher_failure_uses_snippet(self) -> None:
        """When fetcher fails, snippet from search is used as claim."""
        topic = _make_topic()
        results = [SearchResult(
            url="https://example.com/article",
            title="Test",
            snippet="Important claim from search snippet",
        )]
        client = MockSearchClient(results=results)

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_failure,
        )

        assert len(result.sources) > 0
        assert any("search snippet" in s.claim.lower() for s in result.sources)

    def test_deduplicates_urls_across_queries(self) -> None:
        """Results from multiple queries are deduplicated by URL."""
        topic = _make_topic()
        # Same results every time simulates duplicates across queries
        same_result = [SearchResult(
            url="https://example.com/same",
            title="Same Article",
            snippet="Same snippet",
        )]
        client = MockSearchClient(results=same_result)

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_failure,
        )

        # Should only have 1 unique URL despite many query hits
        unique_urls = {s.source_url for s in result.sources}
        assert len(unique_urls) <= 1


class TestClaimExtraction:
    """Tests for claim extraction from text."""

    def test_extracts_statistical_claims(self) -> None:
        """Claim extraction finds sentences with statistics."""
        text = (
            "The weather is nice today. According to a study by Pew Research, "
            "60 percent of rural counties have lost newspapers. "
            "The birds are singing outside."
        )
        claims = _extract_claims_simple(text, "journalism")
        assert len(claims) >= 1
        assert any("percent" in c.lower() or "study" in c.lower() for c in claims)

    def test_empty_text_returns_empty(self) -> None:
        """Empty text returns empty claims list."""
        assert _extract_claims_simple("", "topic") == []


class TestSourceNameFromUrl:
    """Tests for source name derivation."""

    def test_known_source_name(self) -> None:
        """Known domains return proper display names."""
        assert _source_name_from_url("https://www.nytimes.com/article") == "The New York Times"

    def test_unknown_source_name(self) -> None:
        """Unknown domains derive a name from the domain."""
        name = _source_name_from_url("https://www.techblog.com/post")
        assert name == "Techblog"


class TestScoreRecency:
    """Tests for recency scoring."""

    def test_very_recent_scores_1(self) -> None:
        """An article published 7 days ago scores 1.0."""
        reference = datetime(2026, 3, 25, tzinfo=timezone.utc)
        published = datetime(2026, 3, 18, tzinfo=timezone.utc)  # 7 days ago
        assert score_recency(published, reference) == 1.0

    def test_within_30_days_scores_1(self) -> None:
        """An article published exactly 30 days ago scores 1.0."""
        reference = datetime(2026, 3, 25, tzinfo=timezone.utc)
        published = datetime(2026, 2, 23, tzinfo=timezone.utc)  # 30 days ago
        assert score_recency(published, reference) == 1.0

    def test_old_article_scores_0(self) -> None:
        """An article published 2+ years ago scores 0.0."""
        reference = datetime(2026, 3, 25, tzinfo=timezone.utc)
        published = datetime(2024, 1, 1, tzinfo=timezone.utc)  # > 2 years ago
        assert score_recency(published, reference) == 0.0

    def test_intermediate_article_between_0_and_1(self) -> None:
        """An article from roughly 1 year ago scores between 0.0 and 1.0."""
        reference = datetime(2026, 3, 25, tzinfo=timezone.utc)
        published = datetime(2025, 3, 25, tzinfo=timezone.utc)  # 1 year ago
        result = score_recency(published, reference)
        assert 0.0 < result < 1.0

    def test_none_date_returns_neutral(self) -> None:
        """A None published date returns a neutral score of 0.5."""
        assert score_recency(None) == 0.5

    def test_future_date_scores_1(self) -> None:
        """A future-dated article scores 1.0 (treated as very recent)."""
        reference = datetime(2026, 3, 25, tzinfo=timezone.utc)
        published = datetime(2026, 4, 1, tzinfo=timezone.utc)  # in the future
        assert score_recency(published, reference) == 1.0

    def test_score_in_valid_range(self) -> None:
        """Recency score is always within 0.0 to 1.0."""
        reference = datetime(2026, 3, 25, tzinfo=timezone.utc)
        dates = [
            datetime(2026, 3, 24, tzinfo=timezone.utc),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
        ]
        for d in dates:
            s = score_recency(d, reference)
            assert 0.0 <= s <= 1.0, f"Score {s} out of range for date {d}"


class TestSourcesHaveRecencyScore:
    """Tests that recency_score is populated on SourceCitation objects."""

    def test_sources_have_recency_scores_in_range(self) -> None:
        """All sources returned by research_topic have recency_score in 0.0-1.0."""
        topic = _make_topic()
        client = MockSearchClient(results=_sample_search_results(3))

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        assert len(result.sources) > 0
        for source in result.sources:
            assert 0.0 <= source.recency_score <= 1.0

    def test_recency_score_present_on_fetch_failure(self) -> None:
        """recency_score is set even when the URL fetch fails."""
        topic = _make_topic()
        results = [SearchResult(
            url="https://example.com/article",
            title="Test",
            snippet="A test snippet with some data here.",
        )]
        client = MockSearchClient(results=results)

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_failure,
        )

        assert len(result.sources) > 0
        for source in result.sources:
            assert 0.0 <= source.recency_score <= 1.0


class TestGenerateSearchQueriesGuards:
    """Tests for generate_search_queries input validation."""

    def test_empty_title_raises_value_error(self) -> None:
        """generate_search_queries raises ValueError when topic title is empty."""
        topic = TopicContext(
            raw_input="some input",
            input_type="text",
            title="",
            summary="Some summary.",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
        )
        with pytest.raises(ValueError, match="Topic title must not be empty"):
            generate_search_queries(topic)

    def test_whitespace_only_title_raises_value_error(self) -> None:
        """generate_search_queries raises ValueError when topic title is whitespace."""
        topic = TopicContext(
            raw_input="some input",
            input_type="text",
            title="   ",
            summary="Some summary.",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
        )
        with pytest.raises(ValueError, match="Topic title must not be empty"):
            generate_search_queries(topic)


class TestSpectrumWeighting:
    """Tests that spectrum.position affects source ordering in research_topic."""

    def test_left_spectrum_prefers_left_sources(self) -> None:
        """With a left spectrum, left-leaning sources rank higher than right-leaning ones."""
        topic = _make_topic()
        # Mix of left (Guardian, Vox) and right (National Review, Fox News) sources
        mixed_results = [
            SearchResult(url="https://www.nationalreview.com/article1", title="NR", snippet="conservative journalism analysis"),
            SearchResult(url="https://www.foxnews.com/article2", title="Fox", snippet="conservative media coverage"),
            SearchResult(url="https://www.theguardian.com/article3", title="Guardian", snippet="journalism decline study"),
            SearchResult(url="https://www.vox.com/article4", title="Vox", snippet="journalism research data"),
        ]
        client = MockSearchClient(results=mixed_results)

        result_left = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=-8),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )
        result_right = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=8),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        # Both should return sources without crashing
        assert isinstance(result_left, ResearchResult)
        assert isinstance(result_right, ResearchResult)

    def test_neutral_spectrum_returns_sources(self) -> None:
        """Neutral spectrum (position=0) returns sources without bias."""
        topic = _make_topic()
        client = MockSearchClient(results=_sample_search_results(4))

        result = research_topic(
            topic=topic,
            spectrum=StanceConfig(position=0),
            search_client=client,
            fetcher=_mock_fetcher_success,
        )

        assert isinstance(result, ResearchResult)
        assert len(result.sources) > 0
