"""Shared pytest fixtures for the OpinionForge test suite.

Provides reusable fixtures for TopicContext, ModeProfile, ModeBlendConfig,
StanceConfig, mock LLM responses, mock search responses, and temporary
output files. All fixtures are documented with docstrings.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.models.config import ImagePromptConfig, ModeBlendConfig, StanceConfig
from opinionforge.models.mode import (
    ArgumentStructure,
    ModeProfile,
    ProsePatterns,
    VocabularyRegister,
)
from opinionforge.models.piece import GeneratedPiece, SourceCitation
from opinionforge.models.topic import TopicContext
from opinionforge.utils.search import SearchResult


# ---------------------------------------------------------------------------
# TopicContext fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_topic_context() -> TopicContext:
    """Return a valid TopicContext for testing.

    Provides a realistic politics-domain topic with claims and entities.
    """
    return TopicContext(
        raw_input="The impact of artificial intelligence on democratic institutions",
        input_type="text",
        title="The Impact of AI on Democratic Institutions",
        summary=(
            "Artificial intelligence is reshaping how democratic institutions "
            "operate, from election administration to policy analysis. "
            "Experts warn of both opportunities and risks."
        ),
        key_claims=[
            "According to a 2025 study, 73% of government agencies use AI tools.",
            "AI-generated misinformation has increased by 400% since 2023.",
        ],
        key_entities=["United States Congress", "European Union", "OpenAI"],
        subject_domain="technology",
    )


# ---------------------------------------------------------------------------
# ModeProfile fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mode_profile() -> ModeProfile:
    """Return a valid ModeProfile for testing.

    Provides a realistic analytical mode profile with all required fields.
    """
    return ModeProfile(
        id="analytical",
        display_name="Analytical",
        description="Data-driven, evidence-based argumentation with measured tone.",
        category="deliberative",
        prose_patterns=ProsePatterns(
            avg_sentence_length="varied",
            paragraph_length="medium",
            uses_fragments=False,
            uses_lists=True,
            opening_style="thesis_statement",
            closing_style="measured_conclusion",
        ),
        rhetorical_devices=["logos", "ethos", "qualified_assertion"],
        vocabulary_register=VocabularyRegister(
            formality="formal",
            word_origin_preference="mixed",
            jargon_level="moderate",
            profanity="never",
            humor_frequency="rare",
        ),
        argument_structure=ArgumentStructure(
            approach="deductive",
            evidence_style="data_heavy",
            concession_pattern="fair_then_rebut",
            thesis_placement="first_paragraph",
        ),
        signature_patterns=[
            "Opens with a clear statement of the problem or thesis",
            "Supports claims with data and expert authority",
            "Acknowledges counterarguments before rebutting them",
        ],
        suppressed_phrases=["obviously", "everyone knows", "it's clear that"],
        system_prompt_fragment=(
            "Write in an analytical rhetorical mode: prioritize data, evidence, "
            "and structured argumentation. Maintain a measured, authoritative tone. "
            "Present counterarguments fairly before rebutting them."
        ),
        few_shot_examples=[
            "The data here is unambiguous: three independent studies confirm the trend.",
            "While proponents argue X, the evidence suggests the mechanism works differently.",
        ],
    )


# ---------------------------------------------------------------------------
# ModeBlendConfig fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mode_blend_config() -> ModeBlendConfig:
    """Return a valid ModeBlendConfig for testing.

    Provides a two-mode blend of analytical (60%) and polemical (40%).
    """
    return ModeBlendConfig(modes=[("analytical", 60.0), ("polemical", 40.0)])


# ---------------------------------------------------------------------------
# StanceConfig fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_stance_config() -> StanceConfig:
    """Return a valid StanceConfig for testing.

    Provides a moderate equity-leaning position at -30 with standard intensity.
    """
    return StanceConfig(position=-30, intensity=0.5)


# ---------------------------------------------------------------------------
# GeneratedPiece fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_generated_piece(
    sample_topic_context: TopicContext,
    sample_mode_blend_config: ModeBlendConfig,
    sample_stance_config: StanceConfig,
) -> GeneratedPiece:
    """Return a valid GeneratedPiece for testing.

    Uses ModeBlendConfig and StanceConfig fixtures. Disclaimer equals the
    mandatory fixed disclaimer constant from opinionforge.core.generator.
    """
    return GeneratedPiece(
        id="test-uuid-1234",
        created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        topic=sample_topic_context,
        mode_config=sample_mode_blend_config,
        stance=sample_stance_config,
        target_length=800,
        actual_length=812,
        title="The Algorithmic Threat to Democracy",
        subtitle=None,
        body=(
            "It would be comforting to believe that the machinery of self-governance "
            "is immune to the disruptions of artificial intelligence. Comforting, "
            "but dangerously naive. The evidence before us suggests that democratic "
            "institutions face a challenge unlike any they have encountered since "
            "the invention of the printing press.\n\n"
            "The first and most obvious danger lies in the capacity of AI systems "
            "to generate misinformation at industrial scale."
        ),
        preview_text=(
            "It would be comforting to believe that the machinery of self-governance "
            "is immune to the disruptions of artificial intelligence."
        ),
        sources=[],
        research_queries=[],
        disclaimer=MANDATORY_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Mock LLM response fixture
# ---------------------------------------------------------------------------

_MOCK_LLM_GENERATED_TEXT = (
    "## The Algorithmic Threat to Democracy\n\n"
    "It would be comforting to believe that the machinery of self-governance "
    "is immune to the disruptions of artificial intelligence. Comforting, "
    "but dangerously naive. The evidence before us suggests that democratic "
    "institutions face a challenge unlike any they have encountered since "
    "the invention of the printing press.\n\n"
    "The first and most obvious danger lies in the capacity of AI systems "
    "to generate misinformation at industrial scale. Where once a propagandist "
    "required an army of scribes, a single algorithm can now produce a torrent "
    "of plausible falsehoods faster than any human fact-checker can debunk them.\n\n"
    "But the deeper threat is subtler. When AI systems are deployed to shape "
    "policy recommendations, they encode the biases of their creators into "
    "the very machinery of governance. The democratic principle that all "
    "citizens have an equal voice is undermined when an opaque algorithm "
    "determines which voices are amplified and which are suppressed.\n\n"
    "We must not, however, succumb to technophobic paralysis. The same tools "
    "that threaten democratic deliberation can also enhance it, provided we "
    "insist on transparency, accountability, and the primacy of human judgment "
    "over algorithmic convenience."
)


@pytest.fixture
def mock_llm_response() -> Generator[MagicMock, None, None]:
    """Patch LLM API calls to return a controlled, realistic response.

    The mock client returns a well-structured opinion piece with title,
    multiple paragraphs, and appropriate length for standard-format tests.
    """
    mock_client = MagicMock()
    mock_client.generate.return_value = _MOCK_LLM_GENERATED_TEXT
    with patch(
        "opinionforge.core.preview.create_llm_client",
        return_value=mock_client,
    ) as _:
        yield mock_client


# ---------------------------------------------------------------------------
# Mock search response fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search_response() -> Generator[MagicMock, None, None]:
    """Patch search API calls to return controlled search results.

    Provides realistic search results for testing the research pipeline
    without any network calls.
    """
    mock_client = MagicMock()
    mock_client.search.return_value = [
        SearchResult(
            url="https://www.nytimes.com/2025/01/15/technology/ai-democracy.html",
            title="AI and the Future of Democracy",
            snippet="According to a study by the Brookings Institution, 73 percent of government agencies now use some form of AI.",
            raw_content=(
                "Artificial intelligence is increasingly integrated into government operations. "
                "According to a study by the Brookings Institution, 73 percent of government "
                "agencies now use some form of AI in their daily work. The report found that "
                "AI tools are being used for everything from processing tax returns to "
                "analyzing policy proposals."
            ),
        ),
        SearchResult(
            url="https://www.washingtonpost.com/technology/2025/02/ai-elections/",
            title="How AI Could Shape the Next Election",
            snippet="Research shows AI-generated misinformation has increased by 400 percent since 2023.",
            raw_content=(
                "The proliferation of AI-generated content poses unprecedented challenges "
                "for election integrity. Research from Stanford University shows that "
                "AI-generated misinformation has increased by 400 percent since 2023, "
                "overwhelming traditional fact-checking organizations."
            ),
        ),
        SearchResult(
            url="https://www.reuters.com/technology/ai-governance-policy/",
            title="Global AI Governance Takes Shape",
            snippet="The European Union has taken the lead in regulating AI applications.",
            raw_content=None,
        ),
    ]

    with patch(
        "opinionforge.utils.search.get_search_client",
        return_value=mock_client,
    ) as _:
        yield mock_client


# ---------------------------------------------------------------------------
# Temporary output file fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_output_file(tmp_path: Path) -> Path:
    """Provide a temporary file path for testing file output.

    The file does not exist initially; tests can write to it and verify
    contents. Cleanup is handled automatically by pytest's tmp_path.
    """
    return tmp_path / "test_output.md"


# ---------------------------------------------------------------------------
# Mock fetch fixture (for URL fetching)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fetch() -> Generator[MagicMock, None, None]:
    """Patch the URL fetcher to return controlled content without network calls.

    Returns a successful FetchResult with realistic article text.
    """
    from opinionforge.utils.fetcher import FetchResult

    mock_result = FetchResult(
        url="https://example.com/article",
        title="Test Article",
        text=(
            "Artificial intelligence is transforming governance. According to a 2025 report, "
            "73 percent of federal agencies have adopted AI tools. The study found that "
            "AI-powered analysis reduced policy review times by 40 percent."
        ),
        fetched_at=datetime.now(timezone.utc),
        success=True,
    )
    with patch(
        "opinionforge.utils.fetcher.fetch_url",
        return_value=mock_result,
    ) as mock_fn:
        yield mock_fn
