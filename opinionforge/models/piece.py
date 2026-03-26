"""GeneratedPiece, ScreeningResult, and SourceCitation models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.topic import TopicContext


class SourceCitation(BaseModel):
    """A single source citation backing a factual claim.

    Attributes:
        claim: The claim as stated in the piece.
        source_name: Publication or site name.
        source_url: Full URL to the source.
        accessed_at: When the source was accessed.
        political_lean: Detected political lean if identifiable.
        credibility_score: Credibility rating from 0.0 to 1.0.
        recency_score: Recency rating from 0.0 (old/unknown) to 1.0 (very recent).
    """

    claim: str
    source_name: str
    source_url: str
    accessed_at: datetime
    political_lean: str | None = None
    credibility_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ScreeningResult(BaseModel):
    """Result of the similarity screening pass on a generated piece.

    Records what the screening module checked and whether the piece passed
    all checks without requiring a rewrite.

    Attributes:
        passed: True if the piece passed all screening checks.
        verbatim_matches: Number of verbatim n-gram matches found against source texts.
        near_verbatim_matches: Number of normalized near-verbatim matches found.
        suppressed_phrase_matches: Number of suppressed phrase matches found.
        structural_fingerprint_score: Cosine similarity score against known structural
            patterns (0.0 = no match, 1.0 = identical pattern).
        rewrite_iterations: Number of rewrite passes required to resolve violations.
        warning: Optional warning message when the piece passed with caveats.
    """

    passed: bool
    verbatim_matches: int
    near_verbatim_matches: int
    suppressed_phrase_matches: int
    structural_fingerprint_score: float
    rewrite_iterations: int
    warning: str | None = None


class GeneratedPiece(BaseModel):
    """A complete generated opinion piece with metadata.

    Attributes:
        id: UUID identifier for the piece.
        created_at: When the piece was generated.
        topic: The normalized topic context.
        mode_config: Rhetorical mode blend configuration used.
        stance: Stance and intensity controls used.
        target_length: Target word count.
        actual_length: Actual word count of the generated piece.
        title: Generated headline.
        subtitle: Optional subtitle or deck.
        body: The opinion piece text in markdown.
        preview_text: Two-to-three sentence preview.
        sources: List of source citations.
        research_queries: Search queries used during research.
        image_prompt: Generated image prompt if requested.
        image_platform: Target platform for image ratio.
        disclaimer: Mandatory fixed disclaimer string.
        screening_result: Similarity screening result, if screening was run.
    """

    id: str
    created_at: datetime
    topic: TopicContext
    mode_config: ModeBlendConfig
    stance: StanceConfig
    target_length: int
    actual_length: int

    title: str
    subtitle: str | None = None
    body: str
    preview_text: str

    sources: list[SourceCitation]
    research_queries: list[str]

    image_prompt: str | None = None
    image_platform: str | None = None

    disclaimer: str
    screening_result: ScreeningResult | None = None
