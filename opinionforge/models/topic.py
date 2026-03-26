"""TopicContext Pydantic model for normalized topic representation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class TopicContext(BaseModel):
    """Normalized topic representation consumed by all downstream features.

    Attributes:
        raw_input: Original user input text.
        input_type: How the topic was provided ('text', 'url', or 'file').
        title: Extracted or generated topic title.
        summary: Two-to-three sentence topic summary.
        key_claims: Extracted factual claims from the source.
        key_entities: People, organizations, and places mentioned.
        subject_domain: Domain classification (e.g. 'politics', 'economics').
        source_url: Original URL if the input was a URL.
        source_text: Full text if provided (truncated to 10k words).
        fetched_at: Timestamp when a URL was fetched.
    """

    raw_input: str
    input_type: Literal["text", "url", "file"]
    title: str
    summary: str
    key_claims: list[str]
    key_entities: list[str]
    subject_domain: str
    source_url: str | None = None
    source_text: str | None = None
    fetched_at: datetime | None = None
