"""ModeProfile Pydantic model and sub-models for rhetorical mode profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProsePatterns(BaseModel):
    """Sentence and paragraph structure patterns for a rhetorical mode.

    Attributes:
        avg_sentence_length: Typical sentence length category.
        paragraph_length: Typical paragraph length category.
        uses_fragments: Whether the mode uses sentence fragments.
        uses_lists: Whether the mode uses inline or bulleted lists.
        opening_style: How pieces in this mode typically open.
        closing_style: How pieces in this mode typically close.
    """

    avg_sentence_length: Literal["short", "medium", "long", "varied"]
    paragraph_length: Literal["short", "medium", "long"]
    uses_fragments: bool
    uses_lists: bool
    opening_style: str
    closing_style: str


class VocabularyRegister(BaseModel):
    """Vocabulary and language register characteristics for a rhetorical mode.

    Attributes:
        formality: Level of formality in word choice.
        word_origin_preference: Preference for Anglo-Saxon vs Latinate words.
        jargon_level: Amount of domain-specific jargon used.
        profanity: Frequency of profanity usage.
        humor_frequency: How often humor appears in writing.
    """

    formality: Literal["colloquial", "conversational", "formal", "academic", "varied"]
    word_origin_preference: Literal["anglo_saxon", "latinate", "mixed"]
    jargon_level: Literal["none", "light", "moderate", "heavy"]
    profanity: Literal["never", "rare", "occasional", "frequent"]
    humor_frequency: Literal["never", "rare", "regular", "constant"]


class ArgumentStructure(BaseModel):
    """How a rhetorical mode builds and presents arguments.

    Attributes:
        approach: Primary argumentative approach.
        evidence_style: Preferred type of evidence.
        concession_pattern: How counterarguments are handled.
        thesis_placement: Where the main thesis appears in the piece.
    """

    approach: Literal["deductive", "inductive", "dialectical", "narrative", "mixed"]
    evidence_style: Literal[
        "data_heavy", "anecdotal", "expert_authority", "historical_analogy", "mixed"
    ]
    concession_pattern: Literal[
        "none", "brief_dismiss", "fair_then_rebut", "steelman_then_dismantle"
    ]
    thesis_placement: Literal[
        "first_paragraph", "after_setup", "end_reveal", "throughout"
    ]


class ModeProfile(BaseModel):
    """Complete profile for a single abstract rhetorical mode.

    No writer names, Wikipedia URLs, era, publication, or ideological_baseline fields.

    Attributes:
        id: Slug identifier (e.g. 'polemical').
        display_name: Human-readable mode name.
        description: One-sentence description of the mode.
        category: Category grouping (confrontational, investigative, deliberative, literary).
        prose_patterns: Sentence and paragraph structure patterns.
        rhetorical_devices: List of preferred rhetorical devices.
        vocabulary_register: Vocabulary and language register characteristics.
        argument_structure: Argumentative structure preferences.
        signature_patterns: Distinctive stylistic habits unique to this mode.
        suppressed_phrases: Phrases and constructions this mode avoids.
        system_prompt_fragment: Core prompt engineering text describing what to do rhetorically.
        few_shot_examples: Short original-composition examples of the mode's style.
    """

    id: str
    display_name: str
    description: str
    category: str

    prose_patterns: ProsePatterns
    rhetorical_devices: list[str] = Field(min_length=1)
    vocabulary_register: VocabularyRegister
    argument_structure: ArgumentStructure
    signature_patterns: list[str] = Field(min_length=1)
    suppressed_phrases: list[str] = Field(default_factory=list)

    system_prompt_fragment: str
    few_shot_examples: list[str] = Field(min_length=2)
