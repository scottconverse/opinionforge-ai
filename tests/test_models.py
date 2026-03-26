"""Unit tests for all Pydantic data models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.models import (
    ArgumentStructure,
    GeneratedPiece,
    ImagePromptConfig,
    ModeBlendConfig,
    ModeProfile,
    ProsePatterns,
    SourceCitation,
    StanceConfig,
    TopicContext,
    VocabularyRegister,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_topic_context(**overrides: object) -> dict:
    """Return a valid TopicContext data dict with optional overrides."""
    base = {
        "raw_input": "The collapse of local journalism",
        "input_type": "text",
        "title": "Local Journalism Collapse",
        "summary": "Local newspapers are closing at an alarming rate.",
        "key_claims": ["Over 2,500 newspapers closed since 2005"],
        "key_entities": ["United States", "Gannett"],
        "subject_domain": "media",
    }
    base.update(overrides)
    return base


def _make_prose_patterns(**overrides: object) -> dict:
    base = {
        "avg_sentence_length": "medium",
        "paragraph_length": "medium",
        "uses_fragments": False,
        "uses_lists": False,
        "opening_style": "declarative",
        "closing_style": "call to action",
    }
    base.update(overrides)
    return base


def _make_vocabulary_register(**overrides: object) -> dict:
    base = {
        "formality": "formal",
        "word_origin_preference": "mixed",
        "jargon_level": "light",
        "profanity": "never",
        "humor_frequency": "rare",
    }
    base.update(overrides)
    return base


def _make_argument_structure(**overrides: object) -> dict:
    base = {
        "approach": "deductive",
        "evidence_style": "data_heavy",
        "concession_pattern": "fair_then_rebut",
        "thesis_placement": "first_paragraph",
    }
    base.update(overrides)
    return base


def _make_mode_profile(**overrides: object) -> dict:
    """Return a valid ModeProfile data dict with optional overrides."""
    base = {
        "id": "analytical",
        "display_name": "Analytical",
        "description": "Data-driven, evidence-based argumentation with measured tone.",
        "category": "deliberative",
        "prose_patterns": _make_prose_patterns(),
        "rhetorical_devices": ["logos", "ethos"],
        "vocabulary_register": _make_vocabulary_register(),
        "argument_structure": _make_argument_structure(),
        "signature_patterns": ["Opens with clear thesis", "Uses data to support claims"],
        "suppressed_phrases": ["obviously", "everyone knows"],
        "system_prompt_fragment": "Write with analytical precision and measured tone.",
        "few_shot_examples": ["Example sentence one.", "Example sentence two."],
    }
    base.update(overrides)
    return base


def _make_source_citation(**overrides: object) -> dict:
    base = {
        "claim": "Over 2,500 newspapers closed since 2005",
        "source_name": "Pew Research",
        "source_url": "https://pewresearch.org/journalism",
        "accessed_at": datetime(2026, 3, 25, tzinfo=timezone.utc),
        "credibility_score": 0.95,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TopicContext tests
# ---------------------------------------------------------------------------

class TestTopicContext:
    def test_valid_creation(self) -> None:
        tc = TopicContext(**_make_topic_context())
        assert tc.title == "Local Journalism Collapse"
        assert tc.input_type == "text"

    def test_valid_url_input_type(self) -> None:
        tc = TopicContext(**_make_topic_context(input_type="url"))
        assert tc.input_type == "url"

    def test_valid_file_input_type(self) -> None:
        tc = TopicContext(**_make_topic_context(input_type="file"))
        assert tc.input_type == "file"

    def test_rejects_invalid_input_type(self) -> None:
        with pytest.raises(ValidationError, match="input_type"):
            TopicContext(**_make_topic_context(input_type="audio"))

    def test_optional_fields_default_none(self) -> None:
        tc = TopicContext(**_make_topic_context())
        assert tc.source_url is None
        assert tc.source_text is None
        assert tc.fetched_at is None

    def test_optional_fields_populated(self) -> None:
        now = datetime.now(tz=timezone.utc)
        tc = TopicContext(**_make_topic_context(
            source_url="https://example.com",
            source_text="Full article text here.",
            fetched_at=now,
        ))
        assert tc.source_url == "https://example.com"
        assert tc.fetched_at == now


# ---------------------------------------------------------------------------
# ModeProfile tests
# ---------------------------------------------------------------------------

class TestModeProfile:
    def test_valid_creation(self) -> None:
        mp = ModeProfile(**_make_mode_profile())
        assert mp.id == "analytical"
        assert mp.display_name == "Analytical"

    def test_prose_patterns_literal_enforcement(self) -> None:
        with pytest.raises(ValidationError):
            ModeProfile(**_make_mode_profile(
                prose_patterns=_make_prose_patterns(avg_sentence_length="tiny")
            ))

    def test_vocabulary_register_literal_enforcement(self) -> None:
        with pytest.raises(ValidationError):
            ModeProfile(**_make_mode_profile(
                vocabulary_register=_make_vocabulary_register(formality="ultra_formal")
            ))

    def test_argument_structure_literal_enforcement(self) -> None:
        with pytest.raises(ValidationError):
            ModeProfile(**_make_mode_profile(
                argument_structure=_make_argument_structure(approach="random")
            ))

    def test_requires_at_least_one_rhetorical_device(self) -> None:
        with pytest.raises(ValidationError):
            ModeProfile(**_make_mode_profile(rhetorical_devices=[]))

    def test_requires_at_least_two_few_shot_examples(self) -> None:
        with pytest.raises(ValidationError):
            ModeProfile(**_make_mode_profile(few_shot_examples=["Only one example."]))

    def test_optional_suppressed_phrases(self) -> None:
        mp = ModeProfile(**_make_mode_profile(suppressed_phrases=[]))
        assert mp.suppressed_phrases == []


# ---------------------------------------------------------------------------
# ModeBlendConfig tests
# ---------------------------------------------------------------------------

class TestModeBlendConfig:
    def test_valid_single_mode(self) -> None:
        mc = ModeBlendConfig(modes=[("analytical", 100.0)])
        assert len(mc.modes) == 1

    def test_valid_two_modes(self) -> None:
        mc = ModeBlendConfig(modes=[("polemical", 60.0), ("analytical", 40.0)])
        assert len(mc.modes) == 2

    def test_valid_three_modes(self) -> None:
        mc = ModeBlendConfig(modes=[("a", 50.0), ("b", 30.0), ("c", 20.0)])
        assert len(mc.modes) == 3

    def test_weights_summing_to_100_passes(self) -> None:
        mc = ModeBlendConfig(modes=[("polemical", 60.0), ("analytical", 40.0)])
        total = sum(w for _, w in mc.modes)
        assert abs(total - 100) < 0.01

    def test_rejects_weights_not_summing_to_100(self) -> None:
        with pytest.raises(ValidationError, match="sum to 100"):
            ModeBlendConfig(modes=[("polemical", 60.0), ("analytical", 30.0)])

    def test_rejects_more_than_3_modes(self) -> None:
        with pytest.raises(ValidationError, match="Maximum 3"):
            ModeBlendConfig(modes=[("a", 25.0), ("b", 25.0), ("c", 25.0), ("d", 25.0)])

    def test_rejects_empty_modes(self) -> None:
        with pytest.raises(ValidationError, match="At least one"):
            ModeBlendConfig(modes=[])


# ---------------------------------------------------------------------------
# StanceConfig tests
# ---------------------------------------------------------------------------

class TestStanceConfig:
    def test_default_position_is_zero(self) -> None:
        sc = StanceConfig()
        assert sc.position == 0

    def test_default_intensity_is_half(self) -> None:
        sc = StanceConfig()
        assert sc.intensity == 0.5

    def test_label_strongly_equity_focused(self) -> None:
        assert StanceConfig(position=-80).label == "strongly_equity_focused"
        assert StanceConfig(position=-100).label == "strongly_equity_focused"
        assert StanceConfig(position=-50).label == "strongly_equity_focused"

    def test_label_equity_leaning(self) -> None:
        assert StanceConfig(position=-49).label == "equity_leaning"
        assert StanceConfig(position=-26).label == "equity_leaning"
        assert StanceConfig(position=-30).label == "equity_leaning"

    def test_label_balanced(self) -> None:
        assert StanceConfig(position=0).label == "balanced"
        assert StanceConfig(position=-25).label == "balanced"
        assert StanceConfig(position=25).label == "balanced"
        assert StanceConfig(position=20).label == "balanced"
        assert StanceConfig(position=-20).label == "balanced"

    def test_label_liberty_leaning(self) -> None:
        assert StanceConfig(position=26).label == "liberty_leaning"
        assert StanceConfig(position=49).label == "liberty_leaning"
        assert StanceConfig(position=30).label == "liberty_leaning"

    def test_label_strongly_liberty_focused(self) -> None:
        assert StanceConfig(position=50).label == "strongly_liberty_focused"
        assert StanceConfig(position=100).label == "strongly_liberty_focused"

    def test_rejects_position_below_negative_100(self) -> None:
        with pytest.raises(ValidationError):
            StanceConfig(position=-101)

    def test_rejects_position_above_100(self) -> None:
        with pytest.raises(ValidationError):
            StanceConfig(position=101)

    def test_intensity_boundaries(self) -> None:
        sc_low = StanceConfig(intensity=0.0)
        assert sc_low.intensity == 0.0
        sc_high = StanceConfig(intensity=1.0)
        assert sc_high.intensity == 1.0

    def test_rejects_intensity_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            StanceConfig(intensity=-0.1)

    def test_rejects_intensity_above_one(self) -> None:
        with pytest.raises(ValidationError):
            StanceConfig(intensity=1.1)


# ---------------------------------------------------------------------------
# SourceCitation tests
# ---------------------------------------------------------------------------

class TestSourceCitation:
    def test_valid_creation(self) -> None:
        sc = SourceCitation(**_make_source_citation())
        assert sc.credibility_score == 0.95

    def test_rejects_credibility_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            SourceCitation(**_make_source_citation(credibility_score=-0.1))

    def test_rejects_credibility_above_one(self) -> None:
        with pytest.raises(ValidationError):
            SourceCitation(**_make_source_citation(credibility_score=1.1))

    def test_credibility_at_boundaries(self) -> None:
        sc_low = SourceCitation(**_make_source_citation(credibility_score=0.0))
        assert sc_low.credibility_score == 0.0
        sc_high = SourceCitation(**_make_source_citation(credibility_score=1.0))
        assert sc_high.credibility_score == 1.0

    def test_optional_political_lean(self) -> None:
        sc = SourceCitation(**_make_source_citation(political_lean="center-left"))
        assert sc.political_lean == "center-left"


# ---------------------------------------------------------------------------
# GeneratedPiece tests
# ---------------------------------------------------------------------------

class TestGeneratedPiece:
    def test_valid_creation(self) -> None:
        gp = GeneratedPiece(
            id="abc-123",
            created_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
            topic=TopicContext(**_make_topic_context()),
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0),
            target_length=800,
            actual_length=780,
            title="The Death of Local News",
            body="Opinion piece body text here.",
            preview_text="A preview of the piece.",
            sources=[SourceCitation(**_make_source_citation())],
            research_queries=["local journalism decline"],
            disclaimer=MANDATORY_DISCLAIMER,
        )
        assert gp.id == "abc-123"
        assert gp.actual_length == 780
        assert gp.subtitle is None
        assert gp.image_prompt is None

    def test_with_optional_fields(self) -> None:
        gp = GeneratedPiece(
            id="abc-456",
            created_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
            topic=TopicContext(**_make_topic_context()),
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=-50),
            target_length=800,
            actual_length=810,
            title="The Death of Local News",
            subtitle="Why it matters more than you think",
            body="Opinion piece body text here.",
            preview_text="A preview of the piece.",
            sources=[],
            research_queries=[],
            image_prompt="A newspaper crumbling into dust.",
            image_platform="substack",
            disclaimer=MANDATORY_DISCLAIMER,
        )
        assert gp.subtitle == "Why it matters more than you think"
        assert gp.image_prompt is not None
        assert gp.image_platform == "substack"

    def test_disclaimer_is_mandatory_constant(self) -> None:
        gp = GeneratedPiece(
            id="abc-789",
            created_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
            topic=TopicContext(**_make_topic_context()),
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0),
            target_length=800,
            actual_length=780,
            title="Test Title",
            body="Test body.",
            preview_text="Test preview.",
            sources=[],
            research_queries=[],
            disclaimer=MANDATORY_DISCLAIMER,
        )
        assert gp.disclaimer == MANDATORY_DISCLAIMER
