"""Unit tests for mode profiles, ModeProfile model, and mode engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from opinionforge.models.config import ModeBlendConfig
from opinionforge.models.mode import (
    ArgumentStructure,
    ModeProfile,
    ProsePatterns,
    VocabularyRegister,
)
from opinionforge.modes import list_modes, load_mode
from opinionforge.core.mode_engine import blend_modes
from opinionforge.core.mode_engine import load_mode as engine_load_mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_MODE_IDS = [
    "analytical",
    "aphoristic",
    "data_driven",
    "dialectical",
    "forensic",
    "measured",
    "narrative",
    "oratorical",
    "polemical",
    "populist",
    "provocative",
    "satirical",
]

# Representative sample of writer names that must not appear in any mode profile
BANNED_WRITER_NAMES = [
    "Hitchens",
    "Ivins",
    "Buckley",
    "Brooks",
    "Krugman",
    "Sullivan",
    "Will",
    "Dowd",
    "Friedman",
    "Douthat",
    "hitchens",
    "ivins",
    "buckley",
    "brooks",
    "krugman",
    "sullivan",
    "dowd",
    "friedman",
    "douthat",
]


def _profile_text(profile: ModeProfile) -> str:
    """Collect all string field values from a ModeProfile for text scanning."""
    parts = [
        profile.id,
        profile.display_name,
        profile.description,
        profile.category,
        profile.system_prompt_fragment,
        profile.prose_patterns.opening_style,
        profile.prose_patterns.closing_style,
    ]
    parts.extend(profile.rhetorical_devices)
    parts.extend(profile.signature_patterns)
    parts.extend(profile.suppressed_phrases)
    parts.extend(profile.few_shot_examples)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tests: ModeProfile model structure
# ---------------------------------------------------------------------------


def test_mode_profile_imports():
    """ModeProfile and sub-models are importable from opinionforge.models.mode."""
    assert ModeProfile is not None
    assert ProsePatterns is not None
    assert VocabularyRegister is not None
    assert ArgumentStructure is not None


def test_mode_profile_model_validate_success():
    """ModeProfile.model_validate() succeeds on a dict with all required fields."""
    data = {
        "id": "test_mode",
        "display_name": "Test Mode",
        "description": "A test mode for validation purposes.",
        "category": "deliberative",
        "prose_patterns": {
            "avg_sentence_length": "medium",
            "paragraph_length": "short",
            "uses_fragments": False,
            "uses_lists": False,
            "opening_style": "Opens with a question.",
            "closing_style": "Closes with a statement.",
        },
        "rhetorical_devices": ["device one", "device two"],
        "vocabulary_register": {
            "formality": "formal",
            "word_origin_preference": "mixed",
            "jargon_level": "light",
            "profanity": "never",
            "humor_frequency": "never",
        },
        "argument_structure": {
            "approach": "deductive",
            "evidence_style": "mixed",
            "concession_pattern": "fair_then_rebut",
            "thesis_placement": "first_paragraph",
        },
        "signature_patterns": ["pattern one", "pattern two"],
        "suppressed_phrases": [],
        "system_prompt_fragment": "Write carefully.",
        "few_shot_examples": ["Example one text.", "Example two text."],
    }
    profile = ModeProfile.model_validate(data)
    assert profile.id == "test_mode"
    assert profile.display_name == "Test Mode"


def test_mode_profile_model_validate_missing_required_field():
    """ModeProfile.model_validate() raises ValidationError when a required field is missing."""
    data = {
        "id": "incomplete",
        "display_name": "Incomplete",
        # missing description, category, prose_patterns, etc.
    }
    with pytest.raises(ValidationError):
        ModeProfile.model_validate(data)


def test_mode_profile_has_no_legacy_fields():
    """ModeProfile does not have writer-specific fields from VoiceProfile."""
    profile = load_mode("polemical")
    assert not hasattr(profile, "name")
    assert not hasattr(profile, "wikipedia_url")
    assert not hasattr(profile, "era")
    assert not hasattr(profile, "publication")
    assert not hasattr(profile, "ideological_baseline")


# ---------------------------------------------------------------------------
# Tests: All 12 YAML profiles parse correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_all_12_profiles_parse(mode_id: str):
    """Each of the 12 mode YAML files parses into a valid ModeProfile."""
    profile = load_mode(mode_id)
    assert isinstance(profile, ModeProfile)
    assert profile.id == mode_id


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_all_profiles_have_required_fields(mode_id: str):
    """Each profile has all required structural fields non-empty."""
    profile = load_mode(mode_id)
    assert profile.display_name
    assert profile.description
    assert profile.category
    assert profile.prose_patterns.opening_style
    assert profile.prose_patterns.closing_style
    assert len(profile.rhetorical_devices) > 0
    assert len(profile.signature_patterns) > 0
    assert profile.system_prompt_fragment
    assert len(profile.few_shot_examples) >= 2


@pytest.mark.parametrize("mode_id", EXPECTED_MODE_IDS)
def test_no_writer_names_in_profiles(mode_id: str):
    """No ModeProfile loaded from YAML contains any banned writer names."""
    profile = load_mode(mode_id)
    text = _profile_text(profile)
    for name in BANNED_WRITER_NAMES:
        assert name not in text, (
            f"Profile '{mode_id}' contains banned writer name '{name}'"
        )


# ---------------------------------------------------------------------------
# Tests: Category assignments
# ---------------------------------------------------------------------------


def test_polemical_category():
    """polemical.yaml has category 'confrontational'."""
    assert load_mode("polemical").category == "confrontational"


def test_analytical_category():
    """analytical.yaml has category 'deliberative'."""
    assert load_mode("analytical").category == "deliberative"


def test_populist_category():
    """populist.yaml has category 'confrontational'."""
    assert load_mode("populist").category == "confrontational"


def test_satirical_category():
    """satirical.yaml has category 'literary'."""
    assert load_mode("satirical").category == "literary"


def test_forensic_category():
    """forensic.yaml has category 'investigative'."""
    assert load_mode("forensic").category == "investigative"


def test_data_driven_display_name():
    """data_driven.yaml has display_name 'Data-Driven'."""
    assert load_mode("data_driven").display_name == "Data-Driven"


def test_data_driven_category():
    """data_driven.yaml has category 'investigative'."""
    assert load_mode("data_driven").category == "investigative"


# ---------------------------------------------------------------------------
# Tests: list_modes()
# ---------------------------------------------------------------------------


def test_list_modes_returns_12():
    """list_modes() returns a list of exactly 12 ModeProfile instances."""
    modes = list_modes()
    assert len(modes) == 12


def test_list_modes_sorted_by_id():
    """list_modes() results are sorted by id."""
    modes = list_modes()
    ids = [m.id for m in modes]
    assert ids == sorted(ids)


def test_list_modes_contains_all_12_ids():
    """All 12 expected mode ids are present in list_modes() output."""
    ids = {m.id for m in list_modes()}
    assert ids == set(EXPECTED_MODE_IDS)


# ---------------------------------------------------------------------------
# Tests: load_mode() error handling
# ---------------------------------------------------------------------------


def test_load_mode_unknown_id_raises_file_not_found():
    """load_mode raises FileNotFoundError for an unknown id."""
    with pytest.raises(FileNotFoundError):
        load_mode("nonexistent_mode")


def test_engine_load_mode_unknown_id_raises_value_error():
    """mode_engine.load_mode raises ValueError with 'not found' for unknown id."""
    with pytest.raises(ValueError, match="not found"):
        engine_load_mode("nonexistent_mode")


def test_engine_load_mode_fuzzy_suggestion():
    """mode_engine.load_mode provides a fuzzy suggestion for a near-miss id."""
    with pytest.raises(ValueError) as exc_info:
        engine_load_mode("polemicl")
    # Should suggest 'polemical'
    assert "polemical" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: blend_modes()
# ---------------------------------------------------------------------------


def test_blend_modes_single_returns_unmodified():
    """blend_modes with single mode at 100% returns unmodified system_prompt_fragment."""
    profile = load_mode("polemical")
    config = ModeBlendConfig(modes=[("polemical", 100.0)])
    result = blend_modes(config)
    assert result == profile.system_prompt_fragment


def test_blend_modes_two_modes_contains_both():
    """blend_modes with two modes produces output containing text from both modes."""
    polemical = load_mode("polemical")
    analytical = load_mode("analytical")
    config = ModeBlendConfig(modes=[("polemical", 60.0), ("analytical", 40.0)])
    result = blend_modes(config)
    assert polemical.system_prompt_fragment in result
    assert analytical.system_prompt_fragment in result


def test_blend_modes_three_modes_no_error():
    """blend_modes with three modes at 33.3/33.3/33.4 returns non-empty string."""
    config = ModeBlendConfig(
        modes=[("polemical", 33.3), ("analytical", 33.3), ("satirical", 33.4)]
    )
    result = blend_modes(config)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: ModeBlendConfig validation
# ---------------------------------------------------------------------------


def test_mode_blend_config_importable():
    """ModeBlendConfig is importable from opinionforge.models.config."""
    assert ModeBlendConfig is not None


def test_mode_blend_config_valid():
    """ModeBlendConfig accepts a valid single-mode configuration."""
    config = ModeBlendConfig(modes=[("polemical", 100.0)])
    assert config.modes == [("polemical", 100.0)]


def test_mode_blend_config_empty_raises():
    """ModeBlendConfig raises ValueError for empty modes list."""
    with pytest.raises(ValueError):
        ModeBlendConfig(modes=[])


def test_mode_blend_config_four_modes_raises():
    """ModeBlendConfig raises ValueError when more than 3 modes are provided."""
    with pytest.raises(ValueError):
        ModeBlendConfig(
            modes=[
                ("polemical", 25.0),
                ("analytical", 25.0),
                ("satirical", 25.0),
                ("forensic", 25.0),
            ]
        )


def test_mode_blend_config_weights_not_100_raises():
    """ModeBlendConfig raises ValueError when weights sum to 99."""
    with pytest.raises(ValueError):
        ModeBlendConfig(modes=[("polemical", 50.0), ("analytical", 49.0)])


def test_mode_blend_config_three_modes_valid():
    """ModeBlendConfig accepts 3 modes with weights 33.3/33.3/33.4."""
    config = ModeBlendConfig(
        modes=[("polemical", 33.3), ("analytical", 33.3), ("satirical", 33.4)]
    )
    assert len(config.modes) == 3
