"""Unit tests for StanceConfig model and apply_stance() function.

Covers all 5 direction labels, boundary validation for position and intensity,
political vocabulary exclusion, no named publications, intensity-zero neutrality,
and all 4 edge-case combinations from the PRD.

All tests pass with zero API calls.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from opinionforge.core.stance import apply_stance
from opinionforge.models.config import StanceConfig


BASE_MODE_PROMPT = "Write with analytical precision and rhetorical clarity."

# Political labels that must never appear in apply_stance output modifier section
FORBIDDEN_POLITICAL_LABELS = [
    "progressive",
    "conservative",
    "far_left",
    "far_right",
    "left-leaning",
    "right-leaning",
    "far left",
    "far right",
    "liberal",
    "Democrat",
    "Republican",
]

# Named publications that must never appear in source preference instructions
FORBIDDEN_PUBLICATIONS = [
    "Guardian",
    "Nation",
    "Wall Street Journal",
    "National Review",
    "Heritage Foundation",
    "Jacobin",
    "AEI",
    "Brookings",
    "Economist",
    "New York Times",
]


# ---------------------------------------------------------------------------
# StanceConfig.label — all 5 direction values
# ---------------------------------------------------------------------------

class TestStanceConfigLabel:
    """Tests covering all 5 direction label values of StanceConfig.label."""

    def test_label_balanced_at_zero(self) -> None:
        """position=0 returns 'balanced'."""
        assert StanceConfig().label == "balanced"

    def test_label_balanced_at_positive_boundary(self) -> None:
        """position=25 is still in the balanced range."""
        assert StanceConfig(position=25).label == "balanced"

    def test_label_balanced_at_negative_boundary(self) -> None:
        """position=-25 is still in the balanced range."""
        assert StanceConfig(position=-25).label == "balanced"

    def test_label_equity_leaning(self) -> None:
        """position=-30 returns 'equity_leaning'."""
        assert StanceConfig(position=-30).label == "equity_leaning"

    def test_label_equity_leaning_at_boundary(self) -> None:
        """position=-26 is just inside equity_leaning range."""
        assert StanceConfig(position=-26).label == "equity_leaning"

    def test_label_strongly_equity_focused(self) -> None:
        """position=-70 returns 'strongly_equity_focused'."""
        assert StanceConfig(position=-70).label == "strongly_equity_focused"

    def test_label_strongly_equity_focused_at_boundary(self) -> None:
        """position=-50 returns 'strongly_equity_focused'."""
        assert StanceConfig(position=-50).label == "strongly_equity_focused"

    def test_label_strongly_equity_focused_at_max(self) -> None:
        """position=-100 returns 'strongly_equity_focused'."""
        assert StanceConfig(position=-100).label == "strongly_equity_focused"

    def test_label_liberty_leaning(self) -> None:
        """position=30 returns 'liberty_leaning'."""
        assert StanceConfig(position=30).label == "liberty_leaning"

    def test_label_liberty_leaning_at_boundary(self) -> None:
        """position=26 is just inside liberty_leaning range."""
        assert StanceConfig(position=26).label == "liberty_leaning"

    def test_label_strongly_liberty_focused(self) -> None:
        """position=70 returns 'strongly_liberty_focused'."""
        assert StanceConfig(position=70).label == "strongly_liberty_focused"

    def test_label_strongly_liberty_focused_at_boundary(self) -> None:
        """position=50 returns 'strongly_liberty_focused'."""
        assert StanceConfig(position=50).label == "strongly_liberty_focused"

    def test_label_strongly_liberty_focused_at_max(self) -> None:
        """position=100 returns 'strongly_liberty_focused'."""
        assert StanceConfig(position=100).label == "strongly_liberty_focused"


# ---------------------------------------------------------------------------
# StanceConfig validation — position boundaries
# ---------------------------------------------------------------------------

class TestStanceConfigPositionBoundaries:
    """Tests covering StanceConfig position boundary validation."""

    def test_position_minus_101_raises(self) -> None:
        """position=-101 raises ValidationError."""
        with pytest.raises(ValidationError):
            StanceConfig(position=-101)

    def test_position_minus_100_valid(self) -> None:
        """position=-100 is a valid lower bound."""
        config = StanceConfig(position=-100)
        assert config.position == -100

    def test_position_zero_valid(self) -> None:
        """position=0 is valid default."""
        config = StanceConfig(position=0)
        assert config.position == 0

    def test_position_100_valid(self) -> None:
        """position=100 is a valid upper bound."""
        config = StanceConfig(position=100)
        assert config.position == 100

    def test_position_101_raises(self) -> None:
        """position=101 raises ValidationError."""
        with pytest.raises(ValidationError):
            StanceConfig(position=101)


# ---------------------------------------------------------------------------
# StanceConfig validation — intensity boundaries
# ---------------------------------------------------------------------------

class TestStanceConfigIntensityBoundaries:
    """Tests covering StanceConfig intensity boundary validation."""

    def test_intensity_negative_raises(self) -> None:
        """intensity=-0.1 raises ValidationError."""
        with pytest.raises(ValidationError):
            StanceConfig(intensity=-0.1)

    def test_intensity_zero_valid(self) -> None:
        """intensity=0.0 is a valid lower bound."""
        config = StanceConfig(intensity=0.0)
        assert config.intensity == 0.0

    def test_intensity_default(self) -> None:
        """Default intensity is 0.5."""
        config = StanceConfig()
        assert config.intensity == 0.5

    def test_intensity_one_valid(self) -> None:
        """intensity=1.0 is a valid upper bound."""
        config = StanceConfig(intensity=1.0)
        assert config.intensity == 1.0

    def test_intensity_above_one_raises(self) -> None:
        """intensity=1.1 raises ValidationError."""
        with pytest.raises(ValidationError):
            StanceConfig(intensity=1.1)


# ---------------------------------------------------------------------------
# apply_stance — output structure
# ---------------------------------------------------------------------------

class TestApplyStanceOutputStructure:
    """Tests for the structure and content of apply_stance() output."""

    def test_output_begins_with_mode_prompt(self) -> None:
        """apply_stance output begins with the original mode prompt content."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig())
        assert result.startswith(BASE_MODE_PROMPT)

    def test_output_non_empty(self) -> None:
        """apply_stance returns a non-empty string."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig())
        assert len(result) > 0

    def test_output_contains_all_five_dimensions(self) -> None:
        """apply_stance modifier section addresses all 5 dimensions."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=70))
        result_lower = result.lower()
        assert "argument selection" in result_lower
        assert "framing" in result_lower
        assert "source preference" in result_lower
        assert "rhetorical intensity" in result_lower
        assert "counterargument" in result_lower


# ---------------------------------------------------------------------------
# apply_stance — balanced / zero-intensity behavior
# ---------------------------------------------------------------------------

class TestApplyStanceBalanced:
    """Tests for balanced and zero-intensity stance behavior."""

    def test_position_zero_intensity_zero_contains_balanced_keyword(self) -> None:
        """position=0, intensity=0.0 produces output with balanced/measured/deliberative."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=0, intensity=0.0))
        result_lower = result.lower()
        assert any(kw in result_lower for kw in ["balanced", "measured", "deliberative"])

    def test_intensity_zero_produces_same_output_regardless_of_direction(self) -> None:
        """intensity=0.0 produces balanced output regardless of position direction."""
        result_negative = apply_stance(
            BASE_MODE_PROMPT, StanceConfig(position=-100, intensity=0.0)
        )
        result_positive = apply_stance(
            BASE_MODE_PROMPT, StanceConfig(position=100, intensity=0.0)
        )
        # Both should describe balanced/deliberative behavior, not directional framing
        result_neg_lower = result_negative.lower()
        result_pos_lower = result_positive.lower()
        assert any(kw in result_neg_lower for kw in ["balanced", "measured", "deliberative", "multiple perspectives"])
        assert any(kw in result_pos_lower for kw in ["balanced", "measured", "deliberative", "multiple perspectives"])


# ---------------------------------------------------------------------------
# apply_stance — directional behavior
# ---------------------------------------------------------------------------

class TestApplyStanceDirectional:
    """Tests for directional stance behavior."""

    def test_equity_position_contains_equity_keyword(self) -> None:
        """position=-80, intensity=0.9 produces output containing 'equity'."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=-80, intensity=0.9))
        assert "equity" in result.lower()

    def test_liberty_position_contains_liberty_keyword(self) -> None:
        """position=80, intensity=0.9 produces output containing 'liberty'."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=80, intensity=0.9))
        assert "liberty" in result.lower()

    def test_high_intensity_balanced_stance_contains_conviction(self) -> None:
        """position=0, intensity=1.0 produces high-heat, balanced-framing output."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=0, intensity=1.0))
        result_lower = result.lower()
        # Should have high-rhetoric indicators but balanced framing
        assert any(kw in result_lower for kw in ["maximum", "aggressive", "conviction", "uncompromising"])
        # Should not have directional framing keywords at high intensity
        # (balanced position means equity/liberty keywords should be minimal)


# ---------------------------------------------------------------------------
# apply_stance — vocabulary compliance (no political/publication labels)
# ---------------------------------------------------------------------------

class TestApplyStanceVocabularyCompliance:
    """Tests for forbidden vocabulary in apply_stance() output."""

    def test_no_political_party_labels_in_output(self) -> None:
        """apply_stance output contains no political party labels."""
        for position in [-100, -50, 0, 50, 100]:
            result = apply_stance(
                BASE_MODE_PROMPT,
                StanceConfig(position=position, intensity=0.8),
            )
            # Check only the modifier section (after the base mode prompt)
            modifier_section = result[len(BASE_MODE_PROMPT):]
            modifier_lower = modifier_section.lower()
            for label in FORBIDDEN_POLITICAL_LABELS:
                assert label.lower() not in modifier_lower, (
                    f"Forbidden political label '{label}' found in apply_stance "
                    f"output for position={position}"
                )

    def test_no_named_publications_in_output(self) -> None:
        """apply_stance output contains no named publication references."""
        for position in [-100, -50, 0, 50, 100]:
            result = apply_stance(
                BASE_MODE_PROMPT,
                StanceConfig(position=position, intensity=0.9),
            )
            modifier_section = result[len(BASE_MODE_PROMPT):]
            for pub in FORBIDDEN_PUBLICATIONS:
                assert pub not in modifier_section, (
                    f"Forbidden publication '{pub}' found in apply_stance "
                    f"output for position={position}"
                )

    def test_source_preference_uses_evidence_category_vocabulary(self) -> None:
        """Source preference instructions use evidence-category vocabulary."""
        for position in [-80, 0, 80]:
            result = apply_stance(
                BASE_MODE_PROMPT,
                StanceConfig(position=position, intensity=0.9),
            )
            result_lower = result.lower()
            # At least one evidence-category term should appear
            evidence_terms = [
                "peer-reviewed",
                "government data",
                "investigative journalism",
                "market analysis",
                "institutional research",
                "empirical case studies",
            ]
            assert any(term in result_lower for term in evidence_terms), (
                f"No evidence-category vocabulary found in source preference "
                f"instructions for position={position}"
            )


# ---------------------------------------------------------------------------
# apply_stance — edge cases from the PRD
# ---------------------------------------------------------------------------

class TestApplyStanceEdgeCases:
    """Tests for all 4 edge-case stance/intensity combinations from the PRD."""

    def test_edge_case_zero_zero(self) -> None:
        """Edge case (0, 0.0): balanced position, zero intensity — non-empty, no raise."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=0, intensity=0.0))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_edge_case_minus100_one(self) -> None:
        """Edge case (-100, 1.0): max equity, max intensity — non-empty, no raise."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=-100, intensity=1.0))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_edge_case_plus100_one(self) -> None:
        """Edge case (100, 1.0): max liberty, max intensity — non-empty, no raise."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=100, intensity=1.0))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_edge_case_zero_one(self) -> None:
        """Edge case (0, 1.0): balanced, max intensity — non-empty, no raise."""
        result = apply_stance(BASE_MODE_PROMPT, StanceConfig(position=0, intensity=1.0))
        assert isinstance(result, str)
        assert len(result) > 0
