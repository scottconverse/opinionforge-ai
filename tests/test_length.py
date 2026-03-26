"""Unit tests for opinionforge.core.length module."""

from __future__ import annotations

import pytest

from opinionforge.core.length import (
    LENGTH_PRESETS,
    MAX_WORD_COUNT,
    MIN_WORD_COUNT,
    get_length_instructions,
    resolve_length,
)


# ---------------------------------------------------------------------------
# resolve_length — preset names
# ---------------------------------------------------------------------------

class TestResolveLengthPresets:
    """Tests that all named presets resolve to the correct word counts."""

    def test_short_preset(self) -> None:
        assert resolve_length("short") == 500

    def test_standard_preset(self) -> None:
        assert resolve_length("standard") == 750

    def test_long_preset(self) -> None:
        assert resolve_length("long") == 1200

    def test_essay_preset(self) -> None:
        assert resolve_length("essay") == 2500

    def test_feature_preset(self) -> None:
        assert resolve_length("feature") == 5000

    def test_preset_case_insensitive(self) -> None:
        assert resolve_length("Short") == 500
        assert resolve_length("STANDARD") == 750

    def test_preset_with_whitespace(self) -> None:
        assert resolve_length("  long  ") == 1200


# ---------------------------------------------------------------------------
# resolve_length — custom integer strings
# ---------------------------------------------------------------------------

class TestResolveLengthCustom:
    """Tests for custom integer-string word counts."""

    def test_custom_integer_string(self) -> None:
        assert resolve_length("1500") == 1500

    def test_minimum_boundary(self) -> None:
        assert resolve_length("200") == 200

    def test_maximum_boundary(self) -> None:
        assert resolve_length("8000") == 8000

    def test_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="below the minimum"):
            resolve_length("50")

    def test_above_maximum_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds the maximum"):
            resolve_length("10000")

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="below the minimum"):
            resolve_length("0")

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="below the minimum"):
            resolve_length("-500")


# ---------------------------------------------------------------------------
# resolve_length — invalid inputs
# ---------------------------------------------------------------------------

class TestResolveLengthInvalid:
    """Tests that unrecognised preset names are rejected."""

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown length preset"):
            resolve_length("novel")

    def test_unknown_preset_lists_valid_presets(self) -> None:
        with pytest.raises(ValueError, match="Valid presets"):
            resolve_length("tweet")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_length("")


# ---------------------------------------------------------------------------
# get_length_instructions
# ---------------------------------------------------------------------------

class TestGetLengthInstructions:
    """Tests that length instructions vary by target size."""

    def test_short_piece_instructions(self) -> None:
        result = get_length_instructions(500)
        assert "single strongest argument" in result
        assert "500 words" in result

    def test_standard_piece_instructions(self) -> None:
        result = get_length_instructions(750)
        assert "single strongest argument" in result  # <= 800

    def test_long_piece_instructions(self) -> None:
        result = get_length_instructions(2500)
        assert "section breaks" in result or "long-form" in result

    def test_feature_instructions(self) -> None:
        result = get_length_instructions(5000)
        assert "section breaks" in result

    def test_short_vs_long_differ(self) -> None:
        short = get_length_instructions(400)
        long = get_length_instructions(3000)
        assert short != long

    def test_includes_tolerance_range(self) -> None:
        result = get_length_instructions(1000)
        # 10% tolerance: 900-1100
        assert "900" in result
        assert "1100" in result
