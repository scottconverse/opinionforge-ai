"""Tests verifying generated output never contains writer names.

Enforces the PRD success criterion: generated output must not contain
any writer name from the historical roster. Also checks that the mandatory
fixed disclaimer is present in all outputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from opinionforge.core.generator import MANDATORY_DISCLAIMER, generate_piece
from opinionforge.core.topic import ingest_text
from opinionforge.models.config import ModeBlendConfig, StanceConfig

# ---------------------------------------------------------------------------
# Writer surnames list — same 100 surnames used in test_name_sanitization.py
# ---------------------------------------------------------------------------

WRITER_SURNAMES: list[str] = [
    "Hitchens", "Ivins", "Buckley", "Brooks", "Krugman", "Sullivan", "Will",
    "Dowd", "Friedman", "Douthat", "Sontag", "Baldwin", "Didion", "Vidal",
    "Orwell", "Mencken", "Chesterton", "Lippmann", "Liebling", "Royko",
    "Molly", "Kinsley", "Noonan", "Krauthammer", "Broder", "Wills", "Morris",
    "Kristol", "Podhoretz", "Murray", "Gill", "Cockburn", "Coyne", "Landers",
    "Quindlen", "Lewis", "Buchwald", "Karkaria",
    "Levin", "Bierce", "Herbert", "Breslin", "Stephens", "Sulzberger",
    "Thomas", "Trillin", "Hiaasen", "Hebert", "Blatchford", "Page",
    "Schultz", "Tucker", "Runyon", "Barry", "Robinson", "Klein",
    "Otoole", "Deford", "Franklin", "Collins", "Weingarten",
    "Monbiot", "Rice", "Caen", "Newfield", "Bouie",
    "Reston", "Cannon", "Alsop", "Parker", "Singh", "Pitts",
    "Pyle", "Wicker", "Greenfield", "Schmich", "Goldberg", "Barnicle",
    "Albom", "Kempton", "Hentoff", "Kristof", "Buchanan", "Hamill",
    "Toynbee", "Smith", "Cohen", "Rovere", "Angell", "Baker",
    "Safire", "Povich", "Jenkins", "Lopez", "Sowell", "Winchell",
    "Pegler", "Twain", "Dionne", "Roosevelt",
]


# A mock LLM output that is clean (no writer names, no impersonation)
_CLEAN_OUTPUT = (
    "## The Crisis of Modern Democracy\n\n"
    "It would be comforting to believe that the machinery of self-governance "
    "is immune to disruption. Comforting, but dangerously naive. The evidence "
    "suggests that democratic institutions face a challenge unlike any they "
    "have encountered since the printing press.\n\n"
    "The first danger lies in the capacity of AI systems to generate "
    "misinformation at industrial scale. Where once a propagandist required "
    "an army of scribes, a single algorithm can now produce a torrent of "
    "plausible falsehoods.\n\n"
    "We must insist on transparency and the primacy of human judgment "
    "over algorithmic convenience."
)


def _make_piece() -> object:
    """Generate a test piece using a mock LLM client.

    Returns:
        A GeneratedPiece produced by generate_piece() with a mock client.
    """
    mock_client = MagicMock()
    mock_client.generate.return_value = _CLEAN_OUTPUT

    topic = ingest_text("Test topic about democracy and AI governance")
    return generate_piece(
        topic=topic,
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=0, intensity=0.5),
        target_length=750,
        client=mock_client,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDisclaimerPresent:
    """Tests that the mandatory fixed disclaimer is present in all outputs."""

    def test_disclaimer_present_in_output(self) -> None:
        """Mandatory disclaimer is present and equals the fixed constant."""
        piece = _make_piece()
        assert piece.disclaimer, "Disclaimer should not be empty"
        assert piece.disclaimer == MANDATORY_DISCLAIMER, (
            f"Disclaimer should equal MANDATORY_DISCLAIMER constant. "
            f"Got: {piece.disclaimer!r}"
        )

    def test_disclaimer_contains_required_phrases(self) -> None:
        """Mandatory disclaimer contains required phrases per PRD."""
        piece = _make_piece()
        assert "AI-assisted" in piece.disclaimer or "AI-generated" in piece.disclaimer, (
            "Disclaimer must reference AI generation"
        )
        assert "original content" in piece.disclaimer, (
            "Disclaimer must state the content is original"
        )
        assert "not written by" in piece.disclaimer or "not the work" in piece.disclaimer, (
            "Disclaimer must state it is not written by any real person"
        )


class TestNoWriterNamesInOutput:
    """Tests that generated output does not contain any writer surname."""

    def test_body_contains_no_writer_surname(self) -> None:
        """Generated body does not contain any writer surname from the 100-name list."""
        piece = _make_piece()
        violations: list[str] = []
        for surname in WRITER_SURNAMES:
            if surname.lower() in piece.body.lower():
                violations.append(surname)
        assert not violations, (
            f"Generated body contains writer surnames: {violations}. "
            "No historical writer names may appear in generated output."
        )

    def test_title_contains_no_writer_surname(self) -> None:
        """Generated title does not contain any writer surname."""
        piece = _make_piece()
        violations: list[str] = []
        for surname in WRITER_SURNAMES:
            if surname.lower() in piece.title.lower():
                violations.append(surname)
        assert not violations, (
            f"Generated title contains writer surnames: {violations}."
        )

    def test_disclaimer_is_fixed_string_not_writer_specific(self) -> None:
        """Disclaimer is the fixed constant, not dynamically constructed from writer names."""
        piece = _make_piece()
        for surname in WRITER_SURNAMES:
            assert surname.lower() not in piece.disclaimer.lower(), (
                f"Disclaimer must not contain writer name '{surname}'. "
                "Disclaimer must be the fixed constant, not writer-specific."
            )

    @pytest.mark.parametrize("surname", WRITER_SURNAMES)
    def test_no_i_am_pattern(self, surname: str) -> None:
        """Generated body does not contain 'I am [surname]' for any writer."""
        piece = _make_piece()
        forbidden = f"I am {surname}"
        assert forbidden.lower() not in piece.body.lower(), (
            f"Generated body must not contain '{forbidden}'"
        )

    @pytest.mark.parametrize("surname", WRITER_SURNAMES)
    def test_no_as_surname_pattern(self, surname: str) -> None:
        """Generated body does not contain 'As [surname]' for any writer."""
        piece = _make_piece()
        forbidden = f"As {surname}"
        assert forbidden.lower() not in piece.body.lower(), (
            f"Generated body must not contain '{forbidden}'"
        )
