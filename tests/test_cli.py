"""CLI integration tests using Typer's CliRunner.

All external calls (LLM, search, network) are mocked so that tests
run entirely offline and deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from opinionforge.cli import app, _parse_mode_blend
from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece
from opinionforge.models.topic import TopicContext
from opinionforge.models.mode import (
    ArgumentStructure,
    ModeProfile,
    ProsePatterns,
    VocabularyRegister,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_mode_profile(
    mode_id: str = "analytical",
    display_name: str = "Analytical",
    category: str = "deliberative",
) -> ModeProfile:
    """Construct a minimal ModeProfile for testing."""
    return ModeProfile(
        id=mode_id,
        display_name=display_name,
        description="A test rhetorical mode.",
        category=category,
        prose_patterns=ProsePatterns(
            avg_sentence_length="medium",
            paragraph_length="medium",
            uses_fragments=False,
            uses_lists=False,
            opening_style="declarative",
            closing_style="summary",
        ),
        rhetorical_devices=["syllogism", "analogy"],
        vocabulary_register=VocabularyRegister(
            formality="formal",
            word_origin_preference="mixed",
            jargon_level="light",
            profanity="never",
            humor_frequency="rare",
        ),
        argument_structure=ArgumentStructure(
            approach="deductive",
            evidence_style="mixed",
            concession_pattern="fair_then_rebut",
            thesis_placement="first_paragraph",
        ),
        signature_patterns=["clear thesis", "logical progression"],
        suppressed_phrases=[],
        system_prompt_fragment="Write like a test analytical mode.",
        few_shot_examples=["Example sentence one.", "Example sentence two."],
    )


def _make_topic_context(title: str = "Test Topic") -> TopicContext:
    """Construct a minimal TopicContext for testing."""
    return TopicContext(
        raw_input=title,
        input_type="text",
        title=title,
        summary="A test topic summary.",
        key_claims=["Claim one."],
        key_entities=["Entity One"],
        subject_domain="politics",
    )


def _make_generated_piece(**overrides) -> GeneratedPiece:
    """Construct a minimal GeneratedPiece for testing."""
    defaults = dict(
        id="test-id-123",
        created_at=datetime.now(timezone.utc),
        topic=_make_topic_context(),
        mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
        stance=StanceConfig(position=0, intensity=0.5),
        target_length=750,
        actual_length=745,
        title="Test Title",
        body="This is the test body of the opinion piece. " * 20,
        preview_text="Preview sentence one. Preview sentence two.",
        sources=[],
        research_queries=["test query"],
        disclaimer=MANDATORY_DISCLAIMER,
    )
    defaults.update(overrides)
    return GeneratedPiece(**defaults)


# ---------------------------------------------------------------------------
# 1. Top-level help
# ---------------------------------------------------------------------------

class TestHelpOutput:
    """Tests for --help on the app and sub-commands."""

    def test_app_help_exits_0(self) -> None:
        """App-level help exits with code 0."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_app_help_contains_editorial_craft_engine(self) -> None:
        """App help text mentions 'editorial craft engine'."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "editorial craft engine" in result.stdout.lower()

    def test_app_help_does_not_contain_writer(self) -> None:
        """App help text does not reference 'writer' as a noun for the app purpose."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # 'writer' must not appear as the primary description of the tool
        assert "AI-powered opinion writing tool" not in result.stdout

    def test_write_help_shows_mode_flag(self) -> None:
        """write --help shows --mode flag."""
        result = runner.invoke(app, ["write", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.stdout

    def test_write_help_shows_stance_flag(self) -> None:
        """write --help shows --stance flag."""
        result = runner.invoke(app, ["write", "--help"])
        assert result.exit_code == 0
        assert "--stance" in result.stdout

    def test_write_help_shows_intensity_flag(self) -> None:
        """write --help shows --intensity flag."""
        result = runner.invoke(app, ["write", "--help"])
        assert result.exit_code == 0
        assert "--intensity" in result.stdout

    def test_write_help_does_not_show_voice_flag(self) -> None:
        """write --help does NOT show --voice flag."""
        result = runner.invoke(app, ["write", "--help"])
        assert result.exit_code == 0
        assert "--voice" not in result.stdout

    def test_write_help_does_not_show_spectrum_flag(self) -> None:
        """write --help does NOT show --spectrum flag."""
        result = runner.invoke(app, ["write", "--help"])
        assert result.exit_code == 0
        assert "--spectrum" not in result.stdout

    def test_write_help_does_not_show_no_disclaimer_flag(self) -> None:
        """write --help does NOT show --no-disclaimer flag."""
        result = runner.invoke(app, ["write", "--help"])
        assert result.exit_code == 0
        assert "--no-disclaimer" not in result.stdout

    def test_preview_help_shows_mode_flag(self) -> None:
        """preview --help shows --mode flag."""
        result = runner.invoke(app, ["preview", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.stdout

    def test_preview_help_shows_stance_flag(self) -> None:
        """preview --help shows --stance flag."""
        result = runner.invoke(app, ["preview", "--help"])
        assert result.exit_code == 0
        assert "--stance" in result.stdout

    def test_modes_help(self) -> None:
        """modes --help exits 0 and shows flags."""
        result = runner.invoke(app, ["modes", "--help"])
        assert result.exit_code == 0
        assert "--search" in result.stdout or "--detail" in result.stdout

    def test_config_help(self) -> None:
        """config --help exits 0."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 2. modes command (replaces voices)
# ---------------------------------------------------------------------------

class TestModesCommand:
    """Tests for the 'modes' command."""

    def test_modes_command_exists(self) -> None:
        """The 'modes' command exists and exits 0 (even if no profiles)."""
        result = runner.invoke(app, ["modes"])
        # Exit 0 whether profiles exist or not
        assert result.exit_code == 0

    def test_modes_lists_all_12_modes(self) -> None:
        """modes command lists all 12 installed mode profiles."""
        result = runner.invoke(app, ["modes"])
        assert result.exit_code == 0
        # There should be 12 mode profiles from Sprint 1
        expected_modes = [
            "analytical", "aphoristic", "data_driven", "dialectical",
            "forensic", "measured", "narrative", "oratorical",
            "polemical", "populist", "provocative", "satirical",
        ]
        for mode_id in expected_modes:
            assert mode_id in result.stdout, f"Expected mode '{mode_id}' in output"

    def test_modes_detail_polemical(self) -> None:
        """modes --detail polemical shows mode info without any writer names."""
        result = runner.invoke(app, ["modes", "--detail", "polemical"])
        assert result.exit_code == 0
        # Must contain mode-specific info
        assert "polemical" in result.stdout.lower() or "Polemical" in result.stdout
        # Must NOT contain writer-specific fields
        assert "Wikipedia" not in result.stdout
        assert "wikipedia" not in result.stdout
        assert "era" not in result.stdout.lower() or "Era:" not in result.stdout
        assert "Publication:" not in result.stdout
        assert "ideological_baseline" not in result.stdout

    def test_modes_detail_not_found(self) -> None:
        """modes --detail nonexistent exits with code 4."""
        result = runner.invoke(app, ["modes", "--detail", "nonexistent_mode_xyz"])
        assert result.exit_code == 4

    def test_modes_search_filter(self) -> None:
        """modes --search filters by query string."""
        result = runner.invoke(app, ["modes", "--search", "polemical"])
        assert result.exit_code == 0
        assert "polemical" in result.stdout.lower()

    def test_modes_category_filter(self) -> None:
        """modes --category filters by category."""
        result = runner.invoke(app, ["modes", "--category", "confrontational"])
        assert result.exit_code == 0

    def test_voices_command_does_not_exist(self) -> None:
        """The 'voices' command does not exist and results in an error."""
        result = runner.invoke(app, ["voices"])
        # Typer will return 'No such command' and non-zero exit code
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 3. config command
# ---------------------------------------------------------------------------

class TestConfigCommand:
    """Tests for the 'config' command."""

    @patch("opinionforge.cli.get_settings")
    def test_config_shows_settings_masked(self, mock_settings: MagicMock) -> None:
        """config command shows settings with API keys masked."""
        settings = MagicMock()
        settings.opinionforge_llm_provider = "anthropic"
        settings.anthropic_api_key = "test-anthropic-very-secret-1234"
        settings.openai_api_key = None
        settings.opinionforge_search_provider = "tavily"
        settings.opinionforge_search_api_key = "tvly-abcdef1234"
        mock_settings.return_value = settings

        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "anthropic" in result.stdout.lower()
        assert "1234" in result.stdout
        assert "very-secret" not in result.stdout

    def test_config_set_api_key_blocked(self) -> None:
        """Setting API keys via CLI is blocked."""
        result = runner.invoke(app, ["config", "--set", "anthropic_api_key", "new-key"])
        assert result.exit_code == 2

    def test_config_set_unknown_key(self) -> None:
        """Setting an unknown config key exits with code 2."""
        result = runner.invoke(app, ["config", "--set", "unknown_key", "value"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# 4. Deprecated flag error messages
# ---------------------------------------------------------------------------

class TestDeprecatedFlagErrors:
    """Tests that deprecated flags produce clear error messages referencing new flags."""

    def test_voice_flag_exits_2_with_mode_message(self) -> None:
        """--voice flag exits code 2 with a message referencing '--mode'."""
        result = runner.invoke(app, ["write", "test topic", "--voice", "some_writer"])
        assert result.exit_code == 2
        output = (result.stdout + (result.stderr or "")).lower()
        assert "--mode" in output

    def test_spectrum_flag_exits_2_with_stance_message(self) -> None:
        """--spectrum flag exits code 2 with a message referencing '--stance'."""
        result = runner.invoke(app, ["write", "test topic", "--spectrum", "0"])
        assert result.exit_code == 2
        output = (result.stdout + (result.stderr or "")).lower()
        assert "--stance" in output

    def test_no_disclaimer_flag_exits_2(self) -> None:
        """--no-disclaimer flag exits code 2."""
        result = runner.invoke(app, ["write", "test topic", "--no-disclaimer"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# 5. write command — input validation
# ---------------------------------------------------------------------------

class TestWriteValidation:
    """Tests for write command input validation."""

    def test_write_no_topic_no_url_no_file(self) -> None:
        """write with no topic, URL, or file exits with code 2."""
        result = runner.invoke(app, ["write", "--no-preview", "--no-research", "--mode", "analytical"])
        assert result.exit_code == 2

    def test_write_nonexistent_mode_exits_4(self) -> None:
        """write with a nonexistent --mode exits with code 4."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--mode", "nonexistent_mode_xyz",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 4

    def test_write_stance_too_high_exits_2(self) -> None:
        """--stance 101 exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--stance", "101",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_write_stance_too_low_exits_2(self) -> None:
        """--stance -101 exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--stance", "-101",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_write_intensity_too_high_exits_2(self) -> None:
        """--intensity 1.1 exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--intensity", "1.1",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_write_intensity_too_low_exits_2(self) -> None:
        """--intensity -0.1 exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--intensity", "-0.1",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_write_invalid_length_below_min(self) -> None:
        """Length below minimum exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--length", "50",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_write_invalid_length_unknown_preset(self) -> None:
        """Unknown length preset exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--length", "tweet",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_write_stance_at_boundary_100_valid(self) -> None:
        """--stance 100 is a valid boundary value (no validation error at parse time)."""
        # This will fail later on mode loading, but not at stance validation
        result = runner.invoke(app, [
            "write", "test topic",
            "--stance", "100",
            "--mode", "nonexistent_mode_for_boundary_test",
            "--no-preview", "--no-research",
        ])
        # Should fail with code 4 (mode not found), NOT 2 (invalid stance)
        assert result.exit_code == 4

    def test_write_stance_at_boundary_minus100_valid(self) -> None:
        """--stance -100 is a valid boundary value."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--stance", "-100",
            "--mode", "nonexistent_mode_for_boundary_test",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 4


# ---------------------------------------------------------------------------
# 6. mode blend syntax
# ---------------------------------------------------------------------------

class TestModeBlendParsing:
    """Tests for mode blend syntax parsing."""

    def test_single_mode_no_colon(self) -> None:
        """Single mode string parses to 100% weight."""
        blend = _parse_mode_blend("analytical")
        assert blend.modes == [("analytical", 100.0)]

    def test_blend_two_modes(self) -> None:
        """Two-mode blend parses correctly."""
        blend = _parse_mode_blend("polemical:60,narrative:40")
        assert len(blend.modes) == 2
        assert ("polemical", 60.0) in blend.modes
        assert ("narrative", 40.0) in blend.modes

    def test_blend_weights_not_100_exits_2(self) -> None:
        """Blend weights that don't sum to 100 exit with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--mode", "polemical:50,narrative:30",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_blend_invalid_weight_exits_2(self) -> None:
        """Non-numeric weight in blend exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--mode", "analytical:abc",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_empty_mode_string_exits_2(self) -> None:
        """Empty --mode string exits with code 2."""
        result = runner.invoke(app, [
            "write", "test topic",
            "--mode", "",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# 7. write command — full flow (mocked)
# ---------------------------------------------------------------------------

class TestWriteFullFlow:
    """Tests for the write command full generation flow."""

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_no_preview_succeeds(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """write with --no-preview generates successfully."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        result = runner.invoke(app, [
            "write", "climate change",
            "--mode", "analytical",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        assert "Test Title" in result.stdout

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_no_research(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """write with --no-research skips research step."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        result = runner.invoke(app, [
            "write", "climate change",
            "--mode", "analytical",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        assert "Test Title" in result.stdout

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_output_to_file(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """write with --output writes to a file."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        out_file = tmp_path / "output.md"
        result = runner.invoke(app, [
            "write", "climate change",
            "--mode", "analytical",
            "--no-preview",
            "--no-research",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "Test Title" in content

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_always_includes_disclaimer(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """write output always includes the mandatory disclaimer."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        result = runner.invoke(app, [
            "write", "topic",
            "--mode", "analytical",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        assert "AI-assisted rhetorical controls" in result.stdout

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_mode_flag_accepted(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """--mode flag is accepted and passed to blend_modes."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        result = runner.invoke(app, [
            "write", "topic",
            "--mode", "polemical",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        # blend_modes should have been called with polemical mode
        mock_blend.assert_called_once()
        call_arg = mock_blend.call_args[0][0]
        assert call_arg.modes == [("polemical", 100.0)]

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_stance_flag_accepted(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """--stance flag is accepted and passed to apply_stance."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        result = runner.invoke(app, [
            "write", "topic",
            "--mode", "analytical",
            "--stance", "-50",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        mock_stance.assert_called_once()
        stance_arg = mock_stance.call_args[0][1]
        assert stance_arg.position == -50

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_intensity_flag_accepted(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """--intensity flag is accepted and passed to apply_stance."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        result = runner.invoke(app, [
            "write", "topic",
            "--mode", "analytical",
            "--intensity", "0.8",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        mock_stance.assert_called_once()
        stance_arg = mock_stance.call_args[0][1]
        assert abs(stance_arg.intensity - 0.8) < 0.001

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def test_write_custom_length(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """Custom word count length is accepted."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        result = runner.invoke(app, [
            "write", "topic",
            "--mode", "analytical",
            "--length", "1500",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 8. Exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:
    """Tests that CLI exits with PRD-specified codes."""

    def test_success_exit_0(self) -> None:
        """--help exits with code 0."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_invalid_stance_exit_2(self) -> None:
        """Out-of-range --stance exits with code 2."""
        result = runner.invoke(app, [
            "write", "topic", "--stance", "999",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 2

    def test_mode_not_found_exit_4(self) -> None:
        """Nonexistent --mode exits with code 4."""
        result = runner.invoke(app, [
            "write", "topic",
            "--mode", "nonexistent_mode_xyz",
            "--no-preview", "--no-research",
        ])
        assert result.exit_code == 4

    def test_similarity_screening_failure_exit_8(self) -> None:
        """Similarity screening failure exits with code 8."""
        with (
            patch("opinionforge.core.topic.ingest_text") as mock_ingest,
            patch("opinionforge.core.mode_engine.blend_modes") as mock_blend,
            patch("opinionforge.core.stance.apply_stance") as mock_stance,
            patch("opinionforge.core.generator.generate_piece") as mock_generate,
        ):
            mock_ingest.return_value = _make_topic_context()
            mock_blend.return_value = "mode prompt"
            mock_stance.return_value = "modified prompt"
            mock_generate.side_effect = RuntimeError(
                "Similarity screening failed — output blocked."
            )

            result = runner.invoke(app, [
                "write", "climate change",
                "--mode", "analytical",
                "--no-preview",
                "--no-research",
            ])
            assert result.exit_code == 8

    @patch("opinionforge.cli.get_settings")
    def test_config_masks_api_keys(self, mock_settings: MagicMock) -> None:
        """config command masks API keys."""
        settings = MagicMock()
        settings.opinionforge_llm_provider = "anthropic"
        settings.anthropic_api_key = "test-anthropic-key-abcd"
        settings.openai_api_key = None
        settings.opinionforge_search_provider = "tavily"
        settings.opinionforge_search_api_key = None
        mock_settings.return_value = settings

        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "test-anthropic-key-abcd"[:-4] not in result.stdout
        assert "abcd" in result.stdout

    def test_missing_api_key_exit_5(self) -> None:
        """Missing API key during generation exits with code 5."""
        with (
            patch("opinionforge.core.topic.ingest_text") as mock_ingest,
            patch("opinionforge.core.mode_engine.blend_modes") as mock_blend,
            patch("opinionforge.core.stance.apply_stance") as mock_stance,
            patch("opinionforge.core.generator.generate_piece") as mock_generate,
        ):
            mock_ingest.return_value = _make_topic_context()
            mock_blend.return_value = "mode prompt"
            mock_stance.return_value = "modified prompt"
            mock_generate.side_effect = SystemExit(5)

            result = runner.invoke(app, [
                "write", "climate change",
                "--mode", "analytical",
                "--no-preview",
                "--no-research",
            ])
            assert result.exit_code == 5


# ---------------------------------------------------------------------------
# 9. Network errors and input modes
# ---------------------------------------------------------------------------

class TestNetworkAndInputModes:
    """Tests for network error path and --url / --file input modes."""

    @patch("opinionforge.core.topic.ingest_url")
    def test_network_error_exit_3(self, mock_ingest_url: MagicMock) -> None:
        """A network failure during URL ingestion exits with code 3."""
        mock_ingest_url.side_effect = Exception("Connection timed out")

        result = runner.invoke(app, [
            "write", "",
            "--url", "https://example.com/article",
            "--mode", "analytical",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 3

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_url")
    def test_write_from_url(
        self,
        mock_ingest_url: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """--url flag ingests topic from a URL and generates successfully."""
        mock_ingest_url.return_value = _make_topic_context("URL Topic")
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece(
            topic=_make_topic_context("URL Topic")
        )

        result = runner.invoke(app, [
            "write", "",
            "--url", "https://example.com/article",
            "--mode", "analytical",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        mock_ingest_url.assert_called_once_with("https://example.com/article")

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_file")
    def test_write_from_file(
        self,
        mock_ingest_file: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--file flag ingests topic from a local file and generates successfully."""
        topic_file = tmp_path / "topic.txt"
        topic_file.write_text("My file topic content", encoding="utf-8")

        mock_ingest_file.return_value = _make_topic_context("File Topic")
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece(
            topic=_make_topic_context("File Topic")
        )

        result = runner.invoke(app, [
            "write",
            "--file", str(topic_file),
            "--mode", "analytical",
            "--no-preview",
            "--no-research",
        ])
        assert result.exit_code == 0
        mock_ingest_file.assert_called_once_with(str(topic_file))


# ---------------------------------------------------------------------------
# 10. Exporter output contains mandatory disclaimer
# ---------------------------------------------------------------------------

class TestWriteExporterDisclaimer:
    """Tests that exported output always includes the mandatory disclaimer."""

    @patch("opinionforge.core.generator.generate_piece")
    @patch("opinionforge.core.stance.apply_stance")
    @patch("opinionforge.core.mode_engine.blend_modes")
    @patch("opinionforge.core.topic.ingest_text")
    def _run_write_with_export(
        self,
        mock_ingest: MagicMock,
        mock_blend: MagicMock,
        mock_stance: MagicMock,
        mock_generate: MagicMock,
        export_fmt: str,
        tmp_path: Path,
    ) -> str:
        """Helper to run write with a given export format and return file content."""
        mock_ingest.return_value = _make_topic_context()
        mock_blend.return_value = "mode prompt"
        mock_stance.return_value = "modified prompt"
        mock_generate.return_value = _make_generated_piece()

        out_file = tmp_path / f"output.{export_fmt}"
        result = runner.invoke(app, [
            "write", "topic",
            "--mode", "analytical",
            "--export", export_fmt,
            "--no-preview",
            "--no-research",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0
        return out_file.read_text(encoding="utf-8")

    def test_substack_export_contains_disclaimer(self, tmp_path: Path) -> None:
        """Substack export always contains the mandatory disclaimer."""
        with (
            patch("opinionforge.core.topic.ingest_text") as mock_ingest,
            patch("opinionforge.core.mode_engine.blend_modes") as mock_blend,
            patch("opinionforge.core.stance.apply_stance") as mock_stance,
            patch("opinionforge.core.generator.generate_piece") as mock_generate,
        ):
            mock_ingest.return_value = _make_topic_context()
            mock_blend.return_value = "mode prompt"
            mock_stance.return_value = "modified prompt"
            mock_generate.return_value = _make_generated_piece()

            out_file = tmp_path / "output.md"
            result = runner.invoke(app, [
                "write", "topic",
                "--mode", "analytical",
                "--export", "substack",
                "--no-preview",
                "--no-research",
                "--output", str(out_file),
            ])
            assert result.exit_code == 0
            content = out_file.read_text(encoding="utf-8")
            assert "AI-assisted rhetorical controls" in content

    def test_medium_export_contains_disclaimer(self, tmp_path: Path) -> None:
        """Medium export always contains the mandatory disclaimer."""
        with (
            patch("opinionforge.core.topic.ingest_text") as mock_ingest,
            patch("opinionforge.core.mode_engine.blend_modes") as mock_blend,
            patch("opinionforge.core.stance.apply_stance") as mock_stance,
            patch("opinionforge.core.generator.generate_piece") as mock_generate,
        ):
            mock_ingest.return_value = _make_topic_context()
            mock_blend.return_value = "mode prompt"
            mock_stance.return_value = "modified prompt"
            mock_generate.return_value = _make_generated_piece()

            out_file = tmp_path / "output.md"
            result = runner.invoke(app, [
                "write", "topic",
                "--mode", "analytical",
                "--export", "medium",
                "--no-preview",
                "--no-research",
                "--output", str(out_file),
            ])
            assert result.exit_code == 0
            content = out_file.read_text(encoding="utf-8")
            assert "AI-assisted rhetorical controls" in content

    def test_wordpress_export_contains_disclaimer(self, tmp_path: Path) -> None:
        """WordPress export always contains the mandatory disclaimer."""
        with (
            patch("opinionforge.core.topic.ingest_text") as mock_ingest,
            patch("opinionforge.core.mode_engine.blend_modes") as mock_blend,
            patch("opinionforge.core.stance.apply_stance") as mock_stance,
            patch("opinionforge.core.generator.generate_piece") as mock_generate,
        ):
            mock_ingest.return_value = _make_topic_context()
            mock_blend.return_value = "mode prompt"
            mock_stance.return_value = "modified prompt"
            mock_generate.return_value = _make_generated_piece()

            out_file = tmp_path / "output.html"
            result = runner.invoke(app, [
                "write", "topic",
                "--mode", "analytical",
                "--export", "wordpress",
                "--no-preview",
                "--no-research",
                "--output", str(out_file),
            ])
            assert result.exit_code == 0
            content = out_file.read_text(encoding="utf-8")
            assert "AI-assisted rhetorical controls" in content

    def test_twitter_export_contains_disclaimer(self, tmp_path: Path) -> None:
        """Twitter export always contains the mandatory disclaimer."""
        with (
            patch("opinionforge.core.topic.ingest_text") as mock_ingest,
            patch("opinionforge.core.mode_engine.blend_modes") as mock_blend,
            patch("opinionforge.core.stance.apply_stance") as mock_stance,
            patch("opinionforge.core.generator.generate_piece") as mock_generate,
        ):
            mock_ingest.return_value = _make_topic_context()
            mock_blend.return_value = "mode prompt"
            mock_stance.return_value = "modified prompt"
            mock_generate.return_value = _make_generated_piece()

            out_file = tmp_path / "output.txt"
            result = runner.invoke(app, [
                "write", "topic",
                "--mode", "analytical",
                "--export", "twitter",
                "--no-preview",
                "--no-research",
                "--output", str(out_file),
            ])
            assert result.exit_code == 0
            content = out_file.read_text(encoding="utf-8")
            assert "AI-assisted rhetorical controls" in content


# ---------------------------------------------------------------------------
# 11. export command
# ---------------------------------------------------------------------------

class TestExportCommand:
    """Tests for the 'export' command."""

    def test_export_command_references_mode_not_voice(self) -> None:
        """export command error message references '--mode' not '--voice'."""
        result = runner.invoke(app, ["export", "some-id", "--format", "substack"])
        # Should exit with code 1 (phase 3 not implemented)
        assert result.exit_code == 1
        # The error message must reference --mode, not --voice
        output = result.stdout + (result.stderr or "")
        assert "--mode" in output
        assert "--voice" not in output

    def test_export_unknown_format_exits_2(self) -> None:
        """export with unknown format exits with code 2."""
        result = runner.invoke(app, ["export", "some-id", "--format", "myspace"])
        assert result.exit_code == 2
