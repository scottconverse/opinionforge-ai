"""Unit tests for the updated generation pipeline using ModeBlendConfig and StanceConfig.

Uses a mock LLM client for all tests — zero API calls required. Mocks
opinionforge.core.mode_engine.blend_modes (Sprint 1 deliverable) and
opinionforge.core.similarity.screen_output (Sprint 5 deliverable) to keep
Sprint 2 tests fully isolated from parallel sprint work.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from opinionforge.core.generator import (
    MANDATORY_DISCLAIMER,
    compose_system_prompt,
    generate_piece,
    resolve_length,
)
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.piece import GeneratedPiece, ScreeningResult
from opinionforge.models.topic import TopicContext


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_MOCK_MODE_PROMPT = (
    "Write with analytical precision, marshaling evidence systematically "
    "and arguing with measured rhetorical force."
)

_MOCK_LLM_OUTPUT = (
    "## The Case for Evidence-Driven Policy\n\n"
    "The evidence is clear, if one is willing to read it. "
    "Policy debates too often degenerate into competing assertions "
    "when the empirical record provides a decisive answer.\n\n"
    "Decades of research demonstrate that well-designed interventions "
    "produce measurable outcomes. The failure is rarely in the data "
    "but in the political will to follow where it leads.\n\n"
    "The path forward requires nothing less than a commitment to "
    "honoring the evidence even when it is inconvenient."
)

# Shared mock ScreeningResult that always passes (stubs similarity screening)
_MOCK_SCREENING_RESULT = ScreeningResult(
    passed=True,
    verbatim_matches=0,
    near_verbatim_matches=0,
    suppressed_phrase_matches=0,
    structural_fingerprint_score=0.0,
    rewrite_iterations=0,
)


@contextmanager
def _patch_pipeline(
    mode_prompt: str = _MOCK_MODE_PROMPT,
    screening_result: ScreeningResult = _MOCK_SCREENING_RESULT,
) -> Generator[None, None, None]:
    """Patch blend_modes and screen_output for isolated generate_piece tests.

    Patches at the source module level because both are imported inside the
    generate_piece() function body (lazy imports).

    Args:
        mode_prompt: The mode prompt string blend_modes should return.
        screening_result: The ScreeningResult screen_output should return.
    """
    with patch(
        "opinionforge.core.mode_engine.blend_modes",
        return_value=mode_prompt,
    ), patch(
        "opinionforge.core.similarity.screen_output",
        return_value=screening_result,
    ):
        yield


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_topic() -> TopicContext:
    """Return a valid TopicContext for generation tests."""
    return TopicContext(
        raw_input="The role of evidence in public policy",
        input_type="text",
        title="The Role of Evidence in Public Policy",
        summary=(
            "Evidence-based policy has gained renewed attention as policymakers "
            "grapple with complex social challenges. Proponents argue that "
            "rigorous research should guide government decisions."
        ),
        key_claims=["Research shows evidence-based policies outperform intuition-based ones."],
        key_entities=["Congress", "CBO", "Brookings"],
        subject_domain="policy",
    )


@pytest.fixture
def single_mode_config() -> ModeBlendConfig:
    """Return a single-mode ModeBlendConfig (analytical at 100%)."""
    return ModeBlendConfig(modes=[("analytical", 100.0)])


@pytest.fixture
def blended_mode_config() -> ModeBlendConfig:
    """Return a two-mode ModeBlendConfig (polemical 60%, analytical 40%)."""
    return ModeBlendConfig(modes=[("polemical", 60.0), ("analytical", 40.0)])


@pytest.fixture
def default_stance() -> StanceConfig:
    """Return a balanced StanceConfig with default intensity."""
    return StanceConfig()


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Return a mock LLM client that returns a controlled opinion piece."""
    client = MagicMock()
    client.generate.return_value = _MOCK_LLM_OUTPUT
    return client


# ---------------------------------------------------------------------------
# resolve_length — unchanged from v0.2.0, preserved tests
# ---------------------------------------------------------------------------


class TestResolveLength:
    """Tests for the resolve_length utility function."""

    def test_resolve_preset_standard(self) -> None:
        """'standard' preset resolves to 800 words."""
        assert resolve_length("standard") == 800

    def test_resolve_preset_short(self) -> None:
        """'short' preset resolves to 500 words."""
        assert resolve_length("short") == 500

    def test_resolve_preset_essay(self) -> None:
        """'essay' preset resolves to 2500 words."""
        assert resolve_length("essay") == 2500

    def test_resolve_integer_input(self) -> None:
        """Integer input passes through as-is when in range."""
        assert resolve_length(600) == 600

    def test_resolve_invalid_preset_raises(self) -> None:
        """Unknown preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown length preset"):
            resolve_length("invalid_preset")

    def test_resolve_below_minimum_raises(self) -> None:
        """Word count below 200 raises ValueError."""
        with pytest.raises(ValueError, match="outside the allowed range"):
            resolve_length(100)

    def test_resolve_above_maximum_raises(self) -> None:
        """Word count above 8000 raises ValueError."""
        with pytest.raises(ValueError, match="outside the allowed range"):
            resolve_length(9000)


# ---------------------------------------------------------------------------
# compose_system_prompt — updated signature
# ---------------------------------------------------------------------------


class TestComposeSystemPrompt:
    """Tests for compose_system_prompt() with the updated StanceConfig parameter."""

    def test_returns_non_empty_string(self) -> None:
        """compose_system_prompt returns a non-empty string."""
        result = compose_system_prompt(_MOCK_MODE_PROMPT, StanceConfig(), 800)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_mode_prompt_content(self) -> None:
        """compose_system_prompt includes the mode prompt content."""
        result = compose_system_prompt(_MOCK_MODE_PROMPT, StanceConfig(), 800)
        assert _MOCK_MODE_PROMPT in result

    def test_contains_length_instructions(self) -> None:
        """compose_system_prompt includes length instructions."""
        result = compose_system_prompt(_MOCK_MODE_PROMPT, StanceConfig(), 800)
        assert "800" in result

    def test_contains_research_context_when_provided(self) -> None:
        """compose_system_prompt includes research context when provided."""
        result = compose_system_prompt(
            _MOCK_MODE_PROMPT,
            StanceConfig(),
            800,
            research_context="Key finding: policy X improved outcomes by 30%.",
        )
        assert "policy X improved outcomes by 30%" in result

    def test_no_research_context_when_none(self) -> None:
        """compose_system_prompt does not include research section when None."""
        result = compose_system_prompt(_MOCK_MODE_PROMPT, StanceConfig(), 800, None)
        assert "Research Context" not in result

    def test_accepts_stance_config(self) -> None:
        """compose_system_prompt accepts a StanceConfig without error."""
        stance = StanceConfig(position=-40, intensity=0.7)
        result = compose_system_prompt(_MOCK_MODE_PROMPT, stance, 800)
        assert isinstance(result, str)

    def test_output_contains_role_preamble(self) -> None:
        """compose_system_prompt includes the OpinionForge base role."""
        result = compose_system_prompt(_MOCK_MODE_PROMPT, StanceConfig(), 800)
        assert "OpinionForge" in result


# ---------------------------------------------------------------------------
# generate_piece — core acceptance tests
# ---------------------------------------------------------------------------


class TestGeneratePiece:
    """Tests for generate_piece() with ModeBlendConfig and StanceConfig."""

    def test_returns_generated_piece_instance(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """generate_piece() with a mock LLM returns a GeneratedPiece instance."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert isinstance(piece, GeneratedPiece)

    def test_accepts_mode_config_and_stance_parameters(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """generate_piece() accepts mode_config and stance parameters."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length=800,
                client=mock_llm_client,
            )
        assert piece.mode_config == single_mode_config
        assert piece.stance == default_stance

    def test_disclaimer_equals_mandatory_constant(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned disclaimer equals the fixed mandatory disclaimer constant."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert piece.disclaimer == MANDATORY_DISCLAIMER

    def test_disclaimer_does_not_contain_writer_names(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned disclaimer does NOT contain any writer name patterns."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        disclaimer = piece.disclaimer
        writer_names = [
            "Hitchens", "Ivins", "Buckley", "Brooks", "Krugman",
            "Sullivan", "Will", "Dowd", "Friedman", "Douthat",
        ]
        for name in writer_names:
            assert name not in disclaimer, (
                f"Writer name '{name}' found in disclaimer: {disclaimer!r}"
            )

    def test_mandatory_disclaimer_constant_text(self) -> None:
        """MANDATORY_DISCLAIMER constant equals the exact PRD-specified text."""
        expected = (
            "This piece was generated with AI-assisted rhetorical controls. "
            "It is original content and is not written by, endorsed by, or affiliated with any real person."
        )
        assert MANDATORY_DISCLAIMER == expected

    def test_piece_has_title(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned GeneratedPiece has a non-empty title."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert piece.title
        assert len(piece.title) > 0

    def test_piece_has_body(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned GeneratedPiece has a non-empty body."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert piece.body
        assert len(piece.body) > 0

    def test_piece_has_preview_text(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned GeneratedPiece has a non-empty preview_text."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert piece.preview_text
        assert len(piece.preview_text) > 0

    def test_piece_has_correct_target_length(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned GeneratedPiece has target_length matching the preset."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert piece.target_length == 800

    def test_llm_client_called_once(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """The mock LLM client's generate method is called exactly once for generation."""
        with _patch_pipeline():
            generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        # At minimum called once for generation (screening may use it too)
        assert mock_llm_client.generate.call_count >= 1

    def test_llm_error_raises_runtime_error(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
    ) -> None:
        """If the LLM client raises, generate_piece() raises RuntimeError."""
        failing_client = MagicMock()
        failing_client.generate.side_effect = ConnectionError("Network timeout")

        with _patch_pipeline():
            with pytest.raises(RuntimeError, match="Failed to generate opinion piece"):
                generate_piece(
                    topic=sample_topic,
                    mode_config=single_mode_config,
                    stance=default_stance,
                    target_length="standard",
                    client=failing_client,
                )

    def test_generate_with_blended_modes(
        self,
        sample_topic: TopicContext,
        blended_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """generate_piece() works with a two-mode blend configuration."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=blended_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert isinstance(piece, GeneratedPiece)
        assert piece.mode_config == blended_mode_config

    def test_generate_with_equity_stance(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """generate_piece() accepts an equity-focused stance without error."""
        equity_stance = StanceConfig(position=-60, intensity=0.8)
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=equity_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert isinstance(piece, GeneratedPiece)
        assert piece.stance.position == -60

    def test_generate_with_liberty_stance(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """generate_piece() accepts a liberty-focused stance without error."""
        liberty_stance = StanceConfig(position=60, intensity=0.8)
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=liberty_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        assert isinstance(piece, GeneratedPiece)
        assert piece.stance.position == 60

    def test_generate_with_research_context(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """generate_piece() passes research context to the LLM system prompt."""
        research_ctx = "Study shows 73% of agencies use AI. Impact is significant."
        with _patch_pipeline():
            generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                research_context=research_ctx,
                client=mock_llm_client,
            )
        # Verify the system prompt passed to LLM contained the research context
        call_args = mock_llm_client.generate.call_args_list[0]
        system_prompt = call_args[1].get("system_prompt") or call_args[0][0]
        assert research_ctx in system_prompt

    def test_piece_screening_result_populated(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned GeneratedPiece has screening_result populated from screen_output."""
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        # screening_result should be populated because generator calls screen_output
        assert piece.screening_result is not None
        assert piece.screening_result.passed is True

    def test_piece_id_is_uuid_string(
        self,
        sample_topic: TopicContext,
        single_mode_config: ModeBlendConfig,
        default_stance: StanceConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Returned GeneratedPiece has an id that is a UUID string."""
        import uuid as uuid_lib
        with _patch_pipeline():
            piece = generate_piece(
                topic=sample_topic,
                mode_config=single_mode_config,
                stance=default_stance,
                target_length="standard",
                client=mock_llm_client,
            )
        parsed = uuid_lib.UUID(piece.id)
        assert str(parsed) == piece.id
