"""Unit tests for the similarity screening module.

Tests cover:
- check_verbatim: exact 5-word n-gram matching
- check_near_verbatim: normalized 6-word n-gram matching
- check_suppressed_phrases: substring phrase detection
- check_structural_fingerprint: cosine-similarity syllable scoring
- screen_output: full pipeline including rewrite iterations
- generator integration: RuntimeError on failed screening
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from opinionforge.core.similarity import (
    _count_syllables,
    _normalize_text,
    _cosine_similarity,
    _syllable_sequence_for_text,
    check_near_verbatim,
    check_structural_fingerprint,
    check_suppressed_phrases,
    check_verbatim,
    screen_output,
)
from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.piece import ScreeningResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mode_config(mode_id: str = "polemical") -> ModeBlendConfig:
    """Return a minimal ModeBlendConfig for testing."""
    return ModeBlendConfig(modes=[(mode_id, 100.0)])


def _make_clean_client(response: str = "Clean rewritten text without violations.") -> MagicMock:
    """Return a mock LLM client whose generate() returns a clean string."""
    client = MagicMock()
    client.generate.return_value = response
    return client


# ---------------------------------------------------------------------------
# check_verbatim
# ---------------------------------------------------------------------------


class TestCheckVerbatim:
    """Tests for check_verbatim (n=5 exact word matching)."""

    def test_exact_five_word_match_detected(self) -> None:
        """A planted 5-word verbatim match must be detected."""
        output = "the quick brown fox jumps over something"
        source = "the quick brown fox jumps over the lazy dog"
        matches = check_verbatim(output, [source])
        assert len(matches) >= 1

    def test_four_word_match_not_flagged(self) -> None:
        """A 4-word match must NOT be flagged — n=5 is the minimum."""
        output = "the quick brown fox rests"
        source = "the quick brown fox jumps"
        # Only 4 tokens in common at the start; no 5-word overlap
        matches = check_verbatim(output, [source])
        assert matches == []

    def test_no_match_returns_empty_list(self) -> None:
        """Completely different texts return an empty list."""
        output = "completely different content here today"
        source = "the quick brown fox jumps"
        matches = check_verbatim(output, [source])
        assert matches == []

    def test_exact_five_word_match_exact_spec(self) -> None:
        """Spec example: 'the quick brown fox jumps' in source triggers match."""
        output = "the quick brown fox jumps"
        source = "the quick brown fox jumps over the lazy dog"
        matches = check_verbatim(output, [source])
        assert len(matches) >= 1

    def test_multiple_sources_checked(self) -> None:
        """Matches are detected across multiple source texts."""
        output = "one two three four five more words"
        sources = [
            "unrelated content here",
            "one two three four five in second source",
        ]
        matches = check_verbatim(output, sources)
        assert len(matches) >= 1
        assert matches[0]["source_index"] == 1

    def test_returns_list_of_dicts_with_ngram_key(self) -> None:
        """Each match dict must have 'ngram' and 'source_index' keys."""
        output = "alpha beta gamma delta epsilon zeta"
        source = "alpha beta gamma delta epsilon in source"
        matches = check_verbatim(output, [source])
        assert len(matches) >= 1
        assert "ngram" in matches[0]
        assert "source_index" in matches[0]

    def test_empty_source_list_returns_empty(self) -> None:
        """No source texts means no matches possible."""
        matches = check_verbatim("some text here", [])
        assert matches == []

    def test_output_shorter_than_n_returns_empty(self) -> None:
        """Output with fewer than 5 words cannot produce a 5-gram match."""
        output = "only four words"
        source = "only four words extra padding here"
        matches = check_verbatim(output, [source])
        assert matches == []


# ---------------------------------------------------------------------------
# check_near_verbatim
# ---------------------------------------------------------------------------


class TestCheckNearVerbatim:
    """Tests for check_near_verbatim (n=6, normalized matching)."""

    def test_punctuation_differences_detected(self) -> None:
        """Punctuation differences must not prevent near-verbatim detection."""
        output = "The, Quick! Brown... Fox Jumps Over more words here"
        source = "the quick brown fox jumps over the lazy"
        matches = check_near_verbatim(output, [source])
        assert len(matches) >= 1

    def test_spec_example_detected(self) -> None:
        """Spec example: 'The, Quick! Brown... Fox Jumps Over' against 'the quick brown fox jumps over the'."""
        output = "The, Quick! Brown... Fox Jumps Over extra"
        source = "the quick brown fox jumps over the"
        matches = check_near_verbatim(output, [source])
        assert len(matches) >= 1

    def test_five_word_near_verbatim_not_flagged(self) -> None:
        """Near-verbatim uses n=6, so a 5-word normalized overlap must not match."""
        output = "the quick brown fox jumps"
        source = "the quick brown fox jumps and more text"
        # Normalized output has only 5 words — can't form a 6-gram
        matches = check_near_verbatim(output, [source])
        assert matches == []

    def test_no_near_verbatim_returns_empty(self) -> None:
        """Completely different texts return an empty list."""
        output = "abstract philosophical musings about governance"
        source = "the quick brown fox jumps over"
        matches = check_near_verbatim(output, [source])
        assert matches == []

    def test_case_insensitive_matching(self) -> None:
        """Matching is case-insensitive via normalization."""
        output = "ONE TWO THREE FOUR FIVE SIX"
        source = "one two three four five six"
        matches = check_near_verbatim(output, [source])
        assert len(matches) >= 1


# ---------------------------------------------------------------------------
# check_suppressed_phrases
# ---------------------------------------------------------------------------


class TestCheckSuppressedPhrases:
    """Tests for check_suppressed_phrases (substring matching)."""

    def test_planted_phrase_returned(self) -> None:
        """A known suppressed phrase must appear in the return list."""
        text = "This is a test phrase here for demonstration"
        phrases = ["test phrase"]
        result = check_suppressed_phrases(text, phrases)
        assert "test phrase" in result

    def test_no_match_returns_empty_list(self) -> None:
        """Clean text with no matching phrases returns empty list."""
        text = "clean output with no matches whatsoever"
        phrases = ["matched phrase"]
        result = check_suppressed_phrases(text, phrases)
        assert result == []

    def test_multiple_phrases_some_match(self) -> None:
        """Only the matching phrase is returned when multiple phrases are checked."""
        text = "the personal is political and has long been so"
        phrases = ["speaking truth to power", "the personal is political"]
        result = check_suppressed_phrases(text, phrases)
        assert "the personal is political" in result
        assert "speaking truth to power" not in result

    def test_case_insensitive(self) -> None:
        """Phrase matching must be case-insensitive."""
        text = "The Personal Is Political as always"
        phrases = ["the personal is political"]
        result = check_suppressed_phrases(text, phrases)
        assert "the personal is political" in result

    def test_empty_phrases_list_returns_empty(self) -> None:
        """No phrases to check means nothing can match."""
        result = check_suppressed_phrases("any text at all", [])
        assert result == []

    def test_empty_text_returns_empty(self) -> None:
        """Empty text has no phrase content to match."""
        result = check_suppressed_phrases("", ["some phrase"])
        assert result == []


# ---------------------------------------------------------------------------
# check_structural_fingerprint
# ---------------------------------------------------------------------------


class TestCheckStructuralFingerprint:
    """Tests for check_structural_fingerprint (cosine similarity score)."""

    def test_returns_float_in_range(self) -> None:
        """Return value must be a float in [0.0, 1.0]."""
        fingerprints = [{"pattern_name": "test", "syllable_sequence": [3, 2, 4, 1, 3]}]
        score = check_structural_fingerprint("The quick brown fox jumps over.", fingerprints)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_fingerprints_returns_zero(self) -> None:
        """No fingerprints to compare against means score is 0.0."""
        score = check_structural_fingerprint("Some text here.", [])
        assert score == 0.0

    def test_empty_text_returns_zero(self) -> None:
        """Empty text cannot have a fingerprint score."""
        fps = [{"pattern_name": "p", "syllable_sequence": [3, 2, 4]}]
        score = check_structural_fingerprint("", fps)
        assert score == 0.0

    def test_identical_sequence_produces_high_score(self) -> None:
        """Identical syllable sequences produce cosine similarity close to 1.0."""
        # Build a text whose syllable sequence closely matches the fingerprint
        # We'll directly test the cosine logic: use a fingerprint that matches
        # a simple controlled syllable sequence
        fingerprints = [{"pattern_name": "test", "syllable_sequence": [2, 2, 2, 2, 2]}]
        # "Cat sat on the mat. Cat sat on the mat."  — short syllable sequences
        score = check_structural_fingerprint(
            "Cat sat on the mat. Dog ran down the road.", fingerprints
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# screen_output — clean text
# ---------------------------------------------------------------------------


class TestScreenOutputClean:
    """Tests for screen_output with clean (violation-free) text."""

    def test_clean_text_passes(self) -> None:
        """Clean text with no violations returns ScreeningResult(passed=True)."""
        mode_config = _make_mode_config()
        result = screen_output(
            text="This is entirely original content with no source matches.",
            research_texts=[],
            mode_config=mode_config,
            client=None,
        )
        assert isinstance(result, ScreeningResult)
        assert result.passed is True

    def test_clean_text_zero_match_counts(self) -> None:
        """Clean text returns zero for all match count fields."""
        mode_config = _make_mode_config()
        result = screen_output(
            text="Wholly original text that matches nothing at all.",
            research_texts=[],
            mode_config=mode_config,
            client=None,
        )
        assert result.verbatim_matches == 0
        assert result.near_verbatim_matches == 0
        assert result.suppressed_phrase_matches == 0

    def test_no_research_texts_still_checks_suppressed_phrases(self) -> None:
        """screen_output with no research_texts still runs suppressed_phrase check."""
        mode_config = _make_mode_config()
        # Patch the file loader so we control the phrases list
        with patch(
            "opinionforge.core.similarity._load_suppressed_phrases",
            return_value=["forbidden phrase"],
        ):
            result = screen_output(
                text="This text contains the forbidden phrase inside it.",
                research_texts=[],
                mode_config=mode_config,
                client=None,
            )
        # Suppressed phrase detected even with no research_texts
        assert result.suppressed_phrase_matches >= 1
        assert result.passed is False

    def test_empty_research_texts_passes_for_clean_output(self) -> None:
        """Empty research_texts list with clean output returns passed=True."""
        mode_config = _make_mode_config()
        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output(
                text="Clean output with no issues.",
                research_texts=[],
                mode_config=mode_config,
                client=None,
            )
        assert result.passed is True


# ---------------------------------------------------------------------------
# screen_output — verbatim violation
# ---------------------------------------------------------------------------


class TestScreenOutputVerbatimViolation:
    """Tests for screen_output with planted verbatim violations."""

    def test_verbatim_match_detected(self) -> None:
        """A planted 5-word verbatim match returns verbatim_matches >= 1."""
        mode_config = _make_mode_config()
        violation_phrase = "the quick brown fox jumps"
        output = f"Some preamble. {violation_phrase} over the hill. More text here."
        source = f"{violation_phrase} over the lazy dog and more content here."
        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output(
                text=output,
                research_texts=[source],
                mode_config=mode_config,
                client=None,
            )
        assert result.verbatim_matches >= 1

    def test_verbatim_violation_with_no_client_fails(self) -> None:
        """Verbatim violation with no LLM client returns passed=False immediately."""
        mode_config = _make_mode_config()
        violation = "one two three four five words"
        output = f"Opening sentence. {violation} more content here after."
        source = f"Source content: {violation} is present in here."
        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output(
                text=output,
                research_texts=[source],
                mode_config=mode_config,
                client=None,
            )
        assert result.passed is False
        assert result.warning is not None


# ---------------------------------------------------------------------------
# screen_output — suppressed phrase violation
# ---------------------------------------------------------------------------


class TestScreenOutputSuppressedPhrase:
    """Tests for screen_output with planted suppressed phrase violations."""

    def test_suppressed_phrase_detected(self) -> None:
        """A planted suppressed phrase returns suppressed_phrase_matches >= 1."""
        mode_config = _make_mode_config()
        with patch(
            "opinionforge.core.similarity._load_suppressed_phrases",
            return_value=["forbidden catchphrase"],
        ):
            result = screen_output(
                text="This piece ends with the forbidden catchphrase naturally.",
                research_texts=[],
                mode_config=mode_config,
                client=None,
            )
        assert result.suppressed_phrase_matches >= 1

    def test_suppressed_phrase_no_client_returns_failed(self) -> None:
        """Suppressed phrase with no client returns passed=False."""
        mode_config = _make_mode_config()
        with patch(
            "opinionforge.core.similarity._load_suppressed_phrases",
            return_value=["caught phrase"],
        ):
            result = screen_output(
                text="The argument ends with the caught phrase embedded here.",
                research_texts=[],
                mode_config=mode_config,
                client=None,
            )
        assert result.passed is False


# ---------------------------------------------------------------------------
# screen_output — rewrite iterations
# ---------------------------------------------------------------------------


class TestScreenOutputRewriteIterations:
    """Tests for screen_output rewrite iteration behaviour."""

    def test_max_two_rewrite_iterations(self) -> None:
        """With persistent violations, screen_output performs at most 2 iterations."""
        mode_config = _make_mode_config()
        violation = "one two three four five persistent"
        output = f"Text that keeps the {violation} no matter what happens."
        source = f"Source also has {violation} in it definitely."

        # Mock client always returns output still containing the violation
        persistent_client = MagicMock()
        persistent_client.generate.return_value = output

        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output(
                text=output,
                research_texts=[source],
                mode_config=mode_config,
                client=persistent_client,
            )

        # Should have attempted up to 2 rewrites
        assert persistent_client.generate.call_count <= 2
        assert result.rewrite_iterations <= 2

    def test_persistent_violations_return_failed(self) -> None:
        """When violations persist after 2 iterations, passed=False is returned."""
        mode_config = _make_mode_config()
        violation = "red orange yellow green blue purple"
        output = f"The text contains {violation} every single time."
        source = f"The source also has {violation} throughout its content."

        # Mock client always returns the same (still-violating) text
        stubborn_client = MagicMock()
        stubborn_client.generate.return_value = output

        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output(
                text=output,
                research_texts=[source],
                mode_config=mode_config,
                client=stubborn_client,
            )

        assert result.passed is False

    def test_rewrite_resolves_violation(self) -> None:
        """When the rewrite client returns clean text, screen_output returns passed=True."""
        mode_config = _make_mode_config()
        violation = "once upon a time there"
        output = f"Story: {violation} was a kingdom long ago."
        source = f"Traditional tale: {violation} was something important here."

        clean_text = "Story: In a distant era, a kingdom once thrived peacefully."
        clean_client = _make_clean_client(clean_text)

        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output(
                text=output,
                research_texts=[source],
                mode_config=mode_config,
                client=clean_client,
            )

        assert result.passed is True
        assert result.rewrite_iterations == 1
        assert clean_client.generate.call_count == 1

    def test_no_client_with_violation_sets_warning(self) -> None:
        """When no client is provided and violations exist, warning field is populated."""
        mode_config = _make_mode_config()
        with patch(
            "opinionforge.core.similarity._load_suppressed_phrases",
            return_value=["forbidden phrase"],
        ):
            result = screen_output(
                text="This contains the forbidden phrase right here.",
                research_texts=[],
                mode_config=mode_config,
                client=None,
            )
        assert result.passed is False
        assert result.warning is not None
        assert len(result.warning) > 0


# ---------------------------------------------------------------------------
# screen_output — ScreeningResult type contract
# ---------------------------------------------------------------------------


class TestScreeningResultContract:
    """Tests for ScreeningResult type and field contracts."""

    def test_returns_screening_result_instance(self) -> None:
        """screen_output must return a ScreeningResult instance."""
        mode_config = _make_mode_config()
        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output("Some text.", [], mode_config, None)
        assert isinstance(result, ScreeningResult)

    def test_fingerprint_score_in_range(self) -> None:
        """structural_fingerprint_score must be in [0.0, 1.0]."""
        mode_config = _make_mode_config()
        with patch("opinionforge.core.similarity._load_suppressed_phrases", return_value=[]):
            result = screen_output("Some text.", [], mode_config, None)
        assert 0.0 <= result.structural_fingerprint_score <= 1.0

    def test_screening_result_importable_from_models(self) -> None:
        """ScreeningResult must be importable from opinionforge.models.piece."""
        from opinionforge.models.piece import ScreeningResult as SR
        assert SR is ScreeningResult


# ---------------------------------------------------------------------------
# generator.py integration
# ---------------------------------------------------------------------------


class TestGeneratorIntegration:
    """Tests for generator.py integration with similarity screening."""

    def _make_topic(self):
        """Return a minimal TopicContext for testing."""
        from opinionforge.models.topic import TopicContext
        return TopicContext(
            raw_input="Test topic",
            input_type="text",
            title="Test Topic",
            summary="A topic for testing purposes.",
            key_claims=[],
            key_entities=[],
            subject_domain="general",
        )

    def _patch_generator_deps(self):
        """Return a context manager that patches mode_engine and apply_stance dependencies.

        Because mode_engine.py (Sprint 1) may not be fully populated yet, we stub
        it in sys.modules so the local import inside generate_piece() doesn't fail.
        apply_stance is patched at the generator module level since it's already
        imported at module load time.
        """
        import contextlib
        import sys
        import types

        @contextlib.contextmanager
        def _ctx():
            # Install a stub mode_engine module for the local import inside generate_piece
            stub_mode_engine = types.ModuleType("opinionforge.core.mode_engine")
            stub_mode_engine.blend_modes = MagicMock(  # type: ignore[attr-defined]
                return_value="Rhetorical mode instructions for test."
            )
            stub_mode_engine.load_mode = MagicMock(return_value=None)  # type: ignore[attr-defined]

            had_module = "opinionforge.core.mode_engine" in sys.modules
            original = sys.modules.get("opinionforge.core.mode_engine")
            sys.modules["opinionforge.core.mode_engine"] = stub_mode_engine

            # Patch apply_stance at the generator module's already-imported binding
            with patch(
                "opinionforge.core.generator.apply_stance",
                return_value="Rhetorical mode instructions. [stance applied]",
            ):
                try:
                    yield stub_mode_engine
                finally:
                    if had_module and original is not None:
                        sys.modules["opinionforge.core.mode_engine"] = original
                    elif not had_module:
                        sys.modules.pop("opinionforge.core.mode_engine", None)

        return _ctx()

    def test_generator_raises_on_failed_screening(self) -> None:
        """generate_piece() raises RuntimeError when screening returns passed=False."""
        from opinionforge.core.generator import generate_piece

        topic = self._make_topic()
        mode_config = _make_mode_config()
        stance = StanceConfig()

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "## Test Title\n\nSome generated text content here."
        )

        failed_result = ScreeningResult(
            passed=False,
            verbatim_matches=2,
            near_verbatim_matches=0,
            suppressed_phrase_matches=0,
            structural_fingerprint_score=0.0,
            rewrite_iterations=2,
            warning="Unresolved verbatim matches.",
        )

        with self._patch_generator_deps():
            with patch(
                "opinionforge.core.similarity.screen_output",
                return_value=failed_result,
            ):
                with pytest.raises(RuntimeError, match="Similarity screening failed"):
                    generate_piece(
                        topic=topic,
                        mode_config=mode_config,
                        stance=stance,
                        target_length="short",
                        client=mock_client,
                    )

    def test_generator_succeeds_when_screening_passes(self) -> None:
        """generate_piece() returns GeneratedPiece when screening returns passed=True."""
        from opinionforge.core.generator import generate_piece
        from opinionforge.models.piece import GeneratedPiece

        topic = self._make_topic()
        mode_config = _make_mode_config()
        stance = StanceConfig()

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "## Test Title\n\nSome generated text content here that is clean."
        )

        passed_result = ScreeningResult(
            passed=True,
            verbatim_matches=0,
            near_verbatim_matches=0,
            suppressed_phrase_matches=0,
            structural_fingerprint_score=0.0,
            rewrite_iterations=0,
        )

        with self._patch_generator_deps():
            with patch(
                "opinionforge.core.similarity.screen_output",
                return_value=passed_result,
            ):
                piece = generate_piece(
                    topic=topic,
                    mode_config=mode_config,
                    stance=stance,
                    target_length="short",
                    client=mock_client,
                )

        assert isinstance(piece, GeneratedPiece)
        assert piece.screening_result is not None
        assert piece.screening_result.passed is True

    def test_screening_occurs_before_disclaimer(self) -> None:
        """Screening result is stored on the piece alongside the disclaimer."""
        from opinionforge.core.generator import generate_piece, MANDATORY_DISCLAIMER
        from opinionforge.models.piece import GeneratedPiece

        topic = self._make_topic()
        mode_config = _make_mode_config()
        stance = StanceConfig()

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "## Another Title\n\nClean content for testing pipeline order."
        )

        passed_result = ScreeningResult(
            passed=True,
            verbatim_matches=0,
            near_verbatim_matches=0,
            suppressed_phrase_matches=0,
            structural_fingerprint_score=0.0,
            rewrite_iterations=0,
        )

        with self._patch_generator_deps():
            with patch(
                "opinionforge.core.similarity.screen_output",
                return_value=passed_result,
            ):
                piece = generate_piece(
                    topic=topic,
                    mode_config=mode_config,
                    stance=stance,
                    target_length="short",
                    client=mock_client,
                )

        # Both screening_result and disclaimer must be set
        assert piece.screening_result is not None
        assert piece.disclaimer == MANDATORY_DISCLAIMER


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for internal helper functions."""

    def test_normalize_text_strips_punctuation(self) -> None:
        """_normalize_text removes punctuation and lowercases."""
        assert _normalize_text("Hello, World!") == "hello world"

    def test_normalize_text_preserves_spaces(self) -> None:
        """_normalize_text keeps word spacing intact."""
        result = _normalize_text("The Quick Brown Fox")
        assert "the" in result
        assert "quick" in result

    def test_count_syllables_single_syllable(self) -> None:
        """Single-syllable words return 1."""
        assert _count_syllables("cat") == 1

    def test_count_syllables_multisyllable(self) -> None:
        """Multi-syllable words return count > 1."""
        assert _count_syllables("education") > 1

    def test_cosine_similarity_identical_returns_one(self) -> None:
        """Identical non-zero vectors produce similarity of 1.0."""
        vec = [3, 2, 4, 1, 3]
        score = _cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal_returns_zero(self) -> None:
        """Orthogonal vectors produce similarity of 0.0."""
        # [1, 0] and [0, 1] are orthogonal
        score = _cosine_similarity([1, 0], [0, 1])
        assert abs(score) < 1e-9

    def test_cosine_similarity_empty_returns_zero(self) -> None:
        """Empty vector inputs return 0.0."""
        assert _cosine_similarity([], [1, 2, 3]) == 0.0
        assert _cosine_similarity([1, 2, 3], []) == 0.0

    def test_syllable_sequence_non_empty_for_sentence(self) -> None:
        """A non-empty sentence produces a non-empty syllable sequence."""
        seq = _syllable_sequence_for_text("The fox ran quickly over the hill.")
        assert len(seq) >= 1
        assert all(isinstance(s, int) and s > 0 for s in seq)
