"""Confusability Test Suite — 3-Evaluator Pipeline, 108 Test Cases.

Test Matrix:
    - 12 rhetorical modes × 3 topics × 3 stance/intensity combos = 108 test cases

Modes (12):
    analytical, aphoristic, data_driven, dialectical, forensic, measured,
    narrative, oratorical, polemical, populist, provocative, satirical

Topics (3):
    - "universal basic income"
    - "artificial intelligence regulation"
    - "urban housing policy"

Stance/intensity combos (3):
    - (0, 0.5)   — balanced, moderate intensity
    - (-60, 0.8) — equity-leaning, high intensity
    - (60, 0.8)  — liberty-leaning, high intensity

Threshold Rationale:
    LLM judge: A test fails when the judge's confidence that the text was written
    by any specific identifiable real person reaches or exceeds 0.6. This threshold
    is intentionally strict — only clear, strong stylistic matches cause failure.
    Outputs that read as generic formal or opinion writing pass at 0.0.

    Lexical scorer: Uses Jaccard token overlap against a small public-domain
    reference corpus (Project Gutenberg essays). Threshold is 0.25. The lexical
    check is a floor guard against outputs that reproduce unusual vocabulary
    clusters from recognizable style corpora, independent of the LLM judge.

    Regression set: Binary pass/fail. Fails immediately if any previously
    flagged text fragment appears in the generated output. Starts empty; no
    test case fails via this evaluator until the regression set is populated.

A test FAILS if ANY evaluator indicates confusability above its threshold.

Marks:
    All tests are marked @pytest.mark.slow AND @pytest.mark.confusability.
    Run with: pytest -m confusability
    Exclude with: pytest -m 'not slow'
    These tests are expected to run nightly in CI, not on every commit.

API Key Behavior:
    Tests that require API access skip gracefully (pytest.skip) when neither
    ANTHROPIC_API_KEY nor OPENAI_API_KEY is present in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from opinionforge.models.config import ModeBlendConfig, StanceConfig
from opinionforge.models.topic import TopicContext

# ---------------------------------------------------------------------------
# Import the confusability helpers
# ---------------------------------------------------------------------------
from tests.confusability_helpers import (
    LEXICAL_SIMILARITY_THRESHOLD,
    LLM_JUDGE_THRESHOLD,
    ConfusabilityResult,
    lexical_similarity_eval,
    llm_judge_eval,
    regression_eval,
)


# ---------------------------------------------------------------------------
# Test matrix constants
# ---------------------------------------------------------------------------

_MODES: list[str] = [
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

_TOPICS: list[str] = [
    "universal basic income",
    "artificial intelligence regulation",
    "urban housing policy",
]

# (position, intensity) combos
_STANCES: list[tuple[int, float]] = [
    (0, 0.5),
    (-60, 0.8),
    (60, 0.8),
]


def _make_test_id(mode: str, topic: str, position: int, intensity: float) -> str:
    """Construct a deterministic test ID from parametrize arguments.

    Args:
        mode: Rhetorical mode ID slug.
        topic: Topic string.
        position: Stance position integer.
        intensity: Rhetorical intensity float.

    Returns:
        A string safe for use as a pytest test ID.
    """
    topic_slug = topic.replace(" ", "_")
    intensity_str = str(intensity).replace(".", "p")
    return f"{mode}_{topic_slug}_{position:+d}_{intensity_str}"


# Build the full 108-case parameter list
_PARAM_CASES: list[tuple[str, str, int, float]] = [
    (mode, topic, position, intensity)
    for mode in _MODES
    for topic in _TOPICS
    for (position, intensity) in _STANCES
]

_PARAM_IDS: list[str] = [
    _make_test_id(mode, topic, position, intensity)
    for (mode, topic, position, intensity) in _PARAM_CASES
]

assert len(_PARAM_CASES) == 108, (
    f"Expected 108 test cases (12 modes × 3 topics × 3 stances), got {len(_PARAM_CASES)}"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression_cases() -> list[dict[str, Any]]:
    """Load regression cases from the YAML file.

    Returns:
        List of regression case dicts (may be empty on initial run).
    """
    yaml_path = Path(__file__).parent / "confusability_regression_set.yaml"
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("regression_cases", []) or []


@pytest.fixture(scope="module")
def llm_client() -> Any:
    """Return a real LLM client if an API key is available, else skip the module.

    Checks ANTHROPIC_API_KEY first, then OPENAI_API_KEY. Skips the entire
    module if neither is present.

    Returns:
        An LLMClient instance backed by Anthropic or OpenAI.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if anthropic_key:
        from opinionforge.core.preview import AnthropicLLMClient

        return AnthropicLLMClient(api_key=anthropic_key)

    if openai_key:
        from opinionforge.core.preview import OpenAILLMClient

        return OpenAILLMClient(api_key=openai_key)

    pytest.skip(
        "No API key available (ANTHROPIC_API_KEY or OPENAI_API_KEY required). "
        "Confusability tests skipped."
    )


@pytest.fixture(scope="module")
def judge_client() -> Any:
    """Return a real LLM client for the judge evaluator if an API key is available.

    Separate from ``llm_client`` so the judge and generation clients can be
    independently overridden in mock-based tests. Skips the module if no key
    is available.

    Returns:
        An LLMClient instance.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if anthropic_key:
        from opinionforge.core.preview import AnthropicLLMClient

        return AnthropicLLMClient(api_key=anthropic_key)

    if openai_key:
        from opinionforge.core.preview import OpenAILLMClient

        return OpenAILLMClient(api_key=openai_key)

    pytest.skip(
        "No API key available (ANTHROPIC_API_KEY or OPENAI_API_KEY required). "
        "Confusability tests skipped."
    )


# ---------------------------------------------------------------------------
# Topic builder
# ---------------------------------------------------------------------------


def _build_topic_context(topic: str) -> TopicContext:
    """Build a minimal TopicContext for a given topic string.

    Args:
        topic: The raw topic string (e.g. 'universal basic income').

    Returns:
        A TopicContext with a title, brief summary, and empty claim/entity lists.
    """
    titles = {
        "universal basic income": "Universal Basic Income: Economic Security for All?",
        "artificial intelligence regulation": "Regulating Artificial Intelligence: The Policy Debate",
        "urban housing policy": "Urban Housing Policy: Affordability, Density, and Reform",
    }
    summaries = {
        "universal basic income": (
            "Universal basic income proposes unconditional cash payments to all citizens "
            "regardless of employment status. Proponents argue it reduces poverty and "
            "increases economic security; critics warn of fiscal costs and work-disincentive effects."
        ),
        "artificial intelligence regulation": (
            "Artificial intelligence regulation debates whether governments should impose "
            "legal requirements on AI development and deployment. Advocates emphasize safety "
            "and accountability; opponents warn of innovation costs and regulatory overreach."
        ),
        "urban housing policy": (
            "Urban housing policy addresses affordability, zoning, and density in cities. "
            "Reform advocates push for increased supply through upzoning; skeptics "
            "emphasize neighborhood character and infrastructure constraints."
        ),
    }
    return TopicContext(
        raw_input=topic,
        input_type="text",
        title=titles[topic],
        summary=summaries[topic],
        key_claims=[],
        key_entities=[],
        subject_domain="policy",
    )


# ---------------------------------------------------------------------------
# Core evaluation helper
# ---------------------------------------------------------------------------


def _run_confusability_evaluation(
    mode: str,
    topic: str,
    position: int,
    intensity: float,
    generated_text: str,
    judge_client: Any,
    regression_cases: list[dict[str, Any]],
) -> ConfusabilityResult:
    """Run all three evaluators and return a ConfusabilityResult.

    Args:
        mode: Rhetorical mode ID.
        topic: Topic string.
        position: Stance position integer.
        intensity: Rhetorical intensity float.
        generated_text: The generated opinion piece body text.
        judge_client: LLM client for the judge evaluator.
        regression_cases: List of regression case dicts.

    Returns:
        A ConfusabilityResult recording all three evaluator scores.
    """
    # Evaluator 1: LLM judge
    judge_result = llm_judge_eval(generated_text, judge_client)

    # Evaluator 2: Lexical similarity
    lexical_score = lexical_similarity_eval(generated_text)

    # Evaluator 3: Regression set
    reg_passed = regression_eval(generated_text, regression_cases)

    return ConfusabilityResult(
        mode_id=mode,
        topic=topic,
        stance_position=position,
        intensity=intensity,
        llm_judge_author=judge_result["identified_author"],
        llm_judge_confidence=judge_result["confidence"],
        lexical_score=lexical_score,
        regression_passed=reg_passed,
    )


# ---------------------------------------------------------------------------
# 108-case confusability test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.confusability
@pytest.mark.parametrize(
    "mode,topic,position,intensity",
    _PARAM_CASES,
    ids=_PARAM_IDS,
)
def test_confusability(
    mode: str,
    topic: str,
    position: int,
    intensity: float,
    llm_client: Any,
    judge_client: Any,
    regression_cases: list[dict[str, Any]],
) -> None:
    """Verify a generated opinion piece is not confusable with any specific author.

    Generates a piece using the given mode, topic, and stance combination, then
    runs all three confusability evaluators. The test fails if any evaluator
    indicates confusability above its threshold.

    Evaluators:
        1. LLM judge: confidence < 0.6 required.
        2. Lexical similarity: Jaccard overlap < 0.25 required.
        3. Regression set: no flagged fragments found.

    Args:
        mode: Parametrized rhetorical mode ID.
        topic: Parametrized topic string.
        position: Parametrized stance position integer.
        intensity: Parametrized intensity float.
        llm_client: Module-scoped LLM client fixture (skips if no API key).
        judge_client: Module-scoped LLM judge client fixture (skips if no API key).
        regression_cases: Module-scoped regression case list fixture.
    """
    from opinionforge.core.generator import generate_piece
    from opinionforge.core.similarity import screen_output

    topic_context = _build_topic_context(topic)
    mode_config = ModeBlendConfig(modes=[(mode, 100.0)])
    stance = StanceConfig(position=position, intensity=intensity)

    # Generate the piece using the real engine
    piece = generate_piece(
        topic=topic_context,
        mode_config=mode_config,
        stance=stance,
        target_length="short",
        client=llm_client,
    )

    generated_text = piece.body

    result = _run_confusability_evaluation(
        mode=mode,
        topic=topic,
        position=position,
        intensity=intensity,
        generated_text=generated_text,
        judge_client=judge_client,
        regression_cases=regression_cases,
    )

    failure_messages: list[str] = []

    if result.llm_judge_confidence >= LLM_JUDGE_THRESHOLD:
        failure_messages.append(
            f"LLM judge flagged confusability: author={result.llm_judge_author!r}, "
            f"confidence={result.llm_judge_confidence:.3f} >= threshold {LLM_JUDGE_THRESHOLD}"
        )

    if result.lexical_score >= LEXICAL_SIMILARITY_THRESHOLD:
        failure_messages.append(
            f"Lexical scorer exceeded threshold: score={result.lexical_score:.3f} "
            f">= threshold {LEXICAL_SIMILARITY_THRESHOLD}"
        )

    if not result.regression_passed:
        failure_messages.append(
            "Regression evaluator failed: generated text contains a previously flagged fragment."
        )

    assert not failure_messages, (
        f"Confusability failures for {result!r}:\n"
        + "\n".join(f"  - {msg}" for msg in failure_messages)
    )


# ---------------------------------------------------------------------------
# Mock-based tests (run without API keys, always execute)
# ---------------------------------------------------------------------------


class TestLLMJudgeEvalMocked:
    """Tests for llm_judge_eval using a mock LLM client."""

    def test_low_confidence_response_passes(self) -> None:
        """A judge response with confidence 0.0 should return clean result."""
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '{"identified_author": null, "confidence": 0.0}'
        )
        result = llm_judge_eval("Some generic opinion text here.", mock_client)
        assert result["identified_author"] is None
        assert result["confidence"] == 0.0

    def test_high_confidence_response_returns_correct_values(self) -> None:
        """A judge response flagging an author should return their name and score."""
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '{"identified_author": "Jane Doe", "confidence": 0.85}'
        )
        result = llm_judge_eval("Some distinctive text.", mock_client)
        assert result["identified_author"] == "Jane Doe"
        assert result["confidence"] == pytest.approx(0.85)

    def test_confidence_clamped_above_one(self) -> None:
        """Confidence values above 1.0 in raw response must be clamped to 1.0."""
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '{"identified_author": "Someone", "confidence": 1.5}'
        )
        result = llm_judge_eval("text", mock_client)
        assert result["confidence"] <= 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        """Confidence values below 0.0 in raw response must be clamped to 0.0."""
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '{"identified_author": null, "confidence": -0.3}'
        )
        result = llm_judge_eval("text", mock_client)
        assert result["confidence"] >= 0.0

    def test_malformed_json_falls_back_safely(self) -> None:
        """A malformed JSON response should return a safe default (confidence 0.0)."""
        mock_client = MagicMock()
        mock_client.generate.return_value = "I cannot determine authorship."
        result = llm_judge_eval("text", mock_client)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_api_error_returns_safe_default(self) -> None:
        """An API exception during judge call should return confidence 0.0, not raise."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("API unavailable")
        result = llm_judge_eval("text", mock_client)
        assert result["confidence"] == 0.0
        assert result["identified_author"] is None

    def test_result_dict_keys_present(self) -> None:
        """Result dict must always contain 'identified_author' and 'confidence' keys."""
        mock_client = MagicMock()
        mock_client.generate.return_value = '{"confidence": 0.1}'
        result = llm_judge_eval("text", mock_client)
        assert "identified_author" in result
        assert "confidence" in result

    def test_none_author_string_normalized(self) -> None:
        """String 'none' or 'null' for identified_author should be normalized to None."""
        mock_client = MagicMock()
        mock_client.generate.return_value = (
            '{"identified_author": "none", "confidence": 0.1}'
        )
        result = llm_judge_eval("text", mock_client)
        assert result["identified_author"] is None


class TestLexicalSimilarityEval:
    """Tests for lexical_similarity_eval."""

    def test_empty_text_returns_zero(self) -> None:
        """Empty text should produce a similarity score of 0.0."""
        score = lexical_similarity_eval("")
        assert score == 0.0

    def test_returns_float_in_range(self) -> None:
        """Score must be a float in [0.0, 1.0]."""
        score = lexical_similarity_eval("Some opinion about housing policy.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_corpus_returns_zero(self) -> None:
        """An empty reference corpus should return 0.0."""
        score = lexical_similarity_eval("Some text here.", reference_corpus=[])
        assert score == 0.0

    def test_identical_text_returns_one(self) -> None:
        """Text identical to a corpus document should return Jaccard similarity of 1.0."""
        corpus = ["the quick brown fox jumps over the lazy dog"]
        score = lexical_similarity_eval(
            "the quick brown fox jumps over the lazy dog",
            reference_corpus=corpus,
        )
        assert score == pytest.approx(1.0)

    def test_unrelated_text_returns_low_score(self) -> None:
        """Generic modern policy writing should score below the threshold."""
        modern_policy_text = (
            "The government must act swiftly to implement comprehensive regulatory "
            "frameworks that address the emerging challenges posed by technological "
            "disruption in labor markets and economic systems."
        )
        score = lexical_similarity_eval(modern_policy_text)
        assert score < LEXICAL_SIMILARITY_THRESHOLD, (
            f"Expected modern policy text to score below {LEXICAL_SIMILARITY_THRESHOLD}, "
            f"got {score:.3f}"
        )

    def test_custom_corpus_used_when_provided(self) -> None:
        """A custom corpus should be used instead of the default when provided."""
        custom_corpus = ["alpha beta gamma delta epsilon zeta"]
        # Text that overlaps heavily with custom corpus
        score_custom = lexical_similarity_eval(
            "alpha beta gamma delta epsilon zeta", reference_corpus=custom_corpus
        )
        # Same text against empty corpus
        score_empty = lexical_similarity_eval(
            "alpha beta gamma delta epsilon zeta", reference_corpus=[]
        )
        assert score_custom > score_empty

    def test_partial_overlap_produces_intermediate_score(self) -> None:
        """Partially overlapping text should produce a score between 0.0 and 1.0."""
        corpus = ["apple banana cherry date elderberry fig grape honeydew kiwi lemon"]
        text = "apple banana mango papaya raspberry strawberry"
        score = lexical_similarity_eval(text, reference_corpus=corpus)
        assert 0.0 < score < 1.0


class TestRegressionEval:
    """Tests for regression_eval."""

    def test_empty_cases_returns_true(self) -> None:
        """An empty regression set should always return True (vacuous pass)."""
        assert regression_eval("any text", []) is True

    def test_no_match_returns_true(self) -> None:
        """Text that does not contain any flagged fragment should return True."""
        cases = [{"flagged_text_fragment": "very specific unique flagged phrase"}]
        assert regression_eval("completely different text", cases) is True

    def test_exact_match_returns_false(self) -> None:
        """Text containing an exact flagged fragment should return False."""
        cases = [{"flagged_text_fragment": "flagged fragment here"}]
        assert regression_eval("some text with flagged fragment here embedded", cases) is False

    def test_partial_match_returns_false(self) -> None:
        """A flagged fragment that is a substring of the text should return False."""
        cases = [{"flagged_text_fragment": "must fail"}]
        assert regression_eval("this text contains must fail words", cases) is False

    def test_multiple_cases_any_match_fails(self) -> None:
        """If any one regression case matches, the result should be False."""
        cases = [
            {"flagged_text_fragment": "first fragment"},
            {"flagged_text_fragment": "second fragment"},
        ]
        # Only second fragment matches
        assert regression_eval("text with second fragment inside", cases) is False

    def test_empty_fragment_does_not_match(self) -> None:
        """An empty flagged_text_fragment should not trigger a failure."""
        cases = [{"flagged_text_fragment": ""}]
        assert regression_eval("any text here", cases) is True

    def test_missing_fragment_key_skipped(self) -> None:
        """A case dict missing the flagged_text_fragment key should be skipped safely."""
        cases = [{"mode_id": "polemical", "topic": "housing"}]
        assert regression_eval("any text here", cases) is True


class TestConfusabilityResult:
    """Tests for the ConfusabilityResult container."""

    def _make_result(
        self,
        llm_confidence: float = 0.0,
        lexical_score: float = 0.0,
        regression_passed: bool = True,
    ) -> ConfusabilityResult:
        """Build a minimal ConfusabilityResult for testing."""
        return ConfusabilityResult(
            mode_id="analytical",
            topic="universal basic income",
            stance_position=0,
            intensity=0.5,
            llm_judge_author=None,
            llm_judge_confidence=llm_confidence,
            lexical_score=lexical_score,
            regression_passed=regression_passed,
        )

    def test_all_below_threshold_passes(self) -> None:
        """Result with all scores below thresholds should pass."""
        result = self._make_result(llm_confidence=0.1, lexical_score=0.1)
        assert result.passed is True

    def test_llm_at_threshold_fails(self) -> None:
        """LLM confidence at or above 0.6 should cause failure."""
        result = self._make_result(llm_confidence=LLM_JUDGE_THRESHOLD)
        assert result.passed is False

    def test_lexical_at_threshold_fails(self) -> None:
        """Lexical score at or above threshold should cause failure."""
        result = self._make_result(lexical_score=LEXICAL_SIMILARITY_THRESHOLD)
        assert result.passed is False

    def test_regression_fail_causes_failure(self) -> None:
        """A regression failure should cause the overall result to fail."""
        result = self._make_result(regression_passed=False)
        assert result.passed is False

    def test_repr_includes_key_fields(self) -> None:
        """__repr__ should include mode, topic, and pass/fail status."""
        result = self._make_result()
        r = repr(result)
        assert "analytical" in r
        assert "PASS" in r

    def test_all_attributes_accessible(self) -> None:
        """All declared attributes should be accessible on the result object."""
        result = self._make_result(llm_confidence=0.3, lexical_score=0.1)
        assert result.mode_id == "analytical"
        assert result.topic == "universal basic income"
        assert result.stance_position == 0
        assert result.intensity == 0.5
        assert result.llm_judge_author is None
        assert result.llm_judge_confidence == pytest.approx(0.3)
        assert result.lexical_score == pytest.approx(0.1)
        assert result.regression_passed is True


class TestRegressionSetYAML:
    """Tests for the confusability_regression_set.yaml file structure."""

    def test_yaml_loads_without_error(self) -> None:
        """The YAML file must load via yaml.safe_load without raising."""
        yaml_path = Path(__file__).parent / "confusability_regression_set.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data is not None

    def test_top_level_key_is_regression_cases(self) -> None:
        """Top-level key must be 'regression_cases'."""
        yaml_path = Path(__file__).parent / "confusability_regression_set.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "regression_cases" in data

    def test_regression_cases_is_a_list(self) -> None:
        """'regression_cases' value must be a list (possibly empty)."""
        yaml_path = Path(__file__).parent / "confusability_regression_set.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cases = data.get("regression_cases")
        assert isinstance(cases, list) or cases is None

    def test_initial_regression_set_is_empty(self) -> None:
        """On initial setup the regression set must be empty (no pre-existing cases)."""
        yaml_path = Path(__file__).parent / "confusability_regression_set.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cases = data.get("regression_cases") or []
        assert len(cases) == 0, (
            f"Expected 0 regression cases initially, found {len(cases)}. "
            "Populate this file only when actual confusability failures are found."
        )

    def test_populated_cases_have_required_fields(self) -> None:
        """Any populated regression case must have all required fields."""
        required_fields = {
            "mode_id",
            "topic",
            "stance_position",
            "intensity",
            "flagged_text_fragment",
            "flagged_author",
            "evaluator",
        }
        yaml_path = Path(__file__).parent / "confusability_regression_set.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cases = data.get("regression_cases") or []
        for i, case in enumerate(cases):
            missing = required_fields - set(case.keys())
            assert not missing, (
                f"Regression case {i} is missing required fields: {missing}"
            )


class TestParametrizeMatrix:
    """Tests verifying the parametrize matrix is built correctly."""

    def test_exactly_108_cases_generated(self) -> None:
        """The parameter list must contain exactly 108 cases."""
        assert len(_PARAM_CASES) == 108

    def test_all_12_modes_represented(self) -> None:
        """All 12 modes must appear in the parameter list."""
        modes_in_cases = {mode for (mode, _, _, _) in _PARAM_CASES}
        expected_modes = set(_MODES)
        assert modes_in_cases == expected_modes

    def test_all_3_topics_represented(self) -> None:
        """All 3 topics must appear in the parameter list."""
        topics_in_cases = {topic for (_, topic, _, _) in _PARAM_CASES}
        expected_topics = set(_TOPICS)
        assert topics_in_cases == expected_topics

    def test_all_3_stances_represented(self) -> None:
        """All 3 stance/intensity combos must appear in the parameter list."""
        stances_in_cases = {(pos, inten) for (_, _, pos, inten) in _PARAM_CASES}
        expected_stances = set(_STANCES)
        assert stances_in_cases == expected_stances

    def test_test_ids_unique(self) -> None:
        """All 108 test IDs must be unique."""
        assert len(_PARAM_IDS) == len(set(_PARAM_IDS))

    def test_test_ids_count(self) -> None:
        """There must be exactly 108 test IDs."""
        assert len(_PARAM_IDS) == 108

    def test_topic_slugs_in_ids(self) -> None:
        """Test IDs must contain topic words (with spaces replaced by underscores)."""
        for test_id in _PARAM_IDS:
            has_topic = any(
                topic.replace(" ", "_") in test_id for topic in _TOPICS
            )
            assert has_topic, f"No topic slug found in test ID: {test_id!r}"

    def test_mode_in_ids(self) -> None:
        """Each test ID must start with a mode ID."""
        for test_id, (mode, _, _, _) in zip(_PARAM_IDS, _PARAM_CASES):
            assert test_id.startswith(mode), (
                f"Test ID {test_id!r} does not start with mode {mode!r}"
            )
