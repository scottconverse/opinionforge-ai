"""Helper functions for the confusability evaluator pipeline.

Provides three independent evaluators used by the confusability test suite:

1. llm_judge_eval — a prompted LLM call that asks whether the text reads like
   any specific identifiable real person. Returns an identified author (if any)
   and a confidence score in [0.0, 1.0].

2. lexical_similarity_eval — computes token-level overlap (Jaccard similarity)
   between the generated text and a small reference corpus of public-domain
   writing samples. Returns a float in [0.0, 1.0]. A high score indicates the
   text shares unusual vocabulary clusters with a known style corpus.

3. regression_eval — checks whether the generated text contains any fragment
   from a list of previously flagged outputs. Returns True when the text is
   clean (passes), False when it matches a flagged fragment (fails).

All three evaluators accept plain Python types and are independently mockable
in tests.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Public threshold constants (callers may read these for assertion messages)
# ---------------------------------------------------------------------------

LLM_JUDGE_THRESHOLD: float = 0.6
"""Confidence threshold above which the LLM judge flags an output as confusable."""

LEXICAL_SIMILARITY_THRESHOLD: float = 0.25
"""Jaccard overlap threshold above which the lexical scorer flags an output."""


# ---------------------------------------------------------------------------
# Reference corpus (public-domain text fragments)
#
# Sources: Project Gutenberg essays and editorials — all pre-1928 publications,
# firmly in the public domain in the United States.
# These are short samples chosen to represent a range of 19th/early-20th century
# essay and editorial prose styles. They are used as a FLOOR CHECK: if generated
# text has very high token overlap with these samples, it may be reproducing
# archaic or genre-specific language clusters rather than producing original prose.
# ---------------------------------------------------------------------------

_PUBLIC_DOMAIN_CORPUS: list[str] = [
    # Thomas Paine, "Common Sense" (1776)
    (
        "Society in every state is a blessing, but government even in its best state "
        "is but a necessary evil; in its worst state an intolerable one; for when we "
        "suffer, or are exposed to the same miseries by a government, which we might "
        "expect in a country without government, our calamity is heightened by reflecting "
        "that we furnish the means by which we suffer."
    ),
    # Ralph Waldo Emerson, "Self-Reliance" (1841)
    (
        "To believe your own thought, to believe that what is true for you in your private "
        "heart is true for all men — that is genius. Speak your latent conviction, and it "
        "shall be the universal sense; for the inmost in due time becomes the outmost, and "
        "our first thought is rendered back to us by the trumpets of the Last Judgment."
    ),
    # Henry David Thoreau, "Civil Disobedience" (1849)
    (
        "That government is best which governs least; and I should like to see it acted "
        "up to more rapidly and systematically. Carried out, it finally amounts to this, "
        "which also I believe — That government is best which governs not at all; and "
        "when men are prepared for it, that will be the kind of government which they "
        "will have."
    ),
    # John Stuart Mill, "On Liberty" (1859)
    (
        "The object of this Essay is to assert one very simple principle, as entitled "
        "to govern absolutely the dealings of society with the individual in the way "
        "of compulsion and control, whether the means used be physical force in the "
        "form of legal penalties, or the moral coercion of public opinion."
    ),
    # Frederick Douglass, "What to the Slave is the Fourth of July?" (1852)
    (
        "Fellow-citizens, above your national, tumultuous joy, I hear the mournful wail "
        "of millions! whose chains, heavy and grievous yesterday, are, to-day, rendered "
        "more intolerable by the jubilee shouts that reach them. If I do forget, if I do "
        "not faithfully remember those bleeding children of sorrow this day, may my right "
        "hand forget her cunning, and may my tongue cleave to the roof of my mouth!"
    ),
    # Susan B. Anthony, speech after arrest (1873)
    (
        "Friends and fellow citizens: I stand before you tonight under indictment for "
        "the alleged crime of having voted at the last presidential election, without "
        "having a lawful right to vote. It shall be my work this evening to prove to "
        "you that in thus voting, I not only committed no crime, but, instead, simply "
        "exercised my citizen's rights, guaranteed to me and all United States citizens "
        "by the National Constitution."
    ),
    # W.E.B. Du Bois, "The Souls of Black Folk" (1903)
    (
        "Between me and the other world there is ever an unasked question: unasked by "
        "some through feelings of delicacy; by others through the difficulty of rightly "
        "framing it. All, nevertheless, flutter round it. They approach me in a "
        "half-hesitant sort of way, eye me curiously or compassionately, and then, "
        "instead of saying directly, How does it feel to be a problem? they say, "
        "I know an excellent colored man in my town."
    ),
    # Mark Twain, "License of the Press" (1873)
    (
        "The press has done much good, but it has done more evil. Its influence for "
        "good or evil is inestimable. It is a power which can make or unmake kings, "
        "and which exercises an immense influence over the minds of men. It has its "
        "advantages, but it has its disadvantages, and when these latter preponderate "
        "the disadvantages are serious indeed."
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Normalize and tokenize text into a set of lowercase word tokens.

    Strips punctuation, lowercases, and filters tokens shorter than 3 characters
    to reduce noise from common function words (articles, prepositions).

    Args:
        text: Input text string.

    Returns:
        A set of normalized word tokens (length >= 3).
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return {t for t in tokens if len(t) >= 3}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Returns 0.0 when both sets are empty.

    Args:
        set_a: First token set.
        set_b: Second token set.

    Returns:
        Jaccard index in [0.0, 1.0].
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _parse_llm_judge_response(raw: str) -> dict[str, Any]:
    """Parse the LLM judge's JSON response into a structured result dict.

    Attempts JSON extraction from the raw response. Falls back to a safe
    default (confidence 0.0, no identified author) if parsing fails.

    Args:
        raw: Raw string response from the LLM.

    Returns:
        Dict with keys 'identified_author' (str | None) and 'confidence' (float).
    """
    # Try to extract a JSON object from the response
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            author = data.get("identified_author") or data.get("author")
            if author and str(author).strip().lower() in ("none", "null", "", "no"):
                author = None
            return {
                "identified_author": str(author) if author else None,
                "confidence": confidence,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: scan for a confidence number in the text
    confidence_match = re.search(
        r"(?:confidence|score)[:\s]+([0-9]+(?:\.[0-9]+)?)", raw, re.IGNORECASE
    )
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.0
    else:
        confidence = 0.0

    return {"identified_author": None, "confidence": confidence}


# ---------------------------------------------------------------------------
# Evaluator 1: LLM Judge
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM_PROMPT = """You are a literary style expert and confusability evaluator.
Your task is to read a text and determine whether it reads as if it was written by any
specific, identifiable real person — a named journalist, essayist, public intellectual,
or any other real individual whose writing style is publicly recognizable.

You must respond ONLY with a JSON object in this exact format:
{"identified_author": <string or null>, "confidence": <float between 0.0 and 1.0>}

Rules:
- If the text does NOT strongly resemble any specific real person's style, return:
  {"identified_author": null, "confidence": 0.0}
- If the text DOES strongly resemble a specific real person's style, return their name
  and your confidence (0.0 = no similarity, 1.0 = certain this person wrote it).
- Do NOT name generic movements, fictional characters, or organizations — only real
  identifiable individual humans.
- Be conservative: only flag clear, strong stylistic matches. Generic formal or
  conversational prose should score 0.0.
- Threshold for flagging: only return confidence >= 0.4 when you have a clear,
  specific match with distinctive and recognizable stylistic markers."""

_LLM_JUDGE_USER_PROMPT_TEMPLATE = """Read the following opinion piece excerpt and evaluate
whether it reads as if written by any specific, identifiable real person.

TEXT TO EVALUATE:
{text}

Respond ONLY with a JSON object:
{{"identified_author": <name string or null>, "confidence": <0.0-1.0 float>}}"""


def llm_judge_eval(text: str, client: Any) -> dict[str, Any]:
    """Evaluate whether the generated text reads like a specific identifiable author.

    Sends a prompted LLM request asking whether the text resembles any specific
    real person's writing style. Returns the identified author (if any) and a
    confidence score in [0.0, 1.0].

    The LLM judge is mockable: pass any object with a ``generate(system_prompt,
    user_prompt, max_tokens)`` method. In tests, pass a MagicMock configured to
    return a JSON string.

    Args:
        text: The generated opinion piece text to evaluate.
        client: An LLM client implementing the LLMClient protocol (has a
            ``generate(system_prompt, user_prompt, max_tokens)`` method).

    Returns:
        A dict with keys:
            - ``identified_author`` (str | None): Name of identified author, or None.
            - ``confidence`` (float): Confidence score in [0.0, 1.0].
    """
    user_prompt = _LLM_JUDGE_USER_PROMPT_TEMPLATE.format(text=text[:3000])

    try:
        raw_response = client.generate(
            system_prompt=_LLM_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=200,
        )
    except Exception:
        # If the LLM call fails, return a safe default (no confusability detected)
        return {"identified_author": None, "confidence": 0.0}

    result = _parse_llm_judge_response(raw_response)

    # Clamp confidence to valid range
    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

    return result


# ---------------------------------------------------------------------------
# Evaluator 2: Lexical Similarity Scorer
# ---------------------------------------------------------------------------


def lexical_similarity_eval(
    text: str,
    reference_corpus: list[str] | None = None,
) -> float:
    """Compute the maximum Jaccard token-overlap between text and a reference corpus.

    Compares the generated text's token set against each document in the
    reference corpus and returns the maximum overlap score. A high score
    indicates the text shares an unusual number of vocabulary tokens with one
    of the reference documents, which can be a signal of stylistic imitation.

    The reference corpus defaults to the module-level public-domain corpus
    (``_PUBLIC_DOMAIN_CORPUS``) when ``None`` is passed.

    Args:
        text: The generated opinion piece text to evaluate.
        reference_corpus: A list of reference document strings to compare
            against. Defaults to the built-in public-domain corpus when None.

    Returns:
        A float in [0.0, 1.0] representing the maximum Jaccard overlap between
        the text and any single document in the reference corpus. Returns 0.0
        if the corpus is empty or the text is empty.
    """
    if reference_corpus is None:
        reference_corpus = _PUBLIC_DOMAIN_CORPUS

    if not text.strip() or not reference_corpus:
        return 0.0

    text_tokens = _tokenize(text)

    max_similarity = 0.0
    for ref_doc in reference_corpus:
        ref_tokens = _tokenize(ref_doc)
        similarity = _jaccard_similarity(text_tokens, ref_tokens)
        if similarity > max_similarity:
            max_similarity = similarity

    return max_similarity


# ---------------------------------------------------------------------------
# Evaluator 3: Regression Set Checker
# ---------------------------------------------------------------------------


def regression_eval(
    text: str,
    regression_cases: list[dict[str, Any]],
) -> bool:
    """Check whether the generated text contains any previously flagged fragment.

    Iterates through all regression cases and tests whether the generated text
    contains the ``flagged_text_fragment`` as a substring. Returns True when the
    text is clean (no matches — the test PASSES), False when a flagged fragment
    is found (the test FAILS).

    When ``regression_cases`` is empty (the initial state of the regression set),
    this evaluator vacuously returns True for all inputs.

    Args:
        text: The generated opinion piece text to evaluate.
        regression_cases: A list of regression case dicts, each containing at
            minimum a ``flagged_text_fragment`` key with a string value.

    Returns:
        True if no flagged fragments are found in the text (passes).
        False if any flagged fragment is found in the text (fails).
    """
    if not regression_cases:
        return True

    for case in regression_cases:
        fragment = case.get("flagged_text_fragment", "")
        if fragment and fragment in text:
            return False

    return True


# ---------------------------------------------------------------------------
# ConfusabilityResult container
# ---------------------------------------------------------------------------


class ConfusabilityResult:
    """Container for all three evaluator results from a single confusability test.

    Attributes:
        mode_id: The rhetorical mode used.
        topic: The topic string used.
        stance_position: The stance position integer.
        intensity: The intensity float.
        llm_judge_author: Identified author from the LLM judge (or None).
        llm_judge_confidence: Confidence score from the LLM judge [0.0, 1.0].
        lexical_score: Jaccard overlap score from the lexical evaluator [0.0, 1.0].
        regression_passed: Whether the regression evaluator passed (True = clean).
    """

    def __init__(
        self,
        mode_id: str,
        topic: str,
        stance_position: int,
        intensity: float,
        llm_judge_author: str | None,
        llm_judge_confidence: float,
        lexical_score: float,
        regression_passed: bool,
    ) -> None:
        """Initialize a ConfusabilityResult.

        Args:
            mode_id: Rhetorical mode ID.
            topic: Topic string.
            stance_position: Stance position integer (-100 to +100).
            intensity: Rhetorical intensity float (0.0 to 1.0).
            llm_judge_author: Author name identified by LLM judge, or None.
            llm_judge_confidence: LLM judge confidence score [0.0, 1.0].
            lexical_score: Lexical similarity score [0.0, 1.0].
            regression_passed: True if regression check passed (no flagged matches).
        """
        self.mode_id = mode_id
        self.topic = topic
        self.stance_position = stance_position
        self.intensity = intensity
        self.llm_judge_author = llm_judge_author
        self.llm_judge_confidence = llm_judge_confidence
        self.lexical_score = lexical_score
        self.regression_passed = regression_passed

    @property
    def passed(self) -> bool:
        """Return True when ALL three evaluators are below their thresholds.

        Returns:
            True if the text passes all three confusability checks.
        """
        return (
            self.llm_judge_confidence < LLM_JUDGE_THRESHOLD
            and self.lexical_score < LEXICAL_SIMILARITY_THRESHOLD
            and self.regression_passed
        )

    def __repr__(self) -> str:
        """Return a debug-friendly representation of the result.

        Returns:
            A string summarizing all evaluator scores.
        """
        return (
            f"ConfusabilityResult("
            f"mode={self.mode_id!r}, topic={self.topic!r}, "
            f"pos={self.stance_position}, intensity={self.intensity}, "
            f"llm_confidence={self.llm_judge_confidence:.3f}, "
            f"lexical={self.lexical_score:.3f}, "
            f"regression={'pass' if self.regression_passed else 'FAIL'}, "
            f"overall={'PASS' if self.passed else 'FAIL'})"
        )
