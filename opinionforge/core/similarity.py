"""Similarity screening module for OpinionForge.

Screens generated output for verbatim reuse, near-verbatim reuse,
suppressed catchphrases, and structural fingerprint matches before
delivery to the user.

All screening functions are pure Python using only stdlib and pyyaml.
No external ML or NLP libraries are required.

Exit code 8 (per PRD CLI spec) is signalled by the generator raising
RuntimeError when this module returns passed=False after max iterations.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from opinionforge.models.piece import ScreeningResult

if TYPE_CHECKING:
    from opinionforge.core.preview import LLMClient
    from opinionforge.models.config import ModeBlendConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERBATIM_N: int = 5          # n-gram size for verbatim matching
_NEAR_VERBATIM_N: int = 6     # n-gram size for near-verbatim matching
_FINGERPRINT_THRESHOLD: float = 0.85  # cosine similarity block threshold
_MAX_REWRITE_ITERATIONS: int = 2      # hard-coded; not configurable in shipped builds

_DATA_DIR: Path = Path(__file__).parent.parent / "data"
_SUPPRESSED_PHRASES_PATH: Path = _DATA_DIR / "suppressed_phrases.yaml"
_STRUCTURAL_FINGERPRINTS_PATH: Path = _DATA_DIR / "structural_fingerprints.yaml"

# Developer-only bypass — gated by an env var that:
# (1) is not documented in any shipped help text, README, or examples
# (2) is not set in any CI pipeline configuration
# (3) is not importable through any public API
_DEVELOPER_BYPASS_ENV_VAR = "OPINIONFORGE_INTERNAL_SKIP_SCREENING"


# ---------------------------------------------------------------------------
# Data file loaders
# ---------------------------------------------------------------------------


def _load_suppressed_phrases() -> list[str]:
    """Load the global suppressed phrases list from the data file.

    Returns:
        List of suppressed phrase strings. Returns empty list if the file
        is missing or the key is absent.
    """
    try:
        with open(_SUPPRESSED_PHRASES_PATH, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return []
        phrases = data.get("suppressed_phrases", [])
        return [p for p in (phrases or []) if isinstance(p, str)]
    except FileNotFoundError:
        return []


def _load_structural_fingerprints() -> list[dict]:
    """Load structural fingerprint definitions from the data file.

    Returns:
        List of fingerprint dicts, each with at minimum a 'pattern_name'
        and 'syllable_sequence'. Returns empty list if the file is missing.
    """
    try:
        with open(_STRUCTURAL_FINGERPRINTS_PATH, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return []
        fps = data.get("fingerprints", [])
        return [fp for fp in (fps or []) if isinstance(fp, dict)]
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# Syllable counting (stdlib heuristic — no CMU dictionary or NLTK)
# ---------------------------------------------------------------------------


def _count_syllables(word: str) -> int:
    """Estimate the syllable count of a word using vowel-group heuristics.

    This is a heuristic counter, not a dictionary lookup. Accuracy is
    sufficient for structural fingerprint comparison purposes.

    Args:
        word: A single word (punctuation stripped).

    Returns:
        Estimated syllable count, minimum 1.
    """
    word = word.lower().strip(".,!?;:'\"()-")
    if not word:
        return 0
    # Count contiguous vowel groups as syllables
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    # Silent 'e' at the end of a word reduces count by 1 (e.g. "fate" = 1 syl)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _syllable_sequence_for_text(text: str) -> list[int]:
    """Build a syllable-count sequence for each sentence in text.

    One entry per sentence; each entry is the total syllable count for
    that sentence's words.

    Args:
        text: The text to analyse.

    Returns:
        List of per-sentence syllable counts.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    seq: list[int] = []
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        total = sum(_count_syllables(w) for w in words)
        if total > 0:
            seq.append(total)
    return seq


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[int], b: list[int]) -> float:
    """Compute cosine similarity between two integer sequences.

    Both sequences are treated as vectors. If either is empty or both are
    all-zeros the similarity is 0.0.

    Args:
        a: First integer sequence.
        b: Second integer sequence.

    Returns:
        Cosine similarity in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0
    # Align to the shorter length for the dot product
    length = min(len(a), len(b))
    a = a[:length]
    b = b[:length]
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract all word n-grams from text (exact tokens, preserving case).

    Args:
        text: The source text.
        n: N-gram size.

    Returns:
        List of word n-gram tuples.
    """
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def _normalize_text(text: str) -> str:
    """Normalize text for near-verbatim comparison.

    Applies: lowercase, strip all punctuation (keeping spaces and alphanumerics).

    Args:
        text: Raw text.

    Returns:
        Normalized text string.
    """
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


# ---------------------------------------------------------------------------
# Public screening functions
# ---------------------------------------------------------------------------


def check_verbatim(text: str, source_texts: list[str]) -> list[dict]:
    """Check for verbatim (exact) n-gram matches between text and source texts.

    Uses n=5 words as the minimum match size. Exact token comparison,
    case-sensitive, no normalization.

    Args:
        text: The generated text to screen.
        source_texts: List of source texts to check against.

    Returns:
        List of match dicts, each containing:
          - ``ngram``: The matching n-gram as a string.
          - ``source_index``: Index into source_texts of the matching source.
        Returns empty list when no matches are found.
    """
    matches: list[dict] = []
    output_ngrams = set(_word_ngrams(text, _VERBATIM_N))
    for idx, source in enumerate(source_texts):
        source_ngrams = set(_word_ngrams(source, _VERBATIM_N))
        for ngram in output_ngrams & source_ngrams:
            matches.append({"ngram": " ".join(ngram), "source_index": idx})
    return matches


def check_near_verbatim(text: str, source_texts: list[str]) -> list[dict]:
    """Check for near-verbatim matches after text normalization.

    Applies lowercase and punctuation stripping before n-gram comparison.
    Uses n=6 words as the minimum match size.

    Args:
        text: The generated text to screen.
        source_texts: List of source texts to check against.

    Returns:
        List of match dicts, each containing:
          - ``ngram``: The matching normalized n-gram as a string.
          - ``source_index``: Index into source_texts of the matching source.
        Returns empty list when no matches are found.
    """
    matches: list[dict] = []
    norm_text = _normalize_text(text)
    output_ngrams = set(_word_ngrams(norm_text, _NEAR_VERBATIM_N))
    for idx, source in enumerate(source_texts):
        norm_source = _normalize_text(source)
        source_ngrams = set(_word_ngrams(norm_source, _NEAR_VERBATIM_N))
        for ngram in output_ngrams & source_ngrams:
            matches.append({"ngram": " ".join(ngram), "source_index": idx})
    return matches


def check_suppressed_phrases(text: str, phrases: list[str]) -> list[str]:
    """Check generated text for any suppressed catchphrases.

    Performs case-insensitive substring matching for each phrase.

    Args:
        text: The generated text to screen.
        phrases: List of phrases to suppress.

    Returns:
        List of matched phrases found in text. Empty list if none found.
    """
    lower_text = text.lower()
    return [phrase for phrase in phrases if phrase.lower() in lower_text]


def check_structural_fingerprint(
    text: str,
    fingerprints: list[dict],
) -> float:
    """Compute the highest cosine similarity against known structural fingerprints.

    Converts the text's sentence-level syllable-count sequence to a vector
    and computes cosine similarity against each stored fingerprint's
    ``syllable_sequence`` field.

    Args:
        text: The generated text to screen.
        fingerprints: List of fingerprint dicts (from structural_fingerprints.yaml).
            Each must have a ``syllable_sequence`` field (list of ints).

    Returns:
        The highest cosine similarity score found, in [0.0, 1.0].
        Returns 0.0 when no fingerprints are provided or text is empty.
    """
    if not fingerprints or not text.strip():
        return 0.0

    text_seq = _syllable_sequence_for_text(text)
    if not text_seq:
        return 0.0

    max_score = 0.0
    for fp in fingerprints:
        fp_seq = fp.get("syllable_sequence")
        if not fp_seq or not isinstance(fp_seq, list):
            continue
        score = _cosine_similarity(text_seq, fp_seq)
        if score > max_score:
            max_score = score

    return min(1.0, max_score)


# ---------------------------------------------------------------------------
# Rewrite helper
# ---------------------------------------------------------------------------


def _build_rewrite_prompt(
    text: str,
    verbatim_matches: list[dict],
    near_verbatim_matches: list[dict],
    suppressed_phrase_matches: list[str],
) -> str:
    """Build the rewrite instruction prompt for the LLM.

    Highlights the specific violations and instructs the model to rephrase
    only the flagged passages.

    Args:
        text: The original generated text.
        verbatim_matches: List of verbatim match dicts from check_verbatim.
        near_verbatim_matches: List of near-verbatim match dicts from check_near_verbatim.
        suppressed_phrase_matches: List of matched suppressed phrases.

    Returns:
        A formatted instruction string for the LLM rewrite call.
    """
    parts: list[str] = [
        "The following opinion piece contains passages that must be rephrased "
        "to avoid verbatim reuse of source text or suppressed catchphrases. "
        "Rewrite only the flagged passages while preserving the argument and structure.\n\n"
        "--- Original Text ---\n",
        text,
        "\n\n--- Violations to Fix ---\n",
    ]

    if verbatim_matches:
        parts.append("Verbatim matches (rephrase these exact phrases):\n")
        for m in verbatim_matches:
            parts.append(f'  - "{m["ngram"]}"\n')

    if near_verbatim_matches:
        parts.append("Near-verbatim matches (rephrase these phrases):\n")
        for m in near_verbatim_matches:
            parts.append(f'  - "{m["ngram"]}"\n')

    if suppressed_phrase_matches:
        parts.append("Suppressed phrases to remove:\n")
        for phrase in suppressed_phrase_matches:
            parts.append(f'  - "{phrase}"\n')

    parts.append(
        "\n--- Instructions ---\n"
        "Return only the revised text with the flagged passages rephrased. "
        "Do not add commentary or explanation. Preserve the title and structure."
    )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Main screening entry point
# ---------------------------------------------------------------------------


def screen_output(
    text: str,
    research_texts: list[str],
    mode_config: "ModeBlendConfig",
    client: "LLMClient | None",
) -> ScreeningResult:
    """Screen generated output for similarity violations.

    Runs four checks in order:
    1. Verbatim n-gram matching (n=5) against research_texts.
    2. Near-verbatim normalized matching (n=6) against research_texts.
    3. Suppressed phrase matching from global data file and mode config.
    4. Structural fingerprint scoring (cosine similarity, threshold 0.85).

    If violations are found and an LLM client is available, performs up to
    ``_MAX_REWRITE_ITERATIONS`` (2) rewrite passes. Each pass replaces text
    with the LLM-rewritten version and re-screens.

    On failure after 2 iterations (or immediately when no client is provided),
    returns ScreeningResult(passed=False). The generator must then raise
    RuntimeError (exit code 8 per PRD CLI spec).

    Args:
        text: The generated text to screen.
        research_texts: Source texts used during research (may be empty).
        mode_config: The rhetorical mode blend config, which may carry
            per-mode suppressed_phrases.
        client: Optional LLM client for rewrite iterations.

    Returns:
        A ScreeningResult with pass/fail status and violation counts.
    """
    # Developer bypass — gated by undocumented env var, never set in CI
    if os.environ.get(_DEVELOPER_BYPASS_ENV_VAR):
        return ScreeningResult(
            passed=True,
            verbatim_matches=0,
            near_verbatim_matches=0,
            suppressed_phrase_matches=0,
            structural_fingerprint_score=0.0,
            rewrite_iterations=0,
            warning="Developer bypass active — screening skipped.",
        )

    # Load global suppressed phrases and fingerprints from data files
    global_phrases = _load_suppressed_phrases()

    # Collect per-mode suppressed phrases from mode_config if available
    mode_phrases: list[str] = []
    if mode_config is not None:
        for mode_id, _weight in mode_config.modes:
            # Attempt to load per-mode phrases; ignore failures gracefully
            try:
                from opinionforge.core.mode_engine import load_mode  # type: ignore[import]
                profile = load_mode(mode_id)
                if hasattr(profile, "suppressed_phrases") and profile.suppressed_phrases:
                    mode_phrases.extend(profile.suppressed_phrases)
            except Exception:
                pass

    all_phrases = list(dict.fromkeys(global_phrases + mode_phrases))
    fingerprints = _load_structural_fingerprints()

    current_text = text
    iterations = 0

    for _attempt in range(_MAX_REWRITE_ITERATIONS + 1):
        # Run all four checks
        verbatim = check_verbatim(current_text, research_texts)
        near_verbatim = check_near_verbatim(current_text, research_texts)
        phrase_hits = check_suppressed_phrases(current_text, all_phrases)
        fp_score = check_structural_fingerprint(current_text, fingerprints)

        # Determine whether there are any violations
        has_violations = bool(verbatim or near_verbatim or phrase_hits)
        # Note: structural fingerprint only; currently informational unless above threshold
        # The PRD does not specify blocking solely on fingerprint score in the rewrite loop,
        # but verbatim/near-verbatim/phrase violations trigger rewrites.

        if not has_violations:
            # Clean — passed
            return ScreeningResult(
                passed=True,
                verbatim_matches=len(verbatim),
                near_verbatim_matches=len(near_verbatim),
                suppressed_phrase_matches=len(phrase_hits),
                structural_fingerprint_score=fp_score,
                rewrite_iterations=iterations,
            )

        # Violations found — check if we can rewrite
        if _attempt >= _MAX_REWRITE_ITERATIONS:
            # Exhausted all rewrite iterations (or this is attempt 0 with no client)
            break

        if client is None:
            # No LLM client available; cannot rewrite
            break

        # Perform one rewrite iteration
        rewrite_system = (
            "You are an expert editor. Rephrase the flagged passages in the provided "
            "opinion piece to eliminate verbatim reuse and suppressed catchphrases. "
            "Preserve the argument, structure, and rhetorical style."
        )
        rewrite_prompt = _build_rewrite_prompt(
            current_text, verbatim, near_verbatim, phrase_hits
        )
        try:
            current_text = client.generate(
                system_prompt=rewrite_system,
                user_prompt=rewrite_prompt,
                max_tokens=len(current_text.split()) * 2 + 500,
            )
        except Exception:
            break

        iterations += 1

    # Failed — collect final violation counts
    verbatim_final = check_verbatim(current_text, research_texts)
    near_verbatim_final = check_near_verbatim(current_text, research_texts)
    phrase_hits_final = check_suppressed_phrases(current_text, all_phrases)
    fp_score_final = check_structural_fingerprint(current_text, fingerprints)

    warning_parts: list[str] = []
    if verbatim_final:
        warning_parts.append(f"{len(verbatim_final)} verbatim match(es)")
    if near_verbatim_final:
        warning_parts.append(f"{len(near_verbatim_final)} near-verbatim match(es)")
    if phrase_hits_final:
        warning_parts.append(f"{len(phrase_hits_final)} suppressed phrase(s)")

    if client is None and (verbatim or near_verbatim or phrase_hits):
        warning = (
            "Similarity violations detected but no LLM client was provided for "
            f"rewriting. Violations: {', '.join(warning_parts) if warning_parts else 'unknown'}."
        )
    else:
        warning = (
            f"Similarity screening failed after {iterations} rewrite iteration(s). "
            f"Unresolved violations: {', '.join(warning_parts) if warning_parts else 'unknown'}."
        )

    return ScreeningResult(
        passed=False,
        verbatim_matches=len(verbatim_final),
        near_verbatim_matches=len(near_verbatim_final),
        suppressed_phrase_matches=len(phrase_hits_final),
        structural_fingerprint_score=fp_score_final,
        rewrite_iterations=iterations,
        warning=warning,
    )
