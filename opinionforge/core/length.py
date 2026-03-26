"""Length control logic: maps preset names to word counts and validates custom lengths.

Provides resolve_length() for normalizing user input to a target word count,
and get_length_instructions() for producing LLM-ready writing instructions
appropriate to the target length.
"""

from __future__ import annotations


# Sprint-4 preset definitions
LENGTH_PRESETS: dict[str, int] = {
    "short": 500,
    "standard": 750,
    "long": 1200,
    "essay": 2500,
    "feature": 5000,
}

MIN_WORD_COUNT = 200
MAX_WORD_COUNT = 8000


def resolve_length(length_input: str) -> int:
    """Resolve a preset name or numeric string to a target word count.

    Accepts preset names ('short', 'standard', 'long', 'essay', 'feature')
    or integer strings representing custom word counts.

    Args:
        length_input: A preset name or integer string.

    Returns:
        The target word count as an integer.

    Raises:
        ValueError: If the preset name is unrecognised, or the custom count
            falls outside the 200-8000 allowed range.
    """
    key = length_input.strip().lower()

    if key in LENGTH_PRESETS:
        return LENGTH_PRESETS[key]

    # Try parsing as an integer string
    try:
        count = int(key)
    except ValueError:
        valid = ", ".join(sorted(LENGTH_PRESETS.keys()))
        raise ValueError(
            f"Unknown length preset '{length_input}'. "
            f"Valid presets: {valid}. Or provide a word count between "
            f"{MIN_WORD_COUNT} and {MAX_WORD_COUNT}."
        ) from None

    if count < MIN_WORD_COUNT:
        raise ValueError(
            f"Word count {count} is below the minimum of {MIN_WORD_COUNT}."
        )
    if count > MAX_WORD_COUNT:
        raise ValueError(
            f"Word count {count} exceeds the maximum of {MAX_WORD_COUNT}."
        )

    return count


def get_length_instructions(target: int) -> str:
    """Return LLM-ready writing instructions for the given target word count.

    Short pieces (<= 800 words) get instructions emphasising a single
    strongest argument.  Long pieces (>= 2500 words) get instructions for
    section breaks and fully developed arguments.

    Args:
        target: The target word count (should already be validated).

    Returns:
        An instruction string suitable for inclusion in a system prompt.
    """
    tolerance = int(target * 0.1)

    if target <= 800:
        structure = (
            "This is a short piece. Focus on the single strongest argument. "
            "Get to the thesis immediately with minimal preamble. "
            "Every sentence must earn its place."
        )
    elif target <= 1200:
        structure = (
            "This is a standard column. Present the thesis clearly, develop "
            "2-3 supporting arguments with evidence, and close with a strong "
            "conclusion."
        )
    elif target < 2500:
        structure = (
            "This is an extended essay. Develop multiple arguments with "
            "supporting evidence. Include a clear thesis, substantive body "
            "sections, and a compelling conclusion."
        )
    else:
        # >= 2500
        structure = (
            "This is a long-form piece. Use section breaks and subheadings "
            "where stylistically appropriate. Develop each argument fully "
            "with rich evidence, narrative examples, and nuanced analysis. "
            "Build a sustained, layered argument across multiple sections."
        )

    return (
        f"Target word count: {target} words "
        f"(acceptable range: {target - tolerance}-{target + tolerance}). "
        f"{structure}"
    )
