"""Text processing utilities: word counting, text truncation, and citation formatting.

Provides helper functions used across the research and generation pipeline.
"""

from __future__ import annotations

from opinionforge.models.piece import SourceCitation


def word_count(text: str) -> int:
    """Return the number of words in a text string.

    Handles edge cases: empty strings and whitespace-only strings return 0.

    Args:
        text: The text to count words in.

    Returns:
        The number of words.
    """
    if not text or not text.strip():
        return 0
    return len(text.split())


def truncate_text(text: str, max_words: int) -> str:
    """Truncate text to a maximum number of words at a word boundary.

    If the text is already at or under the limit, it is returned unchanged.

    Args:
        text: The text to truncate.
        max_words: Maximum number of words to keep.

    Returns:
        The (possibly truncated) text.
    """
    if max_words <= 0:
        raise ValueError(f"max_words must be a positive integer, got {max_words}.")
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def format_citations(sources: list[SourceCitation]) -> str:
    """Format a list of source citations into the PRD citation appendix format.

    Each citation is formatted as:
        "Claim text" -- [Source Name](URL), accessed YYYY-MM-DD

    Args:
        sources: List of SourceCitation objects to format.

    Returns:
        A formatted string with all citations, one per line.
    """
    if not sources:
        return ""

    lines: list[str] = []
    for source in sources:
        accessed = source.accessed_at.strftime("%Y-%m-%d")
        line = (
            f'"{source.claim}" '
            f"-- [{source.source_name}]({source.source_url}), "
            f"accessed {accessed}"
        )
        lines.append(line)

    return "\n".join(lines)
